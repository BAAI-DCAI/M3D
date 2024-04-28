from typing import List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..lamed_arch import LamedMetaModel, LamedMetaForCausalLM


class LamedConfig(LlamaConfig):
    model_type = "lamed_llama"


class LamedLlamaModel(LamedMetaModel, LlamaModel):
    config_class = LamedConfig
    def __init__(self, config: LlamaConfig):
        super(LamedLlamaModel, self).__init__(config)


class LamedLlamaForCausalLM(LamedMetaForCausalLM, LlamaForCausalLM):
    config_class = LamedConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LamedLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            images: Optional[torch.FloatTensor] = None,
            input_ids: torch.LongTensor = None,
            labels: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            segs: Optional[torch.FloatTensor] = None,

            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        input_ids_pre = input_ids

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
            )

        try:
            seg_ids = torch.nonzero(torch.sum(segs, dim=(1, 2, 3, 4))).flatten().tolist()
        except:
            seg_ids = []

        if self.get_model().seg_enable and seg_ids:
            outputs = super().forward(
                                    input_ids=input_ids,
                                    inputs_embeds=inputs_embeds,
                                    attention_mask=attention_mask,
                                    labels=labels,
                                    output_hidden_states=True,

                                    position_ids=position_ids,
                                    past_key_values=past_key_values,
                                    use_cache=use_cache,
                                    output_attentions=output_attentions,
                                    return_dict=return_dict
                                )

            output_hidden_states = outputs.hidden_states

            last_hidden_state = output_hidden_states[-1]

            seg_token_mask = input_ids_pre[:, 1:] == self.config.seg_token_id
            seg_token_mask = torch.cat(
                [
                    seg_token_mask,
                    torch.zeros((seg_token_mask.shape[0], 1), dtype=seg_token_mask.dtype).cuda(),
                ],
                dim=1,
            )

            seg_prompts = []
            for i in seg_ids:
                if torch.sum(seg_token_mask[i]) == 1:
                    seg_token = last_hidden_state[i][seg_token_mask[i]]
                    seg_prompt = self.get_model().seg_projector(seg_token)
                elif torch.sum(seg_token_mask[i]) > 1:
                    seg_tokens = last_hidden_state[i][seg_token_mask[i]]
                    seg_token = torch.mean(seg_tokens, dim=0, keepdim=True)
                    seg_prompt = self.get_model().seg_projector(seg_token)
                else:
                    seg_prompt = torch.zeros([1, self.config.mm_hidden_size], dtype=last_hidden_state.dtype,
                                             device=last_hidden_state.device)
                seg_prompts.append(seg_prompt)

            seg_prompts = torch.cat(seg_prompts, dim=0)
            logits = self.get_model().seg_module(images[seg_ids], text_emb=seg_prompts)
            loss_dice = self.get_model().dice_loss(logits, segs[seg_ids])
            loss_bce = self.get_model().bce_loss(logits, segs[seg_ids])
            seg_loss = loss_dice + loss_bce
            outputs.loss = outputs.loss + seg_loss
            return outputs
        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )


    @torch.no_grad()
    def generate(
        self,
        images: Optional[torch.Tensor] = None,
        inputs: Optional[torch.Tensor] = None,
        seg_enable: bool = False,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor, Any]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        if seg_enable:
            outputs = super().generate(
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                return_dict_in_generate=True,
                **kwargs
            )

            output_hidden_states = outputs.hidden_states
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.config.seg_token_id

            last_tensors = [tuple[-1] for tuple in output_hidden_states]
            last_hidden_state = torch.cat(last_tensors[1:], dim=1)

            seg_prompts = []
            noseg_ids = []
            for i in range(len(seg_token_mask)):
                if torch.sum(seg_token_mask[i]) == 1:
                    seg_token = last_hidden_state[i][seg_token_mask[i]]
                    seg_prompt = self.get_model().seg_projector(seg_token)
                elif torch.sum(seg_token_mask[i]) > 1:
                    seg_tokens = last_hidden_state[i][seg_token_mask[i]]
                    seg_token = torch.mean(seg_tokens, dim=0, keepdim=True)
                    seg_prompt = self.get_model().seg_projector(seg_token)
                else:
                    noseg_ids.append(i)
                    seg_prompt = torch.zeros([1, self.config.mm_hidden_size], dtype=last_hidden_state.dtype,
                                             device=last_hidden_state.device)
                seg_prompts.append(seg_prompt)

            seg_prompts = torch.cat(seg_prompts, dim=0)
            logits = self.get_model().seg_module(images, seg_prompts)
            logits[noseg_ids] = -torch.inf

            return output_ids, logits
        else:
            output_ids = super().generate(
                inputs_embeds=inputs_embeds,
                **kwargs
            )
            return output_ids


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        return inputs


AutoConfig.register("lamed_llama", LamedConfig)
AutoModelForCausalLM.register(LamedConfig, LamedLlamaForCausalLM)