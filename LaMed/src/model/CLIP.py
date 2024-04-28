import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig, BertModel, AutoConfig, AutoModel
from LaMed.src.model.multimodal_encoder.vit import ViT
from LaMed.src.utils.dist_utils import gather_features


class M3DCLIPConfig(PretrainedConfig):
    model_type = "m3d_clip"

    def __init__(
        self,
        language_model_name_or_path: str = "",
        local_loss: bool = False,
        gather_loss: bool = True,
        in_channels: int = 1,
        img_size: tuple = (32, 256, 256),
        patch_size: tuple = (4, 16, 16),
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        dropout_rate: float = 0,
        spatial_dims: int = 3,
        max_text_len: int = 128,
        vocab_size: int = 30522,
        **kwargs,
    ):
        self.language_model_name_or_path = language_model_name_or_path
        self.in_channels = in_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pos_embed = pos_embed
        self.dropout_rate = dropout_rate
        self.spatial_dims = spatial_dims
        self.local_loss = local_loss
        self.gather_loss = gather_loss
        self.max_text_len = max_text_len
        self.vocab_size = vocab_size
        super().__init__(**kwargs)




class M3DCLIP(PreTrainedModel):
    config_class = M3DCLIPConfig

    def __init__(self, config):
        super().__init__(config)
        self.vision_encoder = ViT(
            in_channels=config.in_channels,
            img_size=config.img_size,
            patch_size=config.patch_size,
            hidden_size=config.hidden_size,
            mlp_dim=config.mlp_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            pos_embed=config.pos_embed,
            dropout_rate=config.dropout_rate,
            spatial_dims=config.spatial_dims,
            classification=True,
        )

        self.language_encoder = BertModel.from_pretrained(config.language_model_name_or_path)

        self.mm_vision_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.mm_language_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.local_loss = config.local_loss
        self.gather_loss = config.gather_loss

    def encode_image(self, image):
        image_feats, _ = self.vision_encoder(image)
        image_feats = self.mm_vision_proj(image_feats)
        image_feats = F.normalize(image_feats, dim=-1)

        return image_feats

    def encode_text(self, input_id, attention_mask):
        text_feats = self.language_encoder(input_id, attention_mask=attention_mask)["last_hidden_state"]
        text_feats = self.mm_language_proj(text_feats)
        text_feats = F.normalize(text_feats, dim=-1)

        return text_feats


    def forward(self, images, input_ids, attention_mask, labels, **kwargs):
        image_features = self.encode_image(images)[:, 0]
        text_features = self.encode_text(input_ids, attention_mask)[:, 0]

        if self.gather_loss:
            all_image_features, all_text_features = gather_features(image_features, text_features)
            if self.local_loss:
                logits_per_image = self.logit_scale * image_features @ all_text_features.T
                logits_per_text = self.logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = self.logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = self.logit_scale * image_features @ text_features.T
            logits_per_text = self.logit_scale * text_features @ image_features.T

        loss = (
                           F.cross_entropy(logits_per_image, labels) +
                           F.cross_entropy(logits_per_text, labels)
                   ) / 2

        ret = {
            "loss": loss,
            "logits": (logits_per_image + logits_per_text) / 2.0,
        }

        return ret

AutoConfig.register("m3d_clip", M3DCLIPConfig)
AutoModel.register(M3DCLIPConfig, M3DCLIP)