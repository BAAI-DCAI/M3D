import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SegVol(nn.Module):
    def __init__(self, 
                image_encoder, 
                mask_decoder,
                prompt_encoder,
                roi_size,
                patch_size,
                ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.feat_shape = np.array(roi_size)/np.array(patch_size)

    def forward(self, image, text_emb=None, text=None, boxes=None, points=None):
        bs = image.shape[0]
        img_shape = (image.shape[2], image.shape[3], image.shape[4])
        image_embedding, _ = self.image_encoder(image)

        image_embedding = image_embedding.transpose(1, 2).view(bs, -1, 
            int(self.feat_shape[0]), int(self.feat_shape[1]), int(self.feat_shape[2]))

        logits = self.forward_decoder(image_embedding, img_shape, text_emb=text_emb, text=text, boxes=boxes, points=points)

        return logits

    def forward_decoder(self, image_embedding, img_shape, text_emb=None, text=None, boxes=None, points=None):
        text_embedding = text_emb
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            text_embedding=text_embedding,
        )

        dense_pe = self.prompt_encoder.get_dense_pe()

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            text_embedding = text_embedding,
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
          )
        logits = F.interpolate(low_res_masks, size=img_shape, mode='trilinear', align_corners=False)

        return logits