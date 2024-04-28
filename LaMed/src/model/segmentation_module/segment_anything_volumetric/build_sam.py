# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial
from pathlib import Path
import urllib.request
import torch

from .modeling import (
    ImageEncoderViT,
    MaskDecoder,
    PromptEncoder,
    Sam,
    TwoWayTransformer,
)
import numpy as np
from .modeling.image_encoder_swin import SwinTransformer
from monai.networks.nets import ViT
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT

from monai.utils import ensure_tuple_rep, optional_import


"""
Examples::
            # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
            >>> net = SwinUNETR(img_size=(96,96,96), in_channels=1, out_channels=4, feature_size=48)
            # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
            >>> net = SwinUNETR(img_size=(128,128,128), in_channels=4, out_channels=3, depths=(2,4,2,2))
            # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
            >>> net = SwinUNETR(img_size=(96,96), in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)
"""

def build_sam_vit_3d(args, checkpoint=None):
    print('build_sam_vit_3d...')
    return _build_sam(
        image_encoder_type='vit',
        embed_dim=768,
        patch_size=args.patch_size,
        checkpoint=checkpoint,
        image_size=args.image_size,
    )

sam_model_registry = {
    "vit": build_sam_vit_3d,
}


def _build_sam(
    image_encoder_type,
    embed_dim,
    patch_size,
    checkpoint,
    image_size,
):
    mlp_dim = 3072
    num_layers = 12
    num_heads = 12
    pos_embed = 'perceptron'
    dropout_rate = 0.0
    
    image_encoder=ViT(
        in_channels=1,
        img_size=image_size,
        patch_size=patch_size,
        hidden_size=embed_dim,
        mlp_dim=mlp_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        pos_embed=pos_embed,
        classification=False,
        dropout_rate=dropout_rate,
    )
    image_embedding_size = [int(item) for item in (np.array(image_size) / np.array(patch_size))]

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location='cpu')['state_dict']
            encoder_dict = {k.replace('model.encoder.', ''): v for k, v in state_dict.items() if 'model.encoder.' in k}
        image_encoder.load_state_dict(encoder_dict)
        print(f'===> image_encoder.load_param: {checkpoint}')
    sam = Sam(
        image_encoder=image_encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=embed_dim,
            image_embedding_size=image_embedding_size,
            input_image_size=image_size,
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            image_encoder_type=image_encoder_type,
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            image_size=np.array(image_size),
            patch_size=np.array(patch_size),
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    return sam
