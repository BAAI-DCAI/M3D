from .segment_anything_volumetric import sam_model_registry
from .segvol import SegVol


def build_segmentation_module(config, **kwargs):
    segmentation_module = getattr(config, 'segmentation_module')
    if 'segvol' in segmentation_module.lower():
        sam_model = sam_model_registry['vit'](args=config, checkpoint=None)
        seg_model = SegVol(
            image_encoder=sam_model.image_encoder,
            mask_decoder=sam_model.mask_decoder,
            prompt_encoder=sam_model.prompt_encoder,
            roi_size=config.image_size,
            patch_size=config.patch_size,
        )
        return seg_model
    else:
        raise ValueError(f'Unknown segmentation module: {segmentation_module}')


