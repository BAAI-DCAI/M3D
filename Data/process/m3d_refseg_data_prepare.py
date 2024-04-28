import os
import nibabel as nib
import numpy as np
import monai.transforms as mtf
import shutil

root_dir = "PATH/M3D_RefSeg/"
output_dir = "PATH/M3D_RefSeg_npy/"


transforms = mtf.Compose([
    mtf.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_max=1.0, b_min=0.0, clip=True),
    mtf.CropForegroundd(keys=["image", "seg"], source_key="image"),
    mtf.Resized(keys=["image", "seg"], spatial_size=[32,256,256],
                mode=['trilinear', 'nearest']),

])


for item in os.listdir(root_dir):
    item_path = os.path.join(root_dir, item)
    if os.path.isdir(item_path):
        ct_file = os.path.join(item_path, "ct.nii.gz")
        mask_file = os.path.join(item_path, "mask.nii.gz")
        if os.path.exists(ct_file) and os.path.exists(mask_file):
            ct_image = nib.load(ct_file).get_fdata().transpose(2, 0, 1)[np.newaxis, ...]
            mask_image = nib.load(mask_file).get_fdata().transpose(2, 0, 1)[np.newaxis, ...]

            pair = {
                "image": ct_image,
                "seg": mask_image,
            }

            items = transforms(pair)
            image = items['image']
            seg = items['seg']

            output_item_dir = os.path.join(output_dir, item)
            os.makedirs(output_item_dir, exist_ok=True)

            np.save(os.path.join(output_item_dir, "ct.npy"), image)
            np.save(os.path.join(output_item_dir, "mask.npy"), seg)

            shutil.copyfile(item_path+"/text.json", output_item_dir+"/text.json")

            print(f"Transformed and saved: {item}")
        else:
            print(f"Missing ct.nii.gz or mask.nii.gz in: {item}")

print("Transformation complete.")
