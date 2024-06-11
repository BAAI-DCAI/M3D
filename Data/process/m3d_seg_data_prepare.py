import monai.transforms as mtf
import torch
from scipy import sparse
import ast
from monai.data import set_track_meta
import os
import numpy as np
import json
from multiprocessing import Pool

source_path = "PATH/data/M3D_Seg/"
target_path = "PATH/data/M3D_Seg_npy/"

transform = mtf.Compose(
    [
        mtf.EnsureTyped(keys=["image", "label"], track_meta=False),
        mtf.CropForegroundd(keys=["image", "label"], source_key="image"),
        mtf.Resized(keys=["image", "label"], spatial_size=[32,256,256],
                    mode=['bilinear', 'nearest']),  # trilinear
    ]
)

set_track_meta(False)

folders = [f for f in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, f))]

def process_data(dir):
    dataset_path = os.path.join(source_path, dir)
    json_path = os.path.join(dataset_path, dir + ".json")

    with open(json_path, 'r') as f:
        data = json.load(f)

    train_list = data["train"]
    test_list = data["test"]

    new_train_list = []
    for pair in train_list:
        image_path = os.path.join(source_path, pair["image"])
        mask_path = os.path.join(source_path, pair["label"])

        mask_shape = ast.literal_eval(mask_path.split('.')[-2].split('_')[-1])  # C*512*512*265
        image_array = np.load(image_path)  # 1*512*512*265
        allmatrix_sp = sparse.load_npz(mask_path)
        mask_array = allmatrix_sp.toarray().reshape(mask_shape)

        # reshape C*D*H*W
        image_array = np.swapaxes(image_array, -1, -3)
        mask_array = np.swapaxes(mask_array, -1, -3)

        # normalization
        image_array = image_array - image_array.min()
        image_array = image_array / np.clip(image_array.max(), a_min=1e-8, a_max=None)

        item = {
            'image': image_array,
            'label': mask_array,
        }

        # crop and resize
        items = transform(item)

        image = items['image'].numpy()
        mask = items['label'].numpy().astype(bool)

        target_image_path = os.path.join(target_path, pair["image"])
        target_image_folder = os.path.dirname(target_image_path)
        os.makedirs(target_image_folder, exist_ok=True)

        target_mask_path = os.path.join(target_path, pair["label"])
        target_mask_folder = os.path.join(os.path.dirname(target_mask_path), "masks")
        os.makedirs(target_mask_folder, exist_ok=True)

        np.save(target_image_path, image)
        for i in range(mask.shape[0]):
            if mask[i:i + 1].sum() != 0:
                np.save(os.path.join(target_mask_folder, "mask_" + str(i) + ".npy"), mask[i:i + 1])
                new_train_list.append({
                    "image": target_image_path,
                    "label": os.path.join(target_mask_folder, "mask_" + str(i) + ".npy")
                })
            else:
                # save negative samples
                mask_empty = np.zeros((1,1,1,1), dtype=np.int8)
                np.save(os.path.join(target_mask_folder, "mask_" + str(i) + ".npy"), mask_empty)
                new_train_list.append({
                    "image": target_image_path,
                    "label": os.path.join(target_mask_folder, "mask_" + str(i) + ".npy")
                })


    new_test_list = []
    for pair in test_list:
        image_path = os.path.join(source_path, pair["image"])
        mask_path = os.path.join(source_path, pair["label"])

        mask_shape = ast.literal_eval(mask_path.split('.')[-2].split('_')[-1])  # C*512*512*265
        image_array = np.load(image_path)  # 1*512*512*265
        allmatrix_sp = sparse.load_npz(mask_path)
        mask_array = allmatrix_sp.toarray().reshape(mask_shape)

        # reshape C*D*H*W
        image_array = np.swapaxes(image_array, -1, -3)
        mask_array = np.swapaxes(mask_array, -1, -3)

        # normalization
        image_array = image_array - image_array.min()
        image_array = image_array / np.clip(image_array.max(), a_min=1e-8, a_max=None)

        item = {
            'image': image_array,
            'label': mask_array,
        }

        # crop and resize
        items = transform(item)

        image = items['image'].numpy()
        mask = items['label'].numpy().astype(bool)

        target_image_path = os.path.join(target_path, pair["image"])
        target_image_folder = os.path.dirname(target_image_path)
        os.makedirs(target_image_folder, exist_ok=True)

        target_mask_path = os.path.join(target_path, pair["label"])
        target_mask_folder = os.path.join(os.path.dirname(target_mask_path), "masks")
        os.makedirs(target_mask_folder, exist_ok=True)

        np.save(target_image_path, image)
        for i in range(mask.shape[0]):
            if mask[i:i + 1].sum() != 0:
                np.save(os.path.join(target_mask_folder, "mask_" + str(i) + ".npy"), mask[i:i + 1])
                new_test_list.append({
                    "image": target_image_path,
                    "label": os.path.join(target_mask_folder, "mask_" + str(i) + ".npy")
                })
            else:
                mask_empty = np.zeros((1,1,1,1), dtype=np.int)
                np.save(os.path.join(target_mask_folder, "mask_" + str(i) + ".npy"), mask_empty)
                new_test_list.append({
                    "image": target_image_path,
                    "label": os.path.join(target_mask_folder, "mask_" + str(i) + ".npy")
                })

    data["train"] = new_train_list
    data["test"] = new_test_list
    with open(os.path.join(target_path, dir, dir + ".json"), 'w') as f:
        json.dump(data, f, indent=4)

with Pool(processes=8) as pool:
    pool.map(process_data, folders)