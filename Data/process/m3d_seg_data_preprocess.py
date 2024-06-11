import os
import numpy as np
import multiprocessing
import argparse
from scipy import sparse
from sklearn.model_selection import train_test_split
import json

join = os.path.join

from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    Orientationd,
)


def set_parse():
    # %% set up parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-category",
                        default=['liver', 'right kidney', 'spleen', 'pancreas', 'aorta', 'inferior vena cava',
                                 'right adrenal gland', 'left adrenal gland', 'gallbladder', 'esophagus', 'stomach',
                                 'duodenum', 'left kidney'], type=list)
    parser.add_argument("-image_dir", default="", type=str, required=True)
    parser.add_argument("-label_dir", default="", type=str, required=True)
    parser.add_argument("-dataset_code", default="0025", type=str, required=True)
    parser.add_argument("-save_root", default="/mnt/hpfs/baaidcai/baifan/data/M3D_Seg/", type=str, required=True)
    parser.add_argument("-test_ratio", default=0, type=float, required=True)

    args = parser.parse_args()
    return args


args = set_parse()

# get ct&gt dir
image_list_all = [item for item in sorted(os.listdir(args.image_dir))]
label_list_all = [item for item in sorted(os.listdir(args.label_dir))]
assert len(image_list_all) == len(label_list_all)
print('dataset size ', len(image_list_all))

# build dataset
data_path_list_all = []
for idx in range(len(image_list_all)):
    img_path = join(args.image_dir, image_list_all[idx])
    label_path = join(args.label_dir, label_list_all[idx])
    name = image_list_all[idx].split('.')[0]
    info = (idx, name, img_path, label_path)
    data_path_list_all.append(info)

img_loader = Compose(
    [
        LoadImaged(keys=['image', 'label']),
        AddChanneld(keys=['image', 'label']),
        # Orientationd(keys=['image', 'label'], axcodes="RAS"),
    ]
)

# save
save_path = join(args.save_root, args.dataset_code)
os.makedirs(save_path, exist_ok=True)

# ct_save_path = join(save_path, 'ct')
# gt_save_path = join(save_path, 'gt')
# if not os.path.exists(ct_save_path):
#     os.makedirs(ct_save_path)
# if not os.path.exists(gt_save_path):
#     os.makedirs(gt_save_path)

# exist file:
exist_file_list = os.listdir(save_path)
print('exist_file_list ', exist_file_list)


def normalize(ct_narray):
    ct_voxel_ndarray = ct_narray.copy()
    ct_voxel_ndarray = ct_voxel_ndarray.flatten()
    # for all data
    thred = np.mean(ct_voxel_ndarray)
    voxel_filtered = ct_voxel_ndarray[(ct_voxel_ndarray > thred)]
    # for foreground data
    upper_bound = np.percentile(voxel_filtered, 99.95)
    lower_bound = np.percentile(voxel_filtered, 00.05)
    mean = np.mean(voxel_filtered)
    std = np.std(voxel_filtered)
    ### transform ###
    ct_narray = np.clip(ct_narray, lower_bound, upper_bound)
    ct_narray = (ct_narray - mean) / max(std, 1e-8)
    return ct_narray


def run(info):
    idx, file_name, case_path, label_path = info

    item = {}
    if file_name in exist_file_list:
        print(file_name + ' exist, skip')
        return
    print('process ', idx, '---', file_name)
    # generate ct_voxel_ndarray
    item_load = {
        'image': case_path,
        'label': label_path,
    }
    item_load = img_loader(item_load)
    ct_voxel_ndarray = item_load['image']
    gt_voxel_ndarray = item_load['label']

    ct_shape = ct_voxel_ndarray.shape
    item['image'] = ct_voxel_ndarray

    # generate gt_voxel_ndarray
    gt_voxel_ndarray = np.array(gt_voxel_ndarray).squeeze()
    present_categories = np.unique(gt_voxel_ndarray)
    gt_masks = []
    for cls_idx in range(len(args.category)):
        cls = cls_idx + 1
        if cls not in present_categories:
            gt_voxel_ndarray_category = np.zeros(ct_shape)
            gt_masks.append(gt_voxel_ndarray_category)
            print('case {} ==> zero category '.format(idx) + args.category[cls_idx])
            print(gt_voxel_ndarray_category.shape)
        else:
            gt_voxel_ndarray_category = gt_voxel_ndarray.copy()
            gt_voxel_ndarray_category[gt_voxel_ndarray != cls] = 0
            gt_voxel_ndarray_category[gt_voxel_ndarray == cls] = 1
            gt_masks.append(gt_voxel_ndarray_category)
    gt_voxel_ndarray = np.stack(gt_masks, axis=0)

    assert gt_voxel_ndarray.shape[0] == len(args.category), str(gt_voxel_ndarray.shape[0])
    assert gt_voxel_ndarray.shape[1:] == ct_voxel_ndarray.shape[1:]
    item['label'] = gt_voxel_ndarray.astype(np.int32)
    print(idx, ' load done!')

    #############################
    item['image'] = normalize(item['image'])
    print(idx, ' transform done')

    ############################
    print(file_name + ' ct gt <--> ', item['image'].shape, item['label'].shape)
    case_path = join(save_path, file_name)
    os.makedirs(case_path, exist_ok=True)

    np.save(join(case_path, 'image.npy'), item['image'])
    allmatrix_sp = sparse.csr_matrix(item['label'].reshape(item['label'].shape[0], -1))
    sparse.save_npz(join(case_path, 'mask_' + str(item['label'].shape)), allmatrix_sp)
    print(file_name + ' save done!')


def generate_dataset_json(root_dir, output_file, test_ratio=0.2):
    cases = os.listdir(root_dir)
    ct_paths, gt_paths = [], []
    for case_name in cases:
        case_files = sorted(os.listdir(join(root_dir, case_name)))
        ct_path = join(root_dir, case_name, case_files[0])
        gt_path = join(root_dir, case_name, case_files[1])
        ct_paths.append(ct_path)
        gt_paths.append(gt_path)

    data = list(zip(ct_paths, gt_paths))
    train_data, val_data = train_test_split(data, test_size=test_ratio)
    labels = {}
    labels['0'] = 'background'
    for idx in range(len(args.category)):
        label_name = args.category[idx]
        label_id = idx + 1
        labels[str(label_id)] = label_name
    dataset = {
        'name': f'{args.dataset_code} Dataset',
        'description': f'{args.dataset_code} Dataset',
        'tensorImageSize': '4D',
        'modality': {
            '0': 'CT',
        },
        'labels': labels,
        'numTrain': len(train_data),
        'numTest': len(val_data),
        'train': [{'image': ct_path, 'label': gt_path} for ct_path, gt_path in train_data],
        'test': [{'image': ct_path, 'label': gt_path} for ct_path, gt_path in val_data]
    }
    with open(output_file, 'w') as f:
        print(f'{output_file} dump')
        json.dump(dataset, f, indent=2)


if __name__ == "__main__":
    with multiprocessing.Pool(processes=64) as pool:
        pool.map(run, data_path_list_all)
    print('Process Finished!')

    generate_dataset_json(root_dir=save_path,
                          output_file=join(save_path, f'{args.dataset_code}.json'),
                          test_ratio=args.test_ratio)
    print('Json Split Done!')
