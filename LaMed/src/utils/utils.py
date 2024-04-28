import re
import torch

def mask2box(mask_3d):
    non_zero_indices = torch.nonzero(mask_3d, as_tuple=True)

    min_indices = [torch.min(indices) for indices in non_zero_indices]
    max_indices = [torch.max(indices) for indices in non_zero_indices]

    min_x = min_indices[0].item()
    min_y = min_indices[1].item()
    min_z = min_indices[2].item()
    max_x = max_indices[0].item()
    max_y = max_indices[1].item()
    max_z = max_indices[2].item()

    X, Y, Z = mask_3d.shape

    min_x_norm = round(min_x / X, 3)
    min_y_norm = round(min_y / Y, 3)
    min_z_norm = round(min_z / Z, 3)

    max_x_norm = round(max_x / X, 3)
    max_y_norm = round(max_y / Y, 3)
    max_z_norm = round(max_z / Z, 3)

    return [min_x_norm, min_y_norm, min_z_norm, max_x_norm, max_y_norm, max_z_norm]


def extract_box_from_text(text):
    match = re.search(r'\[([-+]?\d*\.\d+|[-+]?\d+), ([-+]?\d*\.\d+|[-+]?\d+), ([-+]?\d*\.\d+|[-+]?\d+), ([-+]?\d*\.\d+|[-+]?\d+), ([-+]?\d*\.\d+|[-+]?\d+), ([-+]?\d*\.\d+|[-+]?\d+)\]', text)
    if match:
        box_coordinates = [float(coord) for coord in match.groups()]
        return box_coordinates
    else:
        return None

def calculate_iou(box1, box2):
    x1, y1, z1, x2, y2, z2 = box1
    x3, y3, z3, x4, y4, z4 = box2

    inter_x = max(0, min(x2, x4) - max(x1, x3))
    inter_y = max(0, min(y2, y4) - max(y1, y3))
    inter_z = max(0, min(z2, z4) - max(z1, z3))

    union_x = max(x2, x4) - min(x1, x3)
    union_y = max(y2, y4) - min(y1, y3)
    union_z = max(z2, z4) - min(z1, z3)

    vol_intersection = inter_x * inter_y * inter_z
    vol_union = union_x * union_y * union_z

    iou = vol_intersection / (vol_union + 1e-6)
    return iou