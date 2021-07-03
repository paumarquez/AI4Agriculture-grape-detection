from torch import Tensor, device
from typing import List, Tuple
from collections import namedtuple

import torch
import numpy as np

BoxCoords = namedtuple("BoxCoords", ["x", "y", "w", "h"])

def boxcoords2masks_bounds(boxes: Tensor, shape: Tuple[int, int], d: int) -> Tuple[Tensor, Tensor]:
    '''
    boxes: Tensor N x 4
    Given a list of bounding boxes, compute the mask of each row and column with width d
        so for each box there are ~ box_width // d + box_height // d masks.
        In each of those masks there should be at least d positive pixels as stated by the tightness prior
    '''
    device: device = boxes.device
    masks_list: List[Tensor] = []
    bounds_list: List[float] = []

    for box_coords in boxes:
        box = BoxCoords(
            box_coords[1].int(),
            box_coords[0].int(),
            (box_coords[3]-box_coords[1]).int(),
            (box_coords[2]-box_coords[0]).int()
        )
        for i in range(box.w // d):
            mask = torch.zeros(shape, dtype=torch.float32)
            mask[box.x + i * d:box.x + (i + 1) * d, box.y:box.y + box.h + 1] = 1
            masks_list.append(mask)
            bounds_list.append(d)

        if box.w % d:
            mask = torch.zeros(shape, dtype=torch.float32)
            mask[box.x + box.w - (box.w % d):box.x + box.w + 1, box.y:box.y + box.h + 1] = 1
            masks_list.append(mask)
            bounds_list.append(box.w % d)

        for j in range(box.h // d):
            mask = torch.zeros(shape, dtype=torch.float32)
            mask[box.x:box.x + box.w + 1, box.y + j * d:box.y + (j + 1) * d] = 1
            masks_list.append(mask)
            bounds_list.append(d)

        if box.h % d:
            mask = torch.zeros(shape, dtype=torch.float32)
            mask[box.x:box.x + box.w + 1, box.y + box.h - (box.h % d):box.y + box.h + 1] = 1
            masks_list.append(mask)
            bounds_list.append(box.h % d)

    bounds = torch.tensor(bounds_list, dtype=torch.float32) if bounds_list else torch.zeros((0,), dtype=torch.float32)
    masks = torch.stack(masks_list) if masks_list else torch.zeros((0, *shape), dtype=torch.float32)
    assert masks.shape == (len(masks_list), *shape)
    assert masks.dtype == torch.float32
    assert bounds.shape == (len(masks_list),)

    return masks, bounds
