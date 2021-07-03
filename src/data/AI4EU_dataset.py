import numpy as np
import os
from numpy.lib.function_base import select
import pandas as pd
from skimage import io
import torch
from PIL import ImageFile
import xmltodict
from itertools import repeat, chain
import json
from random import shuffle
from itertools import chain

ImageFile.LOAD_TRUNCATED_IMAGES = True

def bbox2yolo(bbox, width, height):
    cx = (bbox[0] + bbox[2]) / (2 * width)
    cy = (bbox[1] + bbox[3]) / (2 * height)
    w = (bbox[2] - bbox[0]) / width
    h = (bbox[3] - bbox[1]) / height
    return (cx, cy, w, h)


def add_mask(mask, boxes, fill_value=1):
    for box in boxes:
        mask[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = fill_value

def filter_slices_fn(slices):
    return [
        slice for slice in slices if abs((slice[1].stop - slice[1].start)/(slice[0].stop - slice[0].start) - 1) < 0.00001
    ]
    
def get_slices(img_shape, patch_size):
    w, h, d = img_shape
    if patch_size > w and patch_size > h:
        return [(slice(0, w), slice(0, h), slice(0, d))]
    overlap = patch_size // 4
    step = patch_size - overlap
    slices = []
    x_splits = list(range(0, w, step if patch_size < w else w))
    y_splits = list(range(0, h, step if patch_size < h else h))
    if x_splits[-1] != w:
        add_splits_x = [(w-patch_size, w)]
    if y_splits[-1] != h:
        add_splits_y = [(h-patch_size, h)]
    for x_start in x_splits:
        x_end = min(x_start + patch_size, w)
        if x_end == w:
            x_start = x_end - patch_size
        for y_start in y_splits:
            y_end = min(y_start + patch_size, h)
            if y_end == h:
                y_start = y_end - patch_size
            slices.append((
                slice(x_start, x_end),
                slice(y_start, y_end),
                slice(0, d)
            ))
            if y_end == h:
                break
        if x_end == w:
            break
    return slices


def get_slices3(img_shape, patch_size):
    w, h, d = img_shape
    if patch_size > w and patch_size > h:
        return [(slice(0, w), slice(0, h), slice(0, d))]
    overlap = patch_size // 4
    step = patch_size - overlap
    slices = []
    x_splits = list(range(0, w, step if patch_size < w else w))
    y_splits = list(range(0, h, step if patch_size < h else h))
    add_splits_x = []
    add_splits_y = []
    if x_splits[-1] != w:
        add_splits_x = [(w-patch_size, w)]
    if y_splits[-1] != h:
        add_splits_y = [(h-patch_size, h)]
    for x_start, x_stop in chain(zip(x_splits[:-1], x_splits[1:]), add_splits_x):
        for y_start, y_stop in chain(zip(y_splits[:-1], y_splits[1:]), add_splits_y):
            slices.append((
                slice(max(x_start, 0), min(w, x_stop)),
                slice(max(0, y_start), min(h, y_stop)),
                slice(0, d)
            ))
    return slices

def get_slices_k(img_shape, patch_size):
    """
        Get array of slices where each slice is a partition of the image of shape `patch_size`.
        The last dimension (channels dimension) is not partitioned, only the spatial dimensions (width and height).
    """
    w, h, d = img_shape
    overlap = patch_size // 4
    step = patch_size - overlap
    x_lim = (w - patch_size) + (w - patch_size) % step
    y_lim = (h - patch_size) + (h - patch_size) % step
    # Limits have to be set since step is less than patch_size.
    # Otherwise, there could be (many) repeated slices
    bxs, bys, bzs = np.mgrid[0:w:step, 0:h:step, 0:d:d]
    patches = [
        tuple([slice(e if i == 2 or e + patch_size <= m else m - patch_size,
                    min(e + patch_size,
                        m) if i != 2 else m)
                for i, (e, m) in enumerate(zip(origin, img_shape))])
        for origin in zip(bxs.flatten(), bys.flatten(), bzs.flatten())
            if origin[0] <= x_lim and origin[1] <= y_lim
    ]
    return patches


def get_slices_2(img_shape, patch_size):
    """
	Legacy function (if get_slices does not work, use this one)
        Get array of slices where each slice is a partition of the image of shape `patch_size`.
        The last dimension (channels dimension) is not partitioned, only the spatial dimensions (width and height).
    """
    w, h, d = img_shape
    overlap = patch_size // 4
    step = patch_size - overlap
    bxs, bys, bzs = np.mgrid[0:w:step, 0:h:step, 0:d:d]
    patches = [tuple([slice(e,
                            min(e + patch_size,
                                m) if i != 2 else m)
                        for i, (e, m) in enumerate(zip(origin, img_shape))])
                for origin in zip(bxs.flatten(), bys.flatten(), bzs.flatten())]
    return patches

class AI4EU_Dataset(torch.utils.data.Dataset):
    def __init__(
        self, labels_dir, images_dir, split_file, split = 'train', annotators=None,
        annotator_draw=0, load_images=True, min_area=0, min_pct_area_per_box=0.7,
        calc_mask=False, partitioning_patch_size=None, n_images=None):
        self.labels_dir = labels_dir
        self.images_dir = images_dir
        self.split_file = split_file
        self.split = split
        self.annotators = annotators
        self.annotator_draw = annotator_draw
        self.load_images = load_images
        self.min_area = min_area
        self.calc_mask = calc_mask
        self.partitioning_patch_size = partitioning_patch_size
        self.n_images = n_images
        if self.annotators is not None:
            assert annotator_draw in annotators    
        self.ids = self.load_ids()

    @staticmethod
    def name():
        return "AI4EU"
    
    def load_ids(self):
        with open(self.split_file, 'r') as fd:
            splits = json.load(fd)
        if self.split in splits:
            ids = list(set(splits[self.split]))
        else:
            annotators = self.annotators or list(splits.keys())
            ids = list(set(chain(*[
                splits[str(ann)][split] for split in ([self.split] if self.split != "all" else ["train", "val", "test"]) for ann in annotators
            ])))
        sorted_ids = sorted(ids)
        if self.n_images:
            sorted_ids = sorted_ids[:self.n_images]
        return sorted_ids

    def get_image_info(self, image_id):
        with open(os.path.join(self.labels_dir, f'{image_id}.xml'), 'rb') as fd:
            xml_dict = xmltodict.parse(fd)
        bboxes = pd.DataFrame(
            [(r['name'], *r['bndbox'].values()) for r in xml_dict["annotation"]["object"]],
            columns=['annotator', 'xmin', 'ymin', 'xmax', 'ymax']).astype(np.int)
        return bboxes
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple)) : return [self.__getitem__(i) for i in idx]
        if isinstance(idx, str):
            image_id = idx
            idx = [i for i, id in enumerate(self.ids) if id == idx][0]
        else:
            image_id = self.ids[idx]
        data = self.get_image_info(image_id)
        curr_annotators = data.annotator.unique()
        if len(curr_annotators) > 1:
            if self.annotator_draw in curr_annotators:
                data = data[data.annotator == self.annotator_draw]
            else:
                data = data[data.annotator == min(curr_annotators)]
        
        # Filter bounding boxes with min_area
        area = (data['xmax'] - data['xmin']) * (data['ymax'] - data['ymin'])
        data = data[area >= self.min_area]

        img_name = os.path.join(self.images_dir, f'{image_id}.jpg')
        boxes = torch.FloatTensor(data[['xmin', 'ymin', 'xmax', 'ymax']].values)
        data_dict = {}
        data_dict['target'] = {}
        if self.load_images:
            image = io.imread(img_name) / 255.0
            data_dict['image'] = torch.from_numpy(image.copy()).float()
            if self.partitioning_patch_size:
                slices = get_slices(image.shape, self.partitioning_patch_size)
                #if self.filter_slices:
                #    slices = filter_slices_fn(slices)
                data_dict["target"]["slices"] = slices
                #if self.split == "train":
                #    shuffle(data_dict["target"]["slices"])

        data_dict['target']['boxes'] = boxes
        data_dict['target']['labels'] = torch.Tensor(list(repeat(1, len(data)))).long()
        data_dict['target']['image_id'] = torch.Tensor([idx]).long()
        data_dict['target']['image_id_str'] = image_id
        data_dict['target']['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        if self.calc_mask:
            mask=torch.ones(data_dict['image'].shape[:-1])
            add_mask(mask, boxes, fill_value=0)
            data_dict['target']['masks'] = mask.unsqueeze(0).float() # Class 0 (background)
        return data_dict

