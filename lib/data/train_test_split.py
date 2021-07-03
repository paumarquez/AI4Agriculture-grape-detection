import os
import random
import json
import xmltodict
import pandas as pd
import numpy as np
from itertools import chain

LABELS_DIR = './data/all_labels'
OUTPUT_FILE = './data/splits/split3.json'
TRAIN_PROP = 0.7
VAL_PROP = 0.15
TEST_PROP = 0.15
SEED=1
assert TRAIN_PROP + VAL_PROP + TEST_PROP == 1

def invert_dictionary(dictionary):
    new_dict = {}
    for k,v in dictionary.items():
        for x in v:
            new_dict.setdefault(x,[]).append(k)
    return new_dict

def get_image_info(image_id):
    with open(os.path.join(LABELS_DIR, f'{image_id}.xml'), 'rb') as fd:
        xml_dict = xmltodict.parse(fd)
    if "object" not in xml_dict["annotation"]:
        xml_dict["annotation"]["object"] = []
    bboxes = pd.DataFrame(
        [(r['name'], *r['bndbox'].values()) for r in xml_dict["annotation"]["object"]],
        columns=['annotator', 'xmin', 'ymin', 'xmax', 'ymax']).astype(np.int)
    return bboxes

def get_annotators_per_image(image_ids):
    return { image_id: get_image_info(image_id)["annotator"].unique().astype('str') for image_id in image_ids }

def split_list(l, seed=None):
    if seed is not None:
        random.seed(1)
    shuffled_list = l.copy()
    random.shuffle(shuffled_list)
    train_idx = int(len(shuffled_list)*TRAIN_PROP)
    val_idx = train_idx + int(len(shuffled_list)*VAL_PROP)
    train = shuffled_list[:train_idx]
    val = shuffled_list[train_idx:val_idx]
    test = shuffled_list[val_idx:]
    return {
        "train": train,
        "val": val,
        "test": test
    }

def test_split(splits):
    '''
    Args:
        splits: [dict(str, dict(str,list(str)))] split per annotator
    
    Check there's no intersection between sets
    '''
    train_ids = set(chain(*[split['train'] for ann, split in splits.items()]))
    val_ids = set(chain(*[split['val'] for ann, split in splits.items()]))
    test_ids = set(chain(*[split['test'] for ann, split in splits.items()]))

    print(f'Train set size: {len(train_ids)}, %.2f' % (len(train_ids)/(len(train_ids) + len(val_ids) + len(test_ids))))
    print(f'Validation set size: {len(val_ids)}, %.2f' % (len(val_ids)/(len(train_ids) + len(val_ids) + len(test_ids))))
    print(f'Test set size: {len(test_ids)}, %.2f' % (len(test_ids)/(len(train_ids) + len(val_ids) + len(test_ids))))

    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(test_ids.intersection(val_ids)) == 0

def main():
    """
    split train/val/test per annotator
    the images that 
    """
    image_ids = [file_name.split('.')[0] for file_name in os.listdir(LABELS_DIR)]
    
    annotators_per_image = get_annotators_per_image(image_ids)
    images_per_annotator = invert_dictionary(annotators_per_image)

    annotators_per_image_intersection = {image_id: annotators for image_id, annotators in annotators_per_image.items() if len(annotators) > 1}
    images_per_annotator_intersection = invert_dictionary(annotators_per_image_intersection)

    image_ids_intersection = set(annotators_per_image_intersection.keys())

    split_per_annotator = { str(ann): split_list(list(set(ids) - image_ids_intersection), seed=SEED) for ann, ids in images_per_annotator.items() }
    
    intersection_split = split_list(list(image_ids_intersection), seed=SEED)

    for ann, split in split_per_annotator.items():
        set_images = set(images_per_annotator_intersection.get(ann, []))
        split["train"] += list(set_images.intersection(set(intersection_split["train"])))
        split["val"] += list(set_images.intersection(set(intersection_split["val"])))
        split["test"] += list(set_images.intersection(set(intersection_split["test"])))
    
    test_split(split_per_annotator)

    with open(OUTPUT_FILE, 'w') as fd:
        json.dump(split_per_annotator, fd)

if __name__ == "__main__":
    main()

