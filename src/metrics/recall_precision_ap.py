r"""
Reference:
    https://github.com/ternaus/iglovikov_helper_functions/blob/master/iglovikov_helper_functions/metrics/map.py

Script to calculate mean average precision for a fixed Intersection Over Union (IOU)
Expects input data in the COCO format.
Based on https://github.com/ucbdrive/bdd-data/blob/master/bdd_data/evaluate.py
"""
import argparse
import json
from collections import OrderedDict
from typing import Any, Dict, List, Tuple
from collections import defaultdict
import numpy as np
from itertools import chain
import copy

from .ap import get_ap

def group_by_key(list_dicts: List[dict], key: Any) -> defaultdict:
    """Groups list of dictionaries by key.
    >>> c = [{"a": 1, "b": "Wednesday"}, {"a": (1, 2, 3), "b": 16.5}]
    defaultdict(list,
            {1: [{'a': 1, 'b': 'Wednesday'}],
             (1, 2, 3): [{'a': (1, 2, 3), 'b': 16.5}]})
    Args:
        list_dicts:
        key:
    Returns:
    """
    groups: defaultdict = defaultdict(list)
    for detection in list_dicts:
        groups[detection[key]].append(detection)
    return groups


def get_overlaps(gt_boxes: np.ndarray, box: np.ndarray) -> np.ndarray:
    i_xmin = np.maximum(gt_boxes[:, 0], box[0])
    i_ymin = np.maximum(gt_boxes[:, 1], box[1])

    i_xmax = np.minimum(gt_boxes[:, 2], box[2])
    i_ymax = np.minimum(gt_boxes[:, 3], box[3])

    iw = np.maximum(i_xmax - i_xmin, 0.0)
    ih = np.maximum(i_ymax - i_ymin, 0.0)

    intersection = iw * ih

    union = (box[3] - box[1]) * (box[2] - box[0]) + (gt_boxes[:, 3] - gt_boxes[:, 1]) * (gt_boxes[:, 2] - gt_boxes[:, 0]) - intersection

    overlaps = intersection / (union + 1e-7)

    return overlaps

def recall_precision_ap(
    gt: np.ndarray, predictions: np.ndarray, iou_threshold: float, score_threshold = 0.0
) -> Tuple[np.array, np.array, np.array]:
    """
    Parameters:
    predictions and gt: [{
        "image_id",
        "boxes": [xmin, ymin, xmax, ymax]
        "score": only for predictions
    }]
    iou_threshold: IOU threshold from where boxes are considered true positives
    score_threshold: Only boxes with score higher than `score_threhsold` will be taken into account
    """
    predictions = copy.deepcopy(predictions)
    gt = copy.deepcopy(gt)
    for t in gt:
        t["image_id"] = str(t["image_id"].item())
    image_gts = group_by_key(gt, "image_id")
    image_gt_boxes = {
        img_id: t[0]["boxes"].cpu().detach().numpy() for img_id, t in image_gts.items()
    }
    image_gt_checked = {img_id: np.zeros(len(boxes)) for img_id, boxes in image_gt_boxes.items()}
    predictions = list(chain(*[
        [{
            "box": box.cpu().detach().numpy(),
            "score": score.cpu().detach().numpy(),
            "image_id": gt_i["image_id"]
        }
            for box, score in zip(pred["boxes"], pred["scores"])] for gt_i, pred in zip(gt, predictions)]
    ))
    if len(predictions) == 0:
        return 0, 0, 0

    predictions = sorted(predictions, key=lambda x: x["score"])
    # go down dets and mark TPs and FPs
    num_predictions = len(predictions)
    tp = np.zeros(num_predictions)
    fp = np.zeros(num_predictions)

    for prediction_index, prediction in enumerate(predictions):
        box = prediction["box"]
        max_overlap = -np.inf
        jmax = -1

        try:
            gt_boxes = image_gt_boxes[prediction["image_id"]]  # gt_boxes per image
            gt_checked = image_gt_checked[prediction["image_id"]]  # gt flags per image
        except KeyError:
            gt_boxes = []
            gt_checked = None

        if len(gt_boxes) > 0:
            overlaps = get_overlaps(gt_boxes, box)
            max_overlap = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if max_overlap >= iou_threshold:
            if gt_checked[jmax] == 0:
                tp[prediction_index] = 1.0
                gt_checked[jmax] = 1
            else:
                fp[prediction_index] = 1.0
        else:
            fp[prediction_index] = 1.0
    score_idx = [
        prediction_index
            for prediction_index, prediction in enumerate(predictions)
                if prediction["score"] >= score_threshold
    ][0]
    num_gts = sum(map(len,image_gt_boxes.values()))
    
    n_predictions_threshold = len(predictions) - score_idx
    precision = sum(tp[score_idx:])/n_predictions_threshold
    recall = sum(tp[score_idx:])/num_gts

    # compute precision recall
    fp_cumsum = np.cumsum(fp, axis=0)
    tp_cumsum = np.cumsum(tp, axis=0)

    recalls_ap = tp_cumsum / float(num_gts)

    # avoid divide by zero in case the first detection matches a difficult ground truth
    precisions_ap = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)

    ap = get_ap(recalls_ap, precisions_ap)

    return recall, precision, ap
