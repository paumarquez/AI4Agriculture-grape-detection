import numpy as np
from scipy.stats import hmean

from .ap import get_ap


def add_mask(mask, boxes):
        for box in boxes:
            mask[box[1]:box[3], box[0]:box[2]] = 1

def get_metrics_from_masks(pred_mask, target_mask):
    tp = np.sum(pred_mask * target_mask)
    fp = np.sum(pred_mask * (1-target_mask))
    recall = tp/target_mask.sum()
    if tp+fp == 0:
        precision = 0
    else:
        precision = tp/(tp+fp)
    f1_score = hmean([recall, precision]) if recall > 0 and precision > 0 else 0
    return recall, precision, f1_score

def get_mask_metrics(target_boxes, result_boxes, result_scores, score_threshold, calc_ap=True):
    all_boxes = np.concatenate([result_boxes, target_boxes])
    width, height = all_boxes.max(0)[[2,3]]
    
    pred_mask = np.zeros((height, width)).astype(np.int8)
    pred_mask_thr = np.zeros((height, width)).astype(np.int8)
    target_mask = np.zeros((height, width)).astype(np.int8)
    
    add_mask(target_mask, target_boxes)
    
    recalls = np.array([0.0]*len(result_boxes))
    precisions = np.array([0.0]*len(result_boxes))
    sorted_boxes_scores = sorted(
        zip(result_boxes, result_scores),
        key = lambda box_score: box_score[1]
    )
    ap=-10
    # Calc for boxes with score greater or equal than threshold only
    add_mask(
        pred_mask_thr,
        [box for box, score in sorted_boxes_scores if score >= score_threshold]
    )
    final_recall, final_precision, final_f1_score = get_metrics_from_masks(
        pred_mask_thr, target_mask
    )
    if calc_ap:
        for i, (box, _) in enumerate(sorted_boxes_scores):
            add_mask(pred_mask, [box])
            recall, precision, _ = get_metrics_from_masks(pred_mask, target_mask)
            recalls[i] = recall
            precisions[i] = precision
        ap = get_ap(recalls, precisions)

    

    return final_recall, final_precision, final_f1_score, ap