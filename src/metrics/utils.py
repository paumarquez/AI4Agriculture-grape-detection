import numpy as np

from .recall_precision_ap import recall_precision_ap
from .mask_metrics import get_mask_metrics

def get_metrics_per_threshold(target, results, score_threshold, mask_metrics=True, calc_ap=True):
    if not mask_metrics:
        # Get Box Metrics
        recall, precision, ap = recall_precision_ap(target, results, 0.7, score_threshold=score_threshold)
        f1_score = 2/(1/np.maximum(recall, np.finfo(np.float64).eps)+1/np.maximum(precision, np.finfo(np.float64).eps))
    else:
        recall, precision, f1_score, ap = list(map(np.nanmean, zip(*[
            get_mask_metrics(
                target_boxes=t["boxes"].detach().cpu().numpy().astype(np.int),
                result_boxes=res["boxes"].detach().cpu().numpy().astype(np.int),
                result_scores=res["scores"].detach().cpu().numpy(),
                score_threshold=score_threshold,
                calc_ap=calc_ap
            ) for res, t in zip(results, target)
        ])))
    return recall, precision, ap, f1_score