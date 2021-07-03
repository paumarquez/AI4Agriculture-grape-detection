from itertools import chain
import numpy as np
from scipy.stats import hmean

from .recall_precision_ap import recall_precision_ap
from .mask_metrics import get_mask_metrics
from .utils import get_metrics_per_threshold

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Metrics():
    def __init__(self, metrics):
        self.stats = {}
        self.metrics = metrics
        self.init_stats()

    def init_stats(self):
        for metric in set(chain(self.metrics, self.stats.keys())):
            self.create_metric(metric)
    
    def create_metric(self, key):
        self.stats[key] = AverageMeter()
    

    def get_metric(self, metric):
        return self.stats[metric].avg
    
    def get_metrics(self, prefix, not_prefix = None):
        return {
            metric: val.avg for metric, val in self.stats.items()
                if metric.startswith(prefix) and ((not not_prefix) or not metric.startswith(not_prefix))
        }

    def update_metric(self, metric, value, n):
        if metric not in self.stats:
            self.create_metric(metric)
            print(f"WARNING: {metric} metric was not created")
        self.stats[metric].update(value, n)

    def reset_metric(self, metric):
        self.stats[metric].reset()


    def calc_metrics(self, key, arguments):
        if key == "OD_basic_metrics":
            return recall_precision_ap(*arguments)
        elif key == "mask_metrics":
            return get_mask_metrics(*arguments)
        else:
            raise Exception(f"Unexpected metric {key}")

    
    def get_metrics_per_threshold(self, target, results, score_threshold, mask_metrics=True, calc_ap=True):
        return get_metrics_per_threshold(
            target=target,
            results=results,
            score_threshold=score_threshold,
            mask_metrics=mask_metrics,
            calc_ap=calc_ap
        )

    def get_metrics_from_masks(self, pred_mask, target_mask):
        tp = np.sum(pred_mask * target_mask)
        fp = np.sum(pred_mask * (1-target_mask))
        recall = tp/target_mask.sum()
        if tp+fp == 0:
            precision = 0
        else:
            precision = tp/(tp+fp)
        f1_score = hmean([recall, precision]) if recall > 0 and precision > 0 else 0
        return recall, precision, f1_score
