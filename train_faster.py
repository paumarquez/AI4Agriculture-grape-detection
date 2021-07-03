import pandas as pd
import numpy as np
import torch

from options.train_options import TrainOptions
from src.dl_template import DLTemplate
from src.losses.losses import create_loss

class FasterHandler(DLTemplate):
    def __init__(self, *args, **kwargs):
        super(FasterHandler, self).__init__(*args, **kwargs)
    
    def forward(self, model, data, target=None):
        if self.opt.phase == "train":
            assert target is not None
            return model(data, target)
        return model(data)
    
    def transform_data(self, data, target=None):
        data   = [d.permute(2, 0, 1).to(self.device) for d in data]
        if target is not None:
            target = [{key : elem.to(self.device) if type(key) == torch.Tensor else elem for key, elem in t.items()} for t in target]
            return data, target
        return data

    def create_optimizer(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.opt.lr,
            momentum=self.opt.momentum,
            weight_decay=self.opt.wd,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.opt.lr_scheduler_factor,
            patience=self.opt.lr_scheduler_patience,
            threshold=self.opt.lr_scheduler_threshold,
            threshold_mode='abs',
            verbose=True,
            eps=self.opt.lr_scheduler_eps
        )
        return optimizer
    
    def get_loss_from_model(self, res, target):
        return sum(res.values())

    def on_train_start(self):
        self.metrics.init_stats()

    def on_train_forward(self, res, loss, n, loss_info=None):
        for metric, value in res.items():
            self.metrics.update_metric(
                f'train_{metric}', value.item(), n=n
            )

    def on_train_display(self):
        self.log_dash(self.metrics.get_metrics(prefix="train_loss"), self.total_steps)
        self.log(" | ".join(
            [f"{metric}: {round(val,4)}" for metric, val in self.metrics.get_metrics("train_loss").items()]),
            training=True
        )

    def on_val_start(self, split, dataset_name):
        self.final_threshold = None
        self.curr_val_metrics = None
        self.val_threshold = 0.0

    def on_val_end(self, split, dataset_name):
        if split == 'val' and dataset_name == 'AI4EU':
            curr_metric_value = self.metrics.get_metric(self.opt.optim_metric_name)
            if curr_metric_value > self.max_metric:
                self.max_metric = curr_metric_value
                self.log(
                    f'Best metric achieved, saving weights at epoch {self.epoch} with {self.opt.optim_metric_name} %.5f' % (curr_metric_value),
                    training=True
                )
                self.store_checkpoint(self.BEST_CHECKPOINT_NAME)
        dataset_suffix = "_WGISD" if dataset_name == "WGISD" else ""
        self.log_dash(self.metrics.get_metrics(f"{split}{dataset_suffix}", "train_loss"), self.total_steps)
        self.log(
            f"{dataset_name} | " + " | ".join([
                f"{metric}: {round(val,4)}" for metric, val in self.metrics.get_metrics(
                    f"{split}{dataset_suffix}", "train_loss" if dataset_suffix else ("train_loss", f"{split}_WGISD")
                ).items()
            ]) + f" | threshold: {round(self.val_threshold,2)}",
            training=True
        )
        if self.opt.lr_scheduler:
            self.lr_scheduler.step(
                self.metrics.get_metric(self.opt.optim_metric_name)
            )

    def on_val_forward(self, i, set_length, results, data, target, split, dataset_name):
        recall, precision, ap, f1_score = self.metrics.get_metrics_per_threshold(
            target=target, results=results, score_threshold=self.val_threshold, mask_metrics=False
        )
        mask_recall, mask_precision, mask_ap, mask_f1_score = self.metrics.get_metrics_per_threshold(
            target=target, results=results, score_threshold=self.val_threshold, mask_metrics=True, calc_ap=True
        )
        n = len(target)
        dataset_suffix = "_WGISD" if dataset_name == "WGISD" else ""
        self.metrics.update_metric(f'{split}{dataset_suffix}_recall', recall, n)
        self.metrics.update_metric(f'{split}{dataset_suffix}_precision', precision, n)
        self.metrics.update_metric(f'{split}{dataset_suffix}_f1', f1_score, n)
        self.metrics.update_metric(f'{split}{dataset_suffix}_ap', ap, n)

        self.metrics.update_metric(f'{split}{dataset_suffix}_mask_recall', mask_recall, n)
        self.metrics.update_metric(f'{split}{dataset_suffix}_mask_precision', mask_precision, n)
        self.metrics.update_metric(f'{split}{dataset_suffix}_mask_f1', mask_f1_score, n)
        self.metrics.update_metric(f'{split}{dataset_suffix}_mask_ap', mask_ap, n)

METRICS = [
    "train_loss_classifier",
    "train_loss_box_reg",
    "train_loss_objectness",
    "train_loss_rpn_box_reg",
    "train_loss",
    "train_f1",
    "train_recall",
    "train_precision",
    "train_ap",
    "train_mask_f1",
    "train_mask_recall",
    "train_mask_precision",
    "train_mask_ap",
    "val_f1",
    "val_recall",
    "val_precision",
    "val_ap",
    "val_mask_f1",
    "val_mask_recall",
    "val_mask_precision",
    "val_mask_ap"
]

if __name__ == "__main__":
    opt = TrainOptions().parse()
    torch.manual_seed(opt.seed)

    trainer = FasterHandler(opt, METRICS)

    trainer.train()
