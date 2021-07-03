from copy import deepcopy
import pandas as pd
import numpy as np
import torch

from torchvision.models.detection.transform import GeneralizedRCNNTransform

from options.train_options import TrainOptions
from src.dl_template import DLTemplate
from src.losses.losses import create_loss

from matplotlib.colors import colorConverter
import matplotlib as mpl
import matplotlib.pyplot as plt

IMAGE_MEAN=torch.tensor([0.485, 0.456, 0.406])
IMAGE_STD=torch.tensor([0.229, 0.224, 0.225])

def create_masked_plot(image, mask, boxes, title):
    fig, ax = plt.subplots(figsize=(10,15))
    
    color1 = colorConverter.to_rgba('white')
    color2 = colorConverter.to_rgba('red')
    cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',[color1,color2],256)
    cmap2._init()
    cmap2._lut[:,-1] = 0.25
    ax.imshow(image)
    ax.imshow(mask > 0.8, cmap=cmap2)
    for bbox in boxes:
        bbox=bbox.int()
        rect = mpl.patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=2, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)
    plt.title(title)
    return fig

class CustomTransform(GeneralizedRCNNTransform):
    def batch_images(self, images, size_divisible=32):
        return super().batch_images(images, 1)

class MultiStepLR():
    def __init__(self, opt, milestones, gamma):
        self.optimizer=opt
        self.milestones=milestones
        self.gamma=gamma

    def update_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.gamma
            print(f"Learning rate updated. Current value: {param_group['lr']}")

    def step(self, acum_steps):
        if acum_steps in self.milestones:
            self.update_lr()
        
class UNetHandler(DLTemplate):
    def __init__(self, *args, **kwargs):
        super(UNetHandler, self).__init__(*args, **kwargs)
    
    ### Initializations
    def init_loss(self):
        self.weak_loss = create_loss(self.opt)

    def init_transform(self):
        self.transform = CustomTransform(
            min_size=self.opt.transform_min_size,
            max_size=self.opt.transform_max_size,
            image_mean=IMAGE_MEAN.numpy(),
            image_std=IMAGE_STD.numpy()
        )
            

    def create_optimizer(self):
        if self.opt.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.opt.lr,
                momentum=self.opt.momentum,
                weight_decay=self.opt.wd
            )
        elif self.opt.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.opt.lr,
                weight_decay=self.opt.wd
            )
        else:
            raise Exception("Optimizer not defiend")
        if self.opt.lr_scheduler:
            if self.opt.lr_scheduler_method == "plateau":
                self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode=self.opt.lr_scheduler_mode,
                    factor=self.opt.lr_scheduler_factor,
                    patience=self.opt.lr_scheduler_patience,
                    threshold=self.opt.lr_scheduler_threshold,
                    threshold_mode="abs",
                    verbose=True,
                    eps=self.opt.lr_scheduler_eps
                )
            elif self.opt.lr_scheduler_method == "tickBased":
                self.lr_scheduler = MultiStepLR(
                    optimizer,
                    milestones=self.opt.lr_scheduler_nticks,
                    gamma=self.opt.lr_scheduler_factor
                )
            else:
                raise Exception("Wrong learning rate scheduler")
        return optimizer
    
    ### Data
    def custom_collate(self, loader):
        batch = []
        for data, target in loader:
            for img, target_i in zip(data, target):
                for slice_i in target_i["slices"]:
                    sliced_target = deepcopy({
                        **target_i,
                        "boxes": "",
                        "area": "",
                        "labels": "",
                        "masks": ""
                    })
                    sliced_img = img[slice_i]

                    h_start, h_end = slice_i[0].start, slice_i[0].stop
                    w_start, w_end = slice_i[1].start, slice_i[1].stop

                    min_lim = torch.tensor([w_start, h_start, w_start, h_start], dtype=torch.float32)
                    max_lim = torch.tensor([w_end, h_end, w_end, h_end], dtype=torch.float32)
                    
                    new_boxes = target_i["boxes"].max(min_lim).min(max_lim)
                    boxes_toconsider = (new_boxes[:,0]!=new_boxes[:,2]) & (new_boxes[:,1]!=new_boxes[:,3])
                    sliced_target["boxes"] = new_boxes[boxes_toconsider] - torch.tensor([w_start, h_start, w_start, h_start])
                    if self.opt.skip_notbbox_slices and len(sliced_target["boxes"]) == 0:
                        continue
                    if self.opt.skip_bbox_slices and len(sliced_target["boxes"]) > 0:
                        continue
                    sliced_target["area"] = target_i["area"][boxes_toconsider]
                    sliced_target["labels"] = target_i["labels"][boxes_toconsider]
                    
                    sliced_target["masks"] = target_i["masks"][:,slice_i[0],slice_i[1]]

                    sliced_target["current_slice"] = slice_i
                    
                    batch.append((sliced_img, sliced_target))
                    if len(batch) == self.opt.step_batch_size:
                        yield map(list,zip(*batch))
                        batch = []

    def transform_data(self, data, target=None):
        data   = [d.permute(2, 0, 1).to(self.device) for d in data]
        if target:
            target = [{key : elem.to(self.device) if type(elem) == torch.Tensor else elem for key, elem in t.items()} for t in target]
        image_list, target = self.transform(data, target)
        # update area for bounding boxes
        if target:
            for t in target:
                t['area'] = (t["boxes"][:, 3] - t["boxes"][:, 1]) * (t["boxes"][:, 2] - t["boxes"][:, 0])
            for t in target:
                t["masks"] = t["masks"].float()
        if target:
            return image_list.tensors, target
        else:
            return image_list.tensors

    ### Misc
    def update_scheduler(self):
        self.weak_loss.update_time(self.opt.constraints_time_factor)
        t = self.weak_loss.t
        self.log(f"Updating time loss parameter... new t: {t}", True)

    def forward(self, model, data, target=None):
        inf = model(data)
        return inf

    def get_loss_from_model(self, res, target, t=None):
        loss_info = self.weak_loss(res, target, t=t)
        return loss_info["loss_reduced"], loss_info
    
    ### Events
    def on_train_start(self):
        self.metrics.init_stats()

    def on_train_forward(self, res, loss, n, loss_info=None):
        for metric, value in loss_info.items():
            self.metrics.update_metric(
                f'train_{metric}', value.item(), n=n
            )

    def on_train_display(self):
        self.log_dash(self.metrics.get_metrics(prefix="train_loss"), self.total_steps)
        self.log(" | ".join(
            [f"{metric}: {round(val,4)}" for metric, val in self.metrics.get_metrics("train_loss").items()]),
            training=True
        )
        self.metrics.init_stats()

    def on_val_start(self, split, dataset_name):
        self.final_threshold = None
        self.curr_val_metrics = None
        self.val_threshold = 0.5
        self.metrics.init_stats()

    def on_val_forward(self, i, set_length, results, data, target, split, dataset_name):
        if i == 1:
            plot = create_masked_plot(
                image=((data[0].permute(1,2,0).cpu() * IMAGE_STD) + IMAGE_MEAN).numpy(),
                mask=results[0, 1,...].cpu().numpy(),
                boxes=target[0]["boxes"],
                title=f"steps: {self.total_steps}"
            )
            self.log_dash({"val_masked_plot": plot })
        _, loss_dict = self.get_loss_from_model(results, target, t=10)
        for metric, value in loss_dict.items():
            self.metrics.update_metric(
                f'{split}_{metric}', value.item(), n=len(target)
            )
        """
        metrics=[]
        for t, r in zip(
            target,
            results
        ):
            target_mask_i = t["masks"][:, 0, ...].detach().cpu().numpy()
            pred_mask_i = r[:, 1, ...].detach().cpu().numpy()
            curr_metrics = self.metrics.get_metrics_from_masks(
                target_mask=target_mask_i, pred_mask=pred_mask_i > self.val_threshold
            )
            metrics.append(curr_metrics)
        mask_recall, mask_precision, mask_f1_score = list(map(np.mean, zip(*metrics)))
        n = len(target)
        dataset_suffix = "_WGISD" if dataset_name == "WGISD" else ""

        self.metrics.update_metric(f"{split}{dataset_suffix}_mask_recall", mask_recall, n)
        self.metrics.update_metric(f"{split}{dataset_suffix}_mask_precision", mask_precision, n)
        self.metrics.update_metric(f"{split}{dataset_suffix}_mask_f1", mask_f1_score, n)
        """
    def on_val_end(self, split, dataset_name):
        self.log_dash(self.metrics.get_metrics(f"{split}_"), self.total_steps)
        self.log(
            f"{dataset_name} | " + " | ".join([
                f"{metric}: {round(val,4)}" for metric, val in self.metrics.get_metrics(
                    f"{split}_"
                ).items()
            ]),
            training=True
        )
        if split == "val" and dataset_name == "AI4EU":
            curr_metric_value = self.metrics.get_metric(self.opt.optim_metric_name)
            if curr_metric_value < self.min_metric:
                self.min_metric = curr_metric_value
                self.log(
                    f"Best metric achieved, saving weights at epoch {self.epoch} with {self.opt.optim_metric_name} %.5f" % (curr_metric_value),
                    training=True
                )
                self.store_checkpoint(self.BEST_CHECKPOINT_NAME)
        return
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
        if self.opt.lr_scheduler_method == "plateau":
            self.lr_scheduler.step(
                self.metrics.get_metric(self.opt.optim_metric_name)
            )

METRICS = [
    "train_loss",
    "train_loss_negative",
    "train_loss_box_prior",
    "train_loss_pos_size",
    "train_loss_reduced",
    "val_loss",
    "val_loss_negative",
    "val_loss_box_prior",
    "val_loss_pos_size",
    "val_loss_reduced",
]

if __name__ == "__main__":
    opt = TrainOptions().parse()
    torch.manual_seed(opt.seed)

    trainer = UNetHandler(opt, METRICS)

    trainer.train()
