import torch

from .box_prior import BoxPrior
from .pos_size_constraints_priors import PosSizeConstraintsPriors
from .NegSizeLoss import NegSizeLoss

def get_att_from_listofdicts(l, att):
    return [elem[att] for elem in l]

class WeakSemanticSegmentationLoss():
    def __init__(self, init_t, d, margins, lambd, alpha, cache_folder):
        self.alpha = alpha
        self.lambd = lambd
        self.t = init_t
        self.box_prior_loss = BoxPrior(d, cache_folder)
        self.neg_size_loss = NegSizeLoss()
        self.pos_size_constraints_priors = PosSizeConstraintsPriors(margins)

    def update_time(self, factor):
        self.t = min(self.t * factor, 50)

    def loss_negative(self, pred_probs_mask, target, t):
        masks_batch = torch.stack(
            get_att_from_listofdicts(target, "masks"),
            dim=0
        )
        return self.neg_size_loss(pred_probs_mask, masks_batch, t)

    def loss_box_prior(self, pred_probs_mask, target, t):
        return self.box_prior_loss(
            pred_probs_mask,
            get_att_from_listofdicts(target, "boxes"),
            get_att_from_listofdicts(target, "image_id_str"),
            t=t
        )

    def loss_pos_size_constraints_prior(self, pred_probs_mask, target, t):
        masks_batch = torch.stack(
            get_att_from_listofdicts(target, "masks"),
            dim=0
        )
        return self.pos_size_constraints_priors(
            pred_probs_mask,
            masks_batch,
            t=t
        )

    def __call__(self, pred_probs_mask, target, t=None):
        """
        mask: [Tensor] B, K, W, H
        target: List[Dict[str, Tensor]] B, key: "boxes", N, 4
        """
        loss_negative = self.loss_negative(pred_probs_mask, target, t or self.t)
        loss_box_prior = self.loss_box_prior(pred_probs_mask, target, t or self.t)
        loss_pos_size_constraints_prior = self.loss_pos_size_constraints_prior(pred_probs_mask, target, t or self.t)
        total_loss = (
            self.alpha * loss_negative +
            self.lambd * loss_box_prior +
            loss_pos_size_constraints_prior
        )
        loss_info = {
            "loss_negative": loss_negative,
            "loss_box_prior": loss_box_prior,
            "loss_pos_size": loss_pos_size_constraints_prior,
            "loss_reduced": total_loss
        }
        return loss_info
