import torch
import numpy as np
from torch import Tensor
from typing import List
from operator import mul, add
from functools import reduce

class NegSizeLoss():
    def __init__(self):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc_target: List[int] = [0] # Apply only to negative class (background)
        self.idc_pred: List[int] = [1] # Apply only to negative class (background)
        print(f"Initialized {self.__class__.__name__}")

    def penalty(self, z: Tensor, t: float) -> Tensor:
        assert z.shape == ()

        if z <= - 1 / t**2:
            return - torch.log(-z) / t
        else:
            return t * z + -np.log(1 / (t**2)) / t + 1 / t

    def __call__(self, probs: Tensor, masks: Tensor, t: float) -> Tensor:
        b: int
        b, _, *im_shape = probs.shape
        
        probs_m: Tensor = probs[:, self.idc_pred, ...]
        masks_m: Tensor = masks[:, self.idc_target, ...]
        
        # Compute the size for each class, masked by the target pixels (where target ==1)
        masked_sizes: Tensor = torch.einsum(f"bkwh,bkwh->bk", probs_m, masks_m).flatten()

        # We want that size to be <= so no shift is needed
        res: Tensor = reduce(add, (self.penalty(e, t) for e in masked_sizes))  # type: ignore

        loss: Tensor = res / reduce(mul, im_shape)
        assert loss.shape == ()
        assert loss.requires_grad == probs.requires_grad  # Handle the case for validation

        return loss