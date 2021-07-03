import torch
from torch import Tensor

from functools import reduce
from operator import mul, add
import numpy as np
from typing import cast
from typing import List
from operator import mul
from torch.nn.functional import relu

def get_box_pos_size_bounds(masks: Tensor, margins: Tensor) -> Tensor:
    """
    parameters:
        masks: [Tensor[B,K,W,H]]] Binary masks for each K classes
        margins: [float] List of two floats: min and max pct of positive box area 

    returns:
        bounds: [Tensor] [B, 1, 2] 1 = one positive class, 2 = [lower bound, upper bound]
    """
    bounds: Tensor = (1 - masks[:, 0, :, :]).sum((-1,-2)).unsqueeze(1) * margins
    assert (bounds[..., 0] <= bounds[..., 1]).all()

    return bounds.unsqueeze(1) # unsqueeze -> there's only one class
    
class PosSizeConstraintsPriors():
    def __init__(self, margins):
        self.idc: List[int] = [1]
        self.C = len(self.idc)
        self.margins: Tensor = torch.tensor(margins, requires_grad=False)
        print(f"Initialized {self.__class__.__name__}")

    def penalty(self, z: Tensor, t: int) -> Tensor:
        assert z.shape == ()

        if z <= - 1 / t**2:
            return - torch.log(-z) / t
        else:
            return t * z + -np.log(1 / (t**2)) / t + 1 / t
    
    def __call__(self, probs: Tensor, masks: Tensor, t: float) -> Tensor:
        if self.margins.device != masks.device:
            self.margins = self.margins.to(masks.device)
        bounds = get_box_pos_size_bounds(masks, self.margins)
        # b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        b: int
        b, _, *im_shape = probs.shape
        _, k, two = bounds.shape  # scalar or vector
        assert two == 2

        value: Tensor = cast(Tensor, (probs[:, self.idc, ...] * (1 - masks[:, 0, ...])).sum((-1,-2)))
        lower_b = bounds[:, :, 0]
        upper_b = bounds[:, :, 1]

        assert value.shape == (b, self.C), value.shape
        assert lower_b.shape == upper_b.shape == (b, self.C), lower_b.shape

        upper_z: Tensor = cast(Tensor, (value - upper_b).type(torch.float32)).flatten()
        lower_z: Tensor = cast(Tensor, (lower_b - value).type(torch.float32)).flatten()

        upper_penalty: Tensor = relu(reduce(add, (self.penalty(e, t) for e in upper_z)))
        lower_penalty: Tensor = relu(reduce(add, (self.penalty(e, t) for e in lower_z)))
        res: Tensor = upper_penalty + lower_penalty

        loss: Tensor = res.sum() / reduce(mul, im_shape)
        assert loss.requires_grad == probs.requires_grad  # Handle the case for validation

        return loss
