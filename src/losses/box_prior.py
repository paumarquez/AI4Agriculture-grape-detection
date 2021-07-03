from functools import reduce
from operator import add

from torch import Tensor
from typing import List

import torch
import numpy as np
import pickle
import os

from .utils import boxcoords2masks_bounds

class BoxPrior():
    def __init__(self, d: int, cache_folder: str, gpu_friendly: bool = False):
        self.d: int = d
        self.gpu_friendly: bool = gpu_friendly
        
        self.cache_folder = cache_folder
        self.cached_image_ids = {}
        
        self.idc: List[int] = [1] # Evaluate box prior only in the positive class

    def barrier(self, z: Tensor, t: float) -> Tensor:
        assert z.shape == ()

        if z <= - 1 / t**2:
            return - torch.log(-z) / t
        else:
            return t * z + -np.log(1 / (t**2)) / t + 1 / t
  
    def store_cache(self, priors, image_id):
        with open(os.path.join(self.cache_folder, f'{image_id}_box_priors.pkl'), 'wb') as fd:
            pickle.dump(priors, fd)

    def get_priors_from_cache(self, image_id):
        with open(os.path.join(self.cache_folder, f'{image_id}_box_priors.pkl'), 'rb') as fd:
            priors = pickle.load(fd)
        return priors

    def generate_box_priors(self, boxes, shape, d, image_id):
        if False and f"{image_id}_slice_i" in self.cached_image_ids:
            print(f'image id {image_id} in priors cached')
            priors = self.get_priors_from_cache(image_id)
        else:
            # print(f'image id {image_id} priors stored in cache')
            priors = boxcoords2masks_bounds(boxes, shape, d)
            #self.store_cache(priors, image_id)
            #self.cached_image_ids[image_id] = 1
        return priors

    def __call__(self, probs: Tensor, boxes: List[Tensor], image_ids: List[str], t: float) -> Tensor:
        """
        probs: [B, K, W, H] -> mask probabilities for each class and image
        boxes: List[Tensor[N, 4]] -> List of B elements containing the N bounding boxes
        B: # batch size
        K: # classes
        N: # bounding boxes
        For each bounding box' row and column with width "d", assure tightness constraint
        """
        B, K = probs.shape[:2]
        im_shape = probs.shape[2:4]
        box_prior = [[
            self.generate_box_priors(boxes[b], im_shape, self.d, image_ids[b]) if k in self.idc else [] # Calc only for classes to take into account
                for k in range(K)
        ] for b in range(B)]
        sublosses = []
        for b in range(B):
            for k in self.idc:
                masks, bounds = box_prior[b][k]

                sizes: Tensor = None
                if self.gpu_friendly:
                    #sizes = torch.einsum('wh,nwh->n', probs[b, k], masks)
                    sizes = (torch.unsqueeze(probs[b, k], 0) * masks.to(probs.device)).sum((1,2))
                else:
                    sizes = torch.tensor([
                        (probs[b, k] * m.to(probs.device)).sum()
                            for m in masks
                    ], device=probs.device)
                
                assert sizes.shape == bounds.shape == (masks.shape[0],), (sizes.shape, bounds.shape, masks.shape)
                shifted: Tensor = bounds.to(probs.device) - sizes

                init = torch.zeros((), dtype=torch.float32, requires_grad=probs.requires_grad, device=probs.device)
                sublosses.append(reduce(add, (self.barrier(v, t) for v in shifted), init))

        loss: Tensor = reduce(add, sublosses)

        assert loss.dtype == torch.float32
        assert loss.shape == (), loss.shape

        return loss
