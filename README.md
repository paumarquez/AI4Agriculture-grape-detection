# Object Detection and Weakly Supervised Semantic Segmentation for Grape Detection

This project contains the source code to train, evaluate and analyse the object detection and semantic segmentation algorithms built during the final thesis of my bachelor's degree on [Data Science and Engineering at UPC](https://dse.upc.edu/en). This project has been carried out within the _AI4Agriculture_ pilot, in the [AI4EU](https://www.ai4europe.eu) platform.

The models are embedded in a class that normalizes methods to both evaluate and train. The base interface is in `src/dl_template.py` and both models inherit and implement its functions in `train_faster.py` and `train_unet.py`. Both are implemented using PyTorch.

In order to train the models, the `train_unet.sh` or `train_faster.sh` can be called. They load the main not default options. The options are contained in the folder `options` but can be listed by running `python3 train_unet.py --help`.

In order to use them for evaluation, a dot access dictionary (such as [DotMap](https://pypi.org/project/dotmap/)) has to indicate the options of the model. They can be stored in a JSON options file (or just load it as a dictionary). The following implementation shows how to run it:

```python
# In case we are not in the root folder, append the path
import sys
sys.path.append("..")

import torch
import os
import json

from dotmap import DotMap
from train_unet import UNetHandler
from train_faster import FasterHandler

def get_options(checkpoint_dir):
    checkpoint_name = 'model.pt'
    opt_dir = os.path.join(checkpoint_dir, 'opt.json')
    weights_dir = os.path.join(checkpoint_dir, checkpoint_name)

    with open(opt_dir, 'r') as fd:
        opt = json.load(fd)
    
    opt = DotMap({
        # If we are using the Faster R-CNN, the default parameter of trainable_backbone_layers is None
        # but DotMap returns DotMap objects for non initialized entries
        "trainable_backbone_layers": None,
        **{k: '.' + v if type(v) == str and v[:2] == './' else v for k, v in opt.items()},
        
        # Overwrite
        "box_nms_thresh": 0.4, # Set the nms threshold for the Faster R-CNN as you wish
        "purpose": "val", # [val|train] -> val does not shuffle the data loader
        "phase": "val", # [train|val|test|all]
        "log_dash": False, # if True, WANDB is used (only for training)
        "device": "cuda", # set to either cuda or cpu
        "step_batch_size": 1, # Batch loader will load a batch of one element (easier to analyse)
        "checkpoint": weights_dir # add checkpoint to be loaded
    })
    return opt
    
CHECKPOINT_DIR = "../checkpoints_dir/"

opt = get_options(CHECKPOINT_DIR)

model_handler = UNetHandler(opt).init_train() # same with FasterHandler
model_handler.model.eval()

def inference_generator(model_handler, split="train"):
    if split == "train":
        loader = model_handler.train_loader
    elif split == "val":
        loader = model_handler.val_loaders[0][1]
    else:
        raise ValueError("Wrong split assigned")
    for data_d, target_d in model_handler.custom_collate(loader):
        data = model_handler.transform_data(data_d)
        with torch.no_grad():
            res = model_handler.forward(model_handler.model, data)[0]
        yield res
```

If we are not importing the model from the root folder of the project, we will need to append the relative rute to the roots. In this example we are adding `".."` before relative paths.

Note that UNet runs a forward for each patch (slice of an image), so they have to be merged. A function that handles this functionality is contained in `notebooks/utils.py`. To run it:

```python
from notebooks.utils import get_unet_masks

# Get test loader if desired
test_loader = model_handler.get_data_loader(model_handler.opt, 'test')
# Create generator
unet_masks_generator = get_unet_masks(model_handler, loader = test_loader)
# Get
first_image_inference = next(unet_masks_generator)

mask_with_crf = first_image_inference["crf_mask_0,5"].toarray()
mask_raw = first_image_inference["unet_mask_0,9"].toarray()

raw_full_image = first_image_inference["full_image"]
```

The models weights and options are contained in [this drive folder](https://drive.google.com/drive/folders/1cv7yCdyoysEcgNFGvJ9-w-ezKBJ8SsSY?usp=sharing). You should download them and specify the folders that contain the data in the `get_options` function.

If we wanted to train one of the models, we would load the model handler as above or use the options parser, as in `train_faster.py` and `train_unet.py`. After it is loaded, we would just run the following command:

```
model_handler.train()
```

## Data
This project handles two datasets: a novel dataset from the _AI4Agriculture_ project (which is still not public) and the [WGISD](https://github.com/thsant/wgisd) dataset. To use this dataset, set the `dataset` options to `WGISD` and fill the other options listed in `options/base_options.py`.

## Models
### Object Detection
The object detection neural network implemented in this project is the Faster R-CNN from torchvision.

### Weakly Supervised Semantic Segmentation
The implementation of a Weakly Supervised Semantic Segmentation with bounding boxes has been built. It implements the algorithm described in this [paper](https://github.com/LIVIAETS/boxes_tightness_prior). The implementation is strongly based on the project [boxes_tightness_prior](https://github.com/LIVIAETS/boxes_tightness_prior), but we have wrapped the constraints and the loss functions into a single module. The losses are included in the `src/losses` folder and the input parameters are explained by running the help command stated above.

The semantic segmentation neural network is a Residual U-Net. Also extracted from the [boxes_tightness_prior](https://github.com/LIVIAETS/boxes_tightness_prior) project.

## Analysis
Notebooks show some of the analysis performed during the project, even though memory issues did not let us upload the whole analysis (images were too big).

## Aknowledgements
A big thanks to [LIVIA](https://github.com/LIVIAETS) to make the code used in their paper [Bounding boxes for weakly supervised segmentation: Global constraints get close to full supervision](http://proceedings.mlr.press/v121/kervadec20a.html) open source.
