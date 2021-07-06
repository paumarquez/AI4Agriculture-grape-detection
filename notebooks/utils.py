import torch
import numpy as np
import xmltodict
import os
import pandas as pd
import torch

from matplotlib import pyplot as plt
import matplotlib.patches as patches

from src.utils.transforms import resize as resize_data
from src.metrics.recall_precision_ap import recall_precision_ap

from PIL import Image
from scipy.sparse import csr_matrix

def get_metrics_per_threshold(results, target, score_threshold, get_data=False):
    final_results = [{
                        **res,
                        "boxes": res["boxes"][res["scores"] > score_threshold],
                        "labels": res["labels"][res["scores"] > score_threshold],
                        "scores": res["scores"][res["scores"] > score_threshold]
                    } for res in results]
    recall, precision, ap = recall_precision_ap(target, final_results, 0.7)
    f1_score = round(2/(1/np.maximum(recall, np.finfo(np.float64).eps)+1/np.maximum(precision, np.finfo(np.float64).eps)),3)
    ret = recall, precision, ap, f1_score
    if get_data:
        return (*ret, final_results)
    return ret

def get_image_info(labels_dir, image_id):
    '''
    Args:
        labels_dir (str): Path to the xml files containing the annotations of the images 
        image_id (str): Image id which forms the name of the xml file

    Returns:
            Pandas.DataFrame: Dataframe containing the bounding boxes and the annotator of each bounding box
    '''
    with open(os.path.join(labels_dir, f'{image_id}.xml'), 'rb') as fd:
        xml_dict = xmltodict.parse(fd)
    if 'object' not in xml_dict["annotation"]:
        data=[]
    else:
        data=[(str(image_id), r['name'], *r['bndbox'].values()) for r in xml_dict["annotation"]["object"]]
    bboxes = pd.DataFrame(
        data,
        columns=['image_id','annotator', 'xmin', 'ymin', 'xmax', 'ymax']).astype(np.int)
    return bboxes


def get_item(data_holder, image_id, resize=False, min_size=None, max_size=None, device=torch.device("cpu")):
    '''
    Args:
        data_holder (CustomDatasetDataLoader): Object containing the attributes dataset and dataloader
        image_id (str): image id of the item that want to be collected
        resize (bool): Whether to resize the image or not. The same way torchvision resizes the images
        min_size (int): if resize is true, the minimum size that an axis can get when resizing.
        max_size (int): if resize is true, the maximum size that an axis can get when resizing.
        device (torch.device): torch device to store the collected data

    Returns:
            np.array: Image of shape WxHx3
            dict: Contains the target in the same format torchvision works 
    '''
    image, target = data_holder.dataloader.collate_fn([data_holder.dataset[image_id]])
    target = [{key : elem.to(device) if type(elem) == torch.Tensor else elem for key, elem in t.items()} for t in target][0]
    image = image[0]
    if resize:
        image = image.permute(2, 0, 1).to(device)
        image, target = resize_data(image, target, min_size, max_size)
        image = image.permute(1, 2, 0).to(device)
    return image, target

def plot_image(data_holder, image_id, boxes=None, show_gt=True, title='', resize=False, min_size=None, max_size=None):
    fig, ax = plt.subplots(figsize=(20,30))
    scores = None
    image, target = get_item(data_holder, image_id, resize=resize, min_size=min_size, max_size=max_size)
    image = image.cpu().numpy()
    ax.imshow(image)
    
    if show_gt:
        for bbox in target['boxes'].cpu().numpy():
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
    if boxes is not None:
        for bbox in (boxes if type(boxes) != torch.Tensor else boxes.cpu().numpy()):
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=2, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)

    plt.title(title)
    fig.show()
    
def apply_nms(results, nms_threshold):
    nms_idx = [torch.ops.torchvision.nms(r["boxes"], r["scores"], nms_threshold) for r in results]
    results_nms = [{
                        **res,
                        "boxes": res['boxes'][nms_idx[i]],
                        "labels": res['labels'][nms_idx[i]],
                        "scores": res['scores'][nms_idx[i]]
                    } for i, res in enumerate(results)]
    return results_nms
    
def get_unet_masks(trainer, image_ids=None, loader=None):
    if loader is None:
        loader = trainer.val_loaders[0][1]
    if image_ids is None:
        image_ids = loader.dataset.ids
    for image_id in image_ids:
        item = loader.dataset[image_id]
        data_dd = [item["image"]]
        target_dd = [item["target"]]
        
        print(f"starting with {image_id}")
        
        full_im_shape = data_dd[0].shape[:-1]
        full_mask = torch.zeros(full_im_shape)
        full_image = torch.zeros(data_dd[0].shape)
        full_mask_count = torch.zeros(full_im_shape)
        target_dd[0]["boxes"]= torch.tensor([[0,0,0,0]])
        target_dd[0]["masks"]= torch.zeros((1,*full_im_shape))
        target_dd[0]["area"]= torch.tensor([0])
        target_dd[0]["labels"]= torch.tensor([0])
        for data_d, target_d in trainer.custom_collate([[data_dd, target_dd]]):
            data, target = trainer.transform_data(data_d, target_d)
            with torch.no_grad():
                res = trainer.forward(trainer.model, data, target)
            curr_pred_probs = res[0][1].cpu().numpy()
            slice_i = target[0]["current_slice"][:-1]
            full_mask[slice_i] += torch.from_numpy(np.array(Image.fromarray(curr_pred_probs).resize(data_d[0].shape[:-1], Image.NEAREST)))
            full_mask_count[slice_i] += 1
            full_image[slice_i] = data_d[0].cpu()
        full_mask_probs = full_mask/full_mask_count
        mask_probs_crf = torch.zeros((2,*full_mask_probs.shape))
        mask_probs_crf[0] = 1 - full_mask_probs
        mask_probs_crf[1] = full_mask_probs
        image = loader.dataset.__getitem__(image_id)["image"]
        image = np.ascontiguousarray((image*255).type(torch.uint8).cpu().numpy())
        crf_probs = np.array([1,2])#crf_post_process(image, mask_probs_crf, *image.shape[:-1])[1]
        yield {
            "image_id": image_id,
            "unet_mask_0,9": csr_matrix((full_mask_probs.cpu().numpy() > 0.9).astype(np.byte)),
            "crf_mask_0,5": csr_matrix((crf_probs > 0.5).astype(np.byte)),
            "crf_mask_0,9": csr_matrix((crf_probs > 0.9).astype(np.byte)),
            "full_image": full_image.cpu().numpy()
        }