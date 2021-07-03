import torch
from .fasterRCNN_resnet import fasterrcnn_resnet50_fpn
from .residualUNet import ResidualUNet

from torchvision.models.detection.anchor_utils import AnchorGenerator

def copy_parameters(init_model, pretrained_model):
    init_params = dict(init_model.named_parameters())
    params_not_to_copy = tuple(['rpn.head.cls', 'rpn.head.bbox_pred', 'roi_heads'])
    for name, param in pretrained_model.named_parameters():
        if not name.startswith(params_not_to_copy):
            init_params[name].data.copy_(param.data)

def create_model(opt):

    if opt.model == 'fasterRCNN':
        if opt.custom_anchor_widths:
            anchor_generator = AnchorGenerator(
                sizes=opt.anchor_widths, aspect_ratios=opt.anchor_ar# stride_factor=opt.anchor_stride_factor
            )
        model = fasterrcnn_resnet50_fpn(
            pretrained=opt.pretrained,
            pretrained_backbone=True,
            trainable_backbone_layers=opt.trainable_backbone_layers,
            progress=False,
            num_classes = (2 if (not opt.pretrained) else 91),
            rpn_anchor_generator = anchor_generator if opt.custom_anchor_widths else None,
            box_nms_thresh=opt.box_nms_thresh,
            returned_layers=opt.backbone_return_layers,
            min_size=opt.transform_min_size,
            max_size=opt.transform_max_size
        )
        if opt.partially_pretrained:
            pretrained_model = fasterrcnn_resnet50_fpn(pretrained=True, progress=False)
            print('Copying not custom parameters from pretrained model...')
            copy_parameters(model, pretrained_model)
            del pretrained_model
    elif opt.model == "ResidualUNet":
        model = ResidualUNet(3, 2) # 3 = RGB, 2 = N_Classes
        model.init_weights()
    else:
        raise Exception(f'Model {opt.model} not found')

    if opt.checkpoint:
        print(f'Loading checkpoint...')
        model.load_state_dict(torch.load(opt.checkpoint, map_location=torch.device('cpu')))
    
    print("model [%s] was created" % (opt.model))

    return model
