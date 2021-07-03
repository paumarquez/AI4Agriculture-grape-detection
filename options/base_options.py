import argparse
import os
import torch
from datetime import datetime
import json

def str_to_type(s, type=int):
    if s is None:
        return s
    str_numbers = s.split(',')
    numbers = []
    for str_number in str_numbers:
        number = type(str_number)
        if number >= 0:
            numbers.append(number)
    return numbers

def load_hyperparameters(opt):
    if opt.custom_anchor_widths:
        # From small to larger
        # The length must match the number of feature layers extracted from backbone
        # Check out https://gist.github.com/samson-wang/a6073c18f2adf16e0ab5fb95b53db3e6 to see
        # the receptive field of each resnet layer
        # k means scales and AR:
        # [(30.,), (54,), (71,), (110,), (154,)]
        # [(1.02,), (0.84,), (1.06,), (0.58,), (0.68,)]
        if opt.backbone_return_layers == [1,2,3,4]:
            #opt.anchor_widths = [(30., 54.), (71, 110,150), (180, 260, 300), (300,400,500), (350, 450, 600)]
            #opt.anchor_ar = [(1.02, 0.84, 1.06), (0.58, 0.68), (1, 0.6), (0.75,1.0), (0.75, 1.0)]
            opt.anchor_widths = [(27, 40), (57, 83,107), (57, 83,107), (143, 193, 260), (300, 350, 400)][::-1]
            opt.anchor_ar = [(0.9, 0.84, 1.15), (1.51, 0.9),(1.51, 0.9), (1.5, 1), (1.5,1.0)][::-1]
        elif opt.backbone_return_layers == [1,2]:
            #opt.anchor_widths = [(30, 54, 71), (110, 154, 180), (180, 250, 300)]
            #opt.anchor_ar = [(1.02, 0.84), (0.58, 0.68), (0.75, 1.2)]
            opt.anchor_widths = [(27, 80), (107, 193), (260, 400)]
            opt.anchor_ar = [(1), (1.3), (1.3)]
        elif opt.backbone_return_layers == [1,2, 3]:
            # v7-3-3
            opt.anchor_widths = [(30., 54.), (71, 110,150), (180, 260, 300), (300,400,500)]
            opt.anchor_ar = [(1.02, 0.84, 1.06), (0.58, 0.68), (1, 0.6), (0.75,1.0)]
            #opt.anchor_widths = [(27, 40), (57, 83,107), (143, 193, 260), (300, 350, 400)]
            #opt.anchor_ar = [(0.9, 0.84, 1.15), (1.51, 0.9), (1.5, 1), (1.5,1.0)]
            #opt.anchor_widths = [(30.0, 54.0), (71, 110, 150), (180, 260, 300), (300, 400, 500)]
            #opt.anchor_ar = [(1.0, 0.84, 1.1), (1.0, 1.3), (1.0, 1.4), (1.5, 1.0)]
        elif opt.backbone_return_layers == [1]:
            opt.anchor_widths = [(27, 40, 57, 83), (143, 193, 260, 350)]
            opt.anchor_ar = [(0.9, 1.15), (1.51, 0.9)]
        else:
            raise Exception('custom anchor widths are not set correctly with backbone_return_layers')
    return opt

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--version', type=str, default='debug', help='Version of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu id to set the device (currently only one supported)')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--checkpoint', type=str, default='', help='state dictionary to load to the model')
        self.parser.add_argument('--model', type=str, default='fasterRCNN', help='model: fasterRCNN, cascade')
        self.parser.add_argument('--dataset', type=str, default='AI4EU', help='dataset: AI4EU|WGISD|AI4EU_WGISD')
        self.parser.add_argument('--seed', type=int, default=1, help='pyTorch seed')
        self.parser.add_argument('--log_dash', type=int, default=1, help='whether to log to a dash (wandb or tensorboard)')

        # hyperparameters
        self.parser.add_argument('--pretrained', type=bool, nargs= '?', const=True, default=False, help='')
        self.parser.add_argument('--partially_pretrained', type=bool, nargs= '?', const=True, default=False, help='')
        self.parser.add_argument('--backbone_return_layers', type=str, default='1,2,3,4', help='WARNING: Set anchor widths and AR accordingly')
        self.parser.add_argument('--trainable_backbone_layers', type=int, default=None, help='Number of backbone layers to train (from 0 to 5)')
        self.parser.add_argument('--box_nms_thresh', type=float, default=0.4)

        self.parser.add_argument('--custom_anchor_widths', type=bool, nargs= '?', const=True, default=False, help='')
        self.parser.add_argument('--anchor_stride_factor', type=float, default=1.0)

        self.parser.add_argument('--transform_min_size', type=int, default=800)
        self.parser.add_argument('--transform_max_size', type=int, default=1333)


        # input/output sizes
        self.parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
        self.parser.add_argument('--step_batch_size', type=int, default=-1, help='input size per forward, used if data don\'t find in GPU')

        # for setting inputs (data related)
        self.parser.add_argument('--labels_dir', type=str, help='Labels directory')
        self.parser.add_argument('--split_path', type=str, help='Split file or folder')
        self.parser.add_argument('--annotators', type=str, help='Annotators')
        self.parser.add_argument('--images_dir', type=str,help='Images directory')
        self.parser.add_argument('--annotator_draw', type=int, default=0)
        self.parser.add_argument('--min_area', type=int, default=0, help='filter bounding boxes with an area smaller than this value')
        self.parser.add_argument('--data_calc_mask', type=bool, nargs= '?', const=True, default=False, help='')
        self.parser.add_argument('--n_images', type=int, default=None)
        self.parser.add_argument('--skip_notbbox_slices', type=bool, nargs= '?', const=True, default=False, help='Skip slices without bbox when loading data (custom_collator)')
        self.parser.add_argument('--skip_bbox_slices', type=bool, nargs= '?', const=True, default=False, help='')
        self.parser.add_argument('--partitioning_patch_size', default=None, type=int, help='If None, partition is not done. Otherwise, it\'s the size of the squared partitions of the image to apply to the forward.')
        
        # For wgisd dataset
        self.parser.add_argument('--wgisd_labels_dir', type=str,default=None, help='Labels directory')
        self.parser.add_argument('--wgisd_split_path', type=str,default=None, help='Split file or folder')
        self.parser.add_argument('--wgisd_images_dir', type=str,default=None, help='Images directory')

        # for data augmentation
        self.parser.add_argument('--nThreads', default=0, type=int, help='# threads for loading data')

        # for weakly supervised learning
        self.parser.add_argument('--weak_loss_d', default=5, type=int, help='width of the lines for each constrain in the tightness prior loss')
        self.parser.add_argument('--weak_loss_init_t', default=5, type=int, help='initialization of the t parameter for log barrer extensions')
        self.parser.add_argument('--weak_loss_margins', default='0.5,0.85', type=str, help='percentage of positive area of the bounding box (min, max)')
        self.parser.add_argument('--weak_loss_lambda', default=0.0001, type=float, help='percentage of positive area of the bounding box (min, max)')
        self.parser.add_argument('--weak_loss_alpha', default=1, type=float, help='multiplier to negative loss')
        self.parser.add_argument('--weak_loss_cache_folder', default='./cache', type=str, help='percentage of positive area of the bounding box (min, max)')


    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        self.opt.gpu_ids = str_to_type(self.opt.gpu_ids)
        self.opt.annotators = str_to_type(self.opt.annotators)
        self.opt.backbone_return_layers = str_to_type(self.opt.backbone_return_layers)
        self.opt.weak_loss_margins = str_to_type(self.opt.weak_loss_margins, type=float)
        self.opt.lr_scheduler_nticks = str_to_type(self.opt.lr_scheduler_nticks)
        
        if self.opt.n_images is not None:
            raise Exception("Training with a limited number of images! Are you sure?")
        
        assert not (self.opt.skip_notbbox_slices and self.opt.skip_notbbox_slices == self.opt.skip_bbox_slices), "both can not be true"

        if self.opt.step_batch_size == -1:
            self.opt.step_batch_size = self.opt.batch_size
        # step Batch Size must be multiple of step batch size
        if self.opt.batch_size % self.opt.step_batch_size != 0:
            raise Exception("step_batch_size must be multiple of step_batch_size")

        if self.opt.anchor_stride_factor != 1 and not sel.opt.custom_anchor_widths:
            raise Exception('Custom anchor widths is not set, anchor stride factor can not be different than 1')

        self.opt = load_hyperparameters(self.opt)
        self.opt.run_name = f"{self.opt.model}_{self.opt.version}_{str(datetime.now())[:-7]}"

        # set gpu ids
        if self.opt.device != "cpu" and len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk        
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.run_name)
        os.mkdir(expr_dir)
        if save:
            file_name = os.path.join(expr_dir, 'opt.json')
            with open(file_name, 'wt') as opt_file:
                json.dump(args, opt_file)
        return self.opt
