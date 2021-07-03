import torch
import copy
from itertools import chain, cycle

from .data_loader import CreateDataLoader

def AI4EU_collate_data_fn(batch):
    data = [item['image'] for item in batch]
    target = [item['target'] for item in batch]
    return [data, target]

class DataLoaderFusion():
    def __init__(self, a, b):
        if len(a) < len(b):
            self.a = a
            self.b = b
        else:
            self.b = a
            self.a = b

    def __iter__(self):
        return chain.from_iterable(zip(cycle(self.a), self.b))

    def __len__(self):
        return len(self.a) + len(self.b)

def CreateDataset(opt):
    dataset = None

    if opt.dataset == 'AI4EU':
        from .AI4EU_dataset import AI4EU_Dataset
        dataset = AI4EU_Dataset(
            labels_dir=opt.labels_dir,
            images_dir=opt.images_dir,
            split_file=opt.split_path,
            split=opt.phase,
            annotators=opt.annotators,
            annotator_draw=opt.annotator_draw,
            min_area=opt.min_area if opt.purpose == "train" else 0,
            partitioning_patch_size=opt.partitioning_patch_size,
            calc_mask=opt.data_calc_mask,
            n_images=opt.n_images
        )
    elif opt.dataset == 'WGISD':
        from .WGISD_dataset import WGISD_Dataset
        dataset = WGISD_Dataset(
            labels_dir=opt.wgisd_labels_dir,
            images_dir=opt.wgisd_images_dir,
            split_file=opt.wgisd_split_path,
            split=opt.phase,
            min_area=opt.min_area if opt.purpose == "train" else 0
        )
    elif opt.dataset == "AI4EU_WGISD":
        new_opt = copy.deepcopy(opt)
        new_opt.dataset = 'AI4EU'
        AI4EU_dataset = CreateDataset(new_opt)
        new_opt.dataset = 'WGISD'
        WGISD_dataset = CreateDataset(new_opt)
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset)

    print("dataset [%s] was created" % (dataset.name()))
    return dataset

class CustomDatasetDataLoader():
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        self.opt = opt
        if '_' in opt.dataset:
            dataloaders = []
            assert len(opt.dataset.split('_')) == 2
            for dataset_name in opt.dataset.split('_'):
                new_opt = copy.deepcopy(opt)
                new_opt.dataset = dataset_name
                dataset = CreateDataset(new_opt)
                dataloaders.append(torch.utils.data.DataLoader(
                    dataset,
                    batch_size=opt.step_batch_size,
                    shuffle=opt.purpose == "train",
                    num_workers=int(opt.nThreads),
                    collate_fn=AI4EU_collate_data_fn
                ))
            self.dataloader = DataLoaderFusion(*dataloaders)
        else:
            self.dataset = CreateDataset(opt)
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.step_batch_size,
                shuffle=opt.purpose == "train",
                num_workers=int(opt.nThreads),
                collate_fn=AI4EU_collate_data_fn
            )


    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for data in self.dataloader:
            yield data
