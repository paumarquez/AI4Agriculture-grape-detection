
import torch
from torch import optim
from torch.utils.data import DataLoader

import os
from copy import deepcopy
from datetime import datetime
import json
import numpy as np

import logging
import wandb

from .data import CreateDataLoader
from .models import create_model
from .metrics import Metrics

class DLTemplate:
    def __init__(self, opt, metrics_list=[]):
        self.opt = opt
        self.init_device()
        self.metrics = Metrics(metrics_list)
        self.LATEST_CHECKPOINT_NAME = "model_latest.pt"
        self.BEST_CHECKPOINT_NAME = "model_best.pt"
        self.init_logger()
        self.init_loss()
        self.init_transform()
 
    def init_loss(self):
        pass

    def init_transform(self):
        pass

    def init_logger(self):
        self.logging = logging
        self.logging.basicConfig(format=f'{str(datetime.now())[:19]}:%(message)s', level=logging.INFO)

    def log(self, msg, training=False):
        #self.logging.info
        if training:
            print(f'{str(datetime.now())[:19]} | [{self.opt.phase.upper()}] | Epoch: {self.epoch+1} Steps: {self.total_steps} | {msg}')
        else:
            print(f'{str(datetime.now())[:19]} | {msg}')

    def init_device(self):
        if torch.cuda.is_available() and self.opt.device != "cpu":
            self.device = torch.device("cuda")
            self.log('Using CUDA device')
        else:
            self.device = torch.device("cpu")
            self.log('Using CPU device')
    
    def set_device(self, new_device):
        self.model.to(torch.device(new_device))
        self.device = torch.device(new_device)
    
    def get_data_loader(self, opt, split, purpose=None, dataset_name=None):
        if purpose is None:
            purpose=split
        new_opt = deepcopy(opt)
        new_opt.phase = split
        new_opt.purpose = purpose
        if dataset_name:
            new_opt.dataset = dataset_name
        loader = CreateDataLoader(new_opt)
        return loader

    def create_optimizer(self):
        raise Exception("Implement get_optimizer function")

    def init_model(self):
        self.model = create_model(self.opt).to(self.device)

    def get_loss_from_model(self, res, target):
        return res

    def update_scheduler(self):
        pass

    def forward(self, model, data, target):
        return model(data, target)

    def transform_data(self, data, target):
        return data, target

    def custom_collate(self, loader):
        return loader

    def train_epoch(self):
        self.model.train()
        self.metrics.init_stats()
        for data_d, target_d in self.custom_collate(self.train_loader):
            data, target = self.transform_data(data_d, target_d)
            self.total_steps += len(data)
            res = self.forward(self.model, data, target)
            loss = self.get_loss_from_model(res, target)
            loss_info = None
            if type(loss) != torch.Tensor:
                loss, loss_info = loss
            self.on_train_forward(res, loss=loss, n=len(data), loss_info=loss_info)
            loss.backward()

            self.metrics.update_metric("train_loss", loss.item(), n=len(data))

            if self.total_steps % self.opt.batch_size == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()


            if self.total_steps // self.opt.display_freq != (self.total_steps-len(data)) // self.opt.display_freq:
                self.on_train_display()
                self.opt.phase = "val"
                for dataset_name, data_loader in self.val_loaders:
                    self.validate(data_loader, split = "val", dataset_name=dataset_name)
                self.opt.phase = "train"
                if self.opt.validate_train_split:
                    for dataset_name, data_loader in self.train_val_loaders:
                        self.validate(data_loader, split = "train", dataset_name=dataset_name)

            if self.total_steps // self.opt.save_latest_freq != (self.total_steps-len(data)) // self.opt.save_latest_freq:
                self.store_checkpoint(self.LATEST_CHECKPOINT_NAME)
            
            if self.opt.save_periodically_freq and self.total_steps // self.opt.save_periodically_freq != (self.total_steps-len(data)) // self.opt.save_periodically_freq:
                self.store_checkpoint(f"model_step_{self.total_steps}.pt")
            
            if self.total_steps // self.opt.update_constraints_scheduler_freq != (self.total_steps-len(data)) // self.opt.update_constraints_scheduler_freq:
                self.update_scheduler()
            
            if self.opt.lr_scheduler and self.opt.lr_scheduler_method == "tickBased":
                for i in range(self.opt.step_batch_size):
                    self.lr_scheduler.step(self.total_steps)

    def store_checkpoint(self, name):
        if not os.path.isdir(os.path.join(self.opt.checkpoints_dir, self.run_name)):
            os.mkdir(os.path.join(self.opt.checkpoints_dir, self.run_name))
        checkpoint_path = os.path.join(self.opt.checkpoints_dir, self.run_name, name)
        torch.save(self.model.state_dict(), checkpoint_path)

    def on_train_forward(self, res):
        raise Exception("on_train_forward not implemented")

    def on_train_start(self):
        raise Exception("on_train_start not implemented")

    def on_train_display(self):
        raise Exception("on_train_display not implemented")


    def on_val_forward(self, i, n, results, target, split):
        raise Exception("on_val_forward not implemented")

    def on_val_start(self, split):
        raise Exception("on_val_start not implemented")

    def on_val_end(self, split):
        raise Exception("on_val_end not implemented")

    def validate(self, data_loader, split, dataset_name):
        was_training = self.model.training
        if was_training:
            self.model.eval()
        self.metrics.init_stats()
        self.on_val_start(split, dataset_name)

        with torch.no_grad():
            for i, (data_d, target_d) in enumerate(self.custom_collate(self.train_loader)):
                data, target = self.transform_data(data_d, target_d)
                results = self.forward(self.model, data, target)

                self.on_val_forward(i * len(data), len(data_loader), results, data, target, split, dataset_name)
                if i * len(data) >= self.opt.max_val_samples:
                    break
        self.on_val_end(split, dataset_name)

        if was_training:
            self.model.train()

    def set_train_data_loaders(self):
        self.train_loader = self.get_data_loader(self.opt, split='train',purpose='train')
        self.log(f'Train set size: {len(self.train_loader)}')
    
    def set_val_data_loaders(self):
        self.train_val_loaders = []
        self.val_loaders = []
        for dataset_name in self.opt.dataset.split('_'):
            self.train_val_loaders.append(
                (dataset_name, self.get_data_loader(self.opt, split='train', purpose='val', dataset_name=dataset_name))
            )
            self.log(f'Train set size for validation, dataset {dataset_name}: {len(self.train_val_loaders[-1][1])}')
            self.val_loaders.append(
                (dataset_name, self.get_data_loader(self.opt, split='val',purpose='val', dataset_name=dataset_name))
            )
            self.log(f'Validation set size, dataset {dataset_name}: {len(self.val_loaders[-1][1])}')

    def set_data_loaders(self):
        self.set_train_data_loaders()
        self.set_val_data_loaders()

    def init_dash(self):
        if not self.opt.log_dash:
            return
        os.environ["WANDB_API_KEY"] = json.load(open('./keys.json'))["wandb_key"]
        self.wandb_run = wandb.init(name=self.run_name, project="AI4Agriculture", entity='paumarquez', config=self.opt)
        #wandb.watch(self.model)

    def log_dash(self, metrics_dict, n=None):
        if not self.opt.log_dash:
            return
        if n == None:
            n = self.total_steps
        wandb.log({
            **metrics_dict
        }, step=n)

    def init_config(self):
        self.run_name = self.opt.run_name

    def init_train(self):
        self.init_config()
        self.set_data_loaders()
        self.init_model()
        self.init_dash()
        self.optimizer = self.create_optimizer()
        self.max_metric = -np.inf
        self.min_metric = np.inf
        self.total_steps = 0
        return self

    def train(self):
        try:
            self.init_timestamp = datetime.now()
            self.init_train()
            for epoch in range(self.opt.nepochs):
                self.epoch = epoch
                self.train_epoch()
            
            self.log("Finished")
        except Exception as err:
            if (datetime.now() - self.init_timestamp).total_seconds() < 60*10:
                print(f"Deleting WANDB run {self.run_name}...")
                #self.wandb_run.delete()
            raise err
