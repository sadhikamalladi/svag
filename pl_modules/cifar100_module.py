import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import utils
from pytorch_lightning.utilities import AttributeDict

import attr
from functools import partial
from typing import List
from typing import Optional
from typing import Union
from .base_module import Base_Module

__all__ = ['CIFAR100_Module']
        
class CIFAR100_Module(Base_Module):        
    def __init__(self, hparams = None,**kwargs):
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]
        super().__init__( hparams,**kwargs) ## self.hparams = hparams
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_dataloader(self):
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(self.mean, self.std)])
        dataset = CIFAR100(root=self.hparams.data_dir, train=True, transform=transform_train)
        dataloader = DataLoader(dataset, batch_size=self.hparams.train_batch_size, num_workers=self.hparams.num_data_workers, shuffle=True, drop_last=self.hparams.drop_last_batch, pin_memory=self.hparams.pin_data_memory)
        return dataloader
    
    def val_dataloader(self):
        transform_val = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(self.mean, self.std)])
        dataset = CIFAR100(root=self.hparams.data_dir, train=False, transform=transform_val)
        dataloader = DataLoader(dataset, batch_size=self.hparams.test_batch_size, num_workers=self.hparams.num_data_workers, pin_memory=self.hparams.pin_data_memory)
        return dataloader

    
    

        
