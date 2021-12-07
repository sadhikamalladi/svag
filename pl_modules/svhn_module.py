import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets import SVHN
from torch.utils.data import DataLoader
import utils
from pytorch_lightning.utilities import AttributeDict

import attr
from functools import partial
from typing import List
from typing import Optional
from typing import Union
from .base_module import Base_Module

__all__ = ['SVHN_Module']
        
class SVHN_Module(Base_Module):        
    def __init__(self, hparams = None,**kwargs):
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        super().__init__( hparams,**kwargs) ## self.hparams = hparams
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_dataloader(self):
        transform_train = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(self.mean, self.std)])
        dataset = SVHN(root=self.hparams.data_dir, split='train', transform=transform_train, download=True)
        dataloader = DataLoader(dataset, batch_size=self.hparams.train_batch_size, num_workers=self.hparams.num_data_workers, shuffle=True, drop_last=self.hparams.drop_last_batch, pin_memory=self.hparams.pin_data_memory)
        return dataloader
    
    def val_dataloader(self):
        transform_val = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(self.mean, self.std)])
        dataset = SVHN(root=self.hparams.data_dir, split='test', transform=transform_val, download=True)
        dataloader = DataLoader(dataset, batch_size=self.hparams.test_batch_size, num_workers=self.hparams.num_data_workers, pin_memory=self.hparams.pin_data_memory)
        return dataloader

    
    

        
