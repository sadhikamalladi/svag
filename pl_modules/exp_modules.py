import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
import utils
from pytorch_lightning.utilities import AttributeDict

import attr
from functools import partial
from typing import List
from typing import Optional
from typing import Union
from .cifar10_module import CIFAR10_Module
from .cifar100_module import CIFAR100_Module
from .svhn_module import SVHN_Module

import os
import pickle

import numpy as np

__all__ = ['SDELimit1_CIFAR10_Module','SDELimit2_CIFAR10_Module',
           'SDELimit1_CIFAR100_Module', 'SDELimit1_SVHN_Module',
           'NGD_Module', 'Subsampled_SGD']
        
class SDELimit1_CIFAR10_Module(CIFAR10_Module):        
    def __init__(self, hparams = None,**kwargs):
        super().__init__( hparams,**kwargs)
        
        self.k = hparams.batch_k
        c = np.sqrt(2*self.k - 1)
        self.c1 = (1+c)/2
        self.c2 = (1-c)/2

    def training_step(self, batch, batch_idx):
        examples, labels = batch
        B = examples.shape[0]
        b1 = (examples[:B//2], labels[:B//2])
        b2 = (examples[B//2:], labels[B//2:])

        loss1, acc1 = self.forward(b1)
        loss2, acc2 = self.forward(b2)

        loss = self.c1 * loss1 + self.c2 * loss2
        acc = (acc1 + acc2) / 2

        logs = {'loss/train': loss, 'accuracy/train': acc}
        self.log_dict(logs)
        if self.hparams.verbose:
            print(logs)
            
        return loss

class SDELimit1_CIFAR100_Module(CIFAR100_Module):        
    def __init__(self, hparams = None,**kwargs):
        super().__init__( hparams,**kwargs)
        
        self.k = hparams.batch_k
        c = np.sqrt(2*self.k - 1)
        self.c1 = (1+c)/2
        self.c2 = (1-c)/2

    def training_step(self, batch, batch_idx):
        examples, labels = batch
        B = examples.shape[0]
        b1 = (examples[:B//2], labels[:B//2])
        b2 = (examples[B//2:], labels[B//2:])

        loss1, acc1 = self.forward(b1)
        loss2, acc2 = self.forward(b2)

        loss = self.c1 * loss1 + self.c2 * loss2
        acc = (acc1 + acc2) / 2

        logs = {'loss/train': loss, 'accuracy/train': acc}
        self.log_dict(logs)
        if self.hparams.verbose:
            print(logs)
            
        return loss

class SDELimit1_SVHN_Module(SVHN_Module):        
    def __init__(self, hparams = None,**kwargs):
        super().__init__( hparams,**kwargs)
        
        self.k = hparams.batch_k
        c = np.sqrt(2*self.k - 1)
        self.c1 = (1+c)/2
        self.c2 = (1-c)/2

    def training_step(self, batch, batch_idx):
        examples, labels = batch
        B = examples.shape[0]
        b1 = (examples[:B//2], labels[:B//2])
        b2 = (examples[B//2:], labels[B//2:])

        loss1, acc1 = self.forward(b1)
        loss2, acc2 = self.forward(b2)

        loss = self.c1 * loss1 + self.c2 * loss2
        acc = (acc1 + acc2) / 2

        logs = {'loss/train': loss, 'accuracy/train': acc}
        self.log_dict(logs)
        if self.hparams.verbose:
            print(logs)
            
        return loss

class SDELimit2_CIFAR10_Module(CIFAR10_Module):        
    def __init__(self, hparams = None,**kwargs):
        super().__init__( hparams,**kwargs)
        
        self.k = hparams.batch_k
        self.c = np.sqrt(self.k - 1)


    def training_step(self, batch, batch_idx):
        examples, labels = batch

        loss, acc = self.forward(batch)

        logs = {'loss/train': loss, 'accuracy/train': acc}
        self.log_dict(logs)
        if self.hparams.verbose:
            print(logs)
            
        return loss*(1+self.c*(2*torch.bernoulli(torch.tensor(0.5))-1))

class DatasetWithNoise(Dataset):
    def __init__(self, data, inds, B):
        self.cifar_data = data
        self.inds = inds
        self.refresh_noise(len(self.inds), B)

    def refresh_noise(self, N, B):
        gaussians = torch.normal(mean=0., std=1., size=(N,))
        avg_noise = gaussians.mean()
        diff = gaussians - avg_noise
        scaled_diff = np.sqrt(N/B) * diff
        self.deltas = 1 + scaled_diff

    def __len__(self):
        return len(self.inds)

    def __getitem__(self, idx):
        cifar_img, cifar_label = self.cifar_data[self.inds[idx]]
        return cifar_img, cifar_label, self.deltas[idx]

class SubsampledDataset(Dataset):
    def __init__(self, data, inds):
        self.cifar_data = data
        self.inds = inds

    def __len__(self):
        return len(self.inds)

    def __getitem__(self, idx):
        return self.cifar_data[self.inds[idx]]

class Subsampled_SGD(CIFAR10_Module):
    def __init__(self, hparams = None, **kwargs):
        self.N = hparams.num_examples
        super().__init__(hparams, **kwargs)

    def train_dataloader(self):
        # no data augmentation
        transform_train = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(self.mean, self.std)])
        dataset = CIFAR10(root=self.hparams.data_dir,
                          train=True,
                          transform=transform_train,
                          download=True)

        # use a fixed random subset of 10240 = 2560*4 examples
        inds_path = os.path.join(self.hparams.data_dir, f'CIFAR10_random{self.N}.pkl')
        if os.path.exists(inds_path):
            inds = pickle.load(open(inds_path, 'rb'))
        else:
            inds = np.random.randint(0, len(dataset), self.N)
            pickle.dump(inds, open(inds_path, 'wb'))
            
        self.sub_data = SubsampledDataset(dataset, inds)
        
        dataloader = DataLoader(self.sub_data,
                                batch_size=self.hparams.train_batch_size,
                                num_workers=self.hparams.num_data_workers,
                                drop_last=False,
                                shuffle=False,
                                pin_memory=self.hparams.pin_data_memory)
        return dataloader

class NGD_Module(CIFAR10_Module):        
    def __init__(self, hparams = None,**kwargs):

        self.N = hparams.num_examples
        self.B = hparams.orig_batch_size
        
        super().__init__( hparams,**kwargs)

        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.train_size = self.N

    def train_dataloader(self):
        # no data augmentation
        transform_train = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(self.mean, self.std)])
        dataset = CIFAR10(root=self.hparams.data_dir,
                          train=True,
                          transform=transform_train,
                          download=True)

        # use a fixed random subset of 10240 = 2560*4 examples
        inds_path = os.path.join(self.hparams.data_dir, f'CIFAR10_random{self.N}.pkl')
        if os.path.exists(inds_path):
            inds = pickle.load(open(inds_path, 'rb'))
        else:
#             inds = np.random.randint(0, len(dataset), self.N)
            inds = np.random.choice(len(dataset), self.N, replace=False)
            pickle.dump(inds, open(inds_path, 'wb'))
            
        self.dataset_with_noise = DatasetWithNoise(dataset, inds, self.B)
        
        dataloader = DataLoader(self.dataset_with_noise,
                                batch_size=self.hparams.train_batch_size,
                                num_workers=self.hparams.num_data_workers,
                                drop_last=False,
                                shuffle=True,
                                pin_memory=self.hparams.pin_data_memory)
        return dataloader

    def validation_step(self, batch, batch_nb):
        losses, acc = self.forward(batch)
        loss = losses.mean() # added this line b/c reduction=none in criterion

        logs = {'loss': loss, 'acc': acc, 'batch_size':batch[0].size(0)}
        return logs
    
    def validation_step_end(self, batch_parts):
        result= self.aggregate(batch_parts)

        logs = {'loss/val': result[0], 'corrects': result[1]}
        if self.hparams.verbose:
            print(logs)
        return logs
    
    def training_step(self, batch, batch_idx):
        examples, labels, noise = batch
        losses, acc = self.forward((examples, labels))
        
        if self.hparams.gaussian_noise:
            losses = losses * noise
#         loss = losses.mean()/self.hparams.grad_accumulate
        loss = losses.mean()
        logs = {'loss': loss, 'acc': acc , 'batch_size':batch[0].size(0)}
        return logs

    def training_step_end(self, batch_parts):
        if self.local_rank == 0:
            self.dataset_with_noise.refresh_noise(self.N, self.B)
        result= self.aggregate(batch_parts)
        loss = result[0]/result[2]
        acc = result[1]/result[2]
#         logs = {'loss/train': loss*self.hparams.grad_accumulate, 'accuracy/train': acc}
        logs = {'loss/train': loss, 'accuracy/train': acc}
        if self.hparams.verbose:
            print(logs)
        self.log_dict(logs)
        return loss
    
    def aggregate(self,batch_parts):

        batch_parts['batch_size'] = torch.tensor(batch_parts['batch_size'], device = 'cuda:0')
        total_loss = (batch_parts['loss']*batch_parts['batch_size']).sum()
        total_acc = (batch_parts['acc']*batch_parts['batch_size']).sum()
        total_batch_size = (batch_parts['batch_size']).sum()
        return total_loss,total_acc,total_batch_size
