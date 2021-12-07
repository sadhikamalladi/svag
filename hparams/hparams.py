import copy
import math
import warnings
import random
from functools import partial
from typing import List
from typing import Optional
from typing import Union

import attr
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities import AttributeDict
from torch.utils.data import DataLoader
import pandas as pd


__all__ = ['HParams', 'CIFAR10_HParams']


@attr.s(auto_attribs=True)
class HParams:
    # model selection
    arch: str = "resnet18"
    train_batch_size: int = 128
    test_batch_size: int = 256  #same for validation
        
    train_size: int = 128
    test_size: int = 256  #same for validation
    task_type: str = 'classification'

    # optimization parameters
    lr: float = 0.5
    momentum: float = 0.9
    weight_decay: float = 1e-4
    max_epochs: int = 160

    optimizer_name: str = "sgd"
    schedule : List[int] = []
    lr_decay_factors : List[int] = []
    wd_decay_factors : List[int] = []

    # W&B parameters
    # entity defines your team on W&B
    wb_entity : str = "sde-limit"
    wb_project : str = "test"

    # data loader parameters
    num_data_workers: int = 4
    drop_last_batch: bool = False
    pin_data_memory: bool = True
    check_val_every_n_epoch: int = 1
    verbose:bool = False
#     suffix:str = ''
    def to_dict(self):
        d = pd.json_normalize(vars(self), sep='_').to_dict(orient='records')[0]
#         print(d)
        
        tmp = {}
        for (k,v) in d.items():
            if not (isinstance(v,int) or isinstance(v,str) or isinstance(v,float) or isinstance(v,bool) or (v is None)):
                tmp[k] = str(v)
            else:
                tmp[k] = v
        return tmp
    
    
@attr.s(auto_attribs=True)
class CIFAR10_HParams(HParams):
    # model selection
    arch: str = "resnet18"
    depth: int = 34
    num_classes :int =10
    norm_method: str = 'BN'  # 'None' for no normalization
        
    is_bias: bool = True
    train_batch_size: int = 128
    test_batch_size: int = 256
    train_size: int = 50000
    test_size: int = 10000  
    fix_last_layer :bool = False
    bn_affine :bool= False 
    widen_factor :int = 4
    widths :List[int]= [16,32,64] 

    homo:bool =True
    ntk_init :bool = False
    last_bn :bool = False
    EN_method : str = None

    random_flip :bool = False
    hori_flip:bool = True
    crop :bool = True
    pretrained :bool= False
    data_dir:str = 'data/'

    ####### experimental HPs #######
    batch_k: int = 1
    measure_variance: bool = False
    use_torchvision: bool = False
    num_examples: int = 2048
    orig_batch_size: int = 128
    gaussian_noise: bool = False
    grad_accumulate :int = 1
    check_val_every_n_epoch : int = 1
    sample_mode :str = 'random_shuffling'

@attr.s(auto_attribs=True)
class CIFAR100_HParams(HParams):
    # model selection
    arch: str = "resnet18"
    depth: int = 34
    num_classes: int = 100
    norm_method: str = 'BN'  # 'None' for no normalization
        
    is_bias: bool = True
    train_batch_size: int = 128
    test_batch_size: int = 256
    train_size: int = 50000
    test_size: int = 10000  
    fix_last_layer :bool = False
    bn_affine :bool= False 
    widen_factor :int = 4
    widths :List[int]= [16,32,64] 

    homo: bool =True
    ntk_init :bool = False
    last_bn :bool = False
    EN_method : str = None

    random_flip :bool = False

    pretrained :bool= False
    data_dir:str = '/n/fs/ptml/zl4/pytorch-classification/data'

    ####### experimental HPs #######
    batch_k: int = 1
    measure_variance: bool = False

@attr.s(auto_attribs=True)
class SVHN_HParams(HParams):
    # model selection
    arch: str = "resnet18"
    depth: int = 34
    num_classes :int =10
    norm_method: str = 'BN'  # 'None' for no normalization
        
    is_bias: bool = True
    train_batch_size: int = 128
    test_batch_size: int = 256
    train_size: int = 50000
    test_size: int = 10000  
    fix_last_layer :bool = False
    bn_affine :bool= False 
    widen_factor :int = 4
    widths :List[int]= [16,32,64] 

    homo:bool =True
    ntk_init :bool = False
    last_bn :bool = False
    EN_method : str = None

    random_flip :bool = False

    pretrained :bool= False
    data_dir:str = '/n/fs/ptml/zl4/pytorch-classification/data'

    ####### experimental HPs #######
    batch_k: int = 1
    measure_variance: bool = False
