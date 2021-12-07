import os,sys
import random,math
import numpy as np
import torch
from typing import Any
from typing import Callable
from typing import Optional
from pytorch_lightning.callbacks import Callback
import torch.utils.data.dataset as dataset
from pytorch_lightning.utilities import AttributeDict
import argparse
import copy

from torch.utils.data import Sampler
import attr
import torch
import torchvision

import pandas as pd

from models import my_models
import torchvision.models

def parse_args(to_be_parsed = sys.argv[1:]):

    parser = argparse.ArgumentParser(description='lightning explorer!')
    # Datasets
    parser.add_argument('-c', '--config', default='base_config', type=str)
    parser.add_argument('-i', '--imagenet', default=False, action='store_true',
                        help='train on imagenet')
    parser.add_argument('-c100', '--cifar_100', default=False, action='store_true',
                        help='train on CIFAR-100')
    parser.add_argument('-svhn', '--svhn', default=False, action='store_true',
                        help='train on SVHN')
    parser.add_argument('--ngd', default=False, action='store_true',
                        help='train with noisy gradient descent')
    parser.add_argument('--sub_sgd', default=False, action='store_true',
                        help='subsample cifar-10 + standard sgd')
    parser.add_argument('--resume', default=False, action='store_true',
                        help='resume run using checkpoint')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--suffix', default='', type=str)
    parser.add_argument('--job-id', default=None, type=int) #for tracking slurm
    parser.add_argument('--task-id', default=None, type=int,nargs='*') #for tracking slurm
    args = parser.parse_args(to_be_parsed)
    return args

# depth-first make the dictionary atrribute-like
def AttributeDict_recusion(d):
    for (k,v) in d.items():
        if isinstance(v,dict):
            d[k] = AttributeDict_recusion(v)
    return AttributeDict(d)

###############
# Sampler/Dataloader utils #
###############

class My_BatchSampler(Sampler):
    def __init__(self, dataset_size : int, batch_size: int, drop_last: bool, sample_mode:str ) -> None:
        assert(drop_last or  (sample_mode in ['random_shuffling']))
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sample_mode = sample_mode
        self.index_list = None
            
    def __iter__(self):
        if self.sample_mode == 'random_shuffling':
            self.index_list = torch.randperm(self.dataset_size).tolist()
            for i in range(len(self)):
                yield self.index_list[i*self.batch_size:(i+1)*self.batch_size]
## if index exceeds self.dataset_size, the batch will be truncated automatically
        elif self.sample_mode == 'without_replacement':
            for i in range(len(self)):
                yield list(np.random.choice(self.dataset_size,self.batch_size, replace=False))
        elif self.sample_mode == 'with_replacement':
            for i in range(len(self)):
                yield list(np.random.choice(self.dataset_size,self.batch_size, replace=True))
        elif self.sample_mode == 'fixed_sequence': ##must drop last
#             assert(self.dataset_size % self.batch_size == 0)
            if self.index_list is None:
                self.index_list = torch.randperm(self.dataset_size).tolist()
            for i in range(len(self)):
                yield self.index_list[i*self.batch_size:(i+1)*self.batch_size]
        elif self.sample_mode == 'two_without_replacement': ##must drop last
            for i in range(len(self)):
                yield list(np.random.choice(self.dataset_size,self.batch_size//2, replace=False))+ list(np.random.choice(self.dataset_size,self.batch_size//2, replace=False))
                
    def __len__(self):
        if self.drop_last:
            return self.dataset_size // self.batch_size  # type: ignore
        else:
            return (self.dataset_size + self.batch_size - 1) // self.batch_size  # type: ignore

# def flatten_nested_dict(d):
#     df = pd.json_normalize(d, sep='_')
#     return(df.to_dict(orient='records')[0])
##############################
# Callbacks:
##############################

class SC_Test(Callback):
    def __init__(self,scale:float=10.,eps:float=0.0001):
        super().__init__()
        self.scale = scale 
        self.eps = eps
        
    def on_sanity_check_start(self,trainer,pl_module):
        """
           check the scale invariance
        """
        outcome = sc_test(pl_module.model, next(iter(pl_module.test_dataloader()))[0].cuda(), scale = self.scale)
        if outcome < self.eps:
            print(f'Passed the scale invariance test. Score: {outcome}. Bar: {self.eps}')
        else:
            print(f'Failed the scale invariance test. Score: {outcome}. Bar: {self.eps}')
            
class Model_logger(Callback):
    def __init__(self,path:str,freq:int=100):
        super().__init__()
        self.freq = freq
        self.count = -1
        self.PATH = path
        
    def on_train_batch_start(self,trainer,pl_module,*args):
        """
        save at the directory of wandb by default
        """
        self.count += 1
        if self.count%self.freq==0:
            torch.save(pl_module.model.state_dict(),os.path.join(self.PATH, f"ckpt{self.count}"))
        

class Norm_projector(Callback):
    def __init__(self, norm_base = 1, method = "layerwise"):
        super().__init__()
        self.norm_base = norm_base
        self.method = method
        print(f"projection callback using {method}")

    def on_train_batch_start(self,trainer,pl_module,*args):
        """
        save at the directory of wandb by default
        """
        model = pl_module.model

        total_norm = 0
        total_layer = 0
        for (n,p) in model.named_parameters():
            if len(p.shape)==4:
                if self.method == "layerwise":
                    p.data=p.data/ p.norm().detach()*self.norm_base*np.sqrt((p.shape)[0])
                elif self.method == "nodewise":
                    p.data= p.data/ p.norm(dim=0,keepdim=True).detach() * self.norm_base
                elif self.method == "global":
                    total_norm += p.norm().detach()**2
                    total_layer += 1

        if self.method == "global":
            for (n,p) in model.named_parameters():
                if len(p.shape)==4:
                    p.data =p.data / torch.sqrt(total_norm)*np.sqrt(total_layer)*self.norm_base
#                 if '.3' in n:
#                     print(p)


"""
All the loggers below are for tensorboard logger.
"""
class Effective_Step_logger(Callback):
    def on_train_batch_start(self,trainer,pl_module,*args):
        logs = {
            'e_steps': trainer.global_step/pl_module.hparams.batch_k,
            'continuous_time': trainer.global_step * pl_module.hparams.lr,
        }
        trainer.logger.log_metrics(logs)
    
#log LR,WD
class LR_WD_Scheduler(Callback):
    def __init__(self,epoch_wise:bool=True):
        super().__init__()
        self.epoch_wise = epoch_wise
        
    def on_init_start(self,trainer):
        self.count = 0
        self.phase = 0
        
    def _schedule_lr_wd(self,trainer,pl_module):
        if self.count in pl_module.hparams.schedule:
            for group in trainer.optimizers[0].param_groups:
                group['lr'] *= pl_module.hparams.lr_decay_factors[self.phase]
                group['weight_decay'] *= pl_module.hparams.wd_decay_factors[self.phase]
            self.phase +=1
        self.count += 1
        
    def on_train_epoch_start(self,trainer,pl_module):
        if self.epoch_wise:
            self._schedule_lr_wd(trainer,pl_module)

        
    def on_train_batch_start(self,trainer,pl_module,*args):
        if not self.epoch_wise:
            self._schedule_lr_wd(trainer,pl_module)

'''
TODO: change the code computing variance
'''
class Variance_Measurement(Callback):
    def __init__(self,layer_wise:bool=False,epoch_wise:bool=False,freq:int=1):
        super().__init__()
        self.layer_wise = layer_wise
        self.epoch_wise = epoch_wise
        self.freq = freq
        self.count = -1

    def get_gradient_variance(self, trainloader, model, criterion, num = 200, p_name = '', num_examples = 12800):
        model.train()
        count = 0
        norm = {}
        total_norm = 0.
        record = {}
        total_weight_norm = 0
        weight_norm = {}
        variance = {}
        train_batch_size = None
        while count< num:
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                if count >= num:
                    break
                if train_batch_size is None:
                    train_batch_size = len(inputs)
                    num = max(num,num_examples//train_batch_size)
                else:
                    assert(train_batch_size == len(inputs)) ### must drop the last batch
                inputs, targets = inputs.cuda(),targets.cuda()
                outputs = model(inputs)
                model.zero_grad()
                loss = criterion(outputs, targets)
                loss.backward()
                #norm += model.module.meta_layers.0.2.conv1.grad.norm()**2
                for (n,p) in model.named_parameters():
                    if p.grad is not None: # TODO later: figure out why `fc.weight` is no grad
                        total_norm += p.grad.norm()**2/num
                        norm[n] = norm[n] +  p.grad.norm()**2/num if norm.__contains__(n) else     p.grad.norm()**2/num      
                        record[n] = record[n] + p.grad/num if count >0 else p.grad/num
                count += 1
        total_grad_norm = 0
        grad_norm = {}
        for (n,p) in model.named_parameters():
            if p.grad is not None:
                total_grad_norm += record[n].norm()**2
                total_weight_norm += p.norm()**2
                grad_norm[n] = grad_norm[n] +  record[n].norm()**2 if grad_norm.__contains__(n) else     record[n].norm()**2
                weight_norm[n] = weight_norm[n] +  p.norm()**2 if weight_norm.__contains__(n) else     p.norm()**2   
    #             grad_norm += record[n].norm()**2
    #             weight_norm += p.norm()**2
        #empirical_variance = torch.sqrt( total_norm - total_grad_norm )
        return total_norm, (total_grad_norm*num-total_norm)/(num-1) ## now they are unbiased
        #return (total_grad_norm, total_norm, grad_norm, norm, total_weight_norm, weight_norm)
        

    def _log_variance(self,trainer,pl_module):
        self.count +=1
        if self.count % self.freq == 0:
            trainloader = pl_module.train_dataloader()
            tot_norm,tot_grad_norm = self.get_gradient_variance(trainloader, pl_module.model, pl_module.criterion)
            empirical_variance = tot_norm - tot_grad_norm
            logs = {
                'variance/empirical_variance': empirical_variance,
                'variance/total_norm': tot_norm,
                'variance/total_grad_norm': tot_grad_norm
            }
            trainer.logger.log_metrics(logs)
            print(f'epoch：{self.count}', logs)
            
        
    def on_train_epoch_start(self,trainer,pl_module, *args):
        self._log_variance(trainer, pl_module)

    def on_train_batch_start(self,trainer,pl_module, *args):
        pass
        
        
#adajust LR and WD, only supports one optimizer and  one parameter group

class LR_WD_Logger(Callback):
    def __init__(self,epoch_wise:bool=True):
        super().__init__()
        self.epoch_wise = epoch_wise
        
    def _log_lr_wd(self,trainer,pl_module):
        logs = {
            'lr': trainer.optimizers[0].param_groups[0]['lr'],
            'weight decay': trainer.optimizers[0].param_groups[0]['weight_decay'] 
        }
        trainer.logger.log_metrics(logs)
            
    def on_train_batch_end(self,trainer,pl_module, *args):
        if not self.epoch_wise:
            self._log_lr_wd(trainer,pl_module)
        
    def on_train_epoch_end(self,trainer,pl_module, *args):
        if self.epoch_wise:
            self._log_lr_wd(trainer,pl_module)

            
#adajust LR and WD, only supports one optimizer
class Norm_Logger(Callback):
    def __init__(self,layer_wise:bool=False,epoch_wise:bool=True,freq:int=1):
        super().__init__()
        self.layer_wise = layer_wise
        self.epoch_wise = epoch_wise
        self.freq = freq
        self.count = -1

    def _log_norm(self,trainer,pl_module):
        self.count +=1
        if self.count % self.freq == 0:
            assert(hasattr(pl_module, 'model'))
            logs = {'norm/total':0}
            for n,p in pl_module.model.named_parameters():
                norm = p.norm().detach()
                if p.requires_grad:
                    logs['norm/total']+=norm**2
                if self.layer_wise:
                    logs['norm/'+ abbrv_module_name(n)] = norm
            logs['norm/total'] = torch.sqrt(logs['norm/total'])
            trainer.logger.log_metrics(logs)
        
    def on_train_epoch_start(self,trainer,pl_module, *args):
        if self.epoch_wise:
            self._log_norm(trainer,pl_module)

    def on_train_batch_start(self,trainer,pl_module, *args):
        if not self.epoch_wise:
            self._log_norm(trainer,pl_module)
        
###############
# Model utils #
###############

def get_model(model_hparams:dict,) -> torch.nn.Module:
    hparams = copy.deepcopy(model_hparams)
    if hparams.arch in my_models.__dict__.keys():
        model = my_models.__dict__[hparams.arch](**hparams)
    elif hparams.arch in torchvision.models.__dict__.keys():
        model = torchvision.models.__dict__[hparams.arch](pretrained=hparams.pretrained) 
    else:
        raise AttributeError(f"Unknown architecture {hparams.arch}")
    if hasattr(model_hparams,'EN_method') and model_hparams.EN_method is not None:
        model = my_models.ntk.EN(model,num_classes= model_hparams.num_classes,method = model_hparams.EN_method)
    print(model)
    return model

####################
# Evaluation utils #
####################

def accuracy_regression(output,targets):
    return (output*targets>0).float().mean().detach()


def accuracy_classification(output,targets):
    return (torch.max(output, 1)[1] == targets.data).float().mean().detach() 


    
def get_norm(model):
        norm = torch.tensor(0.)
        for p in model.parameters():
            norm += p.norm()**2
        return torch.sqrt(norm)
    

def print_weight(model,filter = []):
    if hasattr(model,'module'):
        model = model.module
    weight = []
    s = 'Norm:'
    for (i,(n,p)) in enumerate(model.named_parameters()):       
        if any([x in n for x in filter]):
            continue
        weight.append(p.data)
        s = s + 'P{}{}:{:.3g} '.format(i+1,abbrv_module_name(n), p.data.norm())
    return s

#filter = ['running', 'num_batches_tracked','aux']
def abbrv_module_name(name):
    name = name.replace('module.','')
    name = name.replace('bias','b')
    name = name.replace('weight','w')
    name = name.replace('features','L')        
    split_name = name.split('.')
    name = '.'.join(split_name[:])
    
    return name


# test for scale invariance 
def sc_test(model, batch, scale = 10):
    new_model = copy.deepcopy(model)
    for p in new_model.parameters():
        if p.requires_grad:
            p.data *= scale
            
    output = model(batch).detach()
    new_output = new_model(batch).detach()
    return (output-new_output).norm()/output.norm()


################################
#   for two layer
################################









##### generate_dataset


def gen_quadratic_regression(D = 40,N = 800, train = True, gauss = True,double =False):
    X = torch.randn(N,D) if gauss else -1 +2*torch.bernoulli(0.5*torch.ones(N,D))
    X = X.cuda()
#     y = (X[:,0]**2 - X[:,1]**2)/2
    y = X[:,0]*X[:,1]
    y = y.cuda()
#     y = y>0
#     str_train = 'train' if train else 'test'
    return dataset.TensorDataset(X,y) if not double else dataset.TensorDataset(X.double(),y.double())



def normalize_dataset(dataset):
    dataset.tensors = ((dataset.tensors[0].t()/dataset.tensors[1].abs()).t(), (dataset.tensors[1].t()/dataset.tensors[1].abs()).t())
    return dataset


######## loss 

def linear_loss(output,targets):
    return (-output*targets).mean() + targets.norm()/math.sqrt(len(targets))

def linear_sign_loss(output,targets):
    targets_sign = targets.sign().detach()
    return (-output*targets).mean() + targets.norm()/math.sqrt(len(targets))

def l2_loss(output,targets):
    return ((output- targets)**2).mean()

