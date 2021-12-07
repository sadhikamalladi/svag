import pytorch_lightning as pl 
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pl_modules import CIFAR10_Module,CIFAR100_Module, SVHN_Module
from hparams import CIFAR10_HParams
from hparams import CIFAR10_HParams_Dict as config_dict
from hparams.cifar10_hparams_dict import CIFAR10_HParams_Dict as config_dict
from hparams import CIFAR100_HParams_Dict as cifar100_dict
from hparams import SVHN_HParams_Dict as svhn_dict
import wandb
from attr import evolve
from pytorch_lightning.loggers import TensorBoardLogger
from utils import *
import os

from pl_modules import SDELimit1_CIFAR10_Module,SDELimit2_CIFAR10_Module, NGD_Module, Subsampled_SGD, SDELimit1_CIFAR100_Module, SDELimit1_SVHN_Module

from pytorch_lightning.callbacks import ModelCheckpoint

ROOT = '/n/fs/ptml/smalladi/sde_limit/'
# ROOT = '/n/fs/ptml/zl4/lightning_explorer/'
ckpt_period = 1 

def main(args):
    config_name = args.config
    suffix = args.suffix
    log_name = config_name +'_'+ suffix if len(suffix)>0 else config_name

    seed_everything(args.seed)
    if args.imagenet:
        config = imagenet_dict[config_name]
    elif args.cifar_100:
        config = cifar100_dict[config_name]
    elif args.svhn:
        config = svhn_dict[config_name]
    else: # cifar-10 training
        if args.sadhika:
            config = config_dict[config_name]
        else:
            config = config_dict[config_name]
    run = wandb.init(project=config.wb_project,
                    name = log_name,
                    sync_tensorboard=True,
                    reinit = True,
                    entity = config.wb_entity,
                    save_code = True,
                    config = config.to_dict())
    config.jobid = args.job_id
    config.taskid = args.task_id
    print(config.to_dict())

    if args.cifar_100:
        ds_str = 'cifar-100'
    elif args.svhn:
        ds_str = 'svhn'
    else:
        ds_str = 'cifar-10'
    if not os.path.exists(ds_str):
        os.mkdir(ds_str)
    
    ckpt_period = 1
    if args.cifar_100:
        if 'sde1' in log_name:
            method = SDELimit1_CIFAR100_Module(config)
        elif 'sde2' in log_name or args.sub_sgd or args.ngd or 'ngd' in config_name:
            raise NotImplementedError
        else:
            method = CIFAR100_Module(config)
    elif args.svhn:
        if 'sde1' in log_name:
            method = SDELimit1_SVHN_Module(config)
        elif 'sde2' in log_name or args.sub_sgd or args.ngd or 'ngd' in config_name:
            raise NotImplementedError
        else:
            method = SVHN_Module(config)
    else:
        if 'sde1' in log_name:
            method = SDELimit1_CIFAR10_Module(config)
        elif 'sde2' in log_name:
            method = SDELimit2_CIFAR10_Module(config)
        elif args.sub_sgd:
            method = Subsampled_SGD(config)
        elif args.ngd or 'ngd' in config_name:
            method = NGD_Module(config)
            ckpt_period = config.check_val_every_n_epoch
        else:
            method = CIFAR10_Module(config)
        
    logger = TensorBoardLogger("tb_logs", name=f"{log_name}")
    logger.log_hyperparams(config.to_dict())

    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(ROOT, ds_str, config_name), save_last=True, period = ckpt_period)
    callbacks=[LR_WD_Scheduler(), LR_WD_Logger(epoch_wise =False), Norm_Logger(layer_wise = False)]
#     callbacks+=[SC_Test()] ##sc_test is on single gpu, leads to cuda out of memory
    callbacks.append(Effective_Step_logger())
    callbacks.append(checkpoint_callback)
    
    if hasattr(config, 'measure_variance') and config.measure_variance:
        freq = 1 # default measure variance once per epoch
        if hasattr(config, 'batch_k') and config.batch_k != 1:
            freq *= config.batch_k * 2
        callbacks.append(Variance_Measurement(freq=freq))
    trainer = pl.Trainer(
        gpus=-1, 
        max_epochs=config.max_epochs, 
        logger=logger, 
        callbacks= callbacks,  
        accelerator='dp' if torch.cuda.is_available() else None,
        deterministic=True, 
        log_every_n_steps=1,
        accumulate_grad_batches = config.grad_accumulate if hasattr(config, 'grad_accumulate') else 1,
        check_val_every_n_epoch = config.check_val_every_n_epoch,
        resume_from_checkpoint=os.path.join(ROOT, ds_str, config_name, 'last.ckpt') if args.resume else None,
        progress_bar_refresh_rate = 100,
    ) #
    trainer.fit(method)
    run.finish()
    wandb.finish()
    return method
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
    wandb.finish()
