import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import utils
from pytorch_lightning.utilities import AttributeDict

import attr
from functools import partial
from typing import List
from typing import Optional
from typing import Union

__all__ = ['Base_Module']
        


class Base_Module(pl.LightningModule):
    model: torch.nn.Module
    hparams: AttributeDict
        
    def __init__(self, hparams= None, **kwargs):
        super().__init__()
        self.hparams = utils.AttributeDict_recusion(attr.asdict(hparams))
        self.model = utils.get_model(self.hparams)
        self.train_size = len(self.train_dataloader().dataset)
        self.val_size = len(self.val_dataloader().dataset)
        
    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        if self.hparams.task_type == 'classification':
            accuracy = utils.accuracy_classification(predictions,labels)
        elif self.hparams.task_type == 'regression':
            accuracy = utils.accuracy_regression(predictions,labels)
        else:
            raise AttributeError(f"Unknown task type {self.hparams.task_type}")
            
        return loss, accuracy
    
    def training_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        logs = {'loss/train': loss, 'accuracy/train': accuracy}
        self.log_dict(logs)
        if self.hparams.verbose:
            print(logs)
        return loss
        
    def validation_step(self, batch, batch_nb):
        avg_loss, accuracy = self.forward(batch)
        loss = avg_loss * batch[0].size(0)
        corrects = accuracy * batch[0].size(0)
        logs = {'loss/val': loss, 'corrects': corrects}
        if self.hparams.verbose:
            print(logs)
        return logs
                
    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss/val'] for x in outputs]).sum() / self.val_size
        accuracy = torch.stack([x['corrects'] for x in outputs]).sum() / self.val_size
        logs = {'loss/val': loss, 'accuracy/val': accuracy}
        self.log_dict(logs)
    
    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)
    
    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr,
                                    weight_decay=self.hparams.weight_decay, momentum=self.hparams.momentum)
            
        return [optimizer]
    
    def train_dataloader(self):
        raise NotImplementedError
    
    def val_dataloader(self):
        raise NotImplementedError
    
    def test_dataloader(self):
        return self.val_dataloader()
    
