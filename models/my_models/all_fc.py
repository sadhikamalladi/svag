import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch
from .ntk import *

__all__ = [
    'all_fc', 'all_fc_bn','all_fc_linear','all_fc_bn_linear','all_fc_wn'
]

eps = 0

class ALL_FC(nn.Module):

    def __init__(self, widths = [20,128,128,128], batch_norm=False, num_classes=1, ntk_init=False , is_bias =True,  norm_method = 'BN',  slope = 0, wn = False, wn_scaler = 1,**kwargs):
        
        assert(not wn or (not batch_norm and slope==1.)) # weight normalization for linear net only
        
        super(ALL_FC, self).__init__()
        widths = [3072]  + widths
        self.widths =  widths
        self.depth = len(widths)
        self.batch_norm = batch_norm
        self.is_bias = is_bias
        self.ntk_init = ntk_init
        self.wn = wn
        self.wn_scaler = wn_scaler

        layers = []
        for i in range(self.depth -1):
            if self.batch_norm:
                linear = NTKLinear(widths[i], widths[i+1], ntk_init =self.ntk_init, bias=False)
                layers += [linear, norm1d(widths[i+1], eps = eps, method = norm_method), nn.LeakyReLU(negative_slope = slope, inplace=True)]       
#                 layers += [linear, nn.modules.BatchNorm1d(widths[i+1], eps = eps), nn.ReLU(inplace=True)]       
            else:
                linear = NTKLinear(widths[i], widths[i+1], ntk_init =self.ntk_init, bias=self.is_bias)
                layers += [linear, nn.LeakyReLU(negative_slope = slope, inplace=True)]      
        
        self.layers = nn.Sequential(*layers)
        self.classifier = NTKLinear(widths[-1], num_classes, ntk_init =self.ntk_init, bias=self.is_bias)
        reset_parameters(self)

    def forward(self, x):
        
        result = self.classifier(self.layers(x.view(x.size(0),-1)))
        if self.wn:
            identity = torch.eye(self.widths[0]).cuda()
            factor = torch.norm(self.classifier(self.layers(identity)), p=2, dim=0) * self.wn_scaler
            result = torch.div(result, factor)
        
            
        return result

    def linearify(self, normalized = False):
        assert(self.wn)
        
        identity = identity = torch.eye(self.widths[0]).cuda()
        linear = self.classifier(self.layers(identity))
        if normalized:
            linear = torch.div(linear, torch.norm(linear, p=2, dim=0))
        return linear * self.wn_scaler


def all_fc_linear(**kwargs):
    model = ALL_FC(batch_norm = False, slope =1, **kwargs)
    return model

def all_fc_bn_linear( **kwargs):
    model = ALL_FC(batch_norm = True, slope = 1, **kwargs)    
    return model
    
def all_fc(**kwargs):
    model = ALL_FC(batch_norm = False, **kwargs)
    return model

def all_fc_bn( **kwargs):
    model = ALL_FC(batch_norm = True, **kwargs)    
    return model


def all_fc_wn( scaler = 1, **kwargs):
    kwargs['is_bias'] = False  # no bias 
    model = ALL_FC(batch_norm = False, slope = 1., wn = True, wn_scaler = scaler,  **kwargs)    
    return model
