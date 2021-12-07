'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei 
'''
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
from .ntk import *

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19', 'all_conv', 'all_conv_bn','simple_cnn','hybrid_fc_cnn_relu','hybrid_fc_cnn_quadratic'
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

cfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
eps = 0


class VGG(nn.Module):

    #def __init__(self, features, num_classes=1000):
    def __init__(self, depth=11, batch_norm=False, num_classes=1000, ntk_init=False, pooling = 'average',widths = None, is_bias =True, padding_mode = 'zeros', preact = False, norm_method = 'BN', last_bn = False, bn_affine = True, fix_last_layer = False, **kwargs):
        super(VGG, self).__init__()
        self.depth = depth
        self.batch_norm = batch_norm
        self.is_bias = is_bias
        self.ntk_init = ntk_init
        self.pooling = pooling
        self.bn_affine = bn_affine
        self.padding_mode = padding_mode
        if widths is None:
            config = cfg[self.depth]
            print(config)
        else:
            config = widths
        self.features = self.make_layers(config, method = norm_method)
        
        #find the last width
        index = len(config)-1
        while not isinstance(config[index],int):
            index -= 1
        self.fc = NTKLinear(config[index], num_classes, ntk_init =self.ntk_init, bias = self.is_bias)
        if last_bn:
            print('lbn')
            self.lbn = norm1d(num_classes,eps=eps, method = norm_method, affine = False)
        self.fc.weight.requires_grad = not fix_last_layer
        if self.fc.bias is not None:
            self.fc.bias.requires_grad = not fix_last_layer
        
        reset_parameters(self)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if hasattr(self,'lbn'):
            x = self.lbn(x)
        return x.squeeze()

        

    def make_layers(self,config, method = 'BN'):
        in_channels = 3
        current_block = []
        count = 0
        for v in config:
            if v == 'M':
                if self.pooling == 'max':
                    current_block += [nn.MaxPool2d(kernel_size=2, stride=2)]
                elif self.pooling == 'average':
                    current_block += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                if self.batch_norm:
                    conv2d = NTKConv2d(in_channels, v, 3, ntk_init =self.ntk_init, padding_mode = self.padding_mode,
                                       padding=1+int(self.padding_mode == 'circular'), bias=False)
                    current_block += [conv2d, norm2d(v, eps = eps, method = method, affine =self.bn_affine), nn.ReLU(inplace=True)]       
                else:
                    conv2d = NTKConv2d(in_channels, v, 3, ntk_init =self.ntk_init, padding_mode = self.padding_mode,
                                       padding=1+int(self.padding_mode == 'circular'), bias=self.is_bias)
                    current_block += [conv2d, nn.ReLU(inplace=True)]      
                in_channels = v
                count += 1
        current_block += [nn.AdaptiveAvgPool2d((1,1))]
        return nn.Sequential(*current_block)




def all_conv(is_bias = False, **kwargs):
    model = VGG(batch_norm = False, pooling = 'average', is_bias = is_bias, **kwargs)
    return model

def all_conv_bn(is_bias = False, **kwargs):
    model = VGG(batch_norm = True, pooling = 'average', is_bias = is_bias, **kwargs)    
    return model

def vgg11(**kwargs):
    kwargs = dict(kwargs)
    kwargs.pop('depth',None)
    model = VGG(depth=11, batch_norm = False, **kwargs)
    return model


def vgg11_bn(**kwargs):
    kwargs = dict(kwargs)
    kwargs.pop('depth',None)
    model = VGG(depth=11, batch_norm = True, **kwargs)
    return model


def vgg13(**kwargs):
    kwargs = dict(kwargs)
    kwargs.pop('depth',None)
    model = VGG(depth=13, batch_norm = False, **kwargs)
    return model


def vgg13_bn(**kwargs):
    kwargs = dict(kwargs)
    kwargs.pop('depth',None)
    model = VGG(depth=13, batch_norm = True, **kwargs)
    return model


def vgg16(**kwargs):
    kwargs = dict(kwargs)
    kwargs.pop('depth',None)
    model = VGG(depth=16, batch_norm = False, **kwargs)
    return model


def vgg16_bn(**kwargs):
    kwargs = dict(kwargs)
    kwargs.pop('depth',None)
    model = VGG(depth=16, batch_norm = True, **kwargs)
    return model


def vgg19(**kwargs):
    kwargs = dict(kwargs)
    kwargs.pop('depth',None)
    model = VGG(depth=19, batch_norm = False, **kwargs)
    return model


def vgg19_bn(**kwargs):
    kwargs = dict(kwargs)
    kwargs.pop('depth',None)
    model = VGG(depth=19, batch_norm = True, **kwargs)
    return model



class Squeeze(nn.Module):
    def forward(self, input):
        return torch.squeeze(input)

class Quadratic(nn.Module):
    def forward(self, input):
        return (input)**2 
    

class Conv2Fc(nn.Module):
    def forward(self, input):
        return input.view(len(input),-1)
    
class Fc2Conv(nn.Module):
    def forward(self, input):
        return input.view(len(input),-1,32,32)
    
# def simple_cnn_bn(**kwargs):
#     num_features = 100
#     return torch.nn.Sequential(
#         torch.nn.Conv2d(3, num_features, kernel_size=3, bias=False, padding =1),
#         nn.BatchNorm2d(num_features, eps=0., momentum=0.1, affine=False),
#         torch.nn.ReLU(),
#         torch.nn.Conv2d(num_features, num_features, kernel_size=3, bias=False, padding =1),
#         nn.BatchNorm2d(num_features, eps=0., momentum=0.1, affine=False),
#         torch.nn.ReLU(),
#         torch.nn.Conv2d(num_features, 1, kernel_size=3, bias=False, padding =1),
#         nn.AdaptiveAvgPool2d((1,1)),
#         nn.BatchNorm2d(1, eps=0., momentum=0.1, affine=False),
#         Squeeze(),
#         )

def simple_cnn(widths =[10], num_classes = 2, **kwargs):
    num_features = widths[0]
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, num_features, kernel_size=3, bias=False, padding =1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(num_features, num_classes, kernel_size=3, bias=False, padding =1),
        nn.AdaptiveAvgPool2d((1,1)),
        Squeeze(),
        )

def simple_cnn(widths =[10], num_classes = 2, **kwargs):
    num_features = widths[0]
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, num_features, kernel_size=3, bias=False, padding =1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(num_features, num_classes, kernel_size=3, bias=False, padding =1),
        nn.AdaptiveAvgPool2d((1,1)),
        Squeeze(),
        )

def hybrid_fc_cnn_relu(widths =[10], num_classes = 2, **kwargs):
    num_features = widths[0]
    return torch.nn.Sequential(
        Conv2Fc(),
        torch.nn.Linear(3072,3072,bias =False),
        Fc2Conv(),
        torch.nn.Conv2d(3, num_features, kernel_size=3, bias=False, padding =1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(num_features, num_classes, kernel_size=3, bias=False, padding =1),
        nn.AdaptiveAvgPool2d((1,1)),
        Squeeze(),
        )

def hybrid_fc_cnn_quadratic(widths =[10], num_classes = 2, **kwargs):
    num_features = widths[0]
    return torch.nn.Sequential(
        Conv2Fc(),
        torch.nn.Linear(3072,3072,bias =False),
        Fc2Conv(),
        torch.nn.Conv2d(3, num_features, kernel_size=3, bias=False, padding =1),
        Quadratic(),
        torch.nn.Conv2d(num_features, num_classes, kernel_size=3, bias=False, padding =1),
        nn.AdaptiveAvgPool2d((1,1)),
        Squeeze(),
        )