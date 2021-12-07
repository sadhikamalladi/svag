'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei 
'''
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
from .ntk import *

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19', 'all_conv', 'all_conv_bn'
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
    def __init__(self, depth=11, batch_norm=False, num_classes=1000, ntk_init=False, pooling = 'max',widths = None, is_bias =True, lap = 0, padding_mode = 'zeros',lap_padding_mode = 'no_padding', mirror = [-1], preact = False, all_mirror=False, norm_method = 'BN',  **kwargs):
        super(VGG, self).__init__()
        self.depth = depth
        self.batch_norm = batch_norm
        self.is_bias = is_bias
        self.ntk_init = ntk_init
        self.pooling = pooling
        self.lap = lap
        self.padding_mode = padding_mode
        self.lap_padding_mode = lap_padding_mode
        self.mirror = mirror
        self.all_mirror = all_mirror
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
        lap_multiplier = 1
        if lap>0:
            lap_multiplier = 1024
            if self.lap_padding_mode == 'no_padding':
                lap_multiplier = (32-lap+1)**2
        #self.classifier = NTKLinear(config[index]*lap_multiplier, num_classes, ntk_init =self.ntk_init, bias = self.is_bias)
        self.classifier = NTKLinear(config[index]*lap_multiplier, num_classes, ntk_init =False, bias = self.is_bias)
        
        
        
        reset_parameters(self)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    
#     #old initialization: kept for the purpose of recovering previous results
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 n = m.weight.size(1)
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()


    def make_layers(self,config, method = 'BN'):
        layers = []
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
                    if self.all_mirror:
                        conv2d = Mirror(conv2d)
                    current_block += [conv2d, norm2d(v, eps = eps, method = method), nn.ReLU(inplace=True)]       
                else:
                    conv2d = NTKConv2d(in_channels, v, 3, ntk_init =self.ntk_init, padding_mode = self.padding_mode,
                                       padding=1+int(self.padding_mode == 'circular'), bias=self.is_bias)
                    if self.all_mirror:
                        conv2d = Mirror(conv2d)
                    current_block += [conv2d, nn.ReLU(inplace=True)]      
                    
                if count in self.mirror:
                    layers.append(Mirror(nn.Sequential(*current_block)))
                    current_block = []
                in_channels = v
                count += 1
        assert(count == self.mirror[-1]+1 or self.mirror[0]==-1)
        if len(current_block)>0:
            layers.append(nn.Sequential(*current_block))
        if self.lap==0:
            layers += [nn.AdaptiveAvgPool2d((1,1))]
        else:
            layers += [LocalAvgPool2d(self.lap, self.lap_padding_mode)]
        return nn.Sequential(*layers)




def all_conv(is_bias = False, **kwargs):
    model = VGG(batch_norm = False, pooling = 'average', is_bias = is_bias, **kwargs)
    return model

def all_conv_bn(is_bias = False, **kwargs):
    model = VGG(batch_norm = True, pooling = 'average', is_bias = is_bias, **kwargs)    
    return model

def vgg11(**kwargs):
    model = VGG(depth=11, batch_norm = False, **kwargs)
    return model


def vgg11_bn(**kwargs):
    model = VGG(depth=11, batch_norm = True, **kwargs)
    return model


def vgg13(**kwargs):
    model = VGG(depth=13, batch_norm = False, **kwargs)
    return model


def vgg13_bn(**kwargs):
    model = VGG(depth=13, batch_norm = True, **kwargs)
    return model


def vgg16(**kwargs):
    model = VGG(depth=16, batch_norm = False, **kwargs)
    return model


def vgg16_bn(**kwargs):
    model = VGG(depth=16, batch_norm = True, **kwargs)
    return model


def vgg19(**kwargs):
    model = VGG(depth=19, batch_norm = False, **kwargs)
    return model


def vgg19_bn(**kwargs):
    model = VGG(depth=19, batch_norm = True, **kwargs)
    return model
