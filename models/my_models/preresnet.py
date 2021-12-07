# from __future__ import absolute_import


import torch.nn as nn
import torch
import math
from .ntk import *
from functools import partial
import types
eps = 0

__all__ = ['preresnet']

def conv3x3(in_planes, out_planes, stride=1,padding_mode = 'zeros'):
    "3x3 convolution with padding"
    return NTKConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1+int(padding_mode == 'circular'), bias=False,padding_mode = padding_mode)

class SuperBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, normalization, stride=1, downsample=None,padding_mode='zeros'):
        super(SuperBasicBlock, self).__init__()
        #self.bn1 = nn.BatchNorm2d(inplanes,eps=eps)
        self.bn1 = normalization(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride,padding_mode=padding_mode)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)


        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out
    

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, normalization, stride=1, downsample=None,padding_mode='zeros'):
        super(BasicBlock, self).__init__()
        #self.bn1 = nn.BatchNorm2d(inplanes,eps=eps)
        self.bn1 = normalization(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride,padding_mode=padding_mode)
        #self.bn2 = nn.BatchNorm2d(planes,eps=eps)
        self.bn2 = normalization(planes)
        self.conv2 = conv3x3(planes, planes,padding_mode=padding_mode)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out
    
    


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, normalization,stride=1, downsample=None,padding_mode='zeros'):
        super(Bottleneck, self).__init__()
        self.bn1 = normalization(inplanes)
        self.conv1 = NTKConv2d(inplanes, planes, kernel_size=1, bias=False,padding_mode=padding_mode)
        self.bn2 = norm2d(planes,eps=eps,method = norm_method)
        self.conv2 = NTKConv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False,padding_mode=padding_mode)
        self.bn3 = normalization(planes)
        self.conv3 = NTKConv2d(planes, planes * 4, kernel_size=1, bias=False,padding_mode=padding_mode)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class PreResNet(nn.Module):

    def __init__(self, depth, num_classes=10, block_name='BasicBlock',padding_mode = 'zeros',widths=[16,32,64], strides = [1,2,2], ntk_init=False, is_bias =False, widen_factor = 1, norm_method = 'BN', homo = True, last_bn = False, fix_last_layer = False, bn_affine = True, **kwargs):
        super(PreResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        
        assert(len(widths) == len(strides))
        L = len(widths)
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % (2*L) == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // (2*L)
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % (3*L) == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // (3*L)
            block = Bottleneck
        elif block_name.lower() == 'superbasicblock':
            assert (depth - 2) % L == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // L
            block = SuperBasicBlock
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.inplanes = widths[0]* widen_factor  ## original: 16 ##
        self.homo = homo
        self.conv1 = NTKConv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False)
        
        self.meta_layers = []
        self.bn_affine = bn_affine
        for i in range(len(widths)):
            self.meta_layers.append(self._make_layer(block, widths[i] * widen_factor, n, stride = strides[i] ,padding_mode =padding_mode, method = norm_method))
        self.meta_layers = nn.Sequential(*self.meta_layers)
        self.bn = norm2d(widths[-1]* widen_factor * block.expansion,eps=eps, method = norm_method, affine = self.bn_affine)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.is_bias = is_bias
        self.ntk_init = ntk_init
        self.depth = depth
        self.padding_mode = padding_mode
        if last_bn:
            print('lbn')
            self.lbn = norm1d(num_classes,eps=eps, method = norm_method, affine = False)
            
        prod = 1
        for x in strides:
            prod = prod*x
        assert(32 % prod == 0)
        self.fc = NTKLinear(widths[-1] * widen_factor * block.expansion, num_classes, ntk_init =self.ntk_init, bias = self.is_bias)
        self.fc.weight.requires_grad = not fix_last_layer
        if self.fc.bias is not None:
            self.fc.bias.requires_grad = not fix_last_layer
        self.num_classes = num_classes

    def _make_layer(self, block, planes, blocks, stride=1, padding_mode ='zeros',method = 'BN'):
        layers = []
        downsample_list = []
        if self.homo:
            downsample_list = [norm2d(self.inplanes,eps=eps,method = method,affine = self.bn_affine)] + [NTKConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)]
        elif stride != 1 or self.inplanes != planes * block.expansion:
            downsample_list = [NTKConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)]
        downsample = nn.Sequential(*downsample_list)
        
        normalization = partial(norm2d,eps=eps,method = method, affine =self.bn_affine)
        layers.append(block(self.inplanes, planes, normalization, stride, downsample,padding_mode))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, normalization))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.meta_layers(x)
        x = self.bn(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if hasattr(self,'lbn'):
            x = self.lbn(x)
        return x


def preresnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return PreResNet(**kwargs)


def nikunj_preresnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    kwargs['num_classes'] = 4
    model = PreResNet(**kwargs)
    model.fc2 = NTKLinear(4,10, ntk_init =model.ntk_init, bias = model.is_bias)
    def forward(self, x):
        x = self.conv1(x)
        x = self.meta_layers(x)
        x = self.bn(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.fc2(x)
        if hasattr(self,'lbn'):
            x = self.lbn(x)
        return x
    model.forward = types.MethodType(forward, model)
    return model


def nikunj_preresnet_rank_2(**kwargs):
    """
    Constructs a ResNet model.
    """
    kwargs['num_classes'] = 2
    model = PreResNet(**kwargs)
    model.fc2 = NTKLinear(2,10, ntk_init =model.ntk_init, bias = model.is_bias)
    def forward(self, x):
        x = self.conv1(x)
        x = self.meta_layers(x)
        x = self.bn(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.fc2(x)
        if hasattr(self,'lbn'):
            x = self.lbn(x)
        return x
    model.forward = types.MethodType(forward, model)
    return model