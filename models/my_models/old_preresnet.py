from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
import torch.nn as nn
import torch
import math
from .ntk import *
bn_momentum = 0.1
eps = 0

__all__ = ['v2preresnet']

def conv3x3(in_planes, out_planes, stride=1,padding_mode = 'zeros'):
    "3x3 convolution with padding"
    return NTKConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1+int(padding_mode == 'circular'), bias=False,padding_mode = padding_mode)

class SuperBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,padding_mode='zeros',method='BN'):
        super(SuperBasicBlock, self).__init__()
        #self.bn1 = nn.BatchNorm2d(inplanes,eps=eps)
        self.bn1 = norm2d(inplanes,eps=eps,method = method)
        self.relu = ReLU(inplace=True)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None,padding_mode='zeros',method='BN'):
        super(BasicBlock, self).__init__()
        #self.bn1 = nn.BatchNorm2d(inplanes,eps=eps)
        self.bn1 = norm2d(inplanes,eps=eps,method = method)
        self.relu = ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride,padding_mode=padding_mode)
        #self.bn2 = nn.BatchNorm2d(planes,eps=eps)
        self.bn2 = norm2d(planes,eps=eps,method = method)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None,padding_mode='zeros', method = 'BN'):
        super(Bottleneck, self).__init__()
        #self.bn1 = nn.BatchNorm2d(inplanes,eps=eps)
        self.bn1 = norm2d(inplanes,eps=eps,method = norm_method)
        self.conv1 = NTKConv2d(inplanes, planes, kernel_size=1, bias=False,padding_mode=padding_mode)
        #self.bn2 = nn.BatchNorm2d(planes,eps=eps)
        self.bn2 = norm2d(planes,eps=eps,method = norm_method)
        self.conv2 = NTKConv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False,padding_mode=padding_mode)
        #self.bn3 = nn.BatchNorm2d(planes,eps=eps)
        self.bn3 = norm2d(planes,eps=eps,method = norm_method)
        self.conv3 = NTKConv2d(planes, planes * 4, kernel_size=1, bias=False,padding_mode=padding_mode)
        self.relu = ReLU(inplace=True)
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

    def __init__(self, depth, num_classes=1000, block_name='BasicBlock',padding_mode = 'zeros',widths=[], ntk_init=False, is_bias =True, lap = 0, lap_padding_mode = 'no_padding', widen_factor = 1, norm_method = 'BN', homo2 = False,pre_avearge_pool = False,  **kwargs):
        super(PreResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        elif block_name.lower() == 'superbasicblock':
            assert (depth - 2) % 3 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 3
            block = SuperBasicBlock
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.inplanes = 16* widen_factor
        self.homo2 = homo2
        self.pre_average_pool = pre_avearge_pool
        self.conv1 = NTKConv2d(3, 16* widen_factor, kernel_size=3, padding=1,
                               bias=False)
        self.layer1 = self._make_layer(block, 16* widen_factor, n,padding_mode =padding_mode, method = norm_method)
        self.layer2 = self._make_layer(block, 32* widen_factor, n, stride=2,padding_mode =padding_mode, method = norm_method)
        self.layer3 = self._make_layer(block, 64* widen_factor, n, stride=2, padding_mode =padding_mode, method = norm_method)
        #self.bn = nn.BatchNorm2d(64 * block.expansion,eps=eps)
        #self.bn = norm2d(64* widen_factor * block.expansion,eps=eps, method = 'L1_BN+aux')
        self.bn = norm2d(64* widen_factor * block.expansion,eps=eps, method = norm_method)
        self.relu = ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.is_bias = is_bias
        self.ntk_init = ntk_init
        self.depth = depth
        self.lap = lap
        self.padding_mode = padding_mode
        self.lap_padding_mode = lap_padding_mode
        
        if self.lap==0:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        else:
            self.avgpool = LocalAvgPool2d(self.lap, self.lap_padding_mode)
            
        lap_multiplier = 1
        if lap>0:
            lap_multiplier = 64
            if self.lap_padding_mode == 'no_padding':
                lap_multiplier = (8-lap+1)**2
        self.fc = NTKLinear(64 * widen_factor * block.expansion*lap_multiplier, num_classes, ntk_init =self.ntk_init, bias = self.is_bias)
        ## zero bias
        self.fc.bias.data = torch.zeros(self.fc.bias.data.size(), dtype = torch.float32)


    def _make_layer(self, block, planes, blocks, stride=1, padding_mode ='zeros',method = 'BN'):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample_list = [NTKConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)]
#             if self.homo2:
#                 downsample_list = [norm2d(self.inplanes,eps=eps,method = method)]  + downsample_list
#             downsample = nn.Sequential(*downsample_list)

        layers = []
        if self.pre_average_pool:
            if stirde>1:
                layers.append(nn.AvgPool2d(stride))
                stride = 1

        downsample = None
        if self.homo2:
            downsample_list = [norm2d(self.inplanes,eps=eps,method = method)] + [NTKConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)]
        elif stride != 1 or self.inplanes != planes * block.expansion:
            downsample_list = [NTKConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)]
        downsample = nn.Sequential(*downsample_list)
        

        layers.append(block(self.inplanes, planes, stride, downsample,padding_mode, method = method))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, method = method))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.bn(x)
        #x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def preresnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return PreResNet(**kwargs)

class v2PreResNet(nn.Module):

    def __init__(self, depth, num_classes=1000, block_name='BasicBlock',widths =[64,64,64],padding_mode = 'zeros', widen_factor = 1, norm_method = 'BN',**kwargs):
        super(v2PreResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.inplanes = widths[0]*widen_factor
        self.conv1 = NTKConv2d(3, widths[0]*widen_factor, kernel_size=3, padding=1,
                               bias=False, padding_mode=padding_mode)
        self.layer1 = self._make_layer(block, widths[0]*widen_factor, n, padding_mode=padding_mode, method =norm_method)
        self.layer2 = self._make_layer(block, widths[1]*widen_factor, n, padding_mode=padding_mode, method =norm_method)
        self.layer3 = self._make_layer(block, widths[2]*widen_factor, n, padding_mode=padding_mode, method =norm_method)
        #self.bn = nn.BatchNorm2d(widths[2] * block.expansion,eps=eps,affine=False)
        #self.bn = norm2d(widths[2]*widen_factor * block.expansion,eps=eps,affine=False,method = norm_method)
        self.bn = norm2d(widths[2]*widen_factor * block.expansion,eps=eps,affine=False,method = norm_method)
        self.relu = ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = NTKLinear(widths[2]*widen_factor * block.expansion, num_classes,bias = False)

        for m in self.modules():
            if isinstance(m, NTKConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, GroupNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, padding_mode, stride=1, method = 'BN'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                NTKConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,padding_mode, method=method))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, method=method))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def v2preresnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return v2PreResNet(**kwargs)
