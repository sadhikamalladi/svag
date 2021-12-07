import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
import math
import torch.nn.functional as F
import copy

__all__ = [ 'NTKLinear', 'NTKConv2d', 'reset_parameters', 'LocalAvgPool2d', 'Mirror','norm2d','norm1d','EN']

class Mirror(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.new_module = module
    
    def forward(self,input):
        input = torch.cat((input,torch.flip(input,[3])),0)
        outputs = self.new_module.forward(input)
        outputs = (outputs[:len(outputs)//2]+  torch.flip(outputs[len(outputs)//2:],[3]))/2
        return outputs
    
    
class EN(nn.Module):
    '''
    end-to-end normalization
    '''
    def __init__(self,module, num_classes, method):
        super().__init__()
        self.module = module
        self.norm = norm1d(num_classes,eps=0,affine = False, method=method)
    
    def forward(self,input):
        output = self.module.forward(input)
        output = self.norm(output)
        return output

class BatchNorm(nn.modules.batchnorm._BatchNorm):
    _aux_size = 0
    
    
    def __init__(self, num_features, eps=1e-5,affine=True):
        super(BatchNorm, self).__init__(num_features, eps=eps, momentum=0.1, affine=affine,
                 track_running_stats=True, L1 = False)
        
        assert(self._aux_size>0)
        self.aux_size = self._aux_size
        self.mode = 'average'
        del self._buffers['running_var']
        del self._buffers['running_mean']
        del self._buffers['num_batches_tracked']
        self.L1 = L1
        self.register_parameter('running_var', torch.nn.Parameter(torch.ones(self.num_features), requires_grad = False))
        self.register_parameter('running_mean', torch.nn.Parameter(torch.zeros(self.num_features), requires_grad = False))
        self.register_parameter('num_batches_tracked', torch.nn.Parameter(torch.tensor(0.), requires_grad = False))
        
        self.reset_running_stats()

        
    def forward(self, input):
        y = input.transpose(0,1)
        view = [-1] + [1]* (len(y.size())-1)

        if self.training:
            aux = input[:self.aux_size]
            y_aux = aux.transpose(0,1)
            y_aux = y_aux.contiguous().view(aux.size(1), -1)
            mu = y_aux.mean(dim=1)
            if not self.L1:
                sigma = y_aux.var(dim=1)**.5
            else:
                sigma = torch.abs(y_aux-mu.view(-1,1)).mean(dim=1)
            
                
            if (sigma ==0).any():
                print('Unbelievable! Dividing by 0',self.num_batches_tracked)
                print(sigma)
                
            if torch.isinf(sigma).any():
                print('Unbelievable! Dividing by Inf',self.num_batches_tracked)
                print(sigma)
                
            if (sigma.data != sigma.data).any():
                print('Unbelievable! Found Nan',self.num_batches_tracked)
                print(sigma.data)
                
            if self.num_batches_tracked.data== 0:
                self.running_mean.data = mu.data
                self.running_var.data = sigma.data
            elif self.mode == 'geometric':
                self.running_mean.data = (1-self.momentum)* self.running_mean.data + self.momentum * mu.data
                self.running_var.data = (1-self.momentum)* self.running_var.data + self.momentum * sigma.data
            elif self.mode == 'average':
                self.running_mean.data = (self.num_batches_tracked.data* self.running_mean.data + mu.data)/(self.num_batches_tracked.data+1)
                self.running_var.data = (self.num_batches_tracked.data* self.running_var.data + sigma.data)/(self.num_batches_tracked.data+1)
            
            y = y - mu.view(view)
            y = y / (sigma.view(view) + self.eps)
            self.num_batches_tracked.data += 1
            #print(self.mode)
        else:
            y = y - self.running_mean.data.view(view)
            y = y / (self.running_var.data.view(view) + self.eps)
        if self.affine:
            y = self.weight.view(view) * y + self.bias.view(view)
        

        return y.transpose(0,1)


def norm2d(num_features, eps=1e-5, method = 'BN', affine = True):
    if method == 'BN':
        return nn.BatchNorm2d(num_features,eps=eps,affine = affine)
    elif method == 'GN':
        return nn.GroupNorm(8,num_features, eps=eps,affine = affine)
    elif method == 'LN':
        return nn.GroupNorm(1, num_features, eps=eps,affine = affine)
    elif method == 'BN+aux':
        return BatchNorm(num_features,eps=eps,affine = affine)
    elif method == 'L1_BN+aux':
        return BatchNorm(num_features,eps=eps,affine = affine, L1 =True)
    elif method == 'None':
        return nn.Identity()
    
def norm1d(num_features, eps=1e-5, method = 'BN', affine = True):
    if method == 'BN':
        return nn.BatchNorm1d(num_features,eps=eps,affine = affine)
    elif method == 'GN':
        return nn.GroupNorm(8, num_features, eps=eps,affine = affine)
    elif method == 'LN':
        return nn.GroupNorm(1, num_features, eps=eps,affine = affine)
    elif method == 'BN+aux':
        return BatchNorm(num_features,eps=eps,affine = affine)
    elif method == 'L1_BN+aux':
        return BatchNorm(num_features,eps=eps,affine = affine, L1=True)
    elif method == 'None':
        return nn.Identity()

def reset_parameters(model):
    for m in model.modules():
        ntk_init = False
        if hasattr(m,'ntk_init'):
            ntk_init = m.ntk_init
        if isinstance(m, nn.Conv2d) or isinstance(m, NTKConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            if ntk_init:
                m.weight.data.normal_(0, 1)
            else:
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear) or isinstance(m, NTKLinear):
            n = m.weight.size(1)
            if ntk_init:
                m.weight.data.normal_(0, 1)
            else:
                m.weight.data.normal_(0, math.sqrt(2./ m.in_features))
            if m.bias is not None:
                m.bias.data.zero_()

                
class LocalAvgPool2d(nn.Module):
    def __init__(self, size,padding_mode):
        super().__init__()
        if size% 2 ==0:
            raise ValueError("LAP size should be odd")
        self.size = size
        self.padding = size//2
        self.padding_mode = padding_mode
        
        
    def forward(self, input):
        C = input.size()[1]
        weight = torch.ones(C, 1, self.size, self.size).cuda()/ (self.size*self.size)
        
        if self.padding_mode == 'circular':
            expanded_padding = (self.padding,self.padding,self.padding,self.padding)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, None, 1, (0,0), 1, C)
        elif self.padding_mode == 'no_padding':
            return F.conv2d(input, weight, None, 1, (0,0), 1, C)
        elif self.padding_mode == 'zeros':
            return F.conv2d(input, weight, None, 1, (self.padding,self.padding), 1, C)
                

class NTKConv2d(nn.Conv2d):
    def __init__(self, *args, ntk_init= False, **kwargs):
        super().__init__( *args,**kwargs)
        self.ntk_init = ntk_init
        fan_in = self.kernel_size[0] * self.kernel_size[1] * self.in_channels
        self.scaler = 1
        if ntk_init:
            self.scaler =  math.sqrt(2. / fan_in)
        reset_parameters(self)
            
            
    def forward(self, x):
        return super().forward(x)*self.scaler
    
    
class NTKLinear(nn.Linear):
    
    def __init__(self, *args, ntk_init= False, grad_hook = False, **kwargs):
        super().__init__( *args,**kwargs)
        self.ntk_init = ntk_init
        self.scaler = 1
        if ntk_init:
            self.scaler =  math.sqrt(2. / self.in_features)   
        reset_parameters(self)
            
    def forward(self, x):
        output = super().forward(x)
        return output*self.scaler
    
    
    
