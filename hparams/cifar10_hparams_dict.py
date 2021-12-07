from .hparams import CIFAR10_HParams
from attr import evolve
import math

__all__ = ['CIFAR10_HParams_Dict']


preresnet32 = CIFAR10_HParams(
    arch = 'preresnet',
    depth = 32,
    widths = [16, 32, 64],
    widen_factor = 1,
    norm_method = 'BN',
    lr=0.8,
    weight_decay = 0.0005,
    momentum=0,
    train_batch_size=128,
    test_batch_size=256,
    schedule = [80,120],
    lr_decay_factors = [0.1,0.1],
    wd_decay_factors = [1.,1.],
    max_epochs = 160,
    fix_last_layer = True,
    bn_affine = False,
    homo = True,
    drop_last_batch=True
)

preresnet32_simple = CIFAR10_HParams(
    arch = 'preresnet',
    depth = 32,
    widths = [16, 32, 64],
    widen_factor = 1,
    norm_method = 'BN',
    lr=0.8,
    weight_decay = 0.0005,
    momentum=0,
    train_batch_size=128,
    test_batch_size=256,
    schedule = [250],
    lr_decay_factors = [0.1],
    wd_decay_factors = [1.],
    max_epochs = 300,
    fix_last_layer = True,
    bn_affine = False,
    homo = True,
    drop_last_batch = True,
)

CIFAR10_HParams_Dict = {'preresnet32':preresnet32}

CIFAR10_HParams_Dict['vgg16'] = evolve(preresnet32, arch = 'vgg16', widths = None, depth =16)
CIFAR10_HParams_Dict['vgg19'] = evolve(preresnet32, arch = 'vgg19', widths = None, depth = 19)

######ngd baseline ####

preresnet32_no_crop_no_flip = CIFAR10_HParams(
    arch = 'preresnet',
    depth = 32,
    widths = [16, 32, 64],
    widen_factor = 1,
    norm_method = 'BN',
    lr=0.8,
    weight_decay = 0.0005,
    momentum=0,
    train_batch_size=128,
    test_batch_size=256,
    schedule = [80,120],
    lr_decay_factors = [0.1,0.1],
    wd_decay_factors = [1.,1.],
    max_epochs = 160,
    fix_last_layer = True,
    bn_affine = False,
    homo = True,
    crop = False,
    hori_flip = False,
    drop_last_batch=True
)


preresnet32_no_crop_no_flip_tiny_lr = CIFAR10_HParams(
    arch = 'preresnet',
    depth = 32,
    widths = [16, 32, 64],
    widen_factor = 1,
    norm_method = 'BN',
    lr=0.8,
    weight_decay = 0.0000,
    momentum=0,
    train_batch_size=128,
    test_batch_size=256,
    schedule = [],
    lr_decay_factors = [0.1,0.1],
    wd_decay_factors = [1.,1.],
    max_epochs = 160,
    fix_last_layer = True,
    bn_affine = False,
    homo = True,
    crop = False,
    hori_flip = False,
    drop_last_batch=True
)

preresnet32_4x_no_crop_no_flip = evolve(preresnet32_no_crop_no_flip, widen_factor = 4)

CIFAR10_HParams_Dict['preresnet32_no_crop_no_flip'] = preresnet32_no_crop_no_flip
CIFAR10_HParams_Dict['preresnet32_4x_no_crop_no_flip'] = preresnet32_4x_no_crop_no_flip

CIFAR10_HParams_Dict['preresnet32_no_crop_w_flip'] = evolve(preresnet32_no_crop_no_flip, hori_flip=True)
CIFAR10_HParams_Dict['preresnet32_4x_no_crop_w_flip'] = evolve(preresnet32_4x_no_crop_no_flip, hori_flip =True)

CIFAR10_HParams_Dict['preresnet32_no_crop_no_flip_tiny_lr'] = preresnet32_no_crop_no_flip_tiny_lr

new_dict = {}
for k,v in CIFAR10_HParams_Dict.items():
    new_dict[k.replace('preresnet32','preresnet32_gn')] = evolve(v, norm_method = 'GN')

CIFAR10_HParams_Dict.update(new_dict)

new_dict = {}
for k,v in CIFAR10_HParams_Dict.items():
    new_dict[k+'_b500'] = evolve(v, train_batch_size = 500 ,lr = 3.2)

CIFAR10_HParams_Dict.update(new_dict)

def sde1(hparams,batch_k =1 ):
    return evolve(
        hparams, 
        batch_k=batch_k, 
        max_epochs =hparams.max_epochs*2*batch_k,
        train_batch_size = hparams.train_batch_size*2,
        schedule =  [x*batch_k*2 for x in hparams.schedule],
        lr = hparams.lr / batch_k,
    )

def gen_sde1(d,name,start=0,end=17):
    for i in range(start,end):
        d[name+f'_sde1_k{2**i}'] = sde1(globals()[name],2**i)

def sde2(hparams,batch_k =1):
    return evolve(
        hparams, 
        batch_k=batch_k, 
        max_epochs =hparams.max_epochs*batch_k,
        schedule =  [x*batch_k for x in hparams.schedule],
        lr = hparams.lr /batch_k,
    )

def linscale_rule(hparams, batch_size):
    ## we use LR = 0.8 for batch_size = 128
    og_bs = 128.0
    og_lr = 0.8

    scaling = batch_size / og_bs
    scaled_lr = scaling * og_lr

    return evolve(
        hparams,
        lr = scaled_lr,
        train_batch_size = batch_size,
        measure_variance = True
    )

def single_lr_phase(hparams):
    lr_decays = hparams.lr_decay_factors
    w_decays = hparams.wd_decay_factors
    sched = hparams.schedule
    return evolve(hparams, lr_decay_factors=[lr_decays[0]], wd_decay_factors=[w_decays[0]], schedule=[sched[0]])

def gen_linscale(d,name,start=0,end=17):
    for i in range(start,end):
        d[name+f'_b{2**i}_linscale'] = linscale_rule(globals()[name],2**i)

preresnet32_gn = evolve(preresnet32, norm_method ='GN')
preresnet32_simple_gn = evolve(preresnet32_simple, norm_method = 'GN')

preresnet32_4x = evolve(preresnet32, widen_factor = 4)
preresnet32_4x_gn = evolve(preresnet32_gn, widen_factor = 4)

preresnet32_4x_simple = evolve(preresnet32_simple, widen_factor = 4)
preresnet32_4x_simple_gn = evolve(preresnet32_4x_simple, norm_method='GN')

preresnet32_b64 =  evolve(preresnet32, max_epochs =80, schedule =[40,60],train_batch_size =64)
CIFAR10_HParams_Dict['preresnet32_b64']  = preresnet32_b64

preresnet32_b32 =  evolve(preresnet32, max_epochs =40, schedule =[20,30], train_batch_size =32)
CIFAR10_HParams_Dict['preresnet32_b32']  = preresnet32_b32

CIFAR10_HParams_Dict['preresnet32_b64_fig1'] = evolve(preresnet32, train_batch_size=64, drop_last_batch=False)

for i in range(10):
    CIFAR10_HParams_Dict[f'preresnet32_b128_sde1_k{2**i}'] = sde1(preresnet32, batch_k=2**i)
    CIFAR10_HParams_Dict[f'preresnet32_b128_sde2_k{2**i}'] = sde2(preresnet32, batch_k=2**i)

for i in range(10):
    CIFAR10_HParams_Dict[f'preresnet32_b64_sde1_k{2**i}'] = sde1(preresnet32_b64, batch_k=2**i)
    CIFAR10_HParams_Dict[f'preresnet32_b64_sde2_k{2**i}'] = sde2(preresnet32_b64, batch_k=2**i)

    b64_baseline = CIFAR10_HParams_Dict['preresnet32_b64_fig1']
    CIFAR10_HParams_Dict[f'preresnet32_b64_fig1_sde1_k{2**i}'] = sde1(b64_baseline, batch_k=2**i)

    
for i in range(10):
    CIFAR10_HParams_Dict[f'preresnet32_b32_sde1_k{2**i}'] = sde1(preresnet32_b32, batch_k=2**i)
    CIFAR10_HParams_Dict[f'preresnet32_b32_sde2_k{2**i}'] = sde2(preresnet32_b32, batch_k=2**i)

for i in range(0, 17): # 1 to 65536
    CIFAR10_HParams_Dict[f'preresnet32_b{2**i}_linscale'] = linscale_rule(preresnet32, 2**i)
    
for i in range(0, 17): # 1 to 65536
    CIFAR10_HParams_Dict[f'preresnet32_4x_b{2**i}_linscale'] = linscale_rule(preresnet32_4x, 2**i)

for i in range(0, 17): # 1 to 65536
    CIFAR10_HParams_Dict[f'preresnet32_gn_b{2**i}_linscale'] = linscale_rule(preresnet32_gn, 2**i)

for i in range(0, 17):
    CIFAR10_HParams_Dict[f'preresnet32_simple_b{2**i}_linscale'] = linscale_rule(preresnet32_simple, 2**i)
    CIFAR10_HParams_Dict[f'preresnet32_simple_gn_b{2**i}_linscale'] = linscale_rule(preresnet32_simple_gn, 2**i)
    CIFAR10_HParams_Dict[f'preresnet32_4x_simple_b{2**i}_linscale'] = linscale_rule(preresnet32_4x_simple, 2**i)
    CIFAR10_HParams_Dict[f'preresnet32_4x_simple_gn_b{2**i}_linscale'] = linscale_rule(preresnet32_4x_simple_gn, 2**i)

b1024_baseline = CIFAR10_HParams_Dict['preresnet32_b1024_linscale']
for i in range(10):
    CIFAR10_HParams_Dict[f'preresnet32_b1024_sde1_k{2**i}'] = sde1(b1024_baseline, batch_k=2**i)

# not scale invariant resnet32
resnet32 = CIFAR10_HParams(
    arch = 'resnet',
    depth = 32,
    widths = [16, 32, 64],
    widen_factor = 1,
    norm_method = 'BN',
    lr=0.8,
    weight_decay = 0.0005,
    momentum=0,
    train_batch_size=128,
    test_batch_size=256,
    schedule = [80,120],
    lr_decay_factors = [0.1,0.1],
    wd_decay_factors = [1.,1.],
    max_epochs = 160,
    fix_last_layer = True,
    bn_affine = False,
    homo = True,
    drop_last_batch = True,
)

resnet32_simple = CIFAR10_HParams(
    arch = 'resnet',
    depth = 32,
    widths = [16, 32, 64],
    widen_factor = 1,
    norm_method = 'BN',
    lr=0.8,
    weight_decay = 0.0005,
    momentum=0,
    train_batch_size=128,
    test_batch_size=256,
    schedule = [250],
    lr_decay_factors = [0.1],
    wd_decay_factors = [1.],
    max_epochs = 300,
    fix_last_layer = True,
    bn_affine = False,
    homo = True,
    drop_last_batch = True,
)
CIFAR10_HParams_Dict['resnet32'] = resnet32
CIFAR10_HParams_Dict['resnet32_simple'] = resnet32_simple
resnet32_b64 = evolve(resnet32, train_batch_size=64)

for i in range(10):
    CIFAR10_HParams_Dict[f'resnet32_b64_sde1_k{2**i}'] = sde1(resnet32_b64, batch_k=2**i)

for i in range(17):
    CIFAR10_HParams_Dict[f'resnet32_simple_b{2**i}_linscale'] = linscale_rule(resnet32_simple, 2**i)

# vgg setup
vgg19 = CIFAR10_HParams(
    arch = 'vgg19_bn',
    widths=None,
    depth=19,
    norm_method = 'BN',
    lr=0.8,
    weight_decay = 0.0005,
    momentum=0,
    train_batch_size=128,
    test_batch_size=256,
    schedule = [80,120],
    lr_decay_factors = [0.1,0.1],
    wd_decay_factors = [1.,1.],
    max_epochs = 160,
    fix_last_layer = True,
    bn_affine = False,
    homo = True,
    drop_last_batch = True,
)
vgg19_gn = evolve(vgg19, norm_method = 'GN')
vgg16 = evolve(vgg19, arch = 'vgg19_bn')

CIFAR10_HParams_Dict['vgg19'] = vgg19
CIFAR10_HParams_Dict['vgg19_gn'] = vgg19_gn

for i in range(0, 17): # 1 to 65536
    CIFAR10_HParams_Dict[f'vgg19_b{2**i}_linscale'] = linscale_rule(vgg19, 2**i)


vgg19_simple = CIFAR10_HParams(
    arch = 'vgg19_bn',
    widths=None,
    depth=19,
    norm_method = 'BN',
    lr=0.8,
    weight_decay = 0.0005,
    momentum=0,
    train_batch_size=128,
    test_batch_size=256,
    schedule = [250],
    lr_decay_factors = [0.1],
    wd_decay_factors = [1.],
    max_epochs = 300,
    fix_last_layer = True,
    bn_affine = False,
    homo = True,
    drop_last_batch = True,
)

def warm_up(hparam,kappa = 100.):
    return evolve(hparam, schedule = [0,1]+hparam.schedule, lr_decay_factors = [1/kappa,kappa] + hparam.lr_decay_factors, wd_decay_factors = [1.,1.] + hparam.wd_decay_factors)

vgg19_simple_gn  = evolve(vgg19_simple, norm_method = "GN")
vgg19_no_normalization_simple = evolve(vgg19_simple, norm_method = 'None', lr = 0.01, wb_project = 'appendix-linscale-no-normalization', measure_variance = True)
vgg19_no_normalization_simple_warm_up = warm_up(vgg19_no_normalization_simple,kappa = 100.)

for setting in ['vgg19_simple','vgg19_simple_gn','vgg19_no_normalization_simple','vgg19_no_normalization_simple_warm_up']:
    gen_linscale(CIFAR10_HParams_Dict, setting)

### no normalization


preresnet32_simple_no_normalization = evolve(preresnet32_simple, norm_method = None, lr = 0.01, wb_project = 'appendix-no-normalization', measure_variance = True)

vgg19_simple_no_normalization = evolve(vgg19_simple, norm_method = 'None', lr = 0.01, wb_project = 'appendix-no-normalization', measure_variance = True)
    
    
for setting in ['preresnet32_simple_no_normalization','vgg19_simple_no_normalization']:
    gen_linscale(CIFAR10_HParams_Dict, setting)
    


preresnet32_b128_no_normalization = evolve(preresnet32, norm_method = 'None', lr = 0.01, wb_project = 'appendix-no-normalization', measure_variance = True)
preresnet32_b128_no_normalization_large_lr = evolve(preresnet32_b128_no_normalization, lr=0.1)
preresnet32_b128_no_normalization_small_wd = evolve(preresnet32, norm_method = 'None', lr = 0.01, wb_project = 'appendix-no-normalization', measure_variance = True, weight_decay=0.0001)
preresnet32_b128_no_normalization_warm_up = warm_up(preresnet32_b128_no_normalization, kappa = 100.)
preresnet32_b128_no_normalization_large_lr_warm_up = warm_up(preresnet32_b128_no_normalization_large_lr, kappa=1000.)
preresnet32_b128_no_normalization_small_wd_warm_up = warm_up(preresnet32_b128_no_normalization_small_wd, kappa = 100.)
# preresnet32_b128_no_normalization = evolve(preresnet32, lr = 0.001, wb_project = 'appendix-no-normalization', measure_variance = True)
vgg16_b128_no_normalization = evolve(vgg16, norm_method = 'None', lr = 0.01, wb_project = 'appendix-no-normalization', measure_variance = True)
vgg16_b128_no_normalization_small_wd = evolve(vgg16, norm_method = 'None', lr = 0.01, wb_project = 'appendix-no-normalization', measure_variance = True, weight_decay=0.0001)



preresnet32_4x_b128_no_normalization = evolve(preresnet32_b128_no_normalization, widen_factor = 4)
preresnet32_4x_b128_no_normalization_warm_up = warm_up(preresnet32_4x_b128_no_normalization, kappa=1000.)
preresnet32_4x_b128_no_normalization_small_lr_warm_up = evolve(preresnet32_4x_b128_no_normalization_warm_up, lr=0.001)
### trainable last layer

preresnet32_b128_no_normalization_l_warm_up = evolve(preresnet32_b128_no_normalization_warm_up, fix_last_layer = False)
preresnet32_b128_no_normalization_l_small_wd_warm_up = evolve(preresnet32_b128_no_normalization_small_wd_warm_up, fix_last_layer = False)

for setting in ['preresnet32_b128_no_normalization', 
                'preresnet32_b128_no_normalization_warm_up', 
                'preresnet32_b128_no_normalization_small_wd_warm_up', 
                'vgg16_b128_no_normalization',
                'vgg16_b128_no_normalization_small_wd',
                'preresnet32_b128_no_normalization_l_warm_up',
                'preresnet32_b128_no_normalization_l_small_wd_warm_up',
                'preresnet32_b128_no_normalization_large_lr_warm_up',
                'preresnet32_4x_b128_no_normalization_warm_up',
                'preresnet32_4x_b128_no_normalization_small_lr_warm_up'
               ]:
    gen_sde1(CIFAR10_HParams_Dict, setting)
    
    

    
    
###### cos lr scheduler 


def cos_lr(hparam, lr_min=0.0001):
    new_lr_decay_factors = []
    start = 0
    end = hparam.max_epochs
    T = end-start
    def cos_value(epoch):
        return (hparam.lr-lr_min)* (1+math.cos(epoch *math.pi/T ) )/2. +0.001
    for epoch in range(start,end):
        new_lr_decay_factors.append(cos_value(epoch+1)/ cos_value(epoch) )
    return evolve(hparam, lr_decay_factors = new_lr_decay_factors ,schedule  = range(start,end), wd_decay_factors = [1]*T, wb_project = 'cos_lr_svag')
    
preresnet32_cos = cos_lr(preresnet32)
vgg16_cos = cos_lr(vgg16)

for setting in ['preresnet32_cos','vgg16_cos']:
    gen_sde1(CIFAR10_HParams_Dict, setting)
    
###### cos lr scheduler 


def tri_lr(hparam, lr_min=0.0001, heights = [1.0,0.5]):
    new_lr_decay_factors = []
    start = 0
    end = hparam.max_epochs
    T = end-start
    cycles = len(heights)
    assert(T% (2*cycles) ==0)
    def tri_value(epoch):
        ratio = (abs(T//(2*cycles) - epoch% (T//cycles)))/ ( T/(2*cycles))
        return ((hparam.lr-lr_min)*(1-ratio) +lr_min) * heights[min(epoch // (T//cycles), len(heights)-1)]
    for epoch in range(start,end):
        new_lr_decay_factors.append(tri_value(epoch+1)/ tri_value(epoch) )
    return evolve(hparam, lr_decay_factors = new_lr_decay_factors ,schedule  = range(start,end), wd_decay_factors = [1]*T, wb_project = 'triangle_lr_svag', lr = lr_min)
    
preresnet32_tri = tri_lr(preresnet32,heights = [1.0,0.5])
vgg16_tri = tri_lr(vgg16,heights = [1.0,0.5])

for setting in ['preresnet32_tri','vgg16_tri']:
    gen_sde1(CIFAR10_HParams_Dict, setting)


###### testing different sampling modes
sample_mode_list = ['random_shuffling','without_replacement', 'with_replacement', 'fixed_sequence']
for sample_mode in sample_mode_list:
    CIFAR10_HParams_Dict[sample_mode] = evolve(preresnet32, sample_mode = sample_mode)
