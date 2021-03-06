from .hparams import CIFAR100_HParams
from .cifar10_hparams_dict import sde1, sde2, linscale_rule, gen_linscale
from attr import evolve

__all__ = ['CIFAR100_HParams_Dict']

preresnet32_4x = CIFAR100_HParams(
    arch = 'preresnet',
    depth = 32,
    widths = [16, 32, 64],
    widen_factor = 4, # NOTE: baseline has 4x width
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
    wb_project = 'c100-v2',
    drop_last_batch = True,
    measure_variance=True
)

# NOTE: linscale schedule has 2 decays to get good performance
preresnet32_4x_simple = CIFAR100_HParams(
    arch = 'preresnet',
    depth = 32,
    widths = [16, 32, 64],
    widen_factor = 4, # NOTE: baseline has 4x width
    norm_method = 'BN',
    lr=0.8,
    weight_decay = 0.0005,
    momentum=0,
    train_batch_size=128,
    test_batch_size=256,
    schedule = [80, 250],
    lr_decay_factors = [0.1, 0.1],
    wd_decay_factors = [1., 1.],
    max_epochs = 300,
    fix_last_layer = True,
    bn_affine = False,
    homo = True,
    wb_project = 'c100-v2',
    drop_last_batch = True,
    measure_variance = True
)

CIFAR100_HParams_Dict = {}

preresnet32_4x_b1024 = evolve(preresnet32_4x, train_batch_size=1024)

preresnet32_4x_gn = evolve(preresnet32_4x, norm_method ='GN')
preresnet32_4x_simple_gn = evolve(preresnet32_4x_simple, norm_method = 'GN')


CIFAR100_HParams_Dict['preresnet32_4x_twodecay'] = evolve(preresnet32_4x, schedule=[80, 250], max_epochs=300, wb_project='c100-dbg')
CIFAR100_HParams_Dict['preresnet32_4x_gn_twodecay'] = evolve(preresnet32_4x_gn, schedule=[80, 250], max_epochs=300, wb_project='c100-dbg')


# SVAG configurations
for i in range(10):
    CIFAR100_HParams_Dict[f'preresnet32_4x_b128_sde1_k{2**i}'] = sde1(preresnet32_4x, batch_k=2**i)
    CIFAR100_HParams_Dict[f'preresnet32_4x_b1024_sde1_k{2**i}'] = sde1(preresnet32_4x_b1024, batch_k=2**i)
    CIFAR100_HParams_Dict[f'preresnet32_4x_gn_b128_sde1_k{2**i}'] = sde1(preresnet32_4x_gn, batch_k=2**i)
    
# linscale configurations
for i in range(0, 17):
    CIFAR100_HParams_Dict[f'preresnet32_4x_simple_b{2**i}_linscale'] = linscale_rule(preresnet32_4x_simple, 2**i)
    CIFAR100_HParams_Dict[f'preresnet32_4x_simple_gn_b{2**i}_linscale'] = linscale_rule(preresnet32_4x_simple_gn, 2**i)

####### VGG ########
vgg16 = CIFAR100_HParams(
    arch = 'vgg16_bn',
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
    wb_project = 'c100-v2',
    drop_last_batch = True,
    measure_variance=True,
)
vgg16_gn = evolve(vgg16, norm_method='GN')

for i in range(10):
    CIFAR100_HParams_Dict[f'vgg16_b128_sde1_k{2**i}'] = sde1(vgg16, batch_k=2**i)
    CIFAR100_HParams_Dict[f'vgg16_gn_b128_sde1_k{2**i}'] = sde1(vgg16_gn, batch_k=2**i)

# for linscale and SVAG
vgg19 = CIFAR100_HParams(
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
    wb_project = 'c100-v2',
    drop_last_batch = True,
    measure_variance=True,
)
vgg19_simple = CIFAR100_HParams(
    arch = 'vgg19_bn',
    widths=None,
    depth=19,
    norm_method = 'BN',
    lr=0.8,
    weight_decay = 0.0005,
    momentum=0,
    train_batch_size=128,
    test_batch_size=256,
    schedule = [80, 250],
    lr_decay_factors = [0.1,0.1],
    wd_decay_factors = [1.,1.],
    max_epochs = 300,
    fix_last_layer = True,
    bn_affine = False,
    homo = True,
    wb_project = 'c100-v2',
    drop_last_batch = True,
    measure_variance=True,
)
vgg19_simple_gn = evolve(vgg19_simple, norm_method='GN')
vgg19_gn = evolve(vgg19, norm_method='GN')

for i in range(17):
    CIFAR100_HParams_Dict[f'vgg19_simple_gn_b{2**i}_linscale'] = linscale_rule(vgg19_simple_gn, 2**i)
    CIFAR100_HParams_Dict[f'vgg19_simple_b{2**i}_linscale'] = linscale_rule(vgg19_simple, 2**i)    

for i in range(10):
    CIFAR100_HParams_Dict[f'vgg19_b128_sde1_k{2**i}'] = sde1(vgg19, batch_k=2**i)
    CIFAR100_HParams_Dict[f'vgg19_gn_b128_sde1_k{2**i}'] = sde1(vgg19_gn, batch_k=2**i)
