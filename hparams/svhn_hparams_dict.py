from .hparams import SVHN_HParams
from .cifar10_hparams_dict import sde1, sde2, linscale_rule, gen_linscale, warm_up
from attr import evolve

import numpy as np

__all__ = ['SVHN_HParams_Dict']

preresnet32 = SVHN_HParams(
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
    wb_project = 'appendix-runs',
    drop_last_batch = True,
    measure_variance=True
)

preresnet32_simple = SVHN_HParams(
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
    schedule = [100],
    lr_decay_factors = [0.1],
    wd_decay_factors = [1.],
    max_epochs = 120,
    fix_last_layer = True,
    bn_affine = False,
    homo = True,
    wb_project = 'svhn-v2',
    drop_last_batch = True,
    data_dir = '/n/fs/ptml/datasets/svhn/',
    measure_variance=True
)

SVHN_HParams_Dict = {}
SVHN_HParams_Dict['preresnet32'] = preresnet32_simple

preresnet32_4x_simple = evolve(preresnet32_simple, widen_factor=4)
preresnet32_simple_gn = evolve(preresnet32_simple, norm_method = 'GN')
preresnet32_simple_gn_b1024 = evolve(preresnet32_simple_gn, train_batch_size=1024)
                         
# linscale configurations
for i in range(0, 17):
    SVHN_HParams_Dict[f'preresnet32_simple_b{2**i}_linscale'] = linscale_rule(preresnet32_simple, 2**i)
    SVHN_HParams_Dict[f'preresnet32_simple_gn_b{2**i}_linscale'] = linscale_rule(preresnet32_simple_gn, 2**i)

# SVAG configurations
for i in range(10):
    SVHN_HParams_Dict[f'preresnet32_simple_b128_sde1_k{2**i}'] = sde1(preresnet32_simple, batch_k=2**i)
    SVHN_HParams_Dict[f'preresnet32_simple_gn_b128_sde1_k{2**i}'] = sde1(preresnet32_simple_gn, batch_k=2**i)
    SVHN_HParams_Dict[f'preresnet32_simple_gn_b1024_sde1_k{2**i}'] = sde1(preresnet32_simple_gn_b1024, batch_k=2**i)

vgg16_simple = SVHN_HParams(
    arch = 'vgg16_bn',
    widths=None,
    depth=19,
    norm_method = 'BN',
    lr=0.8,
    weight_decay = 0.0005,
    momentum=0,
    train_batch_size=128,
    test_batch_size=256,
    schedule = [100],
    lr_decay_factors = [0.1,0.1],
    wd_decay_factors = [1.,1.],
    max_epochs = 120,
    fix_last_layer = True,
    bn_affine = False,
    homo = True,
    wb_project = 'svhn-v2',
    drop_last_batch = True,
    data_dir = '/n/fs/ptml/datasets/svhn/',
    measure_variance=True
)
for i in range(10):
    SVHN_HParams_Dict[f'vgg16_simple_b128_sde1_k{2**i}'] = sde1(vgg16_simple, batch_k=2**i)

vgg19 = SVHN_HParams(
    arch = 'vgg19_bn',
    widths=None,
    depth=19,
    norm_method = 'BN',
    lr=0.8,
    weight_decay = 0.0005,
    momentum=0,
    train_batch_size=128,
    test_batch_size=256,
    schedule = [100],
    lr_decay_factors = [0.1,0.1],
    wd_decay_factors = [1.,1.],
    max_epochs = 120,
    fix_last_layer = True,
    bn_affine = False,
    homo = True,
    wb_project = 'svhn-v2',
    drop_last_batch = True,
    data_dir = '/n/fs/ptml/datasets/svhn/',
    measure_variance=True
)
vgg19_gn = evolve(vgg19, norm_method = 'GN')

# linscale configurations
for i in range(0, 17):
    SVHN_HParams_Dict[f'vgg19_simple_b{2**i}_linscale'] = linscale_rule(vgg19, 2**i)
    SVHN_HParams_Dict[f'vgg19_simple_gn_b{2**i}_linscale'] = warm_up(linscale_rule(vgg19_gn, 2**i), kappa=100.)
