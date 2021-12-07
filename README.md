# Pre-Requisites
We provide a `environment.yml` file that contains the packages required. Running `conda env create -f environment.yml` will duplicate the conda environment under the name `sde`. 

All of our models require a GPU. Except for the NGD models, all experiments are designed to be run within 1 week on a single GPU.

We use Weights & Biases for cloud-based logging.

# Code Structure

This codebase uses PyTorch Lightning (1.0.8). To train a model, run `python train.py -c <config_name>`, where `<config_name>` specifies which architecture, LR schedule, and general hyperparameters to use (see below for configurations that reproduce our paper results). By default, the CIFAR-10 dataset will be downloaded to a `data/` file within this folder. If you would like to download it elsewhere, then modify the `data_dir` parameter in `CIFAR10_Module` in `hparams/hparams.py`. 

- `utils.py` contains argument parsing and callbacks for logging and measuring values during training.
- `hparams/cifar10_hparam_dict.py` contains all configurations used to produce results in the paper.
- `pl_modules/exp_modules.py` contains the modules used to run SVAG and NGD. `cifar10_module.py` contains the baseline module that was used to produce the linear scaling rule experiments. More information on reproducing experiments is below.

# Reproducing Paper Figures

Below are the configurations to reproduce the figures presented in the paper. Pass them through the `-c` flag to `train.py` to reproduce the results in the paper. To test with SVHN or CIFAR-100, use the flags `-svhn` and `-c100` respectively. 

### SVAG Convergence

The $B=64$ setting can be run using `preresnet32_b64_sde1_k{l_val}`, where `l_val` is the value of $l$ desired (e.g., 1 to 32). The $B=1024$ setting can be run using `preresnet32_b1024_sde1_k{l_val}`.

### Verification of Linear Scaling Rule Theory

The PreResNet-32 with BatchNorm results (left) can be reproduced with `preresnet32_simple_b{batch_size}_linscale`, where `batch_size` is the desired batch size (in the figure, 2 to 2048). The PreResNet-32 with GroupNorm results (center) can be reproduced with `preresnet32_simple_gn_b{batch_size}_linscale`. The VGG19 with GroupNorm results (right) can be reproduced with `vgg19_simple_gn_b{batch_size}_linscale`.

In all cases, the learning rate will be scaled per the linear scaling rule with a baseline of $B=128$ and $\eta=0.8$. $G_t$ and $N_t$ are measured by taking the average in the last 50 epochs of the first phase (i.e., epochs 200 to 250).

### SGD and NGD Trajectories

The NGD trajectory can be run with `preresnet32_gn_ngd_50k_b500_accumu_5`, and the corresponding SGD baseline is `preresnet32_gn_ngd_base_50k_b500`. Note that we smoothed the training accuracy by dividing the trajectory into 100-step intervals and recording the average over each interval. NGD is compute intensive, so we recommend running it on multiple GPUs (we used 4).

### SVAG Baselines and Large Batch

PreResNet-32 with BN and batch size 128 (left) can be run with `preresnet32_b128_sde1_k{l_val}`where `l_val` is the value of $l$. PreResNet-32 with BN and batch size 1024 (center) can be run with `preresnet32_b1024_sde1_k{l_val}`, and PreResNet-32 with GN and batch size 128 (right) can be run with `preresnet32_gn_b128_sde1_k{l_val}`.

### SVAG with Different Architectures

The wider PreResNet-32 (left) can be run with `preresnet32_4x_b128_sde1_k{l_val}`, where `l_val` is the value of $l$. VGG16 with BN (center) can be run with `vgg16_b128_sde1_k{l_val}`, and VGG16 with GN (right) can be run with `vgg16_gn_b128_sde1_k{l_val}`.

### SVAG on Unnormalized Network

VGG16 without normalization can be run with `vgg16_b128_no_normalization_sde1_k{l_val}` where `l_val` is the value of $l$.

### SVAG on Triangle Learning Rate Schedule

The triangle learning rate schedule can be run with `preresnet32_tri_sde1_k{l_val}` (for PreResNet-32, left) or `vgg16_tri_sde1_k{l_val}` (VGG16, right).

### SVAG on Cosine Learning Rate Schedule

The cosine learning rate schedule can be run with `preresnet32_cos_sde1_k{l_val}` (for PreResNet-32, left) or `vgg16_cos_sde1_k{l_val}` (VGG16, right).

### Further LSR verification

ResNet-32 with BN (not modified to be scale invariant, left) can be run with `resnet32_simple_b{batch_size}_linscale` where `batch_size` is the batch size desired. The 4x wider PreResNet-32 with BN (center) can be run with `preresnet32_4x_simple_b{batch_size}_linscale` and the version with GN (right) can be run with `preresnet32_4x_simple_gn_b{batch_size}_linscale`.

### NGD with Wider PreResNet

The NGD trajectory can be run with `preresnet32_4x_gn_ngd_50k_medium_accumu_10`, and the SGD trajectory can be run with `preresnet32_4x_gn_ngd_base_50k_medium`. Note that the NGD trajectory is compute-intensive and hence should be run multiple GPUs (we used 5).

### NGD with Smaller Batch Size

The NGD trajectory can be run with `preresnet32_gn_ngd_50k_accumu_5` and the SGD trajectory can be run with `preresnet32_gn_ngd_base_50k`. Again, we ran NGD using 5 GPUs.
