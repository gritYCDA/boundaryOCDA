#!/bin/bash
### baseline of GTA(1280, 720) -> BDD(1280, 720): LSGAN
# source only
CUDA_VISIBLE_DEVICES=3 python train.py --cfg configs/source_only_VGG.yml --tensorboard --exp-suffix test

# adaptseg
CUDA_VISIBLE_DEVICES=2 python train.py --cfg configs/adaptseg_VGG.yml --tensorboard

# advent
CUDA_VISIBLE_DEVICES=1 python train.py --cfg configs/advent_VGG.yml --tensorboard



### Round 1   -    baseline of GTA(1280, 720) -> BDD(960, 540): LSGAN - resize version
# source only
CUDA_VISIBLE_DEVICES=2 python train.py --cfg configs/source_only_VGG.yml --tensorboard --exp-suffix resize

# adaptseg
CUDA_VISIBLE_DEVICES=3 python train.py --cfg configs/adaptseg_VGG.yml --tensorboard --exp-suffix resize

# advent
CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/advent_VGG.yml --tensorboard --exp-suffix resize



### Round 2   -    baseline of GTA(1280, 720) -> BDD(960, 540): LSGAN - resize version
# adaptseg
CUDA_VISIBLE_DEVICES=3 python train.py --cfg configs/adaptseg_VGG.yml --tensorboard --exp-suffix resize

# advent
CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/advent_VGG.yml --tensorboard --exp-suffix resize



####################################################
# LSGAN - adaptsegnet
# cfg.TRAIN.LAMBDA_ADV_MAIN = 0.01
# INPUT_SIZE_TARGET = (960, 540)
# OUTPUT_SIZE_TARGET = (1280, 720)
# adaptseg
CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/adaptseg_VGG.yml --tensorboard --exp-suffix lsgan
# R2
CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/adaptseg_VGG.yml --tensorboard --exp-suffix lsgan_R2 | tee ./log/adaptseg_VGG_lsgan_R2

# advent
CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/advent_VGG.yml --tensorboard --exp-suffix lsgan





#####################################################################
### baseline of GTA(1280, 720) -> Cityscapes(1280, 720): LSGAN
#####################################################################

# source only
CUDA_VISIBLE_DEVICES=2 python train.py --cfg configs/source_only_VGG.yml --source GTA --target Cityscapes --tensorboard

# adaptseg baseline
CUDA_VISIBLE_DEVICES=1 python train.py --cfg ./configs/adaptseg_VGG.yml --gan lsgan --source GTA --target Cityscapes --tensorboard --exp-suffix lsgan | tee ./log/adaptseg_VGG_lsgan

# adaptseg baseline
CUDA_VISIBLE_DEVICES=1 python train.py --cfg ./configs/adaptseg_VGG.yml --gan gan --source GTA --target Cityscapes --tensorboard --exp-suffix gan | tee ./log/adaptseg_VGG_gan



























































