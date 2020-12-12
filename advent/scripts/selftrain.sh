#!/bin/bash
######################## self-learning Stage: CRST+IST ########################
# 12.3
########## basic self learning
# self-training basic: adaptseg warm-up + self-training
CUDA_VISIBLE_DEVICES=3 python train.py --cfg configs/ist_VGG.yml --tensorboard | tee ./log/ist_basic.txt
# 30.1

# self-training basic: source-only warm-up + self-training
CUDA_VISIBLE_DEVICES=3 python train.py --cfg configs/ist_source_only_VGG.yml --tensorboard | tee ./log/ist_sourceonly.txt

# self-training basic: boundary source-only warm-up + self-training
CUDA_VISIBLE_DEVICES=2 python train.py --cfg configs/ist_boundary_VGG.yml --tensorboard | tee ./log/ist_sourceonly_b.txt
