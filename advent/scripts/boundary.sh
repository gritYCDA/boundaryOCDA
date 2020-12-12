#!/bin/bash

######################## Boundary Source Only Stage ########################
# 11.30
########## boundary feature fusion refer to DADA
# boundary source only  - DADA style is NOT working: enc-dec & depthloss lamda: 0.01
CUDA_VISIBLE_DEVICES=3 python train.py --cfg configs/b_source_only_VGG.yml --tensorboard


########## classifier parallel design
# boundary loss lamda: 0.5, DICE loss - Dilataion conv twin version => dice only most effective
CUDA_VISIBLE_DEVICES=1 python train.py --cfg configs/DICE_b_source_only_VGG.yml --tensorboard --exp-suffix test_B_twinBoun

# boundary loss lamda: 0.5, BCE loss - Dilataion conv twin version
CUDA_VISIBLE_DEVICES=1 python train.py --cfg configs/BCE_b_source_only_VGG.yml --tensorboard --exp-suffix test_D_twinBoun

# boundary loss lamda: 0.5, (BCE loss + DICE loss) - Dilataion conv twin version
CUDA_VISIBLE_DEVICES=1 python train.py --cfg configs/BD_b_source_only_VGG.yml --tensorboard --exp-suffix test_twinBounary



########## classifier parallel design ** correct outline anti-fact **
# 0.1 * DICE loss
CUDA_VISIBLE_DEVICES=3 python train.py --cfg configs/DICE_b_source_only_VGG.yml --tensorboard --exp-suffix DICE_twinClassfier --LAMBDA_BOUNDARY 0.1 | tee ./log/DICE0p1.txt

# 0.5 * BCE loss
CUDA_VISIBLE_DEVICES=2 python train.py --cfg configs/BCE_b_source_only_VGG.yml --tensorboard --exp-suffix BCE_twinClassfier --LAMBDA_BOUNDARY 0.5  | tee ./log/BCE0p5.txt

# 0.1 * (BCE loss + 1.0 * DICE loss)
CUDA_VISIBLE_DEVICES=1 python train.py --cfg configs/BD_b_source_only_VGG.yml --tensorboard --exp-suffix DICEBCE_twinClassfier --LAMBDA_BOUNDARY 0.1 --LAMBDA_DICE 1.0  | tee ./log/BOTH0p1.txt


########## boundary quality experiments => (1) to use only things parts, (2) things + {load, sidewalk} by BCE-loss
# 12.6
# (1) to use only things parts
CUDA_VISIBLE_DEVICES=2 python train.py --cfg configs/BCE_b_source_only_VGG.yml --tensorboard --exp-suffix things_BCE --LAMBDA_BOUNDARY 0.5  | tee ./log/things_BCE.txt

# (2) things + {load, sidewalk} by BCE-loss
CUDA_VISIBLE_DEVICES=1 python train.py --cfg configs/BCE_b_source_only_VGG.yml --tensorboard --exp-suffix things_BCE78 --LAMBDA_BOUNDARY 0.5  | tee ./log/things_BCE78.txt



########## attention design





######################## Boundary Adaptation Stage ########################

# boundary adaptseg- exept boundary adaptation
CUDA_VISIBLE_DEVICES=3 python train.py --cfg configs/b_adaptseg_VGG.yml --tensorboard --exp-suffix notAdapt_B | tee ./log/b_adaptseg_VGG_notAdapt_B.txt

# boundary adaptseg- DADA style
CUDA_VISIBLE_DEVICES=2 python train.py --cfg configs/b_adaptseg_VGG.yml --tensorboard --exp-suffix DADA_style | tee ./log/b_adaptseg_VGG_DADA_style.txt

# boundary adaptseg- dual dictriminator:   LAMBDA_ADV_AUX: 0.002
CUDA_VISIBLE_DEVICES=1 python train.py --cfg configs/b_adaptseg_VGG.yml --tensorboard --exp-suffix dual_D | tee ./log/b_adaptseg_VGG_dual_D.txt

# boundary adaptseg- dual dictriminator :   LAMBDA_ADV_AUX: 0.0002
CUDA_VISIBLE_DEVICES=1 python train.py --cfg configs/b_adaptseg_VGG.yml --tensorboard --exp-suffix dual_D_0002




# boundary advent
CUDA_VISIBLE_DEVICES=1 python train.py --cfg configs/b_advent_VGG.yml --tensorboard



#####################################################################
    ### baseline of GTA(1280, 720) -> Cityscapes(1280, 720): LSGAN
#####################################################################
########## classifier parallel design
# boundary loss lamda: 0.5, DICE loss - Dilataion conv twin version => dice only most effective
# 0.5 * BCE loss
CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/BCE_b_source_only_VGG.yml --source GTA --target Cityscapes --tensorboard --exp-suffix BCE_twinClassfier --LAMBDA_BOUNDARY 0.5  | tee ./log_G2C/BCE0p5.txt





########## boundary classification feature concat
##  OCDA_METHOD: ad_boundary                                     [ DeeplabVGG_Boundary_Attention, train_Boundary_vgg ]   OCDA_METHOD: ad_boundary
CUDA_VISIBLE_DEVICES=3 python train.py --cfg configs/BCE_b_source_only_VGG.yml --source GTA --target Cityscapes --exp-suffix CAT --tensorboard | tee ./log_G2C/ad_boundary_sourceCAT.txt
# train_ad_Boundary_advent_vgg(): model feature fusion manner    [ DeeplabVGG_Boundary_Attention, train_ad_Boundary_advent_vgg ] OCDA_METHOD: ad_boundary, OPTION: segOnlyD
CUDA_VISIBLE_DEVICES=2 python train.py --cfg configs/b_adaptseg_VGG.yml --source GTA --target Cityscapes --exp-suffix CAT --option segD --tensorboard | tee ./log_G2C/ad_boundary_adaptsegCAT.txt

# train_cat_Boundary_advent_vgg(): output.cat((pred_main, pred_boundary), dim=1) - single discriminator using.  [ DeeplabVGG_Boundary, train_cat_Boundary_advent_vgg ] OCDA_METHOD: boundary, OPTION: catOutD
CUDA_VISIBLE_DEVICES=1 python train.py --cfg configs/b_adaptseg_VGG.yml --source GTA --target Cityscapes --exp-suffix outputCAT --option catOutD --tensorboard | tee ./log_G2C/ad_boundary_adaptsegOutputCAT.txt

# channel attention aggregate
# [ DeeplabVGG_Boundary_Attention_v2, train_cat_Boundary_advent_vgg ]  OCDA_METHOD: attn_boundary, OPTION: catOutD
CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/b_adaptseg_VGG.yml --source GTA --target Cityscapes --exp-suffix attn_outputCAT --option catOutD --tensorboard | tee ./log_G2C/ad_boundary_attn_adaptOutputCAT.txt












































