# --------------------------------------------------------
# Domain adpatation training
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------
import os
import sys
from pathlib import Path

import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
from torchvision.utils import make_grid
from tqdm import tqdm
import copy
from advent.model.discriminator import get_fc_discriminator
from advent.model.conv_abstract import get_conv_abstract
from advent.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from advent.utils.func import loss_calc, bce_loss, mse_loss, reg_loss_calc_ign
from advent.utils.loss import entropy_loss
from advent.utils.simclr_loss import NTXentLoss
from advent.utils.func import prob_2_entropy
from advent.utils.viz_segmask import colorize_mask

import random

def train_advent(model, trainloader, targetloader, cfg):
    ''' UDA training with advent
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # DISCRIMINATOR NETWORK
    # feature-level
    d_aux = get_fc_discriminator(num_classes=num_classes)
    d_aux.train()
    d_aux.to(device)

    # seg maps, i.e. output, level
    d_main = get_fc_discriminator(num_classes=num_classes)
    d_main.train()
    d_main.to(device)

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # discriminators' optimizers
    optimizer_d_aux = optim.Adam(d_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                 betas=(0.9, 0.99))
    optimizer_d_main = optim.Adam(d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1
    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)):

        # reset optimizers
        optimizer.zero_grad()
        optimizer_d_aux.zero_grad()
        optimizer_d_main.zero_grad()
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)

        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        for param in d_aux.parameters():
            param.requires_grad = False
        for param in d_main.parameters():
            param.requires_grad = False
        # train on source
        _, batch = trainloader_iter.__next__()
        images_source, labels, _, _ = batch
        pred_src_aux, pred_src_main = model(images_source.cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = interp(pred_src_aux)
            loss_seg_src_aux = loss_calc(pred_src_aux, labels, device)
        else:
            loss_seg_src_aux = 0
        pred_src_main = interp(pred_src_main)
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        loss.backward()

        # adversarial training ot fool the discriminator
        _, batch = targetloader_iter.__next__()
        images, _, _, _ = batch
        pred_trg_aux, pred_trg_main = model(images.cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux = interp_target(pred_trg_aux)
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
            loss_adv_trg_aux = bce_loss(d_out_aux, source_label)
        else:
            loss_adv_trg_aux = 0
        pred_trg_main = interp_target(pred_trg_main)
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
        loss_adv_trg_main = bce_loss(d_out_main, source_label)
        loss = (cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main
                + cfg.TRAIN.LAMBDA_ADV_AUX * loss_adv_trg_aux)
        loss = loss
        loss.backward()

        # Train discriminator networks
        # enable training mode on discriminator networks
        for param in d_aux.parameters():
            param.requires_grad = True
        for param in d_main.parameters():
            param.requires_grad = True
        # train with source
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = pred_src_aux.detach()
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_src_aux)))
            loss_d_aux = bce_loss(d_out_aux, source_label)
            loss_d_aux = loss_d_aux / 2
            loss_d_aux.backward()
        pred_src_main = pred_src_main.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main)))
        loss_d_main = bce_loss(d_out_main, source_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        # train with target
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux = pred_trg_aux.detach()
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
            loss_d_aux = bce_loss(d_out_aux, target_label)
            loss_d_aux = loss_d_aux / 2
            loss_d_aux.backward()
        else:
            loss_d_aux = 0
        pred_trg_main = pred_trg_main.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
        loss_d_main = bce_loss(d_out_main, target_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        optimizer.step()
        if cfg.TRAIN.MULTI_LEVEL:
            optimizer_d_aux.step()
        optimizer_d_main.step()

        current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
                          'loss_seg_src_main': loss_seg_src_main,
                          'loss_adv_trg_aux': loss_adv_trg_aux,
                          'loss_adv_trg_main': loss_adv_trg_main,
                          'loss_d_aux': loss_d_aux,
                          'loss_d_main': loss_d_main}
        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            torch.save(d_aux.state_dict(), snapshot_dir / f'model_{i_iter}_D_aux.pth')
            torch.save(d_main.state_dict(), snapshot_dir / f'model_{i_iter}_D_main.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_in_tensorboard(writer, images, i_iter, pred_trg_main, num_classes, 'T')
                draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S')


def train_adaptseg(model, trainloader, targetloader, cfg):
    ''' UDA training with advent
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # DISCRIMINATOR NETWORK
    # feature-level
    d_aux = get_fc_discriminator(num_classes=num_classes)
    d_aux.train()
    d_aux.to(device)

    # seg maps, i.e. output, level
    d_main = get_fc_discriminator(num_classes=num_classes)
    d_main.train()
    d_main.to(device)

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # discriminators' optimizers
    optimizer_d_aux = optim.Adam(d_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                 betas=(0.9, 0.99))
    optimizer_d_main = optim.Adam(d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1
    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)):

        # reset optimizers
        optimizer.zero_grad()
        optimizer_d_aux.zero_grad()
        optimizer_d_main.zero_grad()
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)

        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        for param in d_aux.parameters():
            param.requires_grad = False
        for param in d_main.parameters():
            param.requires_grad = False
        # train on source
        _, batch = trainloader_iter.__next__()
        images_source, labels, _, _ = batch
        pred_src_main = model(images_source.cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = interp(pred_src_aux)
            loss_seg_src_aux = loss_calc(pred_src_aux, labels, device)
        else:
            loss_seg_src_aux = 0
        pred_src_main = interp(pred_src_main)
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        loss.backward()

        # adversarial training ot fool the discriminator
        _, batch = targetloader_iter.__next__()
        images, _, _, _ = batch
        pred_trg_main = model(images.cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux = interp_target(pred_trg_aux)
            d_out_aux = d_aux(F.softmax(pred_trg_aux))
            loss_adv_trg_aux = mse_loss(d_out_aux, source_label)
        else:
            loss_adv_trg_aux = 0
        pred_trg_main = interp_target(pred_trg_main)
        d_out_main = d_main(F.softmax(pred_trg_main))
        loss_adv_trg_main = mse_loss(d_out_main, source_label)
        loss = (cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main
                + cfg.TRAIN.LAMBDA_ADV_AUX * loss_adv_trg_aux)
        loss = loss
        loss.backward()

        # Train discriminator networks
        # enable training mode on discriminator networks
        for param in d_aux.parameters():
            param.requires_grad = True
        for param in d_main.parameters():
            param.requires_grad = True
        # train with source
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = pred_src_aux.detach()
            d_out_aux = d_aux(F.softmax(pred_src_aux))
            loss_d_aux = mse_loss(d_out_aux, source_label)
            loss_d_aux = loss_d_aux / 2
            loss_d_aux.backward()
        pred_src_main = pred_src_main.detach()
        d_out_main = d_main(F.softmax(pred_src_main))
        loss_d_main = mse_loss(d_out_main, source_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        # train with target
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux = pred_trg_aux.detach()
            d_out_aux = d_aux(F.softmax(pred_trg_aux))
            loss_d_aux = mse_loss(d_out_aux, target_label)
            loss_d_aux = loss_d_aux / 2
            loss_d_aux.backward()
        else:
            loss_d_aux = 0
        pred_trg_main = pred_trg_main.detach()
        d_out_main = d_main(F.softmax(pred_trg_main))
        loss_d_main = mse_loss(d_out_main, target_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        optimizer.step()
        if cfg.TRAIN.MULTI_LEVEL:
            optimizer_d_aux.step()
        optimizer_d_main.step()

        current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
                          'loss_seg_src_main': loss_seg_src_main,
                          'loss_adv_trg_aux': loss_adv_trg_aux,
                          'loss_adv_trg_main': loss_adv_trg_main,
                          'loss_d_aux': loss_d_aux,
                          'loss_d_main': loss_d_main}
        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            torch.save(d_aux.state_dict(), snapshot_dir / f'model_{i_iter}_D_aux.pth')
            torch.save(d_main.state_dict(), snapshot_dir / f'model_{i_iter}_D_main.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_in_tensorboard(writer, images, i_iter, pred_trg_main, num_classes, 'T')
                draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S')


def train_adaptseg_w_trans(model, trainloader, targetloader, cfg):
    ''' UDA training with advent
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # DISCRIMINATOR NETWORK
    # feature-level
    d_aux = get_fc_discriminator(num_classes=num_classes)
    d_aux.train()
    d_aux.to(device)

    # seg maps, i.e. output, level
    d_main = get_fc_discriminator(num_classes=num_classes)
    d_main.train()
    d_main.to(device)

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # discriminators' optimizers
    optimizer_d_aux = optim.Adam(d_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                 betas=(0.9, 0.99))
    optimizer_d_main = optim.Adam(d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1
    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)
    criterion = nn.MSELoss()
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)):

        # reset optimizers
        optimizer.zero_grad()
        optimizer_d_aux.zero_grad()
        optimizer_d_main.zero_grad()
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)

        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        for param in d_aux.parameters():
            param.requires_grad = False
        for param in d_main.parameters():
            param.requires_grad = False
        # train on source
        _, batch = trainloader_iter.__next__()
        images_source, labels, _, _ = batch
        pred_src_main, _ = model(images_source.cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = interp(pred_src_aux)
            loss_seg_src_aux = loss_calc(pred_src_aux, labels, device)
        else:
            loss_seg_src_aux = 0
        pred_src_main = interp(pred_src_main)
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        loss.backward()

        # adversarial training ot fool the discriminator
        _, batch = targetloader_iter.__next__()
        images, images_aug, _, _, _ = batch
        pred_trg_main, _ = model(images_aug.cuda(device))
        pred_trg_main_real, _ = model(images.cuda(device))

        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux = interp_target(pred_trg_aux)
            d_out_aux = d_aux(F.softmax(pred_trg_aux))
            loss_adv_trg_aux = mse_loss(d_out_aux, source_label)
        else:
            loss_adv_trg_aux = 0
        pred_trg_main = interp_target(pred_trg_main)
        pred_trg_main_real = interp_target(pred_trg_main_real)

        d_out_main = d_main(F.softmax(pred_trg_main))
        loss_adv_trg_main = mse_loss(d_out_main, source_label)
        loss = (cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main
                + cfg.TRAIN.LAMBDA_ADV_AUX * loss_adv_trg_aux)
        loss = loss
        loss.backward()

        # Train discriminator networks
        # enable training mode on discriminator networks
        for param in d_aux.parameters():
            param.requires_grad = True
        for param in d_main.parameters():
            param.requires_grad = True
        # train with source
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = pred_src_aux.detach()
            d_out_aux = d_aux(F.softmax(pred_src_aux))
            loss_d_aux = mse_loss(d_out_aux, source_label)
            loss_d_aux = loss_d_aux / 2
            loss_d_aux.backward()
        pred_src_main = pred_src_main.detach()
        d_out_main = d_main(F.softmax(pred_src_main))
        loss_d_main = mse_loss(d_out_main, source_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        # train with target
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux = pred_trg_aux.detach()
            d_out_aux = d_aux(F.softmax(pred_trg_aux))
            loss_d_aux = mse_loss(d_out_aux, target_label)
            loss_d_aux = loss_d_aux / 2
            loss_d_aux.backward()
        else:
            loss_d_aux = 0
        pred_trg_main = pred_trg_main.detach()
        d_out_main = d_main(F.softmax(pred_trg_main))
        loss_d_main = mse_loss(d_out_main, target_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()
        # import pdb
        # pdb.set_trace()
        loss_consistency = 10 * criterion(F.softmax(pred_trg_main_real), F.softmax(pred_trg_main).detach())
        loss_consistency.backward()


        optimizer.step()
        if cfg.TRAIN.MULTI_LEVEL:
            optimizer_d_aux.step()
        optimizer_d_main.step()

        current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
                          'loss_seg_src_main': loss_seg_src_main,
                          'loss_adv_trg_aux': loss_adv_trg_aux,
                          'loss_adv_trg_main': loss_adv_trg_main,
                          'loss_d_aux': loss_d_aux,
                          'loss_d_main': loss_d_main,
                          'loss_consistency': loss_consistency}
        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            torch.save(d_aux.state_dict(), snapshot_dir / f'model_{i_iter}_D_aux.pth')
            torch.save(d_main.state_dict(), snapshot_dir / f'model_{i_iter}_D_main.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == 0:
                draw_in_tensorboard_trans(writer, images_aug, images, i_iter, pred_trg_main, pred_trg_main_real, num_classes, 'T')
                draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S')

def label_generator(pred_trg, cls_thresh_tot, cfg, i_iter, tot_iter):
    import math
    device = cfg.GPU_ID
    ###   ### 
    output_main = F.softmax(pred_trg,dim=1)
    amax_output = torch.argmax(output_main, dim=1).type(torch.uint8)
    pred_label_trainIDs = amax_output.clone()
    pred_label = amax_output.clone()
    conf, _ = torch.max(output_main, dim=1)

    conf_dict = {k:[] for k in range(cfg.NUM_CLASSES)}
    pred_cls_num = torch.zeros(cfg.NUM_CLASSES)
    for idx_cls  in range(cfg.NUM_CLASSES):
        idx_temp = pred_label == idx_cls

        pred_cls_num[idx_cls] = pred_cls_num[idx_cls] + torch.sum(idx_temp)
        if idx_temp.any():
            conf_cls_temp = conf[idx_temp].type(torch.float32)
            len_cls_temp = len(conf_cls_temp)
            conf_cls = conf_cls_temp[0:len_cls_temp:16]
            conf_dict[idx_cls].extend(conf_cls)

    cls_thresh = torch.ones(cfg.NUM_CLASSES).type(torch.float32)
    cls_sel_size = torch.zeros(cfg.NUM_CLASSES).type(torch.float32)
    tgt_dict_tot = {}
    for idx_cls in range(cfg.NUM_CLASSES):

        if conf_dict[idx_cls] != None:
            # conf_dict[idx_cls].sort(reverse=True) # sort in descending order
            conf_dict[idx_cls], _ = torch.sort(torch.FloatTensor(conf_dict[idx_cls]), descending=True)
            len_cls = len(conf_dict[idx_cls])
            iter_ratio = 1.0-float(i_iter / (tot_iter+1))
            coeff = 0.2 * (iter_ratio ** 0.5)
            cls_sel_size[idx_cls] = int(math.floor(len_cls * coeff))
            len_cls_thresh = int(cls_sel_size[idx_cls])
            if len_cls_thresh != 0:
                cls_thresh[idx_cls] = conf_dict[idx_cls][len_cls_thresh-1]
            conf_dict[idx_cls] = None


    cls_thresh_tot_ = torch.where(cls_thresh_tot==1.0, cls_thresh, 0.9 * cls_thresh_tot + 0.1 * cls_thresh)
    cls_thresh_mask = (cls_thresh == 1.0) * (cls_thresh_tot!=1.0)
    cls_thresh_tot = torch.where(cls_thresh_mask==1.0, cls_thresh_tot, cls_thresh_tot_)
    
    weighted_prob = output_main / cls_thresh_tot.to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    weighted_pred_trainIDs = torch.argmax(weighted_prob, dim=1).type(torch.uint8)
    weighted_conf, _ = torch.max(weighted_prob, dim=1)

    weighted_pred_trainIDs[weighted_conf < 1] = 255

    return weighted_pred_trainIDs, cls_thresh_tot

def train_selfself(model, trainloader, targetloader, cfg):
    ''' UDA training with advent
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)

    # Model clone
    model_runner = copy.deepcopy(model)
    model_runner.eval()
    model_runner.to(device)

    conv3x3_tgt = get_conv_abstract(cfg)
    conv3x3_tgt.train()
    conv3x3_tgt.to(device)

    d_main = get_fc_discriminator(num_classes=num_classes)
    d_main.train()
    d_main.to(device)

    tgt_dict_tot = {}

    cudnn.benchmark = True
    cudnn.enabled = True

    # OPTIMIZERS
    params = list(model.parameters()) + list(conv3x3_tgt.parameters())
    optimizer = optim.SGD(params,
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    cls_thresh = torch.ones(num_classes).type(torch.float32)

    optimizer_d_main = optim.Adam(d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))

    # for round in range(3):

    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)

    source_label = 0
    target_label = 1

    for i_iter in tqdm(range(len(targetloader))):

        # reset optimizers
        optimizer.zero_grad()
        optimizer_d_main.zero_grad()

        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)

        # train on source
        _, batch = trainloader_iter.__next__()
        images_source, labels, _, _ = batch
        pred_src_main, _ = model(images_source.cuda(device))
 
        pred_src_main = interp(pred_src_main)
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)
        loss = cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
        loss.backward()

        # adversarial training ot fool the discriminator
        _, batch = targetloader_iter.__next__()
        images, images_rev, _, _, name, name_next = batch

        pred_trg_main, feat_trg_main = model(images.cuda(device))

        pred_trg_main = interp_target(pred_trg_main)
        
        with torch.no_grad():
            pred_trg_main_run, feat_trg_main_run = model_runner(images.cuda(device))
            pred_trg_main_run = interp_target(pred_trg_main_run)            

        ##### Label generator for target #####
        label_trg, cls_thresh = label_generator(pred_trg_main_run, cls_thresh, cfg, i_iter)


        ##### CE loss for trg
        # MRKLD + Ign Region
        loss_seg_trg_main = reg_loss_calc_ign(pred_trg_main, label_trg, device)
        loss_tgt_seg = cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_trg_main

        loss_tgt_selfsup, tgt_dict_tot = classSimCLR(feat_trg_main, label_trg, conv3x3_tgt, tgt_dict_tot, device)

        loss = loss_tgt_seg + 0.1 * loss_tgt_selfsup
        
        loss.backward()


        ##### Discriminator #####
        for param in d_main.parameters():
            param.requires_grad = False

        pred_trg_main_rev, _ = model(images_rev.cuda(device))
        pred_trg_main_rev  = interp_target(pred_trg_main_rev)
        d_out_main = d_main(F.softmax(pred_trg_main_rev))
        loss_adv_trg_main = mse_loss(d_out_main, source_label)
        loss = cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main
        loss = loss
        loss.backward()


        for param in d_main.parameters():
            param.requires_grad = True


        pred_src_main = pred_src_main.detach()
        d_out_main = d_main(F.softmax(pred_src_main))
        loss_d_main_src = mse_loss(d_out_main, source_label)
        loss_d_main = loss_d_main_src / 2
        loss_d_main.backward()

        pred_trg_main = pred_trg_main.detach()
        d_out_main = d_main(F.softmax(pred_trg_main))
        loss_d_main_trg = mse_loss(d_out_main, source_label)
        loss_d_main = loss_d_main_trg / 2
        loss_d_main.backward()

        pred_trg_main_rev = pred_trg_main_rev.detach()
        d_out_main = d_main(F.softmax(pred_trg_main_rev))
        loss_d_main_trg_rev = mse_loss(d_out_main, target_label)
        loss_d_main = loss_d_main_trg_rev / 2
        loss_d_main.backward()


        ##### Contrastive loss for trg
        # Contrastive loss ()

        optimizer.step()
        optimizer_d_main.step()

        if i_iter+1 % 500 == 0:
            
            model_runner = copy.deepcopy(model)
        #     for param_fol, param_run in zip(model.parameters(), model_runner.parameters()):
        #         param_run.data = param_fol.data

        current_losses = {'loss_seg_trg_main': loss_seg_trg_main,
                          'loss_seg_src_main': loss_seg_src_main,
                          'loss_tgt_selfsup': loss_tgt_selfsup,
                          'loss_adv_trg_main': loss_adv_trg_main,
                          'loss_d_main_src': loss_d_main_src,
                          'loss_d_main_trg': loss_d_main_trg,
                          'loss_d_main_trg_rev': loss_d_main_trg_rev
                          }

        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            torch.save(model_runner.state_dict(), snapshot_dir / f'model_{i_iter}_run.pth')
            torch.save(d_main.state_dict(), snapshot_dir / f'model_{i_iter}_D.pth')

            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == 0:
                # draw_in_tensorboard_trg(writer, images, images_rev, label_trg, i_iter, pred_trg_main, pred_trg_main_rev, num_classes, 'T')
                draw_in_tensorboard(writer, images, i_iter, pred_trg_main, num_classes, 'T')
                # draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S')


#TODO: self-training here !!!
def train_self_domain_swarp(model, trainloader, targetloader, cfg):
    ''' UDA training with advent
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)

    # Model clone
    model_runner = copy.deepcopy(model)
    model_runner.eval()
    model_runner.to(device)

    # conv3x3_tgt = get_conv_abstract(cfg)
    # conv3x3_tgt.train()
    # conv3x3_tgt.to(device)

    # d_main = get_fc_discriminator(num_classes=num_classes)
    # d_main.train()
    # d_main.to(device)

    tgt_dict_tot = {}

    cudnn.benchmark = True
    cudnn.enabled = True

    # OPTIMIZERS
    # params = list(model.parameters()) + list(conv3x3_tgt.parameters())
    optimizer = optim.SGD(model.parameters(),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    cls_thresh = torch.ones(num_classes).type(torch.float32)

    # optimizer_d_main = optim.Adam(d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
    #                               betas=(0.9, 0.99))

    # for round in range(3):

    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)

    source_label = 0
    target_label = 1

    tot_iter = len(targetloader)

    for i_iter in tqdm(range(tot_iter)):

        # reset optimizers
        optimizer.zero_grad()
        # optimizer_d_main.zero_grad()

        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)
        # adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)

        # train on source
        _, batch = trainloader_iter.__next__()
        images_source, labels, _, _ = batch
        pred_src_main, _ = model(images_source.cuda(device))
 
        pred_src_main = interp(pred_src_main)
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)
        loss = cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
        loss.backward()

        # adversarial training ot fool the discriminator
        _, batch = targetloader_iter.__next__()
        images, images_rev, _, _, name, name_next = batch

        pred_trg_main, feat_trg_main = model(images.cuda(device))

        pred_trg_main = interp_target(pred_trg_main)
        
        with torch.no_grad():
            pred_trg_main_run, feat_trg_main_run = model_runner(images.cuda(device))
            pred_trg_main_run = interp_target(pred_trg_main_run)            

        ##### Label generator for target #####
        label_trg, cls_thresh = label_generator(pred_trg_main_run, cls_thresh, cfg, i_iter, tot_iter)


        ##### CE loss for trg
        # MRKLD + Ign Region
        loss_seg_trg_main = reg_loss_calc_ign(pred_trg_main, label_trg, device)
        loss_tgt_seg = cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_trg_main

        ##### Domain swarping ####
        feat_tgt_swarped, tgt_dict_tot, tgt_label = DomainSwarping(feat_trg_main, label_trg, tgt_dict_tot, device)

        ignore_mask = tgt_label == 255

        feat_tgt_swarped = ~ignore_mask*feat_tgt_swarped + ignore_mask*feat_trg_main
        pred_tgt_swarped = model.classifier_(feat_tgt_swarped)
        pred_tgt_swarped = interp_target(pred_tgt_swarped)

        loss_seg_trg_swarped = reg_loss_calc_ign(pred_tgt_swarped, label_trg, device)
        loss_tgt_seg_swarped = cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_trg_swarped

        loss_tgt = loss_tgt_seg + loss_tgt_seg_swarped

        
        loss_tgt.backward()


        optimizer.step()

        current_losses = {'loss_seg_trg_main': loss_seg_trg_main,
                          'loss_seg_src_main': loss_seg_src_main,
                          'loss_seg_trg_swarped': loss_seg_trg_swarped
                          }

        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            torch.save(model_runner.state_dict(), snapshot_dir / f'model_{i_iter}_run.pth')

            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == 0:
                # draw_in_tensorboard_trg(writer, images, images_rev, label_trg, i_iter, pred_trg_main, pred_trg_main_rev, num_classes, 'T')
                draw_in_tensorboard(writer, images, label_trg, i_iter, pred_trg_main, pred_tgt_swarped, num_classes, 'T')
                # draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S')


def classSimCLR(tgt_feat_warped_cat, tgt_label, conv3x3_tgt, tgt_dict_tot, device):

    tgt_feat_warped_cat_abs = conv3x3_tgt(tgt_feat_warped_cat)
    # fnt_tgt_feat_warped_cat_abs = tgt_feat_warped_cat_abs[:,:tgt_feat_warped_cat_abs.size(1)//2,:,:]


    ##### class-wise Simclr #####
    tgt_label = F.interpolate(tgt_label.unsqueeze(0).float(), (tgt_feat_warped_cat.size(2), tgt_feat_warped_cat.size(3)), mode='nearest')
    tgt_label = tgt_label.long()
    tgt_unique = torch.unique(tgt_label)
    tgt_dict = {}
    tgt_dict_tot_temp = {}

    m = nn.AdaptiveAvgPool2d(1)

    for label_ele in tgt_unique.tolist():
        if not label_ele == 255:
            cls_mask = tgt_label == label_ele
            masked_tgt = cls_mask * tgt_feat_warped_cat_abs
            avg_masked_tgt = m(masked_tgt) * (cls_mask.size(2) * cls_mask.size(3) / cls_mask.sum())
            tgt_dict[label_ele] = avg_masked_tgt

            # if label_ele in tgt_dict_tot:
            #     tgt_dict_tot[label_ele] = 0.99 * tgt_dict_tot[label_ele] + 0.01 * tgt_dict[label_ele]
            # else:
            #     tgt_dict_tot[label_ele] = tgt_dict[label_ele]
            if not label_ele in tgt_dict_tot:
                tgt_dict_tot[label_ele] = tgt_dict[label_ele]

            tgt_dict_tot_temp[label_ele] = tgt_dict_tot[label_ele]

            if label_ele in tgt_dict_tot:
                tgt_dict_tot[label_ele] = 0.99 * tgt_dict_tot[label_ele] + 0.01 * tgt_dict[label_ele]

    tgt_dict = dict(sorted(tgt_dict.items()))
    tgt_list = []
    for key, value in tgt_dict.items():
        tgt_list.append(value)

    try:
        tgt_cat = torch.cat(tgt_list,dim=0).squeeze().to(device)
        tgt_cat = F.normalize(tgt_cat, dim=1)

        tgt_dict_tot_temp = dict(sorted(tgt_dict_tot_temp.items()))
        tgt_tot_temp_list = []
        for key, value in tgt_dict_tot_temp.items():
            tgt_tot_temp_list.append(value)


        tgt_dict_temp_cat = torch.cat(tgt_tot_temp_list,dim=0).squeeze().to(device)
        tgt_dict_temp_cat = F.normalize(tgt_dict_temp_cat, dim=1)



        batch_size = tgt_dict_temp_cat.size(0)
        simloss_xent = NTXentLoss(device, batch_size=batch_size, temperature=0.5, use_cosine_similarity=True)
        cls_sim_loss = simloss_xent(tgt_dict_temp_cat.detach(), tgt_cat)
        cls_sim_loss = cls_sim_loss
    except:
        cls_sim_loss = 0 

    # return  src_feat_embedding_loss, tgt_feat_embedding_loss, cls_sim_loss
    return cls_sim_loss, tgt_dict_tot



def DomainSwarping(tgt_feat_warped_cat, tgt_label, tgt_dict_tot, device):

    alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    alpha = random.choice(alpha_list)

    tgt_label = F.interpolate(tgt_label.unsqueeze(0).float(), (tgt_feat_warped_cat.size(2), tgt_feat_warped_cat.size(3)), mode='nearest')
    tgt_label = tgt_label.long()
    tgt_unique = torch.unique(tgt_label)
    tgt_dict = {}
    tgt_dict_tot_temp = {}

    m = nn.AdaptiveAvgPool2d(1)

    new_masked_tgt_init = 0

    for label_ele in tgt_unique.tolist():
        if not label_ele == 255:
            cls_mask = tgt_label == label_ele
            masked_tgt = cls_mask * tgt_feat_warped_cat
            avg_masked_tgt = m(masked_tgt) * (cls_mask.size(2) * cls_mask.size(3) / cls_mask.sum())
            tgt_dict[label_ele] = avg_masked_tgt

            if not label_ele in tgt_dict_tot:
                print('new class info inserted')
                tgt_dict_tot[label_ele] = tgt_dict[label_ele]

            # new_masked_tgt = alpha * tgt_dict_tot[label_ele] + (1-alpha) * masked_tgt
            new_masked_tgt = tgt_dict_tot[label_ele]
            new_masked_tgt_init += cls_mask * new_masked_tgt

            tgt_dict_tot[label_ele] = alpha * tgt_dict_tot[label_ele] + (1-alpha) * tgt_dict[label_ele]

            tgt_dict_tot[label_ele] = tgt_dict_tot[label_ele].detach()

    return new_masked_tgt_init, tgt_dict_tot, tgt_label


def draw_in_tensorboard_trg(writer, images, images_rev, label_trg, i_iter, pred_main, pred_trg_main_rev, num_classes, type_):
    grid_image = make_grid(images[:3].clone().cpu().data, 3, normalize=True)
    writer.add_image(f'Image - {type_}', grid_image, i_iter)

    grid_image = make_grid(images_rev[:3].clone().cpu().data, 3, normalize=True)
    writer.add_image(f'images_rev - {type_}', grid_image, i_iter)

    grid_image = make_grid(torch.from_numpy(np.array(colorize_mask(np.asarray(
        np.argmax(F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0),
                  axis=2), dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)), 3,
                           normalize=False, range=(0, 255))
    writer.add_image(f'Prediction - {type_}', grid_image, i_iter)

    grid_image = make_grid(torch.from_numpy(np.array(colorize_mask(np.asarray(
        np.argmax(F.softmax(pred_trg_main_rev).cpu().data[0].numpy().transpose(1, 2, 0),
                  axis=2), dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)), 3,
                           normalize=False, range=(0, 255))
    writer.add_image(f'Prediction_rev - {type_}', grid_image, i_iter)

    grid_image = make_grid(torch.from_numpy(np.array(colorize_mask(np.asarray(label_trg.cpu().squeeze(), dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)), 3,
                           normalize=False, range=(0, 255))
    writer.add_image(f'Labels_IAST - {type_}', grid_image, i_iter)

def draw_in_tensorboard(writer, images, label_trg, i_iter, pred_main, pred_main_swarp, num_classes, type_):
    grid_image = make_grid(images[:3].clone().cpu().data, 3, normalize=True)
    writer.add_image(f'Image - {type_}', grid_image, i_iter)

    pred_main_cat = torch.cat((pred_main, pred_main_swarp), dim=-1)

    grid_image = make_grid(torch.from_numpy(np.array(colorize_mask(np.asarray(
        np.argmax(F.softmax(pred_main_cat).cpu().data[0].numpy().transpose(1, 2, 0),
                  axis=2), dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)), 3,
                           normalize=False, range=(0, 255))
    writer.add_image(f'Prediction_main_swarp - {type_}', grid_image, i_iter)

    grid_image = make_grid(torch.from_numpy(np.array(colorize_mask(np.asarray(label_trg.cpu().squeeze(), dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)), 3,
                           normalize=False, range=(0, 255))
    writer.add_image(f'Labels_IAST - {type_}', grid_image, i_iter)


    # grid_image = make_grid(torch.from_numpy(np.array(colorize_mask(np.asarray(
    #     np.argmax(F.softmax(pred_main_tgt).cpu().data[0].numpy().transpose(1, 2, 0),
    #               axis=2), dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)), 3,
    #                        normalize=False, range=(0, 255))
    # writer.add_image(f'Prediction_swarped - {type_}', grid_image, i_iter)
    # output_sm = F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0)
    # output_ent = np.sum(-np.multiply(output_sm, np.log2(output_sm)), axis=2,
    #                     keepdims=False)
    # grid_image = make_grid(torch.from_numpy(output_ent), 3, normalize=True,
    #                        range=(0, np.log2(num_classes)))
    # writer.add_image(f'Entropy - {type_}', grid_image, i_iter)

def draw_in_tensorboard_trans(writer, images, images_real, i_iter, pred_main, pred_main_real, num_classes, type_):
    grid_image = make_grid(images[:3].clone().cpu().data, 3, normalize=True)
    writer.add_image(f'Image - {type_}', grid_image, i_iter)

    grid_image = make_grid(images_real[:3].clone().cpu().data, 3, normalize=True)
    writer.add_image(f'Image_real - {type_}', grid_image, i_iter)

    grid_image = make_grid(torch.from_numpy(np.array(colorize_mask(np.asarray(
        np.argmax(F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0),
                  axis=2), dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)), 3,
                           normalize=False, range=(0, 255))
    writer.add_image(f'Prediction - {type_}', grid_image, i_iter)


    grid_image = make_grid(torch.from_numpy(np.array(colorize_mask(np.asarray(
        np.argmax(F.softmax(pred_main_real).cpu().data[0].numpy().transpose(1, 2, 0),
                  axis=2), dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)), 3,
                           normalize=False, range=(0, 255))
    writer.add_image(f'Prediction_real - {type_}', grid_image, i_iter)

    # output_sm = F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0)
    # output_ent = np.sum(-np.multiply(output_sm, np.log2(output_sm)), axis=2,
    #                     keepdims=False)
    # grid_image = make_grid(torch.from_numpy(output_ent), 3, normalize=True,
    #                        range=(0, np.log2(num_classes)))
    # writer.add_image(f'Entropy - {type_}', grid_image, i_iter)


def print_losses(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter = {i_iter} {full_string}')


def log_losses_tensorboard(writer, current_losses, i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value), i_iter)


def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()


def train_domain_adaptation(model, trainloader, targetloader, cfg):
    
    if cfg.TRAIN.DA_METHOD == 'AdvEnt':
        train_advent(model, trainloader, targetloader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'AdaptSeg':
        train_adaptseg(model, trainloader, targetloader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'AdaptSeg_w_trans':
        train_adaptseg_w_trans(model, trainloader, targetloader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'self_domain_swarp':
        train_self_domain_swarp(model, trainloader, targetloader, cfg)    
    else:
        raise NotImplementedError(f"Not yet supported DA method {cfg.TRAIN.DA_METHOD}")
