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
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from advent.model.discriminator import get_fc_discriminator
from advent.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from advent.utils.func import loss_calc, bce_loss, mse_loss, boundary_loss_func, reg_loss_calc_ign
from advent.utils.loss import entropy_loss
from advent.utils.func import prob_2_entropy
from advent.utils.viz_segmask import colorize_mask
import matplotlib.pyplot as plt
from copy import deepcopy
import random


###########################################################################################
# TODO: Source Only for VGG
def train_vgg(model, trainloader, targetloader, cfg):
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

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)


    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)

    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)

    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)):

        # reset optimizers
        optimizer.zero_grad()
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)

        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators


        _, batch = trainloader_iter.__next__()
        images_source, labels, _, weather_name, _ = batch
        mid_feature_src, pred_src_main = model(images_source.cuda(device))

        ###########################
        # train on source for Seg #
        ###########################

        pred_src_main = interp(pred_src_main)
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)
        loss = cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
        loss.backward()

        optimizer.step()

        current_losses = {'loss_seg_src_main': loss_seg_src_main}

        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)
            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S')


# TODO: advent resnet
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


# TODO: advent + VGG backbone baseline
def train_advent_vgg(model, trainloader, targetloader, cfg):
    '''
        UDA training with advent
    '''
    # Create the model and start the training.
    adaptseg_on = (cfg.TRAIN.DA_METHOD == 'AdapSeg')
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
    # d_aux = get_fc_discriminator(num_classes=512, ndf=128)
    # d_aux.train()
    # d_aux.to(device)

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
    # optimizer_d_aux = optim.Adam(d_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
    #                              betas=(0.9, 0.99))

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

    criterion_loss = mse_loss if cfg.GAN == 'lsgan' else bce_loss

    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)):

        # reset optimizers
        optimizer.zero_grad()
        # optimizer_d_aux.zero_grad()
        optimizer_d_main.zero_grad()
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)
        # adjust_learning_rate_discriminator(optimizer_d_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)

        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators

        for param in d_main.parameters():
            param.requires_grad = False

        _, batch = trainloader_iter.__next__()
        images_source, labels, _, weather_name, _ = batch
        mid_feature_src, pred_src_main = model(images_source.cuda(device))

        ###########################
        # train on source for Seg #
        ###########################
        pred_src_main = interp(pred_src_main)
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)
        loss = cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
        loss.backward()

        ###################################
        # train on gan generator for Seg  #
        ###################################

        _, batch = targetloader_iter.__next__()
        images, _, _, _ = batch
        mid_feature_tgt, pred_trg_main = model(images.cuda(device))

        pred_trg_main = interp_target(pred_trg_main)
        if adaptseg_on:
            d_out_main = d_main(F.softmax(pred_trg_main))
        else:
            d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
        loss_adv_trg_main = criterion_loss(d_out_main, source_label)
        loss = cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main
        loss.backward()

        for param in d_main.parameters():
            param.requires_grad = True

        ##################################
        # train on gan discrimin for Seg #
        ##################################

        pred_src_main = pred_src_main.detach()
        if adaptseg_on:
            d_out_main = d_main(F.softmax(pred_src_main))
        else:
            d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main)))
        loss_d_main = criterion_loss(d_out_main, source_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        pred_trg_main = pred_trg_main.detach()
        if adaptseg_on:
            d_out_main = d_main(F.softmax(pred_trg_main))
        else:
            d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
        loss_d_main = criterion_loss(d_out_main, target_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        optimizer.step()
        optimizer_d_main.step()

        current_losses = {'loss_seg_src_main': loss_seg_src_main,
                          'loss_adv_trg_main': loss_adv_trg_main,
                          'loss_d_main': loss_d_main}

        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
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

###########################################################################################
# TODO: Source Only for boundary VGG
def train_Boundary_vgg(model, trainloader, targetloader, cfg):
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

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)


    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)

    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)

    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)):

        # reset optimizers
        optimizer.zero_grad()
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)

        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        _, batch = trainloader_iter.__next__()
        images_source, labels, _, weather_name, labels_things = batch
        mid_feature_src, pred_src_main, pred_src_boundary = model(images_source.cuda(device))

        ###########################
        # train on source for Seg #
        ###########################
        # Boundary Training
        pred_src_boundary = interp(pred_src_boundary)
        loss_boundary_src_main, boundary_targets = boundary_loss_func(pred_src_boundary, labels_things, cfg.TRAIN.BOUNDARY_LOSS, cfg.TRAIN.LAMBDA_DICE)

        # Segmentation Training
        pred_src_main = interp(pred_src_main)
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)

        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main +
                cfg.TRAIN.LAMBDA_BOUNDARY * loss_boundary_src_main)

        loss.backward()

        optimizer.step()

        current_losses = {'loss_seg_src_main': loss_seg_src_main}

        if cfg.TRAIN.BOUNDARY_LOSS == "BCE":
            current_losses['loss_boundary_BCE_{}'.format(cfg.TRAIN.LAMBDA_BOUNDARY)] = loss_boundary_src_main
        elif cfg.TRAIN.BOUNDARY_LOSS == "DICE":
            current_losses['loss_boundary_DICE_{}'.format(cfg.TRAIN.LAMBDA_BOUNDARY)] = loss_boundary_src_main
        elif cfg.TRAIN.BOUNDARY_LOSS == "BCE+DICE":
            current_losses['loss_boundary_BCE+DICE_{}'.format(cfg.TRAIN.LAMBDA_BOUNDARY)] = loss_boundary_src_main
        else:
            raise NotImplementedError(f"Not yet supported {cfg.TRAIN.BOUNDARY_LOSS}")

        current_losses['loss_boundary_src_main'] = cfg.TRAIN.LAMBDA_BOUNDARY * loss_boundary_src_main

        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)
            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S', pred_src_boundary, boundary_targets)

# TODO: seg-only adaptation
def train_ad_Boundary_advent_vgg(model, trainloader, targetloader, cfg):
    '''
        UDA training with advent
    '''
    # Create the model and start the training.
    adaptseg_on = (cfg.TRAIN.DA_METHOD == 'AdapSeg')
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
    # d_aux = get_fc_discriminator(num_classes=512, ndf=128)
    # d_aux.train()
    # d_aux.to(device)

    # seg maps, i.e. output, level
    d_main = get_fc_discriminator(num_classes=num_classes + 1)
    d_main.train()
    d_main.to(device)

    # boundary map
    # d_boundary = get_fc_discriminator(num_classes=1)
    # d_boundary.train()
    # d_boundary.to(device)

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # discriminators' optimizers
    # optimizer_d_aux = optim.Adam(d_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
    #                              betas=(0.9, 0.99))

    optimizer_d_main = optim.Adam(d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))

    # optimizer_d_boundary = optim.Adam(d_boundary.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
    #                               betas=(0.9, 0.99))

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

    # criterion_loss = nn.CrossEntropyLoss(weight=weight_tensor).cuda(device)


    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)):

        #### reset optimizers
        optimizer.zero_grad()
        # optimizer_d_aux.zero_grad()
        optimizer_d_main.zero_grad()
        # optimizer_d_boundary.zero_grad()

        #### adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)
        # adjust_learning_rate_discriminator(optimizer_d_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)
        # adjust_learning_rate_discriminator(optimizer_d_boundary, i_iter, cfg)

        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        for param in d_main.parameters():
            param.requires_grad = False
        # for param in d_boundary.parameters():
        #     param.requires_grad = False

        _, batch = trainloader_iter.__next__()
        images_source, labels, _, weather_name, labels_things = batch
        mid_feature_src, pred_src_main, pred_src_boundary = model(images_source.cuda(device))

        ###########################
        # train on source for Seg #
        ###########################
        # pred_b: [1, 1, 720, 1280] / labels: [1, 720, 1280]
        # Boundary Training
        pred_src_boundary = interp(pred_src_boundary)
        loss_boundary_src_main, _ = boundary_loss_func(pred_src_boundary, labels, cfg.TRAIN.BOUNDARY_LOSS, cfg.TRAIN.LAMBDA_DICE)

        # Segmentation Training
        pred_src_main = interp(pred_src_main)
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)

        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main +
                cfg.TRAIN.LAMBDA_BOUNDARY * loss_boundary_src_main)

        loss.backward()

        ###################################
        # train on gan generator for Seg  #
        ###################################

        _, batch = targetloader_iter.__next__()
        images, _, _, _ = batch
        mid_feature_tgt, pred_trg_main, pred_trg_boundary = model(images.cuda(device))

        pred_trg_main = interp_target(pred_trg_main)
        # pred_trg_fusion = torch.cat((F.softmax(pred_trg_main), pred_trg_boundary), dim=1)
        # pred_trg_boundary_expand = pred_trg_boundary.repeat(1, num_classes, 1, 1)

        if adaptseg_on:
            # d_out_main = d_main(F.softmax(pred_trg_main) * pred_trg_boundary_expand)
            d_out_main = d_main(F.softmax(pred_trg_main))
            # d_out_boundary = d_boundary(pred_trg_boundary)
        else:
            # d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)) * pred_trg_boundary_expand)
            d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
            # d_out_boundary = d_boundary(prob_2_entropy(pred_trg_boundary))

        loss_adv_trg_main = mse_loss(d_out_main, source_label)
        # loss_adv_trg_boundary = mse_loss(d_out_boundary, source_label)

        loss = cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main
        # loss = (cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main +
        #         cfg.TRAIN.LAMBDA_ADV_BOUNDARY * loss_adv_trg_boundary)

        loss.backward()

        ##################################
        # train on gan discrimin for Seg #
        ##################################

        for param in d_main.parameters():
            param.requires_grad = True
        # for param in d_boundary.parameters():
        #     param.requires_grad = True

        ##### train with source #####
        pred_src_main = pred_src_main.detach()
        # pred_src_boundary_expand = pred_src_boundary.repeat(1, num_classes, 1, 1).detach()
        # pred_src_boundary = pred_src_boundary.detach()

        if adaptseg_on:
            # d_out_main = d_main(F.softmax(pred_src_main) * pred_src_boundary_expand)
            d_out_main = d_main(F.softmax(pred_src_main))
            # d_out_boundary = d_boundary(pred_src_boundary)
        else:
            # d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main)) * pred_src_boundary_expand)
            d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main)))
            # d_out_boundary = d_boundary(prob_2_entropy(pred_src_boundary))

        # loss_d_boundary = mse_loss(d_out_boundary, source_label)
        # loss_d_boundary = loss_d_boundary / 2
        # loss_d_boundary.backward()

        loss_d_main = mse_loss(d_out_main, source_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        ##### train with target #####
        pred_trg_main = pred_trg_main.detach()
        # pred_trg_boundary_expand = pred_trg_boundary_expand.detach()
        # pred_trg_boundary = pred_trg_boundary.detach()

        if adaptseg_on:
            # d_out_main = d_main(F.softmax(pred_trg_main) * pred_trg_boundary_expand)
            d_out_main = d_main(F.softmax(pred_trg_main))
            # d_out_boundary = d_boundary(pred_trg_boundary)
        else:
            # d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)) * pred_trg_boundary_expand)
            d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
            # d_out_boundary = d_boundary(prob_2_entropy(pred_trg_boundary))

        # loss_d_boundary = mse_loss(d_out_boundary, target_label)
        # loss_d_boundary = loss_d_boundary / 2
        # loss_d_boundary.backward()

        loss_d_main = mse_loss(d_out_main, target_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        optimizer.step()
        optimizer_d_main.step()
        # optimizer_d_boundary.step()

        current_losses = {
                'loss_seg_src_main': loss_seg_src_main,
                'loss_boundary_src_main': loss_boundary_src_main,
                'loss_adv_trg_main': loss_adv_trg_main,
                # 'loss_adv_trg_boundary': loss_adv_trg_boundary,
                'loss_d_main': loss_d_main
                # 'loss_d_boundary': loss_d_boundary
        }

        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            torch.save(d_main.state_dict(), snapshot_dir / f'model_{i_iter}_D_main.pth')
            # torch.save(d_boundary.state_dict(), snapshot_dir / f'model_{i_iter}_D_boundary.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)
            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_in_tensorboard(writer, images, i_iter, pred_trg_main, num_classes, 'T', pred_trg_boundary)
                draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S', pred_src_boundary)


# TODO: b_advent + VGG backbone boundary
def train_cat_Boundary_advent_vgg(model, trainloader, targetloader, cfg):
    ''' UDA training with advent
    '''
    # Create the model and start the training.
    adaptseg_on = (cfg.TRAIN.DA_METHOD == 'AdapSeg')
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
    # d_aux = get_fc_discriminator(num_classes=512, ndf=128)
    # d_aux.train()
    # d_aux.to(device)

    # seg maps, i.e. output, level
    d_main = get_fc_discriminator(num_classes=num_classes + 1)
    d_main.train()
    d_main.to(device)

    # boundary map
    # d_boundary = get_fc_discriminator(num_classes=1)
    # d_boundary.train()
    # d_boundary.to(device)

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # discriminators' optimizers
    # optimizer_d_aux = optim.Adam(d_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
    #                              betas=(0.9, 0.99))

    optimizer_d_main = optim.Adam(d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))

    # optimizer_d_boundary = optim.Adam(d_boundary.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
    #                               betas=(0.9, 0.99))

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

    # criterion_loss = nn.CrossEntropyLoss(weight=weight_tensor).cuda(device)


    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)):

        #### reset optimizers
        optimizer.zero_grad()
        # optimizer_d_aux.zero_grad()
        optimizer_d_main.zero_grad()
        # optimizer_d_boundary.zero_grad()

        #### adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)
        # adjust_learning_rate_discriminator(optimizer_d_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)
        # adjust_learning_rate_discriminator(optimizer_d_boundary, i_iter, cfg)

        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        for param in d_main.parameters():
            param.requires_grad = False
        # for param in d_boundary.parameters():
        #     param.requires_grad = False

        _, batch = trainloader_iter.__next__()
        images_source, labels, _, weather_name, labels_things = batch
        mid_feature_src, pred_src_main, pred_src_boundary = model(images_source.cuda(device))

        ###########################
        # train on source for Seg #
        ###########################
        # pred_b: [1, 1, 720, 1280] / labels: [1, 720, 1280]
        # Boundary Training
        pred_src_boundary = interp(pred_src_boundary)
        loss_boundary_src_main, _ = boundary_loss_func(pred_src_boundary, labels, cfg.TRAIN.BOUNDARY_LOSS, cfg.TRAIN.LAMBDA_DICE)

        # Segmentation Training
        pred_src_main = interp(pred_src_main)
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)

        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main +
                cfg.TRAIN.LAMBDA_BOUNDARY * loss_boundary_src_main)

        loss.backward()

        ###################################
        # train on gan generator for Seg  #
        ###################################

        _, batch = targetloader_iter.__next__()
        images, _, _, _ = batch
        mid_feature_tgt, pred_trg_main, pred_trg_boundary = model(images.cuda(device))

        pred_trg_main = interp_target(pred_trg_main)
        pred_trg_boundary = interp_target(pred_trg_boundary)
        pred_trg_fusion = torch.cat((F.softmax(pred_trg_main), pred_trg_boundary), dim=1)
        # pred_trg_boundary_expand = pred_trg_boundary.repeat(1, num_classes, 1, 1)

        if adaptseg_on:
            # d_out_main = d_main(F.softmax(pred_trg_main) * pred_trg_boundary_expand)
            # d_out_main = d_main(F.softmax(pred_trg_main))
            # d_out_boundary = d_boundary(pred_trg_boundary)
            d_out_main = d_main(pred_trg_fusion)
        else:
            # d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)) * pred_trg_boundary_expand)
            # d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
            # d_out_boundary = d_boundary(prob_2_entropy(pred_trg_boundary))
            d_out_main = d_main(prob_2_entropy(pred_trg_fusion))

        loss_adv_trg_main = mse_loss(d_out_main, source_label)
        # loss_adv_trg_boundary = mse_loss(d_out_boundary, source_label)

        loss = cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main
        # loss = (cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main +
        #         cfg.TRAIN.LAMBDA_ADV_BOUNDARY * loss_adv_trg_boundary)

        loss.backward()

        ##################################
        # train on gan discrimin for Seg #
        ##################################

        for param in d_main.parameters():
            param.requires_grad = True
        # for param in d_boundary.parameters():
        #     param.requires_grad = True

        ##### train with source #####
        pred_src_main = pred_src_main.detach()
        # pred_src_boundary_expand = pred_src_boundary.repeat(1, num_classes, 1, 1).detach()
        pred_src_boundary = pred_src_boundary.detach()
        pred_src_fusion = torch.cat((F.softmax(pred_src_main), pred_src_boundary), dim=1)

        if adaptseg_on:
            # d_out_main = d_main(F.softmax(pred_src_main) * pred_src_boundary_expand)
            # d_out_main = d_main(F.softmax(pred_src_main))
            # d_out_boundary = d_boundary(pred_src_boundary)
            d_out_main = d_main(pred_src_fusion)
        else:
            # d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main)) * pred_src_boundary_expand)
            # d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main)))
            # d_out_boundary = d_boundary(prob_2_entropy(pred_src_boundary))
            d_out_main = d_main(prob_2_entropy(pred_src_fusion))

        # loss_d_boundary = mse_loss(d_out_boundary, source_label)
        # loss_d_boundary = loss_d_boundary / 2
        # loss_d_boundary.backward()

        loss_d_main = mse_loss(d_out_main, source_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        ##### train with target #####
        pred_trg_main = pred_trg_main.detach()
        # pred_trg_boundary_expand = pred_trg_boundary_expand.detach()
        pred_trg_boundary = pred_trg_boundary.detach()
        pred_trg_fusion = torch.cat((F.softmax(pred_trg_main), pred_trg_boundary), dim=1)

        if adaptseg_on:
            # d_out_main = d_main(F.softmax(pred_trg_main) * pred_trg_boundary_expand)
            # d_out_main = d_main(F.softmax(pred_trg_main))
            # d_out_boundary = d_boundary(pred_trg_boundary)
            d_out_main = d_main(pred_trg_fusion)
        else:
            # d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)) * pred_trg_boundary_expand)
            # d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
            # d_out_boundary = d_boundary(prob_2_entropy(pred_trg_boundary))
            d_out_main = d_main(prob_2_entropy(pred_trg_fusion))

        # loss_d_boundary = mse_loss(d_out_boundary, target_label)
        # loss_d_boundary = loss_d_boundary / 2
        # loss_d_boundary.backward()

        loss_d_main = mse_loss(d_out_main, target_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        optimizer.step()
        optimizer_d_main.step()
        # optimizer_d_boundary.step()

        current_losses = {
                'loss_seg_src_main': loss_seg_src_main,
                'loss_boundary_src_main': loss_boundary_src_main,
                'loss_adv_trg_main': loss_adv_trg_main,
                # 'loss_adv_trg_boundary': loss_adv_trg_boundary,
                'loss_d_main': loss_d_main
                # 'loss_d_boundary': loss_d_boundary
        }

        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            torch.save(d_main.state_dict(), snapshot_dir / f'model_{i_iter}_D_main.pth')
            # torch.save(d_boundary.state_dict(), snapshot_dir / f'model_{i_iter}_D_boundary.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)
            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_in_tensorboard(writer, images, i_iter, pred_trg_main, num_classes, 'T', pred_trg_boundary)
                draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S', pred_src_boundary)


# TODO: boundary and seg multi-adaptations
def train_Boundary_advent_vgg(model, trainloader, targetloader, cfg):
    ''' UDA training with advent
    '''
    # Create the model and start the training.
    adaptseg_on = (cfg.TRAIN.DA_METHOD == 'AdapSeg')
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
    # d_aux = get_fc_discriminator(num_classes=512, ndf=128)
    # d_aux.train()
    # d_aux.to(device)

    # seg maps, i.e. output, level
    d_main = get_fc_discriminator(num_classes=num_classes)
    d_main.train()
    d_main.to(device)

    # boundary map
    d_boundary = get_fc_discriminator(num_classes=1)
    d_boundary.train()
    d_boundary.to(device)

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # discriminators' optimizers
    # optimizer_d_aux = optim.Adam(d_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
    #                              betas=(0.9, 0.99))

    optimizer_d_main = optim.Adam(d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))

    optimizer_d_boundary = optim.Adam(d_boundary.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
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

    # criterion_loss = nn.CrossEntropyLoss(weight=weight_tensor).cuda(device)


    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)):

        #### reset optimizers
        optimizer.zero_grad()
        # optimizer_d_aux.zero_grad()
        optimizer_d_main.zero_grad()
        optimizer_d_boundary.zero_grad()

        #### adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)
        # adjust_learning_rate_discriminator(optimizer_d_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_boundary, i_iter, cfg)

        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        for param in d_main.parameters():
            param.requires_grad = False
        for param in d_boundary.parameters():
            param.requires_grad = False

        _, batch = trainloader_iter.__next__()
        images_source, labels, _, weather_name, labels_things = batch
        mid_feature_src, pred_src_main, pred_src_boundary = model(images_source.cuda(device))

        ###########################
        # train on source for Seg #
        ###########################
        # pred_b: [1, 1, 720, 1280] / labels: [1, 720, 1280]
        # Boundary Training
        pred_src_boundary = interp(pred_src_boundary)
        loss_boundary_src_main, _ = boundary_loss_func(pred_src_boundary, labels, cfg.TRAIN.BOUNDARY_LOSS, cfg.TRAIN.LAMBDA_DICE)

        # Segmentation Training
        pred_src_main = interp(pred_src_main)
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)

        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main +
                cfg.TRAIN.LAMBDA_BOUNDARY * loss_boundary_src_main)

        loss.backward()

        ###################################
        # train on gan generator for Seg  #
        ###################################

        _, batch = targetloader_iter.__next__()
        images, _, _, _ = batch
        mid_feature_tgt, pred_trg_main, pred_trg_boundary = model(images.cuda(device))

        pred_trg_main = interp_target(pred_trg_main)
        pred_trg_boundary = interp_target(pred_trg_boundary)
        # pred_trg_boundary_expand = pred_trg_boundary.repeat(1, num_classes, 1, 1)

        if adaptseg_on:
            # d_out_main = d_main(F.softmax(pred_trg_main) * pred_trg_boundary_expand)
            d_out_main = d_main(F.softmax(pred_trg_main))
            d_out_boundary = d_boundary(pred_trg_boundary)
        else:
            # d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)) * pred_trg_boundary_expand)
            d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
            d_out_boundary = d_boundary(prob_2_entropy(pred_trg_boundary))

        loss_adv_trg_main = mse_loss(d_out_main, source_label)
        loss_adv_trg_boundary = mse_loss(d_out_boundary, source_label)

        # loss = cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main
        loss = (cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main +
                cfg.TRAIN.LAMBDA_ADV_BOUNDARY * loss_adv_trg_boundary)

        loss.backward()

        ##################################
        # train on gan discrimin for Seg #
        ##################################

        for param in d_main.parameters():
            param.requires_grad = True
        for param in d_boundary.parameters():
            param.requires_grad = True

        ##### train with source #####
        pred_src_main = pred_src_main.detach()
        # pred_src_boundary_expand = pred_src_boundary.repeat(1, num_classes, 1, 1).detach()
        pred_src_boundary = pred_src_boundary.detach()

        if adaptseg_on:
            # d_out_main = d_main(F.softmax(pred_src_main) * pred_src_boundary_expand)
            d_out_main = d_main(F.softmax(pred_src_main))
            d_out_boundary = d_boundary(pred_src_boundary)
        else:
            # d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main)) * pred_src_boundary_expand)
            d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main)))
            d_out_boundary = d_boundary(prob_2_entropy(pred_src_boundary))

        loss_d_boundary = mse_loss(d_out_boundary, source_label)
        loss_d_boundary = loss_d_boundary / 2
        loss_d_boundary.backward()

        loss_d_main = mse_loss(d_out_main, source_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        ##### train with target #####
        pred_trg_main = pred_trg_main.detach()
        # pred_trg_boundary_expand = pred_trg_boundary_expand.detach()
        pred_trg_boundary = pred_trg_boundary.detach()

        if adaptseg_on:
            # d_out_main = d_main(F.softmax(pred_trg_main) * pred_trg_boundary_expand)
            d_out_main = d_main(F.softmax(pred_trg_main))
            d_out_boundary = d_boundary(pred_trg_boundary)
        else:
            # d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)) * pred_trg_boundary_expand)
            d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
            d_out_boundary = d_boundary(prob_2_entropy(pred_trg_boundary))

        loss_d_boundary = mse_loss(d_out_boundary, target_label)
        loss_d_boundary = loss_d_boundary / 2
        loss_d_boundary.backward()

        loss_d_main = mse_loss(d_out_main, target_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        optimizer.step()
        optimizer_d_main.step()
        optimizer_d_boundary.step()

        current_losses = {
                'loss_seg_src_main': loss_seg_src_main,
                'loss_boundary_src_main': loss_boundary_src_main,
                'loss_adv_trg_main': loss_adv_trg_main,
                'loss_adv_trg_boundary': loss_adv_trg_boundary,
                'loss_d_main': loss_d_main,
                'loss_d_boundary': loss_d_boundary
        }

        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            torch.save(d_main.state_dict(), snapshot_dir / f'model_{i_iter}_D_main.pth')
            torch.save(d_boundary.state_dict(), snapshot_dir / f'model_{i_iter}_D_boundary.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)
            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_in_tensorboard(writer, images, i_iter, pred_trg_main, num_classes, 'T', pred_trg_boundary)
                draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S', pred_src_boundary)


###########################################################################################
# TODO: Instance-wise Self-Training based on worm-up adversariial trained VGG
def train_IST_vgg(model, trainloader, targetloader, cfg):
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

    # pre-trained model for Self-Training
    model_st = deepcopy(model)
    model_st.eval()
    model_st.to(device)

    # target class-wise cutting threshold dictionary
    target_cut_thresh = {}
    cls_thresh = torch.ones(num_classes).type(torch.float32)

    cudnn.benchmark = True
    cudnn.enabled = True

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)


    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                         align_corners=True)

    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)

    # Iteration follow the targetloader size
    total_iter = len(targetloader)

    for i_iter in tqdm(range(total_iter)):

        # reset optimizers
        optimizer.zero_grad()
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)

        # UDA Training
        _, batch = trainloader_iter.__next__()
        images_source, labels, _, weather_name = batch
        _, pred_src_main = model(images_source.cuda(device))

        ###########################
        # train on source for Seg #
        ###########################
        # Segmentation Training
        pred_src_main = interp(pred_src_main)
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)

        loss = cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
        loss.backward()

        ###################################
        #    Self-training generator      #
        ###################################

        ####### Target inference  #######
        _, batch = targetloader_iter.__next__()
        # images, images_rev, _, _, name, name_next = batch  # contrastive learning
        images, _, _, _ = batch

        feat_trg_main, pred_trg_main = model(images.cuda(device))

        pred_trg_main = interp_target(pred_trg_main)

        ####### Fixed model target inference #######
        with torch.no_grad():
            feat_trg_main_st, pred_trg_main_st = model_st(images.cuda(device))
            pred_trg_main_st = interp_target(pred_trg_main_st)

        ####### pseudo label generator for target #######
        label_trg, cls_thresh = label_generator(pred_trg_main_st, cls_thresh, cfg, i_iter, total_iter)

        ##### CE loss for trg : Confidence Regularized Self-Training #######
        ####### MRKLD + Ign Region for target segmentation  TODO: understanding this parts!!
        loss_seg_trg_main = reg_loss_calc_ign(pred_trg_main, label_trg, device)
        loss_tgt_seg = cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_trg_main


        ####### Domain Swarping #######
        feat_trg_swarped, target_cut_thresh, target_label = DomainSwarping(feat_trg_main, label_trg, target_cut_thresh, device)
        ignore_mask = (target_label == 255)

        feat_trg_swarped = (~ignore_mask * feat_trg_swarped) + (ignore_mask * feat_trg_main)
        pred_trg_swarped = model.classifier_(feat_trg_swarped)
        pred_trg_swarped = interp_target(pred_trg_swarped)

        ###### MRKLD + ign loss for swarped target segmentation #######
        loss_seg_trg_swarped = reg_loss_calc_ign(pred_trg_swarped, label_trg, device)
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
            torch.save(model_st.state_dict(), snapshot_dir / f'model_{i_iter}_st.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)
            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                # draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S')
                st_draw_in_tensorboard_trans(writer, images, label_trg, i_iter, pred_trg_main, pred_trg_swarped, num_classes, 'T')


# TODO: Instance-wise Self-Training for boundary VGG
def train_Boundary_IST_vgg(model, trainloader, targetloader, cfg):
    '''
        UDA training with advent
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

    # pre-trained model for Self-Training
    model_st = deepcopy(model)
    model_st.eval()
    model_st.to(device)

    # target class-wise cutting threshold dictionary
    target_cut_thresh = {}
    cls_thresh = torch.ones(num_classes).type(torch.float32)

    cudnn.benchmark = True
    cudnn.enabled = True

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)


    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                         align_corners=True)

    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)

    # Iteration follow the targetloader size
    total_iter = len(targetloader)

    for i_iter in tqdm(range(total_iter)):

        # reset optimizers
        optimizer.zero_grad()
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)

        # UDA Training
        _, batch = trainloader_iter.__next__()
        images_source, labels, _, weather_name = batch
        _, pred_src_main, pred_src_boundary = model(images_source.cuda(device))

        ###########################
        # train on source for Seg #
        ###########################
        # Boundary Training
        pred_src_boundary = interp(pred_src_boundary)
        loss_boundary_src_main, boundary_targets = boundary_loss_func(pred_src_boundary, labels, cfg.TRAIN.BOUNDARY_LOSS, cfg.TRAIN.LAMBDA_DICE)

        # Segmentation Training
        pred_src_main = interp(pred_src_main)
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)

        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main +
                cfg.TRAIN.LAMBDA_BOUNDARY * loss_boundary_src_main)
        loss.backward()

        current_losses = {'loss_seg_src_main': loss_seg_src_main}

        if cfg.TRAIN.BOUNDARY_LOSS == "BCE":
            current_losses['loss_boundary_BCE_{}'.format(cfg.TRAIN.LAMBDA_BOUNDARY)] = loss_boundary_src_main
        elif cfg.TRAIN.BOUNDARY_LOSS == "DICE":
            current_losses['loss_boundary_DICE_{}'.format(cfg.TRAIN.LAMBDA_BOUNDARY)] = loss_boundary_src_main
        elif cfg.TRAIN.BOUNDARY_LOSS == "BCE+DICE":
            current_losses['loss_boundary_BCE+DICE_{}'.format(cfg.TRAIN.LAMBDA_BOUNDARY)] = loss_boundary_src_main
        else:
            raise NotImplementedError(f"Not yet supported {cfg.TRAIN.BOUNDARY_LOSS}")
        current_losses['loss_boundary_src_main'] = cfg.TRAIN.LAMBDA_BOUNDARY * loss_boundary_src_main

        ###################################
        #    Self-training generator      #
        ###################################

        ####### Target inference  #######
        _, batch = targetloader_iter.__next__()
        # images, images_rev, _, _, name, name_next = batch # contrastive learning
        images, _, _, _ = batch
        _, pred_trg_main, feat_trg_main, pred_trg_boundary = model(images.cuda(device))

        pred_trg_main = interp_target(pred_trg_main)
        pred_trg_boundary = interp_target(pred_trg_boundary)

        ####### Fixed model target inference #######
        with torch.no_grad():
            _, pred_trg_main_st, feat_trg_main_st, pred_trg_boundary = model_st(images.cuda(device))
            pred_trg_main_st = interp_target(pred_trg_main_st)

        ####### pseudo label generator for target #######
        label_trg, cls_thresh = label_generator(pred_trg_main_st, cls_thresh, cfg, i_iter, total_iter)


        ##### CE loss for target segmentation #######
        ####### MRKLD + Ign Region for target segmentation  TODO: understanding this parts!!
        loss_seg_trg_main = reg_loss_calc_ign(pred_trg_main, label_trg, device)
        loss_tgt_seg = cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_trg_main


        ####### Domain Swarping #######
        feat_trg_swarped, target_cut_thresh, target_label = DomainSwarping(feat_trg_main, label_trg, target_cut_thresh, device)

        ignore_mask = (target_label == 255)

        feat_trg_swarped = (~ignore_mask * feat_trg_swarped) + (ignore_mask * feat_trg_main)
        pred_trg_swarped = model.classifier_(feat_trg_swarped)
        pred_trg_swarped = interp_target(pred_trg_swarped)

        ###### MRKLD + ign loss for swarped target segmentation #######
        loss_seg_trg_swarped = reg_loss_calc_ign(pred_trg_swarped, label_trg, device)
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
            torch.save(model_st.state_dict(), snapshot_dir / f'model_{i_iter}_st.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)
            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S', pred_src_boundary, boundary_targets)
                # draw_in_tensorboard(writer, images, label_trg, i_iter, pred_trg_main, pred_tgt_swarped, num_classes, 'T')


###########################################################################################
def label_generator(pred_trg, cls_thresh_tot, cfg, i_iter, tot_iter):
    import math
    device = cfg.GPU_ID
    ###   ###
    output_main = F.softmax(pred_trg, dim=1)
    amax_output = torch.argmax(output_main, dim=1).type(torch.uint8)
    pred_label_trainIDs = amax_output.clone()
    pred_label = amax_output.clone()
    conf, _ = torch.max(output_main, dim=1)

    conf_dict = {k: [] for k in range(cfg.NUM_CLASSES)}
    pred_cls_num = torch.zeros(cfg.NUM_CLASSES)
    for idx_cls in range(cfg.NUM_CLASSES):
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
            iter_ratio = 1.0 - float(i_iter / (tot_iter + 1))
            coeff = 0.2 * (iter_ratio ** 0.5)
            cls_sel_size[idx_cls] = int(math.floor(len_cls * coeff))
            len_cls_thresh = int(cls_sel_size[idx_cls])
            if len_cls_thresh != 0:
                cls_thresh[idx_cls] = conf_dict[idx_cls][len_cls_thresh - 1]
            conf_dict[idx_cls] = None

    cls_thresh_tot_ = torch.where(cls_thresh_tot == 1.0, cls_thresh, 0.9 * cls_thresh_tot + 0.1 * cls_thresh)
    cls_thresh_mask = (cls_thresh == 1.0) * (cls_thresh_tot != 1.0)
    cls_thresh_tot = torch.where(cls_thresh_mask == 1.0, cls_thresh_tot, cls_thresh_tot_)

    weighted_prob = output_main / cls_thresh_tot.to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    weighted_pred_trainIDs = torch.argmax(weighted_prob, dim=1).type(torch.uint8)
    weighted_conf, _ = torch.max(weighted_prob, dim=1)

    weighted_pred_trainIDs[weighted_conf < 1] = 255

    return weighted_pred_trainIDs, cls_thresh_tot


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
            new_masked_tgt_init += cls_mask * new_masked_tgt    # (1, 1024, 67, 120) = (1, 1, 67, 120) * (1, 1024, 1, 1)

            tgt_dict_tot[label_ele] = alpha * tgt_dict_tot[label_ele] + (1-alpha) * tgt_dict[label_ele]

            tgt_dict_tot[label_ele] = tgt_dict_tot[label_ele].detach()

    return new_masked_tgt_init, tgt_dict_tot, tgt_label


def draw_in_tensorboard(writer, images, i_iter, pred_main, num_classes, type_, pred_src_boundary=None, boundary_targets=None):
    grid_image = make_grid(images[:3].clone().cpu().data, 3, normalize=True)
    writer.add_image(f'Image - {type_}', grid_image, i_iter)

    grid_image = make_grid(torch.from_numpy(np.array(colorize_mask(np.asarray(
        np.argmax(F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0),
                  axis=2), dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)), 3,
                           normalize=False, range=(0, 255))
    writer.add_image(f'Prediction - {type_}', grid_image, i_iter)

    output_sm = F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0)
    output_ent = np.sum(-np.multiply(output_sm, np.log2(output_sm)), axis=2,
                        keepdims=False)
    grid_image = make_grid(torch.from_numpy(output_ent), 3, normalize=True,
                           range=(0, np.log2(num_classes)))
    writer.add_image(f'Entropy - {type_}', grid_image, i_iter)

    if pred_src_boundary is not None:
        # pred_src_boundary[pred_src_boundary >= 0.5] = 1
        # pred_src_boundary[pred_src_boundary < 0.5] = 0
        grid_image = make_grid(torch.from_numpy(pred_src_boundary.detach().cpu().numpy().squeeze(0).squeeze(0)), 3, normalize=True)
        writer.add_image(f'Boundary - {type_}', grid_image, i_iter)

    if boundary_targets is not None:
        grid_image = make_grid(torch.from_numpy(boundary_targets.detach().cpu().numpy().squeeze(0).squeeze(0)), 3,
                               normalize=True)
        writer.add_image(f'BoundaryGT - {type_}', grid_image, i_iter)


def st_draw_in_tensorboard_trans(writer, images, label_trg, i_iter, pred_main, pred_main_swarp, num_classes, type_):
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

def train_minent(model, trainloader, targetloader, cfg):
    ''' UDA training with minEnt
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

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP)):

        # reset optimizers
        optimizer.zero_grad()

        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)

        # UDA Training
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

        # adversarial training with minent
        _, batch = targetloader_iter.__next__()
        images, _, _, _ = batch
        pred_trg_aux, pred_trg_main = model(images.cuda(device))
        pred_trg_aux = interp_target(pred_trg_aux)
        pred_trg_main = interp_target(pred_trg_main)
        pred_prob_trg_aux = F.softmax(pred_trg_aux)
        pred_prob_trg_main = F.softmax(pred_trg_main)

        loss_target_entp_aux = entropy_loss(pred_prob_trg_aux)
        loss_target_entp_main = entropy_loss(pred_prob_trg_main)
        loss = (cfg.TRAIN.LAMBDA_ENT_AUX * loss_target_entp_aux
                + cfg.TRAIN.LAMBDA_ENT_MAIN * loss_target_entp_main)
        loss.backward()
        optimizer.step()

        current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
                          'loss_seg_src_main': loss_seg_src_main,
                          'loss_ent_aux': loss_target_entp_aux,
                          'loss_ent_main': loss_target_entp_main}

        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(),
                       osp.join(cfg.TRAIN.SNAPSHOT_DIR, f'model_{i_iter}.pth'))
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_in_tensorboard(writer, images, i_iter, pred_trg_main, num_classes, 'T')
                draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S')


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
    if cfg.TRAIN.MODEL == 'DeepLabv2':
        if cfg.TRAIN.DA_METHOD == 'MinEnt':
            train_minent(model, trainloader, targetloader, cfg)
        elif cfg.TRAIN.DA_METHOD == 'AdvEnt':
            train_advent(model, trainloader, targetloader, cfg)
    elif cfg.TRAIN.MODEL == 'DeepLabv2_VGG':
        if cfg.TRAIN.DA_METHOD == 'source_only':
            if cfg.TRAIN.OCDA_METHOD == 'baseline':
                train_vgg(model, trainloader, targetloader, cfg)
            elif cfg.TRAIN.OCDA_METHOD == 'boundary' or cfg.TRAIN.OCDA_METHOD == 'ad_boundary' or cfg.TRAIN.OCDA_METHOD == 'attn_boundary':
                train_Boundary_vgg(model, trainloader, targetloader, cfg)
            elif cfg.TRAIN.OCDA_METHOD == 'selfTrain' or cfg.TRAIN.OCDA_METHOD == 'selfTrain_boundary':
                train_IST_vgg(model, trainloader, targetloader, cfg)

        elif cfg.TRAIN.DA_METHOD == 'AdapSeg' or cfg.TRAIN.DA_METHOD == 'AdvEnt':
            if cfg.TRAIN.OCDA_METHOD == 'baseline':
                train_advent_vgg(model, trainloader, targetloader, cfg)
            elif cfg.TRAIN.OCDA_METHOD == 'boundary' or cfg.TRAIN.OCDA_METHOD == 'ad_boundary' or cfg.TRAIN.OCDA_METHOD == 'attn_boundary':
                if cfg.TRAIN.OPTION == 'twinD':
                    train_Boundary_advent_vgg(model, trainloader, targetloader, cfg)
                elif cfg.TRAIN.OPTION == 'segOnlyD':
                    train_ad_Boundary_advent_vgg(model, trainloader, targetloader, cfg)
                elif cfg.TRAIN.OPTION == 'catOutD':
                    train_cat_Boundary_advent_vgg(model, trainloader, targetloader, cfg)

            elif cfg.TRAIN.OCDA_METHOD == 'selfTrain':
                train_IST_vgg(model, trainloader, targetloader, cfg)
            else:
                raise NotImplementedError(
                    f"Not yet supported !OCDA! method {cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}_{cfg.TRAIN.OCDA_METHOD}")

    else:
        raise NotImplementedError(f"Not yet supported !DA! method {cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}_{cfg.TRAIN.OCDA_METHOD}")
