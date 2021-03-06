# --------------------------------------------------------
# Domain adpatation evaluation
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------

import os.path as osp
import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from advent.utils.func import per_class_iu, fast_hist
from advent.utils.serialization import pickle_dump, pickle_load

from torchvision.utils import make_grid, save_image
from torch.nn import functional as F
from advent.utils.viz_segmask import colorize_mask
import os
import matplotlib.pyplot as plt
import cv2

result_root = '../../experiments/snapshots/'

def evaluate_domain_adaptation( models, test_loader, cfg,
                                fixed_test_size=True,
                                verbose=True):
    device = cfg.GPU_ID
    interp = None
    if fixed_test_size:
        interp = nn.Upsample(size=(cfg.TEST.OUTPUT_SIZE_TARGET[1], cfg.TEST.OUTPUT_SIZE_TARGET[0]), mode='bilinear', align_corners=True)
    # eval
    if cfg.TEST.MODE == 'single':
        eval_single(cfg, models,
                    device, test_loader, interp, fixed_test_size,
                    verbose)
    elif cfg.TEST.MODE == 'best':
        eval_best(cfg, models,
                  device, test_loader, interp, fixed_test_size,
                  verbose)
    else:
        raise NotImplementedError(f"Not yet supported test mode {cfg.TEST.MODE}")


def eval_single(cfg, models,
                device, test_loader, interp,
                fixed_test_size, verbose):
    assert len(cfg.TEST.RESTORE_FROM) == len(models), 'Number of models are not matched'

    folder_path = cfg.TEST.RESTORE_FROM[0].split('/')[-2]
    folder_path = osp.join(result_root, folder_path, "eval_image")
    if not osp.exists(folder_path):
        os.makedirs(folder_path)

    for checkpoint, model in zip(cfg.TEST.RESTORE_FROM, models):
        load_checkpoint_for_evaluation(model, checkpoint, device)
    # eval
    hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
    for index, batch in tqdm(enumerate(test_loader)):
        image, label, _, name = batch
        if not fixed_test_size:
            interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
        with torch.no_grad():
            output = None
            for model, model_weight in zip(models, cfg.TEST.MODEL_WEIGHT):
                _, pred_main, pred_boundary = model(image.cuda(device))
                output_ = interp(pred_main).cpu().data[0].numpy()
                output_pred = interp(pred_main)
                if output is None:
                    output = model_weight * output_
                else:
                    output += model_weight * output_

                domain = name[0].split('/')[-2]
                save_path = folder_path + "/" + domain + "_" + name[0].split('/')[-1].split('.')[0]

                # segmentation prediction save
                save_image(torch.from_numpy(np.array(colorize_mask(
                    np.asarray(np.argmax(F.softmax(output_pred).cpu().data[0].numpy().transpose(1, 2, 0), axis=2),
                               dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)),
                           save_path+"_3_seg.png", 3, normalize=False, range=(0, 255))

                # boundary prediction save
                pred_boundary = F.interpolate(pred_boundary, label.shape[1:], mode='bilinear')
                save_image(pred_boundary.clone(), save_path+"_1_boundary.png", normalize=True)

                # red boundary visualize
                vis_red_boundary(save_path, pred_boundary.clone())

                # binary boundary prediction save
                save_image_binary(save_path, pred_boundary.clone(), threshold=0.5)

                # color label save
                save_image(torch.from_numpy(np.array(
                    colorize_mask(np.asarray(label.squeeze(0).numpy(),
                                             dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)),
                           save_path+"_2_label.png", 3, normalize=False)


            assert output is not None, 'Output is None'
            output = output.transpose(1, 2, 0)
            output = np.argmax(output, axis=2)
        label = label.numpy()[0]
        hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
    inters_over_union_classes = per_class_iu(hist)
    print(f'mIoU = \t{round(np.nanmean(inters_over_union_classes) * 100, 2)}')
    if verbose:
        display_stats(cfg, test_loader.dataset.class_names, inters_over_union_classes)


def eval_best(cfg, models,
              device, test_loader, interp,
              fixed_test_size, verbose):
    assert len(models) == 1, 'Not yet supported multi models in this mode'
    assert osp.exists(cfg.TEST.SNAPSHOT_DIR[0]), 'SNAPSHOT_DIR is not found'
    start_iter = cfg.TEST.SNAPSHOT_STEP
    step = cfg.TEST.SNAPSHOT_STEP
    max_iter = cfg.TEST.SNAPSHOT_MAXITER
    cache_path = osp.join(cfg.TEST.SNAPSHOT_DIR[0], 'all_res.pkl')
    if osp.exists(cache_path):
        all_res = pickle_load(cache_path)
    else:
        all_res = {}
    cur_best_miou = -1
    cur_best_model = ''
    for i_iter in range(start_iter, max_iter + 1, step):
        restore_from = osp.join(cfg.TEST.SNAPSHOT_DIR[0], f'model_{i_iter}.pth')
        if not osp.exists(restore_from):
            # continue
            if cfg.TEST.WAIT_MODEL:
                print('Waiting for model..!')
                while not osp.exists(restore_from):
                    time.sleep(5)
        print("Evaluating model", restore_from)
        if i_iter not in all_res.keys():
            load_checkpoint_for_evaluation(models[0], restore_from, device)
            # eval
            hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
            # for index, batch in enumerate(test_loader):
            #     image, _, _, name = batch
            test_iter = iter(test_loader)
            for index in tqdm(range(len(test_loader))):
                #TODO: need to check      image, _, label, _, name, _ = next(test_iter)
                image, label, _, name = next(test_iter)
                if not fixed_test_size:
                    interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
                with torch.no_grad():
                    pred_main = models[0](image.cuda(device))[1]
                    output = interp(pred_main).cpu().data[0].numpy()
                    output = output.transpose(1, 2, 0)
                    output = np.argmax(output, axis=2)
                label = label.numpy()[0]
                hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
                if verbose and index > 0 and index % 100 == 0:
                    print('{:d} / {:d}: {:0.2f}'.format(
                        index, len(test_loader), 100 * np.nanmean(per_class_iu(hist))))
            inters_over_union_classes = per_class_iu(hist)
            all_res[i_iter] = inters_over_union_classes
            pickle_dump(all_res, cache_path)
        else:
            inters_over_union_classes = all_res[i_iter]
        computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
        if cur_best_miou < computed_miou:
            cur_best_miou = computed_miou
            cur_best_model = restore_from
        print('\tCurrent mIoU:', computed_miou)
        print('\tCurrent best model:', cur_best_model)
        print('\tCurrent best mIoU:', cur_best_miou)
        if verbose:
            display_stats(cfg, test_loader.dataset.class_names, inters_over_union_classes)


def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint)
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda(device)


def display_stats(cfg, name_classes, inters_over_union_classes):
    for ind_class in range(cfg.NUM_CLASSES):
        print(name_classes[ind_class]
              + '\t' + str(round(inters_over_union_classes[ind_class] * 100, 2)))

def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min + 1e-5)
    return img

def save_image_binary(path, input, threshold = 0.5):
    norm_img = norm_ip(input, input.min(), input.max())
    norm_img[norm_img >= threshold] = 1
    norm_img[norm_img < threshold] = 0

    save_image(norm_img, path+"_4_boundary_binary.png")

def vis_red_boundary(path, input):

    """
    path: image path
    input: tensor to visualize
        - shape (1, H, W)
    """
    colors = [2, 200, 255]
    img = input.squeeze()
    img = (img / 8).tanh().cpu().numpy()

    img = (img + 1) / 2
    img = img / img.max()
    img_viz = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    img_viz[:, :, 0] = colors[0]
    img_viz[:, :, 1] = colors[1]
    img_viz[:, :, 2] = np.uint8(img * 255)
    img_viz = cv2.cvtColor(img_viz, cv2.COLOR_HSV2RGB)

    plt.imsave(path+"_0_vis_boundary.png", img_viz)