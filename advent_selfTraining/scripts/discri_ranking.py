# --------------------------------------------------------
# AdvEnt training
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------
import argparse
import os
import os.path as osp
import pprint
import random
import warnings
from tqdm import tqdm
import numpy as np
import yaml
import torch
from torch.utils import data
import sys
import torch.nn as nn
sys.path.append("/media/user/a9755522-b17e-4bde-96f6-088bbbc3a1401/OCDA/ADVENT")
from advent.model.deeplabv2 import get_deeplab_v2
from advent.model.deeplabv2_vgg import get_deeplab_v2_vgg
from advent.dataset.gta5 import GTA5DataSet
from advent.dataset.cityscapes import CityscapesDataSet
from advent.dataset.bdd import BDDataSet
from advent.domain_adaptation.config import cfg, cfg_from_file
from advent.domain_adaptation.train_UDA import train_domain_adaptation
from advent.model.discriminator import get_fc_discriminator
import torch.nn.functional as F
from advent.utils.func import prob_2_entropy
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from advent.utils.viz_segmask import colorize_mask
from advent.utils.canny import CannyFilter
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for domain adaptation (DA) training")
    parser.add_argument('--cfg', type=str, default=None,
                        help='optional config file', )
    parser.add_argument("--random-train", action="store_true",
                        help="not fixing random seed.")

    parser.add_argument("--exp-suffix", type=str, default=None,
                        help="optional experiment suffix")
    return parser.parse_args()


def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint)

    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda(device)


def f2(x):
    return x[1]

def ranking_target_w_discrim(model_seg, target_loader, output_path, cfg):

    train_lst_ordered = osp.join(output_path,'train_ent_full.txt')

    writer = SummaryWriter(log_dir=cfg.TEST.SNAPSHOT_DIR)

    num_classes = cfg.NUM_CLASSES
    device = cfg.GPU_ID    
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    targetloader_iter = enumerate(target_loader)

    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    dis_dictionary = {}
    l1loss = torch.nn.L1Loss()
    with torch.no_grad():
        for i in tqdm(range(len(target_loader))):
        # for i in tqdm(range(30)):
            _, batch = targetloader_iter.__next__()
            images, image_aug, _, _, name, _ = batch
            pred_trg_main, _ = model_seg(images.cuda(device))

            # edge = CannyFilter(images)


            # pred_trg_main_aug, _ = model_seg(image_aug.cuda(device))

            pred_trg_main = interp_target(pred_trg_main)
            # pred_trg_main_aug = F.softmax(interp_target(pred_trg_main_aug),dim=1)

            # pred_trg_max, _ = torch.max(pred_trg_main,dim=1)
            # pred_trg_main_aug_ = pred_trg_main * pred_trg_main_aug
            # diff = l1loss(pred_trg_main, pred_trg_main_aug_)
            # diff = torch.mean(diff)

            # pred_trg_entropy = model_dis(F.softmax(pred_trg_main))
            pred_trg_entropy = prob_2_entropy(F.softmax(pred_trg_main))
            # pred_trg_entropy = abs(float(pred_trg_entropy.mean()) - 0.5)
            # import pdb
            # pdb.set_trace()

            pred_trg_entropy = float(pred_trg_entropy.mean())

            dis_dictionary[name[0]] = pred_trg_entropy

            # if i % 2 == 0:
            #     draw_in_tensorboard(writer, images, image_aug, i, pred_trg_main, pred_trg_main_aug_, 'S')
            # import pdb
            # pdb.set_trace()
        dis_dictionary = sorted(dis_dictionary.items(),key=f2)
        with open(train_lst_ordered, 'w') as f:
            for i in range(len(dis_dictionary)):
                f.write("%s\t%s\n" % (dis_dictionary[i][0], dis_dictionary[i][1]))



def draw_in_tensorboard(writer, images, images_aug, i_iter, pred_main, pred_main_aug, type_):
    grid_image = make_grid(images[:3].clone().cpu().data, 3, normalize=True)
    writer.add_image(f'Image - {type_}', grid_image, i_iter)

    grid_image = make_grid(images_aug[:3].clone().cpu().data, 3, normalize=True)
    writer.add_image(f'Image_aug - {type_}', grid_image, i_iter)

    grid_image = make_grid(torch.from_numpy(np.array(colorize_mask(np.asarray(
        np.argmax(F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0),
                  axis=2), dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)), 3,
                           normalize=False, range=(0, 255))
    writer.add_image(f'Prediction - {type_}', grid_image, i_iter)

    grid_image = make_grid(torch.from_numpy(np.array(colorize_mask(np.asarray(
        np.argmax(F.softmax(pred_main_aug).cpu().data[0].numpy().transpose(1, 2, 0),
                  axis=2), dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)), 3,
                           normalize=False, range=(0, 255))
    writer.add_image(f'Prediction_aug - {type_}', grid_image, i_iter)

def main():
    # LOAD ARGS
    args = get_arguments()
    print('Called with args:')
    print(args)

    assert args.cfg is not None, 'Missing cfg file'
    cfg_from_file(args.cfg)
    # auto-generate exp name if not specified
    if cfg.EXP_NAME == '':
        cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}'

    if args.exp_suffix:
        cfg.EXP_NAME += f'_{args.exp_suffix}'
    # auto-generate snapshot path if not specified
    # if cfg.TEST.SNAPSHOT_DIR == '':
    cfg.TEST.SNAPSHOT_DIR = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME)
    os.makedirs(cfg.TEST.SNAPSHOT_DIR, exist_ok=True)

    num_classes = cfg.NUM_CLASSES
    device = cfg.GPU_ID

    output_path = osp.join(cfg.TEST.SNAPSHOT_DIR, 'compound_order')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print('Using config:')
    pprint.pprint(cfg)

    # INIT
    _init_fn = None
    if not args.random_train:
        torch.manual_seed(cfg.TRAIN.RANDOM_SEED)
        torch.cuda.manual_seed(cfg.TRAIN.RANDOM_SEED)
        np.random.seed(cfg.TRAIN.RANDOM_SEED)
        random.seed(cfg.TRAIN.RANDOM_SEED)

        def _init_fn(worker_id):
            np.random.seed(cfg.TRAIN.RANDOM_SEED + worker_id)

    if os.environ.get('ADVENT_DRY_RUN', '0') == '1':
        return

    i_iter = 46000
    
    #### Load Discriminator model #####    
    restore_from = osp.join(cfg.TEST.SNAPSHOT_DIR, f'model_{i_iter}.pth')
    model_seg = get_deeplab_v2_vgg(cfg=cfg, num_classes=cfg.NUM_CLASSES, pretrained_model=cfg.TRAIN_VGG_PRE_MODEL)
    load_checkpoint_for_evaluation(model_seg, restore_from, device)

    #### Load Discriminator model #####
    # restore_from_dis = osp.join(cfg.TEST.SNAPSHOT_DIR, f'model_{i_iter}_D2.pth')
    # model_dis = get_fc_discriminator(num_classes=num_classes)
    # load_checkpoint_for_evaluation(model_dis, restore_from_dis, device)

    print('Model loaded')


    target_dataset = BDDataSet(root=cfg.DATA_DIRECTORY_TARGET,
                                       list_path=cfg.DATA_LIST_TARGET,
                                       set=cfg.TRAIN.SET_TARGET,
                                       info_path=cfg.TRAIN.INFO_TARGET,
                                       crop_size=cfg.TEST.INPUT_SIZE_TARGET,
                                       mean=cfg.TRAIN.IMG_MEAN)

    target_loader = data.DataLoader(target_dataset,
                                    batch_size=cfg.TRAIN.BATCH_SIZE_TARGET,
                                    num_workers=cfg.NUM_WORKERS,
                                    shuffle=True,
                                    pin_memory=True,
                                    worker_init_fn=_init_fn)

    with open(osp.join(cfg.TRAIN.SNAPSHOT_DIR, 'train_cfg.yml'), 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)

    # UDA TRAINING
    # ranking_target_w_discrim(model_seg, model_dis, target_loader, output_path, cfg)
    ranking_target_w_discrim(model_seg, target_loader, output_path, cfg)


if __name__ == '__main__':
    main()
