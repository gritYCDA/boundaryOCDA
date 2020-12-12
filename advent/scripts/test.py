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
import warnings

from torch.utils import data

from advent.model.deeplabv2 import get_deeplab_v2
from advent.dataset.cityscapes import CityscapesDataSet
from advent.domain_adaptation.config import cfg, cfg_from_file
from advent.domain_adaptation.eval_UDA import evaluate_domain_adaptation

from advent.dataset.bdd import BDDdataset
from advent.model.deeplabv2_vgg import get_deeplab_v2_vgg
from advent.utils import project_root

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for evaluation")
    parser.add_argument('--cfg', type=str, default=None,
                        help='optional config file', )
    parser.add_argument("--exp-suffix", type=str, default=None,
                        help="optional experiment suffix")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="the number of dataloader workers")
    """
        dataset manager
    """
    parser.add_argument('--source', type=str, default='GTA',
                        help='source dataset [GTA, SYNTHIA]')
    parser.add_argument('--target', type=str, default='BDD',
                        help='target dataset [Cityscapes, BDD]')

    return parser.parse_args()


def main(config_file, exp_suffix):
    # LOAD ARGS
    assert config_file is not None, 'Missing cfg file'
    cfg_from_file(config_file)

    cfg.NUM_WORKERS = args.num_workers

    ### dataset settings
    cfg.SOURCE = args.source
    cfg.TARGET = args.target
    ## source config
    if cfg.SOURCE == 'GTA':
        cfg.DATA_LIST_SOURCE = str(project_root / 'advent/dataset/gta5_list/{}.txt')
        cfg.DATA_DIRECTORY_SOURCE = str(project_root / 'data/GTA5')

    elif cfg.SOURCE == 'SYNTHIA':
        raise NotImplementedError(f"Not yet supported {cfg.SOURCE} dataset")
    else:
        raise NotImplementedError(f"Not yet supported {cfg.SOURCE} dataset")

    ## target config
    if cfg.TARGET == 'Cityscapes':
        cfg.DATA_LIST_TARGET = str(project_root / 'advent/dataset/cityscapes_list/{}.txt')
        cfg.DATA_DIRECTORY_TARGET = str(project_root / 'data/cityscapes')
        cfg.EXP_ROOT = project_root / 'experiments_G2C'
        cfg.EXP_ROOT_SNAPSHOT = osp.join(cfg.EXP_ROOT, 'snapshots_G2C')
        cfg.EXP_ROOT_LOGS = osp.join(cfg.EXP_ROOT, 'logs_G2C')
        cfg.TEST.INPUT_SIZE_TARGET = (1024, 512)
        cfg.TEST.OUTPUT_SIZE_TARGET = (2048, 1024)
        cfg.TEST.INFO_TARGET = str(project_root / 'advent/dataset/cityscapes_list/info.json')

    elif cfg.TARGET == 'BDD':
        cfg.DATA_LIST_TARGET = str(project_root / 'advent/dataset/compound_list/{}.txt')
        cfg.DATA_DIRECTORY_TARGET = str(project_root / 'data/bdd/Compound')
        cfg.EXP_ROOT = project_root / 'experiments'
        cfg.EXP_ROOT_SNAPSHOT = osp.join(cfg.EXP_ROOT, 'snapshots')
        cfg.EXP_ROOT_LOGS = osp.join(cfg.EXP_ROOT, 'logs')
        cfg.TEST.INPUT_SIZE_TARGET = (960, 540)
        cfg.TEST.OUTPUT_SIZE_TARGET = (1280, 720)
        cfg.TEST.INFO_TARGET = str(project_root / 'advent/dataset/compound_list/info.json')

    else:
        raise NotImplementedError(f"Not yet supported {cfg.TARGET} dataset")


    # auto-generate exp name if not specified
    if cfg.EXP_NAME == '':
        cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}_{cfg.TRAIN.OCDA_METHOD}'
    if exp_suffix:
        cfg.EXP_NAME += f'_{exp_suffix}'
    # auto-generate snapshot path if not specified
    if cfg.TEST.SNAPSHOT_DIR[0] == '':
        cfg.TEST.SNAPSHOT_DIR[0] = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME)
        os.makedirs(cfg.TEST.SNAPSHOT_DIR[0], exist_ok=True)

    print('Using config:')
    pprint.pprint(cfg)
    # load models
    models = []
    n_models = len(cfg.TEST.MODEL)
    if cfg.TEST.MODE == 'best':
        assert n_models == 1, 'Not yet supported'
    for i in range(n_models):
        if cfg.TEST.MODEL[i] == 'DeepLabv2':
            model = get_deeplab_v2(num_classes=cfg.NUM_CLASSES,
                                   multi_level=cfg.TEST.MULTI_LEVEL[i])

        elif cfg.TRAIN.MODEL == 'DeepLabv2_VGG':
            model = get_deeplab_v2_vgg(cfg=cfg, num_classes=cfg.NUM_CLASSES, pretrained_model=cfg.TRAIN_VGG_PRE_MODEL)
        else:
            raise NotImplementedError(f"Not yet supported {cfg.TEST.MODEL[i]}")
        models.append(model)

    if os.environ.get('ADVENT_DRY_RUN', '0') == '1':
        return

    # dataloaders
    if cfg.TARGET == 'Cityscapes':
        test_dataset = CityscapesDataSet(root=cfg.DATA_DIRECTORY_TARGET,
                                         list_path=cfg.DATA_LIST_TARGET,
                                         set=cfg.TEST.SET_TARGET,
                                         info_path=cfg.TEST.INFO_TARGET,
                                         crop_size=cfg.TEST.INPUT_SIZE_TARGET,
                                         mean=cfg.TEST.IMG_MEAN,
                                         labels_size=cfg.TEST.OUTPUT_SIZE_TARGET)
        test_loader = data.DataLoader(test_dataset,
                                      batch_size=cfg.TEST.BATCH_SIZE_TARGET,
                                      num_workers=cfg.NUM_WORKERS,
                                      shuffle=False,
                                      pin_memory=True)
    elif cfg.TARGET == 'BDD':
        test_dataset = BDDdataset(root=cfg.DATA_DIRECTORY_TARGET,
                                           list_path=cfg.DATA_LIST_TARGET,
                                           set=cfg.TEST.SET_TARGET,
                                           info_path=cfg.TEST.INFO_TARGET,
                                           crop_size=cfg.TEST.INPUT_SIZE_TARGET,
                                           mean=cfg.TEST.IMG_MEAN,
                                           labels_size=cfg.TEST.OUTPUT_SIZE_TARGET)
        test_loader = data.DataLoader(test_dataset,
                                        batch_size=cfg.TEST.BATCH_SIZE_TARGET,
                                        num_workers=cfg.NUM_WORKERS,
                                        shuffle=False,
                                        pin_memory=True)
    else:
        raise NotImplementedError(f"Not yet supported {cfg.TARGET} datasets")
    # eval
    evaluate_domain_adaptation(models, test_loader, cfg)


if __name__ == '__main__':
    args = get_arguments()
    print('Called with args:')
    print(args)
    main(args.cfg, args.exp_suffix)
