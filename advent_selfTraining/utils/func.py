import numpy as np
import torch
import torch.nn as nn

from advent.utils.loss import cross_entropy_2d
import torch.nn.functional as F


def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)


def mse_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.MSELoss()(y_pred, y_truth_tensor)


def reg_loss_calc_ign(pred, label, device):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w

    mr_weight_kld = 0.1
    ig_weight_ent = 0.05
    ignore_label = 255
    num_classes = 19
    beta = 0.95
    reg_weights = torch.FloatTensor([mr_weight_kld]).to(device)
    num_class = float(num_classes)
    valid_num = torch.sum(label != ignore_label).float()
    label_reg = label[reg_weights != 0,:,:]
    valid_reg_num = torch.sum(label_reg != ignore_label).float()
    invalid_reg_num = torch.sum(label_reg == ignore_label).float()

    softmax = F.softmax(pred, dim=1)   # compute the softmax values
    logsoftmax = F.log_softmax(pred,dim=1)   # compute the log of softmax values

    label_expand = torch.unsqueeze(label, 1).repeat(1,int(num_class),1,1)
    labels = label_expand.clone()
    labels[labels != ignore_label] = 1.0
    labels[labels == ignore_label] = 0.0
    labels_valid = labels.clone()
    labels_invalid = (1.0 - labels.clone())
    # labels = torch.unsqueeze(labels, 1).repeat(1,num_class,1,1)
    labels = torch.cumsum(labels, dim=1)
    labels[labels != label_expand + 1] = 0.0
    del label_expand
    labels[labels != 0 ] = 1.0
    ### check the vectorized labels
    # check_labels = torch.argmax(labels, dim=1)
    # label[label == 255] = 0
    # print(torch.sum(check_labels.float() - label))
    reg_weights = reg_weights.float().view(len(reg_weights),1,1,1)
    ce = torch.sum( -logsoftmax*labels ) # cross-entropy loss with vector-form softmax
    softmax_val = softmax*labels_valid
    logsoftmax_val = logsoftmax*labels_valid
    kld = torch.sum( -logsoftmax_val/num_class*reg_weights )
    ce_pre = -logsoftmax*labels # cross-entropy loss with vector-form softmax

    softmax_inval = softmax*labels_invalid
    logsoftmax_inval = logsoftmax*labels_invalid
    ign_ent = torch.sum(-softmax_inval*logsoftmax_inval)

    if valid_reg_num > 0:
        # reg_ce = ce/valid_num + (mr_weight_kld*kld)/valid_reg_num
        # if invalid_reg_num > 0:
        reg_ce = ce/valid_num + (mr_weight_kld*kld)/valid_reg_num + (ig_weight_ent*ign_ent)/invalid_reg_num
        # reg_ce = ce/valid_num + (mr_weight_kld*kld)/valid_reg_num
        # else:
        # reg_ce = torch.sum(beta * ce_pre + (1-beta) * (-softmax * logsoftmax)) / valid_num + (mr_weight_kld*kld)/valid_reg_num
        # reg_ce = ce/valid_num + (mr_weight_kld*kld)/valid_reg_num
        
    else:
        if valid_num == 0:
            return ce
        reg_ce = ce/valid_num

    loss = reg_ce

    return loss

def loss_calc(pred, label, device):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = label.long().to(device)
    return cross_entropy_2d(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    """ Poly_LR scheduler
    """
    return base_lr * ((1 - float(iter) / max_iter) ** power)


def _adjust_learning_rate(optimizer, i_iter, cfg, learning_rate):
    lr = lr_poly(learning_rate, i_iter, cfg.TRAIN.MAX_ITERS, cfg.TRAIN.POWER)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate(optimizer, i_iter, cfg):
    """ adject learning rate for main segnet
    """
    _adjust_learning_rate(optimizer, i_iter, cfg, cfg.TRAIN.LEARNING_RATE)


def adjust_learning_rate_discriminator(optimizer, i_iter, cfg):
    _adjust_learning_rate(optimizer, i_iter, cfg, cfg.TRAIN.LEARNING_RATE_D)


def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
