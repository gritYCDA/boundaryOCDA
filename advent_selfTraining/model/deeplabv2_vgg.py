import numpy as np
import torch
from torch import nn
from torchvision import models
import pdb
import sys
import torch.nn.functional as F

sys.path.append("/media/user/a9755522-b17e-4bde-96f6-088bbbc3a1401/OCDA/ADVENT")
class Classifier_Module(nn.Module):

    def __init__(self, dims_in, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(dims_in, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out


class DeeplabVGG(nn.Module):
    def __init__(self, num_classes, vgg16_caffe_path=None, pretrained=False):
        super(DeeplabVGG, self).__init__()
        vgg = models.vgg16()
        if pretrained:
         
            vgg.load_state_dict(torch.load(vgg16_caffe_path))

        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        features = nn.Sequential(*(features[i] for i in list(range(23))+list(range(24,30))))
        for i in [23,25,27]:
            features[i].dilation = (2,2)
            features[i].padding = (2,2)

        fc6 = nn.Conv2d(512, 1024, kernel_size=3, padding=4, dilation=4)
        fc7 = nn.Conv2d(1024, 1024, kernel_size=3, padding=4, dilation=4)

        self.features = nn.Sequential(*([features[i] for i in range(len(features))] + [ fc6, nn.ReLU(inplace=True), fc7, nn.ReLU(inplace=True)]))

        self.classifier = Classifier_Module(1024, [6,12,18,24],[6,12,18,24],num_classes)


    def forward(self, x, domain=0):
        x = self.features(x)
        seg_out = self.classifier(x)
        return seg_out, x

    def classifier_(self, x):
        seg_out = self.classifier(x)
        return seg_out

    def optim_parameters(self, args):
        return self.parameters()


def get_deeplab_v2_vgg(cfg, num_classes=19, pretrained_model=None, pretrained=True):

    model = DeeplabVGG(num_classes, pretrained_model, pretrained)
    return model
