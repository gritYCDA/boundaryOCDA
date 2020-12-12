import numpy as np
import torch
from torch import nn
from torchvision import models
import sys
import torch.nn.functional as F

sys.path.append("/media/user/a9755522-b17e-4bde-96f6-088bbbc3a1401/OCDA/ADVENT")


class Classifier_Module(nn.Module):

    def __init__(self, dims_in, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(dims_in, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out

## 1. Deeplab + VGG baseline
class DeeplabVGG(nn.Module):
    def __init__(self, num_classes, vgg16_caffe_path=None, pretrained=False):
        super(DeeplabVGG, self).__init__()
        vgg = models.vgg16()
        if pretrained:
            vgg.load_state_dict(torch.load(vgg16_caffe_path))

        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())
        # remove pool4/pool5
        features = nn.Sequential(*(features[i] for i in list(range(23)) + list(range(24, 30))))
        for i in [23, 25, 27]:
            features[i].dilation = (2, 2)
            features[i].padding = (2, 2)

        fc6 = nn.Conv2d(512, 1024, kernel_size=3, padding=4, dilation=4)
        fc7 = nn.Conv2d(1024, 1024, kernel_size=3, padding=4, dilation=4)

        self.features = nn.Sequential(
            *([features[i] for i in range(len(features))] + [fc6, nn.ReLU(inplace=True), fc7, nn.ReLU(inplace=True)]))

        self.classifier = Classifier_Module(1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

    def forward(self, x):
        x = self.features(x)
        seg_out = self.classifier(x)
        return x, seg_out

    def classifier_(self, x):
        seg_out = self.classifier(x)
        return seg_out

    def optim_parameters(self, args):
        return self.parameters()

## 2. Deeplab + Parallel Boundary Classifier
class DeeplabVGG_Boundary(DeeplabVGG):
    def __init__(self, num_classes, vgg16_caffe_path=None, pretrained=False):
        super().__init__(num_classes, vgg16_caffe_path, pretrained)
        self.classifier_boundary = Classifier_Module(1024, [6, 12, 18, 24], [6, 12, 18, 24], 1)

    def forward(self, x):
        x = self.features(x)
        seg_out = self.classifier(x)
        boundary = self.classifier_boundary(x)
        return x, seg_out, boundary

    def Boundary_classifier_(self, x):
        seg_out = self.classifier(x)
        boundary = self.classifier_boundary(x)
        return seg_out, boundary

## 3. Deeplab + Attention Boundary
class DeeplabVGG_Boundary_Attention(nn.Module):
    def __init__(self, num_classes, vgg16_caffe_path=None, pretrained=False):
        super(DeeplabVGG_Boundary_Attention, self).__init__()

        vgg = models.vgg16()
        if pretrained:
            vgg.load_state_dict(torch.load(vgg16_caffe_path))

        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())
        # remove pool4/pool5
        features = nn.Sequential(*(features[i] for i in list(range(23)) + list(range(24, 30))))
        for i in [23, 25, 27]:
            features[i].dilation = (2, 2)
            features[i].padding = (2, 2)

        fc6 = nn.Conv2d(512, 1024, kernel_size=3, padding=4, dilation=4)
        fc7 = nn.Conv2d(1024, 1024, kernel_size=3, padding=4, dilation=4)

        self.features = nn.Sequential(
            *([features[i] for i in range(len(features))] + [fc6, nn.ReLU(inplace=True), fc7, nn.ReLU(inplace=True)]))

        self.classifier = Classifier_Module(1025, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.classifier_boundary = Classifier_Module(1024, [6, 12, 18, 24], [6, 12, 18, 24], 1)

    def forward(self, x):
        x = self.features(x)

        # encoder
        # x_en = self.enc4_1(x)
        # x_en = self.relu(x_en)
        # x_en = self.enc4_2(x_en)
        # x_en = self.relu(x_en)
        # x_en = self.enc4_3(x_en)
        # x_en = self.relu(x_en)
        # boundary = self.enc4_4(x_en)
        # boundary = torch.mean(x_en, dim=1, keepdim=True)

        # decoder
        # x_de = self.dec4(x_en)
        # x_de = self.relu(x_de)
        # feature fusion
        # x = x + torch.sigmoid(x_de)
        boundary = self.classifier_boundary(x)
        x = torch.cat((x, boundary), dim=1)
        seg_out = self.classifier(x)
        return x, seg_out, boundary

    def optim_parameters(self, args):
        return self.parameters()


class DeeplabVGG_Boundary_Attention_v2(nn.Module):
    def __init__(self, num_classes, vgg16_caffe_path=None, pretrained=False):
        super(DeeplabVGG_Boundary_Attention_v2, self).__init__()

        vgg = models.vgg16()
        if pretrained:
            vgg.load_state_dict(torch.load(vgg16_caffe_path))

        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())
        # remove pool4/pool5
        features = nn.Sequential(*(features[i] for i in list(range(23)) + list(range(24, 30))))
        for i in [23, 25, 27]:
            features[i].dilation = (2, 2)
            features[i].padding = (2, 2)

        fc6 = nn.Conv2d(512, 1024, kernel_size=3, padding=4, dilation=4)
        fc7 = nn.Conv2d(1024, 1024, kernel_size=3, padding=4, dilation=4)

        self.features = nn.Sequential(
            *([features[i] for i in range(len(features))] + [fc6, nn.ReLU(inplace=True), fc7, nn.ReLU(inplace=True)]))

        self.classifier = Classifier_Module(1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.classifier_boundary = Classifier_Module(1024, [6, 12, 18, 24], [6, 12, 18, 24], 1)

        self.fusion_block = FusionBlock(1024)

    def forward(self, x):
        x = self.features(x)

        boundary = self.classifier_boundary(x)
        x = self.fusion_block(x, boundary)

        seg_out = self.classifier(x)
        return x, seg_out, boundary

    def optim_parameters(self, args):
        return self.parameters()


class FusionBlock(nn.Module):
    def __init__(self, in_c=1024):
        super(FusionBlock, self).__init__()

        self.mixer = nn.Sequential(
            nn.Conv2d(in_c + 1, in_c, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU()
        )

        self.mlp = nn.Sequential(
            nn.Linear(in_c, in_c//16),
            nn.ReLU(),
            nn.Linear(in_c//16, in_c)
        )

    def forward(self, x, boundary):
        x = torch.cat((x, boundary), dim=1)
        x = self.mixer(x).squeeze()

        x_avg = F.avg_pool2d(x, (x.size(1), x.size(2)), stride=(x.size(1), x.size(2)))
        x_avg = x_avg.view(x_avg.size(0))
        x_avg_att = self.mlp(x_avg)

        x_max = F.max_pool2d(x, (x.size(1), x.size(2)), stride=(x.size(1), x.size(2)))
        x_max = x_max.view(x_max.size(0))
        x_max_att = self.mlp(x_max)

        x_att = x_avg_att + x_max_att

        scale = F.sigmoid(x_att).view(x_att.size(0), 1, 1)
        x = x * scale

        return x.unsqueeze(0)



def get_deeplab_v2_vgg(cfg, num_classes=19, pretrained_model=None, pretrained=True):
    if cfg.TRAIN.OCDA_METHOD == 'baseline' or cfg.TRAIN.OCDA_METHOD == "selfTrain":
        model = DeeplabVGG(num_classes, pretrained_model, pretrained)
    elif cfg.TRAIN.OCDA_METHOD == 'boundary' or cfg.TRAIN.OCDA_METHOD == 'selfTrain_boundary':
        model = DeeplabVGG_Boundary(num_classes, pretrained_model, pretrained)
    elif cfg.TRAIN.OCDA_METHOD == 'ad_boundary':
        model = DeeplabVGG_Boundary_Attention(num_classes, pretrained_model, pretrained)
    elif cfg.TRAIN.OCDA_METHOD == 'attn_boundary':
        model = DeeplabVGG_Boundary_Attention_v2(num_classes, pretrained_model, pretrained)
    return model
