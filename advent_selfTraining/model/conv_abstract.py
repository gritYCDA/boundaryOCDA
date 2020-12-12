

import torch.nn as nn
import torch.nn.functional as F

class ConvAbstract(nn.Module):
    def __init__(self, cfg):
        super(ConvAbstract, self).__init__()
        # input_dim = cfg.CONTRA_DA.INPUT_DIM
        # output_dim = cfg.CONTRA_DA.OUTPUT_DIM

        input_dim = 1024
        output_dim = 256
        self.conv1 = nn.Conv2d(input_dim, input_dim, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(input_dim, output_dim, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)

        return x

    def optim_parameters(self, args):
        return self.parameters()

def get_conv_abstract(cfg):
    model = ConvAbstract(cfg)
    return model