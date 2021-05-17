from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Conv2D -> BatchNorm2D (if with_bn=True) -> ReLU\n
    Padding is handled to keep outputs WxH the same as inputs WxH
    """
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(ConvBlock, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu
