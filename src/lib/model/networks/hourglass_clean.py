from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel

BN_MOMENTUM = 0.1

class ConvBlock(nn.Module):
    """
    Conv2D -> BatchNorm2D (if with_bn=True) -> ReLU\n
    Padding is handled to keep outputs WxH the same as inputs WxH
    """
    def __init__(self, kernel, inp_dim, out_dim, stride=1, with_bn=True):
        super(ConvBlock, self).__init__()

        pad = (kernel - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel, padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu

class Residual(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(Residual, self).__init__()

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
        self.bn1   = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
        self.bn2   = nn.BatchNorm2d(out_dim)

        self.skip  = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2   = self.bn2(conv2)

        skip  = self.skip(x)
        return self.relu(bn2 + skip)

def make_layer(kernel, inp_dim, out_dim, modules, layer=ConvBlock):
    layers  = [layer(kernel, inp_dim, out_dim)]
    layers += [layer(kernel, out_dim, out_dim) for _ in range(modules - 1)]

    return nn.Sequential(*layers)

def make_hg_layer(kernel, inp_dim, out_dim, modules, layer=ConvBlock):
    layers  = [layer(kernel, inp_dim, out_dim, stride=2)]
    layers += [layer(kernel, out_dim, out_dim) for _ in range(modules - 1)]

    return nn.Sequential(*layers)

def make_layer_revr(kernel, inp_dim, out_dim, modules, layer=ConvBlock):
    layers = [layer(kernel, inp_dim, inp_dim) for _ in range(modules - 1)]
    layers.append(layer(kernel, inp_dim, out_dim))

    return nn.Sequential(*layers)

def make_base_layer(inp_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(inp_dim, out_dim, kernel_size=7, stride=1, padding=3, bias=False),
        nn.BatchNorm2d(out_dim, momentum=BN_MOMENTUM),
        nn.ReLU(inplace=True)
    )

def make_base_layer(inp_dim, out_dim):
    return nn.Sequential(
        ConvBlock(7, inp_dim, 128, stride=2),
        Residual(3, 128, out_dim, stride=2)
    )


class HourglassModule(nn.Module):
    def __init__(
        self, n, dims, modules, layer=Residual,
    ):
        super(HourglassModule, self).__init__()

        curr_mod, next_mod = modules[:2]
        curr_dim, next_dim = dims[:2]

        self.up1  = make_layer(3, curr_dim, curr_dim, curr_mod, layer=layer)

        self.low1 = make_hg_layer(3, curr_dim, next_dim, curr_mod, layer=layer)

        self.low2 = \
        HourglassModule(n - 1, dims[1:], modules[1:], layer=layer) \
        if n > 1 else \
        make_layer(3, next_dim, next_dim, next_mod, layer=layer)

        self.low3 = make_layer_revr(3, next_dim, curr_dim, curr_mod, layer=layer)

        self.up2  = nn.Upsample(scale_factor=2)

    def forward(self, x):
        up1  = self.up1(x)
        low1 = self.low1(x)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return up1 + up2


class HourglassNetwork(nn.Module):
    def __init__(self, dims, modules, num_stacks):
        super(HourglassNetwork, self).__init__()
        assert len(dims) == len(modules)

        n = len(dims) - 1

        self.num_stacks = num_stacks
        
        dim = dims[0] # == cnv_dim

        self.hg_stacks = nn.ModuleList([
            nn.Sequential(
                HourglassModule(n, dims, modules),
                ConvBlock(3, dim, dim)
            )
            for _ in range(num_stacks)
        ])

        self.inters = nn.ModuleList([
            Residual(3, dim, dim)
            for _ in range(num_stacks - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(dim, dim, (1, 1), bias=False),
                    nn.BatchNorm2d(dim)
                ),
                nn.Sequential(
                    nn.Conv2d(dim, dim, (1, 1), bias=False),
                    nn.BatchNorm2d(dim)
                )])
            for _ in range(num_stacks - 1)
        ])

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        outs = []

        for i in range(self.num_stacks):
            feats = self.hg_stacks[i](x)
            
            outs.append(feats)
            
            if i < self.num_stacks - 1:
                inter_x, inter_feats = self.inters_[i]
                x = self.relu(inter_x(x) + inter_feats(feats))
                x = self.inters[i](x)

        return outs


class Hourglass(BaseModel):
    # This class is not used because of compatibility issues with parameter names when trying
    # to finetune a model from CenterNet/CenterTrack/CenterPoly
    # It generates the exact same architecture as the one found in hourglass.py
    def __init__(self, heads, head_convs, opt):
        self.opt = opt

        num_stacks = opt.num_stacks
        print(f'Generating hourglass network with {num_stacks} stacks')

        # TODO make these parameters?
        dims    = [256, 256, 384, 384, 384, 512]
        modules = [2, 2, 2, 2, 2, 4]

        super(Hourglass, self).__init__(
            heads, head_convs, num_stacks, dims[0], opt=opt)

        self.hg = HourglassNetwork(dims, modules, num_stacks)

        if opt.pre_img:
            self.pre_img_layer = make_base_layer(3, dims[0])
            
        if opt.pre_hm:
            self.pre_hm_layer = make_base_layer(1, dims[0])

        self.pre = self.base_layer = make_base_layer(3, dims[0])

    def img2feats(self, x):
        x = self.pre(x)

        return self.hg(x)
    
    def imgpre2feats(self, x, pre_img=None, pre_hm=None):
        x = self.pre(x)

        if pre_img is not None:
            x = x + self.pre_img_layer(pre_img)
        if pre_hm  is not None:
            x = x + self.pre_hm_layer(pre_hm)
        
        return self.hg(x)

def GetHourglass(num_layers, heads, head_convs, opt):
    return Hourglass(heads, head_convs, opt)
