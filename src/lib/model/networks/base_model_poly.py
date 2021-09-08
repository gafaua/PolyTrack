from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
from lib.model.networks.base_modules import ConvBlock

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def make_head(head_kernel, last_channel, conv_dims, out_channels):
    sequence = [ConvBlock(head_kernel, last_channel, conv_dims[0], with_bn=False)]

    for i in range(1, len(conv_dims)):
        sequence.append(nn.Conv2d(conv_dims[i-1], conv_dims[i], (1,1)))
        sequence.append(nn.ReLU(inplace=True))

    sequence.append(nn.Conv2d(conv_dims[-1], out_channels, kernel_size=(1,1)))

    return nn.Sequential(*sequence)

class BaseModelPoly(nn.Module):
  # This class is not used, but you can see it as a clarification of BaseModel
    def __init__(self, heads, head_convs, num_stacks, last_channel, opt=None):
        super(BaseModelPoly, self).__init__()
        if opt is not None and opt.head_kernel != 3:
          print('Using head kernel:', opt.head_kernel)
          head_kernel = opt.head_kernel
        else:
          head_kernel = 3
        self.num_stacks = num_stacks
        self.heads = heads

        # TODO:Check possible compatibility issues when using head_convs in the heads generation
        # For now, for the same behavior, from CenterPoly code head_convs must be [dims[0]]
        for head in self.heads:
            out_channels = self.heads[head]
            conv_dims = head_convs[head]

            module = nn.ModuleList([
                make_head(head_kernel, last_channel, conv_dims, out_channels)
                for _ in self.num_stacks
            ])

            self.__setattr__(head, module)

            if 'hm' in head:
                for heat in self.__getattr__(head):
                    heat[-1].bias.data.fill_(-2.19)


    def img2feats(self, x):
      raise NotImplementedError
    
    def imgpre2feats(self, x, pre_img=None, pre_hm=None):
      raise NotImplementedError

    def forward(self, x, pre_img=None, pre_hm=None):
      if (pre_hm is not None) or (pre_img is not None):
        feats = self.imgpre2feats(x, pre_img, pre_hm)
      else:
        feats = self.img2feats(x)
      out = []
      if self.opt.model_output_list:
        for s in range(self.num_stacks):
          z = []
          for head in sorted(self.heads):
              z.append(self.__getattr__(head)(feats[s]))
          out.append(z)
      else:
        for s in range(self.num_stacks):
          z = {}
          for head in self.heads:
              z[head] = self.__getattr__(head)(feats[s])
          out.append(z)
      return out
