import cv2
import _init_paths

from dataset.dataset_factory import dataset_factory
from opts import opts
from model.model import create_model, load_model, save_model
from tools.vis_polygons_json import add_polys_to_image

import torch
import time
import numpy as np
from PIL import Image, ImageDraw

def test(opt):
    Dataset = dataset_factory['mots']
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt.arch)
    print(opt.gpus)
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print("Creating model...")
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)
    #print(model)

    mots_dataset = Dataset(opt, 'train')
    print("Dataset created")
    train_loader = torch.utils.data.DataLoader(
      mots_dataset, batch_size=1, shuffle=True,
      num_workers=opt.num_workers, pin_memory=True, drop_last=True
    )

    print(f"DataLoader created with batch_size: {opt.batch_size}")

    for batch in train_loader:
        print(batch.keys())
        print(batch['image'][0].shape)
        print('polys', batch['poly'][0].shape)
        print('hm', batch['hm'][0].shape)
        print('hm', batch['hm'][0])

        im = batch['image'][0].numpy().transpose(1, 2, 0)
        im = ((im * mots_dataset.std) + mots_dataset.mean) * 255.
        im = im.astype(np.uint8)
        
        hm = (batch['hm'][0].numpy().transpose(1,2,0) * 255.).astype(np.uint8)
        im = add_polys_to_image(im, batch['poly'][0]*4)
        
        hm = cv2.resize(hm, (im.shape[:2][1], im.shape[:2][0]))
        cv2.imshow('hm', hm)
        cv2.imshow('polys', im)
        # cv2.imshow('polys', im)
        cv2.waitKey(10000)
        #print(batch['poly'])
        #time.sleep(10)
        # python testing.py tracking,polydet --pre_hm --elliptical_gt





if __name__ == '__main__':
  opt = opts().parse()
  test(opt)
