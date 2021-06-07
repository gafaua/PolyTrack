import cv2
import _init_paths

from dataset.dataset_factory import dataset_factory
from opts import opts
from model.model import create_model, load_model, save_model
#from tools.vis_polygons_json import add_polys_to_image
from trainer import Trainer


import torch
import os
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
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    if opt.load_model != '':
      model, optimizer, start_epoch = load_model(
        model, opt.load_model, opt, optimizer)

    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    #print(model)

    mots_dataset = Dataset(opt, 'train')
    print("Dataset created")
    train_loader = torch.utils.data.DataLoader(
      mots_dataset, batch_size=opt.batch_size, shuffle=True,
      num_workers=opt.num_workers, pin_memory=True, drop_last=True
    )

    tot = 1
    print(f"DataLoader created with batch_size: {opt.batch_size}")
    print(f"Training {tot} epoch...")
    for epoch in range(tot):
      trainer.train(epoch, train_loader)
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
            epoch, model, optimizer)

    print("Done!")

    # for batch in train_loader:
    #     print(batch.keys())
    #     print(batch['image'][0].shape)
    #     print('polys', batch['poly'][0].shape)
    #     print('hm', batch['hm'][0].shape)
    #     print('hm', batch['hm'][0])

    #     im = batch['image'][0].numpy().transpose(1, 2, 0)
    #     im = ((im * mots_dataset.std) + mots_dataset.mean) * 255.
    #     im = im.astype(np.uint8)
        
    #     hm = (batch['hm'][0].numpy().transpose(1,2,0) * 255.).astype(np.uint8)
    #     im = add_polys_to_image(im, batch['poly'][0]*4)
        
    #     im = cv2.resize(im, (hm.shape[:2][1], hm.shape[:2][0]))
    #     hmpolys = np.hstack((im, cv2.cvtColor(hm, cv2.COLOR_GRAY2RGB)))
    #     cv2.imshow('hm', hmpolys)
    #     #cv2.imshow('polys', im)
    #     # cv2.imshow('polys', im)
    #     cv2.waitKey(10000)
    #     #print(batch['poly'])
    #     #time.sleep(10)
    #     # python testing.py tracking,polydet --pre_hm --elliptical_gt






if __name__ == '__main__':
  opt = opts().parse()
  test(opt)
