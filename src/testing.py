import cv2
import _init_paths

from dataset.dataset_factory import dataset_factory
from opts import opts
from model.model import create_model, load_model, save_model
#from tools.vis_polygons_json import add_polys_to_image
from trainer import Trainer
from detector import Detector


import torch
import os
import numpy as np
from PIL import Image, ImageDraw

def test(opt):
    Dataset = dataset_factory['mots']
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    # detector = Detector(opt)
    # path = '/home/travail/gasp/PolyTrack/data/MOTS/train/MOTS20-09/img1'
    # imgs = sorted(os.listdir(path))

    # for img in imgs:
    #   detector.run(os.path.join(path, img))

    print("Creating model...")
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    if opt.load_model != '':
      model, optimizer, start_epoch = load_model(
        model, opt.load_model, opt, optimizer)

    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    #print(model)

    mots_dataset = Dataset(opt, 'val')
    print("Dataset created")
    train_loader = torch.utils.data.DataLoader(
      mots_dataset, batch_size=opt.batch_size, shuffle=False,
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






if __name__ == '__main__':
  opt = opts().parse()
  test(opt)
