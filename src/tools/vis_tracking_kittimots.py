import numpy as np
import cv2
import os
import glob
import sys
from collections import defaultdict
from pathlib import Path
import pycocotools.mask as rletools
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

DATA_PATH = '../../data/KITTIMOTS/'
IMG_PATH = DATA_PATH + 'train/'
SAVE_VIDEO = False
IS_GT = True

cats = ['Car', 'Pedestrian']
cat_ids = {cat: i for i, cat in enumerate(cats)}
COLORS = [(255, 0, 255), (122, 122, 255), (255, 0, 0)]

def draw_bbox(img, bboxes, c=(255, 0, 255)):
  for bbox in bboxes:
    color = COLORS[int(bbox[5])]
    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), 
      (int(bbox[2]), int(bbox[3])), 
      color, 2, lineType=cv2.LINE_AA)
    ct = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    txt = '{}'.format(int(bbox[4]))
    cv2.putText(img, txt, (int(ct[0]), int(ct[1])), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                color, thickness=1, lineType=cv2.LINE_AA)

if __name__ == '__main__':
  # seqs = os.listdir(IMG_PATH)
  seqs = ['0001']
  for seq in sorted(seqs):
    print('seq', seq)
    if '.DS_Store' in seq:
      continue

    gt_file = DATA_PATH + 'instances_txt/' + seq + '.txt'

    with open(gt_file, 'r') as f:
      lines = f.readlines()

    lines = [l.split() for l in lines]
    
    frame_count = -1
    im_to_inst = {}

    for l in lines:
      frame, oid, cid, h, w, rle = l

      if int(cid) - 1 not in cat_ids.values():
        continue

      frame = int(frame)
      if frame_count != frame:
        frame_count = frame
        im_to_inst[frame] = []
      
      im_to_inst[frame].append(rle)

    for i in im_to_inst:
      #img = cv2.imread(os.path.join(IMG_PATH, '{}/{:06d}.png'.format(seq, i)))
      img = Image.open(os.path.join(IMG_PATH, '{}/{:06d}.png'.format(seq, i))).convert('RGBA')
      #img.putalpha(128)

      size = [int(h), int(w)]
      merged = np.zeros(size, dtype=np.float)
      print(f'Frame {i}: {len(im_to_inst[i])} masks')
      for mask in im_to_inst[i]:
        m = {'size': size, 'counts': mask.encode(encoding='UTF-8')}
        binary_mask = rletools.decode(m)
        
        merged = np.logical_or(merged, binary_mask)
      
      merged_mask = Image.fromarray(np.uint8(merged * 128), mode='L')
      color = Image.new('RGBA', (size[1], size[0]), (228, 150, 150, 255))
      # plt.imshow(merged_mask)
      # plt.imshow(img)
      # plt.show()
      image = Image.composite(color, img, merged_mask)

      image.save('../../data/KITTIMOTS/examples/{:06d}.png'.format(i))


    # preds = {}
    # for K in range(1, len(sys.argv)):
    #   pred_path = sys.argv[K] + '/{}.txt'.format(seq)
    #   pred_file = open(pred_path, 'r')
    #   preds[K] = defaultdict(list)
    #   for line in pred_file:
    #     tmp = line[:-1].split(' ')
    #     frame_id = int(tmp[0])
    #     track_id = int(tmp[1])
    #     cat_id = cat_ids[tmp[2]]
    #     bbox = [float(tmp[6]), float(tmp[7]), float(tmp[8]), float(tmp[9])]
    #     score = float(tmp[17])
    #     preds[K][frame_id].append(bbox + [track_id, cat_id, score])

    # images_path = '{}/{}/'.format(IMG_PATH, seq)
    # images = os.listdir(images_path)
    # num_images = len([image for image in images if 'png' in image])
    
    # for i in range(num_images):
    #   frame_id = i
    #   file_path = '{}/{:06d}.png'.format(images_path, i)
    #   img = cv2.imread(file_path)
    #   for K in range(1, len(sys.argv)):
    #     img_pred = img.copy()
    #     draw_bbox(img_pred, preds[K][frame_id])
    #     cv2.imshow('pred{}'.format(K), img_pred)
    #   cv2.waitKey()
      # if SAVE_VIDEO:
      #   video.write(img_pred)
    # if SAVE_VIDEO:
    #   video.release()
