import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import numpy as np
import os
import json
import cv2


def add_polys_to_image(image, polys):
    im = Image.fromarray(image)
    ec = (255, 255, 0, 100)
    for ind, poly in enumerate(polys):
        points = []
        for i in range(0, len(poly), 2):
            points.append((poly[i], poly[i+1]))

        ImageDraw.Draw(im, 'RGBA').polygon(points, outline=0, fill=ec)
    return np.array(im, dtype=np.uint8)
    
if __name__ == '__main__':
    base_dir = '../../data/KITTIMOTS/train'
    anno_file = '../../data/KITTIMOTS/json_gt/train_32.json'
    anno = json.load(open(anno_file, 'r'))
    id_to_file = {}
    for image in anno['images']:
        id_to_file[image['id']] = image['file_name']

    anns = json.load(open(anno_file, 'r'))
    image_to_boxes = {}
    for ann in anns['annotations']:
        gt = {'cat_id': ann['category_id']}
        gt['pseudo_depth'] = ann['pseudo_depth']
        gt['poly'] = ann['poly']

        if id_to_file[ann['image_id']] in image_to_boxes:
            image_to_boxes[id_to_file[ann['image_id']]].append(gt)
        else:
            image_to_boxes[id_to_file[ann['image_id']]] = [gt]

    count = 1
    for key in sorted(image_to_boxes):
        im = Image.open(os.path.join(base_dir, key))
        depth_map = Image.fromarray(np.ones((im.size[1], im.size[0])) * 255)
        depths = []
        for poly in sorted(image_to_boxes[key], key=lambda x: x['pseudo_depth'], reverse=True):
            depth = float(poly['pseudo_depth'])
            depths.append(depth)
            label = int(poly['cat_id'])
            ec = (0, 149, 255, 100)
            if label == 1:
                ec = (255, 127, 0, 100)  # car
            elif label == 2:
                ec = (255, 255, 0, 100)  # person

            # print(poly['polygon'])
            points = []
            for i in range(0, len(poly['poly']), 2):
                points.append((poly['poly'][i], poly['poly'][i+1]))
            ImageDraw.Draw(im, 'RGBA').polygon(points, outline=0, fill=ec)

        for poly in sorted(image_to_boxes[key], key=lambda x: x['pseudo_depth'], reverse=True):
            depth = float(poly['pseudo_depth'])
            points = []
            for i in range(0, len(poly['poly']), 2):
                points.append((poly['poly'][i], poly['poly'][i+1]))
            depth_color = (depth-np.min(depths)) / np.max(depths) * 255
            ImageDraw.Draw(depth_map).polygon(points, outline=0, fill=depth_color)

        write_dir = os.path.join(os.path.dirname(anno_file), 'image_examples', os.path.dirname(key).split('/')[0])
        if not os.path.exists(write_dir):
            os.mkdir(write_dir)
        im.save(os.path.join(write_dir, os.path.basename(key)))
        heatmap = cv2.applyColorMap(np.array(depth_map).astype(np.uint8), cv2.COLORMAP_HOT)
        cv2.imwrite(os.path.join(write_dir, os.path.basename(key).replace('.png', '_depth.png')), heatmap)
