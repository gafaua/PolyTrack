import os
import numpy as np
import json
import cv2
import pycocotools.mask as rletools
import polygon_tools
from progress.bar import Bar

DATA_PATH = '../../data/KITTIMOTS/'
OUT_PATH = DATA_PATH + 'json_gt/'
SPLITS = ['test', 'train']
NBR_VERTICES = 32
VAL_SEQ = ['0000', '0001', '0002', '0007', '0016', '0017']

if __name__ == '__main__':
  seqmap = open(os.path.join(DATA_PATH, 'evaluate_mots.seqmap.val'), 'w')

  for split in SPLITS:
    data_path = DATA_PATH + split
    out_path = OUT_PATH + '{}_{}.json'.format(split, NBR_VERTICES)
    out_full = {'images': [], 'annotations': [],
              'categories': [{'id': 1, 'name': 'car'}, {'id': 2, 'name': 'pedestrian'}],
              'videos': []}
    if split == 'train':
      out_train = {'images': [], 'annotations': [],
              'categories': [{'id': 1, 'name': 'car'}, {'id': 2, 'name': 'pedestrian'}],
              'videos': []}
      out_val = {'images': [], 'annotations': [],
              'categories': [{'id': 1, 'name': 'car'}, {'id': 2, 'name': 'pedestrian'}],
              'videos': []}

    seqs = os.listdir(data_path)
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    for seq in sorted(seqs):
      video_cnt += 1
      video = {
        'id': video_cnt,
        'file_name': seq,
        }
      out_full['videos'].append(video)
      
      img_path = os.path.join(data_path, seq)
      ann_path = os.path.join(DATA_PATH, 'instances_txt', f'{seq}.txt')
      images = os.listdir(img_path)
      num_images = len([image for image in images if 'png' in image])

      image_range = [0, num_images - 1]

      if split == 'train':
        if seq in VAL_SEQ:
          out_val['videos'].append(video)
          seqmap.write(f'{seq} empty {image_range[0]} {image_range[1]}\n')
        else:
          out_train['videos'].append(video)

      for i in range(num_images):
        if (i < image_range[0] or i > image_range[1]):
          continue
        image_info = {'file_name': '{}/{:06d}.png'.format(seq, i),
                      'id': image_cnt + i + 1,
                      'frame_id': i + 1 - image_range[0],
                      'prev_image_id': image_cnt + i if i > 0 else -1,
                      'next_image_id': \
                        image_cnt + i + 2 if i < num_images - 1 else -1,
                      'video_id': video_cnt}
        
        if 'height' not in video:
            first_img = cv2.imread(os.path.join(data_path, image_info['file_name']))
            height, width = first_img.shape[:2]
            video['height'] = height
            video['width'] = width

        out_full['images'].append(image_info)

        if split == 'train':
          if seq in VAL_SEQ:
            out_val['images'].append(image_info)
          else:
            out_train['images'].append(image_info)

      print('{}: {} images'.format(seq, num_images))
      if split == 'train':
        anns = np.loadtxt(ann_path, dtype=str, delimiter=',')

        # For depth
        previous_frame = -1
        depth = 0

        bar = Bar(f'{split}: {seq}', max=anns.shape[0])
        for i in range(anns.shape[0]):
          Bar.suffix = f'[{i}/{anns.shape[0]}]|Tot: {bar.elapsed_td} |ETA: {bar.eta_td}'
          bar.next()

          annotation = anns[i].split(' ')
          annotation = [int(item) for item in annotation[:-1]] + [annotation[-1]]
          frame_id = int(annotation[0])

          if previous_frame == frame_id:
              depth += 1
          else:
              previous_frame = frame_id
              depth = 0

          if (frame_id - 1 < image_range[0] or frame_id - 1> image_range[1]):
            continue
          track_id = int(annotation[1])
          category_id = annotation[2]
          if category_id not in [1, 2]:
              continue
          ann_cnt += 1

          mask = {'size': [annotation[3], annotation[4]], 'counts': annotation[5].encode(encoding='UTF-8')}

          # find bbox from mask
          x, y, w, h = rletools.toBbox(mask)

          # find polygon from mask
          binary_mask = rletools.decode(mask)
          polygon = polygon_tools.mask_to_polygon(binary_mask, bbox=[x, y, x+w, y+h], nbr_vertices=NBR_VERTICES)
          polygon = np.reshape(polygon, NBR_VERTICES * 2).tolist()

          ann = {'id': ann_cnt,
                 'category_id': category_id,
                 'image_id': image_cnt + frame_id + 1,
                 'track_id': track_id,
                 'bbox': [x, y, w, h],
                 'poly': polygon,
                 'pseudo_depth': depth,
                 'conf': 1}
          
          out_full['annotations'].append(ann)

          if seq in VAL_SEQ:
            out_val['annotations'].append(ann)
          else:
            out_train['annotations'].append(ann)

        print()

      image_cnt += num_images

    print('loaded {} for {} images and {} samples'.format(
      split, len(out_full['images']), len(out_full['annotations'])))

    if split == 'train':
      json.dump(out_full, open(out_path.replace('train', 'train_full'), 'w'), indent=1)
      json.dump(out_train, open(out_path, 'w'), indent=1)
      json.dump(out_val, open(out_path.replace('train', 'val'), 'w'), indent=1)
    else:
      json.dump(out_full, open(out_path, 'w'), indent=1)

    print(f'Json files were generated for {split} with {NBR_VERTICES} vertices')
