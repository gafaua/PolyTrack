import os
import numpy as np
import json
import cv2
import pycocotools.mask as rletools
import polygon_tools

# Use the same script for MOT16
# DATA_PATH = '../../data/mot16/'
# DATA_PATH = '../../data/mot17/'
DATA_PATH = '../../data/MOTS/'
OUT_PATH = DATA_PATH + 'json_gt/'
SPLITS = ['test', 'train']
NBR_VERTICES = 16

if __name__ == '__main__':
  for split in SPLITS:
    data_path = DATA_PATH + split
    out_path = OUT_PATH + '{}_{}.json'.format(split, NBR_VERTICES)
    out = {'images': [], 'annotations': [], 
           'categories': [{'id': 1, 'name': 'pedestrian'}],
           'videos': []}
    seqs = os.listdir(data_path)
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    for seq in sorted(seqs):
      video_cnt += 1
      out['videos'].append({
        'id': video_cnt,
        'file_name': seq})
      seq_path = '{}/{}/'.format(data_path, seq)
      img_path = seq_path + 'img1/'
      ann_path = seq_path + 'gt/gt.txt'
      images = os.listdir(img_path)
      num_images = len([image for image in images if 'jpg' in image])

      image_range = [0, num_images - 1]
      for i in range(num_images):
        if (i < image_range[0] or i > image_range[1]):
          continue
        image_info = {'file_name': '{}/img1/{:06d}.jpg'.format(seq, i + 1),
                      'id': image_cnt + i + 1,
                      'frame_id': i + 1 - image_range[0],
                      'prev_image_id': image_cnt + i if i > 0 else -1,
                      'next_image_id': \
                        image_cnt + i + 2 if i < num_images - 1 else -1,
                      'video_id': video_cnt}
        out['images'].append(image_info)
      print('{}: {} images'.format(seq, num_images))
      if split != 'test':
        # anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
        anns = np.loadtxt(ann_path, dtype=str, delimiter=',')

        # For depth
        previous_frame = -1
        depth = 0

        for i in range(anns.shape[0]):
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

          ann = {'id': ann_cnt,
                 'category_id': category_id,
                 'image_id': image_cnt + frame_id,
                 'track_id': track_id,
                 'bbox': [x, y, w, h],
                 'poly': polygon,
                 'pseudo_depth': depth,
                 'conf': 1}
          out['annotations'].append(ann)
      image_cnt += num_images
    print('loaded {} for {} images and {} samples'.format(
      split, len(out['images']), len(out['annotations'])))
    json.dump(out, open(out_path, 'w'), indent=1)
    print('wrote json {} file in {}'.format(split, out_path))
        
        

