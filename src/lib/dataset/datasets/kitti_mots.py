from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pycocotools.mask as rletools
import numpy as np

from progress.bar import Bar
from ..generic_dataset import GenericDataset

class KITTIMOTS(GenericDataset):
  num_categories = 2
  default_resolution = [255, 1151] # < 256 x 1152 after padding
  # default_resolution = [383, 1279] # < 384 x 1280 after padding 
  class_name = ['Car', 'Pedestrian']
  cat_ids = {1:1, 2:2}
  max_objs = 50
  def __init__(self, opt, split):
    self.split = split
    self.dataset_version = opt.dataset_version

    data_dir = os.path.join(opt.data_dir, 'KITTIMOTS')

    img_dir = os.path.join(data_dir, 'test' if split == 'test' else 'train')

    if opt.dataset_version == 'train_val':
      ann_path = os.path.join(data_dir, 'json_gt', '{}_{}.json'.format(split, opt.nbr_points))
    elif opt.dataset_version == 'train_full':
      ann_path = os.path.join(data_dir, 'json_gt', '{}_{}.json'
                 .format('train_full' if split == 'train' else 'test', opt.nbr_points))

    self.images = None
    super(KITTIMOTS, self).__init__(opt, split, ann_path, img_dir)
    
    self.num_samples = len(self.images)
    print('Loaded KITTIMOTS/{} - {} set with {} samples'.format(opt.dataset_version, split, self.num_samples))

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    results_dir = save_dir
    if not os.path.exists(results_dir):
      os.makedirs(results_dir)

    for video in self.coco.dataset['videos']:
      video_id = video['id']
      file_name = video['file_name']
      out_path = os.path.join(results_dir, '{}.txt'.format(file_name))
      height, width = video['height'], video['width']
      with open(out_path, 'w') as f:
        images = self.video_to_images[video_id]
        bar = Bar(f'Saving tracking results of {file_name}', max=len(images))
        for j, image_info in enumerate(sorted(images, key=lambda x: x['frame_id'])):
          if not (image_info['id'] in results):
            continue
          result = results[image_info['id']]
          frame_id = int(image_info['frame_id']) - 1
          tracks_in_frame = []

          for item in result:
            if not ('tracking_id' in item):
              item['tracking_id'] = np.random.randint(1000)
            if item['active'] == 0:
              continue
            cat_id = item['class']
            tracking_id = item['tracking_id']
            rle_mask = rletools.frPyObjects([item['poly']], height, width)[0]
            track_id = f'{cat_id}{tracking_id:03}'

            tracks_in_frame.append({
              'track_id': track_id,
              'cat_id': cat_id,
              'mask': rle_mask,
              'pseudo_depth': item['pseudo_depth']
            })

          if len(tracks_in_frame) > 1:
            # make sure no masks overlap in the same frame
            merged = np.ones((height,width), dtype=np.float) * -1
            tracks_in_frame.sort(key=lambda x: x['pseudo_depth'])

            for i, track in enumerate(tracks_in_frame):
              binary_mask = rletools.decode(track['mask'])
              merged *= np.logical_not(binary_mask)
              merged += binary_mask * i
            
            for i, track in enumerate(tracks_in_frame):
              binary_mask = merged == i
              track['mask'] = rletools.encode(np.asfortranarray(binary_mask))

          for track in tracks_in_frame:
            # write line to file
            track_id = track['track_id']
            cat_id = track['cat_id']
            rle_mask = track['mask']['counts'].decode(encoding='UTF-8')

            f.write(f'{frame_id} {track_id} {cat_id} {height} {width} {rle_mask}\n')

          Bar.suffix = f'[{j}/{len(images)}]|Tot: {bar.elapsed_td} |ETA: {bar.eta_td} |Tracks: {len(tracks_in_frame)}'
          bar.next()
        bar.finish()
        print()

  def run_eval(self, results, save_dir):
    # Make sure that the TrackEval repo is in src/tools
    trackers_folder = os.path.join(save_dir, 'results_kitti_mots_{}'.format(self.dataset_version))
    save_dir = os.path.join(trackers_folder, 'PolyTrack', 'data')
    print(f'Saving results in {save_dir}')
    self.save_results(results, save_dir)

    if self.split == 'test':
      print('No tracking data for test set to compute validation results!')
      return
    print('\nRunning eval...')

    gt_dir = os.path.join(self.opt.data_dir, 'KITTIMOTS')
    gt_loc_format = '{gt_folder}/instances_txt/{seq}.txt'

    os.system(f'python tools/TrackEval/scripts/run_kitti_mots.py --GT_FOLDER {gt_dir} --TRACKERS_FOLDER {trackers_folder} --USE_PARALLEL True --GT_LOC_FORMAT {gt_loc_format}')

