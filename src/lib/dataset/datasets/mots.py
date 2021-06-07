import os
import pycocotools.mask as rletools
from ..generic_dataset import GenericDataset

class MOTS(GenericDataset):
  num_categories = 1
  default_resolution = [544, 960]
  class_name = ['pedestrian']
  max_objs = 256
  cat_ids = {2:1}

  def __init__(self, opt, split):
    self.dataset_version = opt.dataset_version

    data_dir = os.path.join(opt.data_dir, 'MOTS')
    img_dir = os.path.join(data_dir, 'test' if split == 'test' else 'train')

    if opt.dataset_version == 'train_val':
      ann_path = os.path.join(data_dir, 'json_gt', '{}_{}.json'.format(split, opt.nbr_points))
    elif opt.dataset_version == 'train_full':
      ann_path = os.path.join(data_dir, 'json_gt', '{}_{}.json'
                 .format('train_full' if split == 'train' else 'test', opt.nbr_points))

    print('Using MOTS version {} with {} points polys'.format(opt.dataset_version, opt.nbr_points))
    print('Annotations file: {}'.format(ann_path))

    self.images = None
    super().__init__(opt=opt, split=split, ann_path=ann_path, img_dir=img_dir)

    self.num_samples = len(self.images)
    print('Loaded MOTS/{} {} {} samples'.format(opt.dataset_version, split, self.num_samples))
  
  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    # TODO save results to text file, convert polygons to masks
    results_dir = os.path.join(save_dir, 'results_mots{}'.format(self.dataset_version))
    if not os.path.exists(results_dir):
      os.mkdir(results_dir)

    for video in self.coco.dataset['videos']:
      video_id = video['id']
      file_name = video['file_name']
      out_path = os.path.join(results_dir, '{}.txt'.format(file_name))
      height, width = video['height'], video['width']
      with open(out_path, 'w') as f:
        images = self.video_to_images[video_id]
        for image_info in sorted(images, key=lambda x: x['frame_id']):
          if not (image_info['id'] in results):
            continue
          result = results[image_info['id']]
          frame_id = image_info['frame_id']

          for item in result:
            if not ('tracking_id' in item):
              item['tracking_id'] = np.random.randint(1000)
            if item['active'] == 0:
              continue
            cat_id = 2 #result['class'] only pedestrians in MOTS
            tracking_id = item['tracking_id']

            rle_mask = rletools.frPyObjects([item['poly']], height, width)[0]['counts'].decode(encoding='UTF-8')
            f.write(f'{frame_id} {cat_id}{tracking_id:03} {cat_id} {height} {width} {rle_mask}\n')

  def run_eval(self, results, save_dir):
    print(f'Saving results in {save_dir}')
    self.save_results(results, save_dir)
    print('Running eval...')
    print('TODO do eval!')
    # TODO run a eval_MOTS_Challenge, get this from the
    # MOTS challenge github dev toolkit