import os

from ..generic_dataset import GenericDataset

class MOTS(GenericDataset):
  num_categories = 1
  default_resolution = [544, 960]
  class_name = ['pedestrian']
  max_objs = 256
  cat_ids = {2:1}

  def __init__(self, opt, split):
    # TODO handle eval split
    data_dir = os.path.join(opt.data_dir, 'MOTS')
    ann_path = os.path.join(data_dir, 'json_gt', '{}_{}.json'.format(split, opt.nbr_points))
    img_dir = os.path.join(data_dir, split)

    print('Using MOTS with {} points polys'.format(opt.nbr_points))

    self.images = None
    super().__init__(opt=opt, split=split, ann_path=ann_path, img_dir=img_dir)

    self.num_samples = len(self.images)
    print('Loaded MOTS {} {} samples'.format(split, self.num_samples))
  
  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    # TODO save results to text file
    ...

  def run_eval(self, results, save_dir):
    # TODO run a eval_MOTS_Challenge, get this from the
    # MOTS challenge github dev toolkit
    ...