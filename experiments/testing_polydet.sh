cd src
# debugging

python testing.py tracking,polydet --exp_id kitti_mots_trainval_hg --dataset_version train_val \
--dataset kitti_mots --test_dataset kitti_mots \
--pre_hm --same_aug --elliptical_gt --debug 2 \
--nbr_points 32  --arch hourglass \
--load_model ../exp/tracking,polydet/kitti_mots_trainval_hg/model_last.pth

#python test.py tracking,polydet --exp_id mots_testing --dataset_version train_val --test_dataset mots --pre_hm --same_aug --elliptical_gt --poly_weight 1 --nbr_points 32 --load_model ../exp/tracking,polydet/mots_testing/model_last.pth

# testing demo on validation set

# python demo.py tracking,polydet \
# --dataset_version train_val --test_dataset kitti_mots --dataset kitti_mots \
# --pre_hm --same_aug --elliptical_gt --nbr_points 32 \
# --load_model ../exp/tracking,polydet/kitti_mots_trainval_hg/model_last.pth \
# --demo ../data/KITTIMOTS/train/0007 \
# --arch hourglass \
# --track_thresh 0.1 --max_age 32 --debug 3
#--save_video

# cd ../results

# ffmpeg -framerate 12 -i demo%d.jpg demo.mp4

cd ..