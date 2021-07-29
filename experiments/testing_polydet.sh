cd src
# debugging

# python testing.py tracking,polydet --exp_id kitti_mots_trainfull_hg --dataset_version train_val \
# --dataset kitti_mots --test_dataset kitti_mots \
# --pre_hm --same_aug --elliptical_gt --debug 3 \
# --nbr_points 32  --arch hourglass \
# --load_model ../exp/tracking,polydet/kitti_mots_trainfull_hg/model_100.pth

#python test.py tracking,polydet --exp_id mots_testing --dataset_version train_val --test_dataset mots --pre_hm --same_aug --elliptical_gt --poly_weight 1 --nbr_points 32 --load_model ../exp/tracking,polydet/mots_testing/model_last.pth

# testing demo on validation set

python demo.py tracking,polydet \
--dataset_version train_val --test_dataset mots --dataset mots \
--pre_hm --same_aug --nbr_points 32 --elliptical_gt \
--load_model ../exp/tracking,polydet/mots_trainval_pre_hg_deep_heads/model_best_3h_2807.pth \
--head_conv 256 --num_head_conv_poly 3 \
--demo ../data/MOTS/train/MOTS20-09/img1 \
--arch hourglass \
--track_thresh 0.6 --max_age 32 --ukf \
--save_video --avg_polys

# --demo ../data/KITTIMOTS/test/0021 \

# cd ../results

# ffmpeg -r 10 -i demo%d.jpg -vcodec libx264 -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" -y -an demo.mp4 -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2"

cd ..