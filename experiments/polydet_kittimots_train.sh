cd src

# Training

# python main.py tracking,polydet \
# --dataset kitti_mots --exp_id kitti_mots_trainval_hg_deep_head --dataset_version train_val \
# --num_epochs 120 --val_intervals 1 \
# --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 \
# --pre_hm --same_aug --elliptical_gt --nbr_points 32 \
# --arch hourglass \
# --lr_step 50,90 \
# --head_conv 256 --num_head_conv_poly 3 \
# --batch_size 4 \
# --load_model ../models/ctdet_coco_hg.pth

# Testing

python test.py tracking,polydet \
--dataset kitti_mots --exp_id kitti_mots_trainval_hg_deep_head --dataset_version train_val \
--test_dataset kitti_mots --arch hourglass \
--head_conv 256 --num_head_conv_poly 3 \
--pre_hm --same_aug --elliptical_gt --nbr_points 32 \
--load_model ../exp/tracking,polydet/kitti_mots_trainval_hg_deep_head/model_last.pth \
--track_thresh 0.3 --max_age 32 --ukf --keep_res

cd ..