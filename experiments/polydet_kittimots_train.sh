cd src

# Training

python main.py tracking,polydet \
--dataset kitti_mots --exp_id kitti_mots_trainval_hg --dataset_version train_val \
--val_intervals 1 --num_epochs 100 \
--hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 \
--pre_hm --same_aug --elliptical_gt --nbr_points 32 \
--arch hourglass \
--lr_step 45,70 \
--batch_size 2 --resume \

# Testing

# python test.py tracking,polydet \
# --dataset mots --exp_id mots_trainval_hg_48p_pre --dataset_version train_val \
# --test_dataset mots --arch hourglass \
# --pre_hm --same_aug --elliptical_gt --nbr_points 48 \
# --load_model ../exp/tracking,polydet/mots_trainval_hg_48p_pre/model_last.pth \
# --track_thresh 0.4 --max_age 32

cd ..