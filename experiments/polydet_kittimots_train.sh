cd src

# Training

python main.py tracking,polydet \
--dataset kitti_mots --exp_id kitti_mots_trainval_hg_deep_head --dataset_version train_full \
--num_epochs 120 \
--hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 \
--pre_hm --same_aug --elliptical_gt --nbr_points 32 \
--arch hourglass \
--save_point 30,50,70,90,110 \
--lr_step 40,70 \
--head_conv 256 --num_head_conv_poly 3 \
--batch_size 4 \
--load_model ../models/ctdet_coco_hg.pth

# Testing

# python test.py tracking,polydet \
# --dataset mots --exp_id mots_trainval_hg_48p_pre --dataset_version train_val \
# --test_dataset mots --arch hourglass \
# --pre_hm --same_aug --elliptical_gt --nbr_points 48 \
# --load_model ../exp/tracking,polydet/mots_trainval_hg_48p_pre/model_last.pth \
# --track_thresh 0.4 --max_age 32

cd ..