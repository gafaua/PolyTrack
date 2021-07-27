cd src

# Training

python main.py tracking \
--dataset mots --exp_id ct_mots_trainval --dataset_version train_val \
--num_epochs 70 --val_intervals 1 \
--hm_disturb 0.05 --lost_disturb 0.5 --fp_disturb 0.1 \
--pre_hm --same_aug \
--arch hourglass \
--lr_step 45 \
--batch_size 2 \
--load_model ../models/ctdet_coco_hg.pth

# Testing

# CUDA_LAUNCH_BLOCKING=1  python test.py tracking,polydet \
# --dataset mots --exp_id mots_trainval_hg_pre --dataset_version train_val \
# --test_dataset mots --arch hourglass \
# --pre_hm --same_aug --nbr_points 32 --elliptical_gt \
# --load_model ../exp/tracking,polydet/mots_trainval_hg_pre/model_best_67_0207.pth \
# --track_thresh 0.6 --max_age 32 --ukf --avg_polys

# --load_model ../exp/tracking,polydet/mots_trainval_pre_hg_round/model_best.pth \
cd ..