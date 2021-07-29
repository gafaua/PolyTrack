cd src

# Training

# python main.py tracking,polydet \
# --dataset mots --exp_id mots_trainval_pre_hg_deep_heads --dataset_version train_val \
# --num_epochs 100 --val_intervals 1 \
# --hm_disturb 0.05 --lost_disturb 0.5 --fp_disturb 0.1 \
# --pre_hm --same_aug --nbr_points 32 --elliptical_gt \
# --arch hourglass \
# --lr_step 50,75 \
# --save_point 30,40,50,60 \
# --head_conv 256 --num_head_conv_poly 5 \
# --batch_size 2 \
# --load_model ../models/ctdet_coco_hg.pth

# Testing

python test.py tracking,polydet \
--dataset mots --exp_id mots_trainval_pre_hg_deep_heads --dataset_version train_val \
--test_dataset mots --arch hourglass \
--pre_hm --same_aug --nbr_points 32 --elliptical_gt \
--head_conv 256 --num_head_conv_poly 3 \
--load_model ../exp/tracking,polydet/mots_trainval_pre_hg_deep_heads/model_best_3h_2807.pth \
--track_thresh 0.6 --max_age 32 --ukf --avg_polys

# --load_model ../exp/tracking,polydet/mots_trainval_pre_hg_round/model_best.pth \
cd ..