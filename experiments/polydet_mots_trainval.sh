cd src

# Training

# python main.py tracking,polydet \
# --dataset mots --exp_id mots_trainval_testing_poly_head --dataset_version train_val \
# --num_epochs 100 --val_intervals 1 \
# --hm_disturb 0.05 --lost_disturb 0.5 --fp_disturb 0.1 \
# --pre_hm --same_aug --nbr_points 32 --elliptical_gt \
# --arch hourglass \
# --lr_step 50,75 \
# --head_conv 256 --num_head_conv_poly 3 \
# --batch_size 2 \
# --load_model ../models/ctdet_coco_hg.pth

# Testing

python test.py tracking,polydet \
--dataset mots --exp_id mots_trainval_testing_poly_head --dataset_version train_val \
--test_dataset mots --arch hourglass \
--pre_hm --same_aug --nbr_points 32 --elliptical_gt \
--head_conv 256 --num_head_conv_poly 3 \
--load_model ../exp/tracking,polydet/mots_trainval_testing_poly_head/model_best.pth \
--track_thresh 0.65 --max_age 32

# --load_model ../exp/tracking,polydet/mots_trainval_pre_hg_round/model_best.pth \
cd ..