cd src

# Training

# python main.py tracking,polydet \
# --dataset mots --exp_id mots --dataset_version train_full \
# --num_epochs 70 \
# --hm_disturb 0.05 --lost_disturb 0.5 --fp_disturb 0.1 \
# --pre_hm --same_aug --nbr_points 32 --elliptical_gt \
# --arch hourglass \
# --head_conv 256 --num_head_conv_poly 3 \
# --lr_step 45 \
# --save_point 30,40,50,60 \
# --batch_size 2 \
# --load_model ../models/ctdet_coco_hg.pth

# Testing

python test.py tracking,polydet \
--dataset mots --exp_id mots --dataset_version train_full \
--test_dataset mots --arch hourglass \
--head_conv 256 --num_head_conv_poly 3 \
--pre_hm --same_aug --nbr_points 32 --elliptical_gt \
--load_model ../exp/tracking,polydet/mots/model_last.pth \
--track_thresh 0.4 --max_age 32 --ukf

cd ..