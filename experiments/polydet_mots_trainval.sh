cd src

# Training

# python main.py tracking,polydet \
# --dataset mots --exp_id mots_trainval --dataset_version train_val \
# --val_intervals 5 --eval_val --num_epochs 100 \
# --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 \
# --pre_hm --same_aug --elliptical_gt --nbr_points 32 \
# --batch_size 4 --arch dla_60 \

# Testing

python test.py tracking,polydet \
--dataset mots --exp_id mots_trainval --dataset_version train_val \
--test_dataset mots \
--pre_hm --same_aug --elliptical_gt --nbr_points 32 \
--load_model ../exp/tracking,polydet/mots_trainval/model_last.pth \
--arch dla_60

cd ..