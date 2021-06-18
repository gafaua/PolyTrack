cd src

# Training

python main.py tracking,polydet \
--dataset mots --exp_id mots_trainval_dla34 --dataset_version train_val \
--val_intervals 2 --num_epochs 100 \
--hm_disturb 0.05 --lost_disturb 0.5 --fp_disturb 0.1 \
--pre_hm --same_aug --elliptical_gt --nbr_points 32 \
--lr_step 45,70
--batch_size 8 \

# Testing

python test.py tracking,polydet \
--dataset mots --exp_id mots_trainval_dla34 --dataset_version train_val \
--test_dataset mots \
--pre_hm --same_aug --elliptical_gt --nbr_points 32 \
--load_model ../exp/tracking,polydet/mots_trainval_dla34/model_best.pth \

cd ..