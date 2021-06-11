cd src

python main.py tracking,polydet \
--exp_id mots_trainval --dataset_version train_val \
--val_intervals 10 --eval_val --num_epochs 100 \
--hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 \
--pre_hm --same_aug --elliptical_gt --nbr_points 32 \
/

cd ..