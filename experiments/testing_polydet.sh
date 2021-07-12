cd src
# debugging

# python testing.py tracking,polydet --exp_id mots_trainval_hg --dataset_version train_val \
# --dataset mots --test_dataset mots \
# --pre_hm --same_aug --elliptical_gt --debug 2 \
# --nbr_points 32  --arch hourglass \
# --load_model ../exp/tracking,polydet/mots_trainval_hg/model_best.pth

#python test.py tracking,polydet --exp_id mots_testing --dataset_version train_val --test_dataset mots --pre_hm --same_aug --elliptical_gt --poly_weight 1 --nbr_points 32 --load_model ../exp/tracking,polydet/mots_testing/model_last.pth

# testing demo on validation set

python demo.py tracking,polydet \
--dataset_version train_val --test_dataset mots \
--pre_hm --same_aug --elliptical_gt --nbr_points 48 \
--load_model ../exp/tracking,polydet/mots_trainval_hg_48p_pre/model_best.pth \
--demo ../data/MOTS/train/MOTS20-09/img1/ --save_video \
--arch hourglass \
--track_thresh 0.6 --max_age 32 

cd ../results

ffmpeg -framerate 12 -i demo%d.jpg demo.mp4

cd ..