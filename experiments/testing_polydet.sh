cd src
# debugging

python testing.py tracking,polydet --exp_id mots_testing --dataset_version train_val \
--dataset mots --test_dataset mots \
--pre_hm --same_aug --elliptical_gt --debug 2 \
--nbr_points 32 --arch dla_60 \
--load_model ../exp/tracking,polydet/mots_trainval/model_last.pth


#python test.py tracking,polydet --exp_id mots_testing --dataset_version train_val --test_dataset mots --pre_hm --same_aug --elliptical_gt --poly_weight 1 --nbr_points 32 --load_model ../exp/tracking,polydet/mots_testing/model_last.pth

# testing demo on validation set

# python demo.py tracking,polydet --exp_id mots_testing \
# --dataset_version train_val --test_dataset mots \
# --pre_hm --same_aug --elliptical_gt --nbr_points 32 \
# --load_model ../exp/tracking,polydet/mots_trainval/model_last.pth \
# --demo ../data/MOTS/train/MOTS20-09/img1/ --save_video \
# --vis_thresh -1 --arch dla_60

cd ..