cd src

#python testing.py tracking,polydet --exp_id mots_testing --pre_hm --same_aug --elliptical_gt --batch_size 8 --debug 1 --nbr_points 32 --load_model ../exp/tracking,polydet/mots_testing/model_last.pth
python test.py tracking,polydet --exp_id mots_testing --dataset_version train_val --test_dataset mots --pre_hm --same_aug --elliptical_gt --poly_weight 1 --nbr_points 32 --load_model ../exp/tracking,polydet/mots_testing/model_last.pth

cd ..