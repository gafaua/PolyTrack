cd src

#python testing.py tracking,polydet --pre_hm --same_aug --elliptical_gt --batch_size 8 --debug 1 --load_model ../exp/tracking,polydet/mots_testing/model_last.pth
python testing.py tracking,polydet --exp_id mots_testing --pre_hm --same_aug --elliptical_gt --batch_size 8 --poly_weight 1 --nbr_points 32

cd ..