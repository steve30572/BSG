while [ true ]; do
  python train_photo.py --dataset Photo --bn --nodeclas_weight_decay 5e-3 --decoder_channels 64 --mask Edge --save_path 3.pt
done
