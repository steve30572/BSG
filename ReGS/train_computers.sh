while [ true ]; do
  python train_nodeclas.py --dataset Computers --bn --encoder_dropout 0.5 --encoder_channels 128 --hidden_channels 256 --eval_period 10 --mask Edge
done
