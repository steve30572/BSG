# while [ true ]; do
#   python train_computers.py --dataset Computers --bn --encoder_dropout 0.5 --encoder_channels 128 --hidden_channels 256 --eval_period 10 --mask Edge --p 0.7
# done

while [ true ]; do
  python train_citeseer.py --dataset Citeseer --bn --l2_normalize --nodeclas_weight_decay 0.1 --lr 0.02 --mask Edge
done
