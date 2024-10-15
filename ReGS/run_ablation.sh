# while [ true ]; do
#   python train_citeseer.py --dataset Citeseer --bn --l2_normalize --nodeclas_weight_decay 0.1  --lr 0.02 --mask Edge --alpha 0 --beta 0
#   python train_citeseer.py --dataset Citeseer --bn --l2_normalize --nodeclas_weight_decay 0.1  --lr 0.02 --mask Edge --beta 0 --gamma 0
#   python train_citeseer.py --dataset Citeseer --bn --l2_normalize --nodeclas_weight_decay 0.1  --lr 0.02 --mask Edge --alpha 0 --gamma 0
# done
# while [ true ]; do
#   python train_physics.py --dataset Physics  --mask Edge --save_path physics.pt
# done
while [ true ]; do
  python train_cs.py --dataset CS  --mask Edge --save_path cs.pt
done