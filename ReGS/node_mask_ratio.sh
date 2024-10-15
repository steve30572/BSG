while [ true ]; do
  python train_original.py --dataset Cora --bn --l2_normalize --mask Edge --eval_period 10 --p 0.1 --alpha 0 --beta 0 --gamma 0 --save_path ori.pt
  python train_original.py --dataset Cora --bn --l2_normalize --mask Edge --eval_period 10 --p 0.2 --alpha 0 --beta 0 --gamma 0 --save_path ori.pt
  python train_original.py --dataset Cora --bn --l2_normalize --mask Edge --eval_period 10 --p 0.3 --alpha 0 --beta 0 --gamma 0 --save_path ori.pt
  python train_original.py --dataset Cora --bn --l2_normalize --mask Edge --eval_period 10 --p 0.4 --alpha 0 --beta 0 --gamma 0 --save_path ori.pt
  python train_original.py --dataset Cora --bn --l2_normalize --mask Edge --eval_period 10 --p 0.5 --alpha 0 --beta 0 --gamma 0 --save_path ori.pt
  python train_original.py --dataset Cora --bn --l2_normalize --mask Edge --eval_period 10 --p 0.6 --alpha 0 --beta 0 --gamma 0 --save_path ori.pt
  python train_original.py --dataset Cora --bn --l2_normalize --mask Edge --eval_period 10 --p 0.7 --alpha 0 --beta 0 --gamma 0 --save_path ori.pt
  python train_original.py --dataset Cora --bn --l2_normalize --mask Edge --eval_period 10 --p 0.8 --alpha 0 --beta 0 --gamma 0 --save_path ori.pt
  python train_original.py --dataset Cora --bn --l2_normalize --mask Edge --eval_period 10 --p 0.9 --alpha 0 --beta 0 --gamma 0 --save_path ori.pt
done
