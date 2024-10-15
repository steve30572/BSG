values="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"

for value in $values
do
    python train_linkpred.py --dataset Cora --bn --mask Edge --p $value
    python train_linkpred.py --dataset Cora --bn --p $value --alpha 0 --beta 0 --gamma 0
done