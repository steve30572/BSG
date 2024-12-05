# BSG

## BSG introduces three loss functions.

## The details of each loss function are included in the model.py 
#### We utilized the implementation of the baseline MaskGAE as our backbone structure.


## Environment

Higher versions should also be available.

- numpy==1.26.3
- torch==2.1.2+cu121
- torch-cluster==1.6.3
- torch_geometric==2.4.0
- torch-scatter==2.1.2
- torch-sparse==0.6.18
- scipy==1.11.4
- texttable==1.7.0

## Newly added hyperparameters


- alpha = $\lambda\_2$
- beta = $\lambda\_1$
- gamma = $\lambda\_3$
- margin = margin $m$


## Reproduction

### Link Prediction

```
python train_link.py --dataset <dataset_name>
```
```py
<dataset_name> = [Cora, Citeseer, Pubmed, Photo, Computers]
```


<img width="403" alt="image" src="https://github.com/user-attachments/assets/030beb73-c022-4e18-b155-968329a48e9a">


### Node Classification
example with Cora
```
python train_node.py --dataset Cora --alpha 0.0002 --beta 0.001 --gamma 0.0009  --margin -0.2 
```
