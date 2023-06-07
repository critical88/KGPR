
<h2 align="center">
Knowledge Graph Pruning for Recommendation
</h2>

This is the Pytorch implementation for KGPR.

## Dependencies
- pytorch==1.11.0
- numpy==1.21.5
- scipy==1.7.3
- torch-scatter==2.0.9
- scikit-learn==0.24.2

## training

### Pruned model training

#### last-fm
```
python main.py --dataset last-fm --is_two_hop 
```

#### yelp2018
```
python main.py --dataset yelp2018 --is_two_hop
```

### Create pruned knowledge graph

edit the `main.py` as following
```
## train()
predict()
```
#### last-fm

```
python main.py --dataset last-fm --is_two_hop --pretrain_model_path=<saved model path>
```

## Training with pruned knowledge graph

- download the SOTA model, such as [KGIN](https://github.com/huangtinglin/Knowledge_Graph_based_Intent_Network/blob/main/modules/KGIN.py), [KGAT](https://github.com/LunaBlack/KGAT-pytorch).
- replace the `kg_final.txt` with our generated file.
- run the base model.
