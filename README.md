
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

#### amazon-book
```
python main.py --dataset amazon-book --model mask_node_final
```

#### last-fm
```
python main.py --dataset last-fm --model mask_node_final 
```

### Create pruned knowledge graph

```
python main.py --pretrain_model_path=<saved model path>
```

Then you can find the pruned kg in your `<saved model path>/saved_kg`

## Training with pruned knowledge graph

- download the SOTA model, such as [KGIN](https://github.com/huangtinglin/Knowledge_Graph_based_Intent_Network/blob/main/modules/KGIN.py), [KGAT](https://github.com/LunaBlack/KGAT-pytorch).
- replace the `kg_final.txt` with our generated file.
- run the base model.
