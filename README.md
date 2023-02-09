# Active Learning with Self-Training GNNs

This repository contains the code and data for the paper: Improving Graph Neural Networks by Combining Active Learning with Self-Training.

## Prerequisites

- Python >= 3.7
- Nvidia GPU
- Pytorch >= 1.12
- Pytorch Geometric >= 2.1

## Citation Data

To run STAL on Cora, Citesser or Pubmed use the python scripts inside `/citation` . To run the script:

```
python3 citations.py --dataset cora --method GCN --hidden_channels 128 --lr 0.001 --dropout 0.5 --epochs 200 --splits 5 --use_AL --AL_strategy uncertainty --use_ST --num_pseudos 0.5
python3 citations.py --dataset citeseer --method GCN --hidden_channels 128 --lr 0.001 --dropout 0.5 --epochs 200 --splits 5 --use_AL --AL_strategy uncertainty --use_ST --num_pseudos 0.5
python3 citations.py --dataset pubmed --method GCN --hidden_channels 128 --lr 0.001 --dropout 0.5 --epochs 200 --splits 5 --use_AL --AL_strategy uncertainty --use_ST --num_pseudos 0.5
```


## Arxiv Data

To run STAL on Arxiv use the python scripts inside `/arxiv` . Run the script:

```
python3 arxiv.py --method GCN --hidden_channels 128 --num_layers 2 --lr 0.001 --dropout 0.3 --epochs 300 --runs 5 --use_AL --AL_strategy uncertainty --use_ST --num_pseudos 0.5

```

## Cite
