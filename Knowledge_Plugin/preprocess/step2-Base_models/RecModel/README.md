# Recommendation Model Baselines

Implementations are referred to the repository for Neural Logic Reasoning (NLR)
https://github.com/rutgerswiselab/NLR or the original paper.

## Environments

```
python==3.7.3
numpy==1.18.1
torch==1.0.1
pandas==0.24.2
scipy==1.3.0
tqdm==4.32.1
scikit_learn==0.23.1
```

## Datasets

- The processed datasets are in  [`../../data`](https://github.com/rutgerswiselab/NLR/tree/master/dataset)

- **ML-100k**: The origin dataset can be found [here](https://grouplens.org/datasets/movielens/100k/). 

- **Amazon Datasets**: The origin dataset can be found [here](http://jmcauley.ucsd.edu/data/amazon/). 

- The codes for processing the data can be found in [`./data_preprocess/`](https://github.com/rutgerswiselab/NLR/tree/master/src/datasets)


## Model Training
```
python main.py --rank 1 --optimizer Adam --metric ndcg@10,precision@1 --random_seed 2018 --gpu 0 --lr 0.001 --dataset ml100k01-1-5 --model_name RecModel
```