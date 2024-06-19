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

## Dataset Preprocessing
```
cd RecModel/data_preprocess
python ml1m.py
```

## RecModel Training
```
cd RecModel
python main.py --rank 1 --optimizer Adam --metric ndcg@10,precision@1 --random_seed 2018 --gpu 0 --lr 0.001 --dataset ml1m --model_name RecModel --sample_type random --path ./data
```

```
python main.py --rank 1 --optimizer Adam --metric ndcg@10,precision@1 --random_seed 2018 --gpu 0 --lr 0.001 --dataset ml1m --model_name RecModel --sample_type pop --path ./data
```

## SASRec Model Training
```
cd SASRec
bash run_ml1m.sh
```