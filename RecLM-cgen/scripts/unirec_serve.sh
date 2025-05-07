# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash 

###############################################################################################
### Please modify the following variables according to your device and mission requirements ###
###############################################################################################
###############################################################################################


# default parameters for local run
MODEL_NAME='SASRec' # [ MF, AvgHist, AttHist, SVDPlusPlus, GRU, SASRec, ConvFormer, FASTConvFormer]
loss_type='fullsoftmax' # [bce, bpr, softmax]
DATASET_NAME="movies"
MODEL_PATH="./unirec/output/movies/SASRec/train/checkpoint_.../SASRec-SASRec-movies.pth"
max_seq_len=10
verbose=2

# overall config
DATA_TYPE='SeqRecDataset'  # BaseDataset SeqRecDataset

### serve ###################################
export PYTHONPATH=$PWD

CUDA_VISIBLE_DEVICES=0 python unirec/asyc_server.py \
    --config_dir="unirec/config" \
    --model=$MODEL_NAME \
    --dataloader=$DATA_TYPE \
    --dataset=$DATASET_NAME \
    --model_path=$MODEL_PATH \
    --embedding_size=128 \
    --hidden_size=128 \
    --use_pre_item_emb=0 \
    --loss_type=$loss_type \
    --max_seq_len=$max_seq_len \
    --has_user_bias=0 \
    --has_item_bias=0 \
    --history_mask_mode='autoregressive' \
    --num_workers=1 \
    --verbose=$verbose \
    --use_wandb=0 \
    --exp_name=sub_movie_server \
    --port=$1
# done