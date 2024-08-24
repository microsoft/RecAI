# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash

LOCAL_ROOT="$HOME/UniRec"  # path to UniRec

# default parameters for local run
MY_DIR=$LOCAL_ROOT
ALL_DATA_ROOT="$LOCAL_ROOT/data"
OUTPUT_ROOT="$LOCAL_ROOT/output"
cd $MY_DIR
export PYTHONPATH=$PWD

MODEL_NAME='MF'   # SAR, UserCF, SLIM, AdmmSLIM, EASE, MultiVAE
DATA_TYPE='BaseDataset' 
DATASET_NAME="amazon_video_games_v3"
learning_rate=0.004432861648876859  #0.002
epochs=100
weight_decay=0
embedding_size=512
loss_type='softmax' # [bce, bpr, softmax, ccl, fullsoftmax]
distance_type='dot' # [cosine, mlp, dot]
n_sample_neg_train=285  #20  #400

test_protocol='one_vs_all'  #'one_vs_all' 'session_aware' 
user_history_filename='user_history' #'user_history'
user_history_file_format='user-item_seq' #"user-item" 

ALL_RESULTS_ROOT="$OUTPUT_ROOT/$DATASET_NAME/$MODEL_NAME"
mkdir -p $ALL_RESULTS_ROOT 
python unirec/main/main.py \
    --config_dir="unirec/config" \
    --model=$MODEL_NAME \
    --dataloader=$DATA_TYPE \
    --dataset=$DATASET_NAME \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
    --output_path=$ALL_RESULTS_ROOT"/RecExplainer" \
    --learning_rate=$learning_rate \
    --use_pre_item_emb=0 \
    --loss_type=$loss_type \
    --has_user_bias=0 \
    --has_item_bias=0 \
    --epochs=$epochs  \
    --batch_size=128 \
    --n_sample_neg_train=$n_sample_neg_train \
    --n_sample_neg_valid=0 \
    --valid_protocol=$test_protocol \
    --test_protocol=$test_protocol \
    --grad_clip_value=0.7467958582006463 \
    --weight_decay=$weight_decay \
    --history_mask_mode='autoregressive' \
    --user_history_filename=$user_history_filename \
    --user_history_file_format=$user_history_file_format \
    --metrics="['hit@1;5;10;20', 'ndcg@1;5;10;20','mrr@1;5;10;20']" \
    --key_metric="ndcg@10" \
    --shuffle_train=1 \
    --seed=436913 \
    --early_stop=5 \
    --embedding_size=$embedding_size \
    --num_workers=6 \
    --num_workers_test=0 \
    --neg_by_pop_alpha=0 \
    --distance_type=$distance_type \
    --scheduler_factor=0.1 \
    --tau=1.0 \
    --optimizer="adamw" \
    --verbose=2 \
    --use_wandb=0 \
    --wandb_file=""