# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash
# pre-train on one locale dataset with feature embedding and text embedding

# root
LOCAL_ROOT="$HOME/UniRec"

MY_DIR=$LOCAL_ROOT
ALL_DATA_ROOT="$LOCAL_ROOT/data"
OUTPUT_ROOT="$LOCAL_ROOT/output"

# default parameters for local run
MODEL_NAME='SASRec' # [AvgHist, AttHist, MF, SVDPlusPlus, GRU4Rec, SASRec, LKNN, MultiVAE]
DATA_TYPE='SeqRecDataset' #AERecDataset BaseDataset SeqRecDataset
DATASET_NAME="amazon_video_games_v3"  #"x-engmt-1m" #"Beauty"   
verbose=2
learning_rate=0.0003112577321994525 # 0.0003
epochs=100
weight_decay=0 #1e-6
dropout_prob=0
loss_type='fullsoftmax' # [bce, bpr, softmax, ccl, fullsoftmax]
distance_type='dot' # [cosine, mlp, dot]
n_sample_neg_train=0  #400
max_seq_len=9
history_mask_mode='autoregressive'
embedding_size=256

cd $MY_DIR
export PYTHONPATH=$PWD


ALL_RESULTS_ROOT="$OUTPUT_ROOT/$DATASET_NAME/$MODEL_NAME"
mkdir -p $ALL_RESULTS_ROOT
### train ###################################
python unirec/main/main.py \
    --config_dir="unirec/config" \
    --model=$MODEL_NAME \
    --dataloader=$DATA_TYPE \
    --dataset=$DATASET_NAME \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
    --output_path=$ALL_RESULTS_ROOT"/RecExplainer" \
    --learning_rate=$learning_rate \
    --dropout_prob=$dropout_prob \
    --use_pre_item_emb=0 \
    --loss_type=$loss_type \
    --max_seq_len=$max_seq_len \
    --has_user_bias=0 \
    --has_item_bias=0 \
    --epochs=$epochs  \
    --batch_size=2048 \
    --n_sample_neg_train=$n_sample_neg_train \
    --n_sample_neg_valid=0 \
    --valid_protocol='one_vs_all' \
    --test_protocol='one_vs_all' \
    --grad_clip_value=10 \
    --weight_decay=$weight_decay \
    --user_history_filename="user_history" \
    --user_history_file_format="user-item_seq"  \
    --history_mask_mode=$history_mask_mode \
    --metrics="['hit@1;5;10;20', 'ndcg@1;5;10;20','mrr@1;5;10;20']" \
    --key_metric="ndcg@10" \
    --shuffle_train=1 \
    --seed=3418 \
    --early_stop=5 \
    --embedding_size=$embedding_size \
    --num_workers=6 \
    --num_workers_test=0 \
    --verbose=$verbose \
    --neg_by_pop_alpha=0 \
    --distance_type=$distance_type \
    --hidden_dropout_prob=0.4954053313841075 \
    --attn_dropout_prob=0.1871672079253271 \
    --scheduler_factor=0.1 \
    --tau=1.0 \
    --optimizer="adamw" \
    --use_wandb=0 \
    --wandb_file=""