# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash 

###############################################################################################
### Please modify the following variables according to your device and mission requirements ###
###############################################################################################
UNIREC_ROOT="./unirec"  # path to UniRec
###############################################################################################


# default parameters for local run
ALL_DATA_ROOT="$UNIREC_ROOT/data"
OUTPUT_ROOT="$UNIREC_ROOT/output"
MODEL_NAME='SASRec' # [ MF, AvgHist, AttHist, SVDPlusPlus, GRU, SASRec, ConvFormer, FASTConvFormer]
loss_type='fullsoftmax' # [bce, bpr, softmax]
DATASET_NAME=$1
max_seq_len=10
verbose=2

export PYTHONPATH=$PWD

# overall config
DATA_TYPE='SeqRecDataset'  # BaseDataset SeqRecDataset

# train
learning_rate=0.001
test_protocol='one_vs_all'  # 'one_vs_k' 'one_vs_all' 'session_aware'


exp_name="$MODEL_NAME-$DATASET_NAME"

ALL_RESULTS_ROOT="$OUTPUT_ROOT/$DATASET_NAME/$MODEL_NAME"
mkdir -p $ALL_RESULTS_ROOT
### train ###################################

if [ ! -d "$ALL_DATA_ROOT/$DATASET_NAME" ]; then
  TOKENIZER_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
  python preprocess/transform2unirec.py \
      --data_path "./data/dataset/${DATASET_NAME}/" \
      --unirec_data_path "$ALL_DATA_ROOT/${DATASET_NAME}" \
      --unirec_config_path "${UNIREC_ROOT}/config/dataset/${DATASET_NAME}.yaml" \
      --tokenizer_path $TOKENIZER_PATH
fi


CUDA_VISIBLE_DEVICES=0 python -m unirec.main.main \
    --config_dir="unirec/config" \
    --model=$MODEL_NAME \
    --dataloader=$DATA_TYPE \
    --dataset=$DATASET_NAME \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
    --output_path=$ALL_RESULTS_ROOT"/train" \
    --learning_rate=$learning_rate \
    --dropout_prob=0.0 \
    --embedding_size=128 \
    --hidden_size=128 \
    --use_pre_item_emb=0 \
    --loss_type=$loss_type \
    --max_seq_len=$max_seq_len \
    --has_user_bias=0 \
    --has_item_bias=0 \
    --epochs=100  \
    --early_stop=10 \
    --batch_size=1024 \
    --n_sample_neg_train=0 \
    --neg_by_pop_alpha=1.0 \
    --valid_protocol=$test_protocol \
    --test_protocol=$test_protocol \
    --grad_clip_value=-1 \
    --weight_decay=0.0 \
    --history_mask_mode='autoregressive' \
    --user_history_filename="user_history" \
    --user_history_file_format="user-item_seq" \
    --metrics="['hit@10;20', 'ndcg@10;20']" \
    --key_metric="ndcg@10" \
    --num_workers=4 \
    --num_workers_test=0 \
    --verbose=$verbose \
    --exp_name=$exp_name \
    --use_wandb=0 \
    --shuffle_train 1 \
    --scheduler_factor 0.5 \
    --gpu_id -1
# done