# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#!/bin/bash
# root
LOCAL_ROOT="$HOME/UniRec"

MY_DIR=$LOCAL_ROOT
ALL_DATA_ROOT="$LOCAL_ROOT/data"

DATASET_NAME="amazon_video_games_v3" 

cd $MY_DIR
export PYTHONPATH=$PWD

model_file="$HOME/UniRec/output/$DATASET_NAME/SASRec/RecExplainer/xxx/SASRec.pth"
output_path="$HOME/UniRec/output/$DATASET_NAME/SASRec/RecExplainer/xxx/"

CUDA_VISIBLE_DEVICES=0 python unirec/main/data4Exp.py \
    --dataset_name="train_ids.csv" \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
    --model_file=$model_file \
    --test_batch_size=100 \
    --user_history_filename="user_history" \
    --user_history_file_format="user-item_seq" \
    --output_path=$output_path"train_top.txt" \
    --sim_item_file=$output_path"sim_item.txt"
### test user ###################################
CUDA_VISIBLE_DEVICES=0 python unirec/main/data4Exp.py \
    --dataset_name="test_ids.csv" \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
    --model_file=$model_file \
    --test_batch_size=100 \
    --user_history_filename="user_history" \
    --user_history_file_format="user-item_seq" \
    --output_path=$output_path"test_top.txt"
