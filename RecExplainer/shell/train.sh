# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

DATA_DIR="$HOME/RecExplainer/data/amazon_video_games_v3/process_data/"
UNIREC_DATA_DIR="$HOME/UniRec/output/amazon_video_games_v3/SASRec/RecExplainer/xxx"
DATA_NAMES="both"
TASK_TYPE="both"
max_hist_len=9
llm_model_name_or_path="lmsys/vicuna-7b-v1.3"
llm_max_length=1024
output_dir=$HOME/RecExplainer/output/amazon_video_games_v3/both_batch1_len1024_lr1e-4_epoch10_wdecay0_accu8_warmup2000_fp16_slow_lora

cd $HOME/RecExplainer

accelerate launch --config_file ./shell/config/single_node.yaml ./src/train.py \
    --data_dir $DATA_DIR \
    --data_names $DATA_NAMES \
    --task_type $DATA_NAMES \
    --sequential_file $DATA_DIR"sequential_data.txt" \
    --cache_dir $HOME/.cache \
    --max_hist_len $max_hist_len \
    --max_example_num_per_dataset 1000000 \
    --llm_model_name_or_path $llm_model_name_or_path \
    --rec_model_name_or_path $UNIREC_DATA_DIR/SASRec.pth \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --llm_max_length $llm_max_length \
    --output_dir $output_dir \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --weight_decay 0 \
    --gradient_accumulation_steps 8 \
    --num_warmup_steps 2000 \
    --checkpointing_steps 1000000 \
    --log_steps 200 \
    --eval_steps 1000000 \
    --preprocessing_num_workers=0 \
    --load_best_model \
    --use_slow_tokenizer \
    --use_lora