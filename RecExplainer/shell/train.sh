# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


### for training SASRec model

export DISABLE_MLFLOW_INTEGRATION=true;
export WANDB_DIR=$HOME/.cache/
export WANDB_PROJECT="RecExplainer"

DATA_DIR="$HOME/RecAI/RecExplainer/data/amazon_video_games_v3"

attn_implementation="flash_attention_2"
model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct"
rec_model_name_or_path=$DATA_DIR/SASRec.pth
rec_model_type="SASRec"
model_max_length=1024
torch_dtype="bfloat16"

train_file=$DATA_DIR/both_train.json
validation_file=$DATA_DIR/both_valid.json
sequential_file=$DATA_DIR/sequential_data.txt
max_hist_len=9
task_type="both"
template_name="llama-3"

output_dir=$DATA_DIR/output/both_flashattn2_llam3-8b_len1024_bf16_lr1e-4_epoch20_batch4_accu4_warmratio0.1_4gpus

cd $HOME/RecAI/RecExplainer

torchrun --nnodes=1 --nproc_per_node 4 --master_port=29501 ./src/sft_training.py \
    --seed 2024 \
    --do_train \
    --do_eval \
    --bf16 \
    --attn_implementation $attn_implementation \
    --rec_model_name_or_path $rec_model_name_or_path \
    --model_name_or_path $model_name_or_path \
    --model_max_length $model_max_length \
    --cache_dir $HOME/.cache \
    --torch_dtype $torch_dtype \
    --train_file $train_file \
    --validation_file $validation_file \
    --sequential_file $sequential_file \
    --max_hist_len $max_hist_len \
    --task_type $task_type \
    --template_name $template_name \
    --rec_model_type $rec_model_type \
    --output_dir $output_dir \
    --learning_rate 1e-4 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --dataloader_drop_last False \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --logging_steps 100 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --report_to wandb \
    --run_name "amazon_video_games_v3_both_flashattn2_llam3-8b_len1024_bf16_lr1e-4_epoch20_batch4_accu4_warmratio0.1_4gpus" > $HOME/RecAI/RecExplainer/training.log 2>&1
    

### for training MF model

export DISABLE_MLFLOW_INTEGRATION=true;
export WANDB_DIR=$HOME/.cache/
export WANDB_PROJECT="RecExplainer"

DATA_DIR="$HOME/RecAI/RecExplainer/data/mf_amazon_video_games_v3"

attn_implementation="flash_attention_2"
model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct"
rec_model_name_or_path=$DATA_DIR/MF.pth
rec_model_type="MF"
model_max_length=1024
torch_dtype="bfloat16"

train_file=$DATA_DIR/both_train.json
validation_file=$DATA_DIR/both_valid.json
sequential_file=$DATA_DIR/sequential_data.txt
max_hist_len=9
task_type="both"
template_name="llama-3"

output_dir=$DATA_DIR/output/both_flashattn2_llam3-8b_len1024_bf16_lr1e-4_epoch20_batch4_accu4_warmratio0.1_4gpus

cd $HOME/RecAI/RecExplainer

torchrun --nnodes=1 --nproc_per_node 4 --master_port=29501 ./src/sft_training.py \
    --seed 2024 \
    --do_train \
    --do_eval \
    --bf16 \
    --attn_implementation $attn_implementation \
    --rec_model_name_or_path $rec_model_name_or_path \
    --model_name_or_path $model_name_or_path \
    --model_max_length $model_max_length \
    --cache_dir $HOME/.cache \
    --torch_dtype $torch_dtype \
    --train_file $train_file \
    --validation_file $validation_file \
    --sequential_file $sequential_file \
    --max_hist_len $max_hist_len \
    --task_type $task_type \
    --template_name $template_name \
    --rec_model_type $rec_model_type \
    --output_dir $output_dir \
    --learning_rate 1e-4 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --dataloader_drop_last False \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --logging_steps 100 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --report_to wandb \
    --run_name "mf_amazon_video_games_v3_both_flashattn2_llam3-8b_len1024_bf16_lr1e-4_epoch20_batch4_accu4_warmratio0.1_4gpus" > $HOME/RecAI/RecExplainer/mf_training.log 2>&1
    