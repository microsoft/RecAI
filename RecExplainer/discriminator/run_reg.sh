# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

DISCRIMINATOR_DATA_DIR="$HOME/RecExplainer/data/amazon_video_games_v3/discriminator"
EX_DIR=$HOME/RecExplainer/discriminator
cd $EX_DIR

CUDA_VISIBLE_DEVICES=0 python run_cls.py \
    --num_labels 1 \
    --is_regression True \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --train_file $DISCRIMINATOR_DATA_DIR/regression_chatgpt_train.csv \
    --validation_file $DISCRIMINATOR_DATA_DIR/regression_chatgpt_test.csv \
    --model_name_or_path bert-base-uncased \
    --cache_dir $HOME/.cache/ \
    --output_dir $EX_DIR/output/discriminator/regression_chatgpt \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --dataloader_drop_last False \
    --dataloader_num_workers=0 \
    --gradient_accumulation_steps 1 \
    --seed 2024 \
    --logging_steps 25 \
    --save_strategy no \
    --evaluation_strategy steps \
    --eval_steps 125 \
    --warmup_ratio 0.1 \
    --report_to none > $EX_DIR/output/regression_chatgpt.log 2>&1

CUDA_VISIBLE_DEVICES=0 python run_cls.py \
    --num_labels 1 \
    --is_regression True \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --train_file $DISCRIMINATOR_DATA_DIR/regression_recexplainer_train.csv \
    --validation_file $DISCRIMINATOR_DATA_DIR/regression_recexplainer_test.csv \
    --model_name_or_path bert-base-uncased \
    --cache_dir $HOME/.cache/ \
    --output_dir $EX_DIR/output/discriminator/regression_recexplainer \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --dataloader_drop_last False \
    --dataloader_num_workers=0 \
    --gradient_accumulation_steps 1 \
    --seed 2024 \
    --logging_steps 25 \
    --save_strategy no \
    --evaluation_strategy steps \
    --eval_steps 125 \
    --warmup_ratio 0.1 \
    --report_to none > $EX_DIR/output/regression_recexplainer.log 2>&1


CUDA_VISIBLE_DEVICES=0 python run_cls.py \
    --num_labels 1 \
    --is_regression True \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --train_file $DISCRIMINATOR_DATA_DIR/regression_llama3_train.csv \
    --validation_file $DISCRIMINATOR_DATA_DIR/regression_llama3_test.csv \
    --model_name_or_path bert-base-uncased \
    --cache_dir $HOME/.cache/ \
    --output_dir $EX_DIR/output/discriminator/regression_llama3 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --dataloader_drop_last False \
    --dataloader_num_workers=0 \
    --gradient_accumulation_steps 1 \
    --seed 2024 \
    --logging_steps 25 \
    --save_strategy no \
    --evaluation_strategy steps \
    --eval_steps 125 \
    --warmup_ratio 0.1 \
    --report_to none > $EX_DIR/output/regression_llama3.log 2>&1