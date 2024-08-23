# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

DISCRIMINATOR_DATA_DIR="$HOME/RecExplainer/data/amazon_video_games_v3/discriminator"
EX_DIR=$HOME/RecExplainer/discriminator
cd $EX_DIR

CUDA_VISIBLE_DEVICES=0 python run_cls.py \
    --num_labels 3 \
    --is_regression False \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --train_file $DISCRIMINATOR_DATA_DIR/classification_train.csv \
    --validation_file $DISCRIMINATOR_DATA_DIR/classification_test.csv \
    --model_name_or_path bert-base-uncased \
    --cache_dir $HOME/.cache/ \
    --output_dir $EX_DIR/output/discriminator \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --dataloader_drop_last False \
    --dataloader_num_workers=0 \
    --gradient_accumulation_steps 1 \
    --seed 2024 \
    --logging_steps 20 \
    --save_strategy no \
    --evaluation_strategy steps \
    --eval_steps 375 \
    --warmup_ratio 0.1 \
    --report_to none