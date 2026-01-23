#!/bin/bash

set -euo pipefail

DATASET="movies"
DATA_PATH="data/${DATASET}/"


INITIAL_MODEL="path/to/your/initial/model"
OUTPUT_BASE=./sensitivate2/$(date "+%m%d")-${DATASET}-RL

mkdir -p "${OUTPUT_BASE}"
CURRENT_MODEL="${INITIAL_MODEL}"

if [ "$DATASET" = "toys" ]; then
    GPU_ID=0,1,2,5
    MAIN_PORT=13355
elif [ "$DATASET" = "movies" ]; then
    GPU_ID=1,2,5,6
    MAIN_PORT=13356
elif [ "$DATASET" = "steam" ]; then
    GPU_ID=0,1,2,5
    MAIN_PORT=13357
fi

nohup accelerate launch --gpu_ids $GPU_ID --config_file ./accelerate.yaml grpo_train.py \
  --data_path "${DATA_PATH}" \
  --model_path "${CURRENT_MODEL}" \
  --output_dir "${OUTPUT_BASE}" \
  --topk 10 \
  --rl_max_samples 10000 \
  --num_generations 16 \
  --num_train_epochs 2 \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --eval_steps 500 \
  --eval_ratio 0.05 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --beta 1e-3 \
  --reward_type ranking \
  --use_control_symbol \
  --idx \
  --logging_steps 5 \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 > ${OUTPUT_BASE}/output.log 2>&1 &

