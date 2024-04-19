#!/bin/bash


OUTPUT_PATH="snap/ICR_SubMovie/"
BACKBONE="snap/Llama-2-7b-hf-chat/"

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --gpu_ids all main.py \
  --seed 0 \
  --data_path data/dataset/sub_movie/ \
  --output_path ${OUTPUT_PATH} \
  --backbone ${BACKBONE} \
  --item_index title64_t \
  --batch_size 1 \
  --topk 10 \
  --clip_grad_norm 1.0 \
  --epoch 40 \
  --gen_max_length 512 \
  --lr 0.001 \
  --gradient_accumulation_steps 16 \
  --train_stage SFT \
  --SFT_actor_lora_r 16 \
  --SFT_actor_lora_a 8 \
  --warmup_ratio 0.0125 \
  --val_batch_size 16 \
  --SFT_train_tasks SFTSeqRec,SFTPersonalControlRec,SFTControlRec_re,SFTPersonalCategoryRate,ShareChatGPT \
  --SFT_val_tasks SFTTestSeqRec,SFT+TestPersonalControlRec,SFT-TestPersonalControlRec,SFTTestPersonalCategoryRateEP_50 \
  --backup_ip 0.0.0.0 \
  --val_epoch 0 \
  --share_chat_gpt_ratio 0.5 \
  --FA2 \
  --llama2_chat_template \
  --idx \
  --teacher_port 12621 \
  --distributed
