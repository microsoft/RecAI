#!/bin/bash


BACKBONE="snap/ICR_SubMovie/SFT_Epoch27/"
OUTPUT_PATH_SUFFIX="RL/"
RL_LOAD="3000step_RL"

# --RL_load is the file saved in RL: snap/ICR_SubMovie/SFT_Epoch27/RL/3000step_RL.pth
# need to keep the setting about model params as same as training, such RL_actor_lora_r, RL_actor_lora_a, RL_critic_lora_r, RL_critic_lora_a and lm_head_full_tune.
CUDA_VISIBLE_DEVICES=0 python main.py \
  --output_path ${BACKBONE}${OUTPUT_PATH_SUFFIX} \
  --backbone $BACKBONE \
  --train_stage RL_Merge \
  --RL_actor_lora_r 4 \
  --RL_actor_lora_a 2 \
  --RL_critic_lora_r 4 \
  --RL_critic_lora_a 2 \
  --RL_load ${BACKBONE}${OUTPUT_PATH_SUFFIX}${RL_LOAD} \
  --lm_head_full_tune \
  --FA2

