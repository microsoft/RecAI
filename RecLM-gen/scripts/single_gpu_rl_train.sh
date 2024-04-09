#!/bin/bash


BACKBONE="snap/ICR_SubMovie/SFT_Epoch27/"
OUTPUT_PATH_SUFFIX="RL/"

CUDA_VISIBLE_DEVICES=0 python main.py \
  --seed 0 \
  --data_path data/dataset/sub_movie/ \
  --output_path ${BACKBONE}${OUTPUT_PATH_SUFFIX} \
  --backbone ${BACKBONE} \
  --item_index title64_t \
  --batch_size 8 \
  --gradient_accumulation_steps 4 \
  --topk 10 \
  --clip_grad_norm 0.5 \
  --epoch 4 \
  --gen_max_length 512 \
  --train_stage RL \
  --RL_actor_lora_r 4 \
  --RL_critic_lora_r 4 \
  --RL_train_tasks RLSeqRec,RL+PersonalControlRec,RL-PersonalControlRec,RLPersonalCategoryRateLP,RLPersonalCategoryRateMP,RLPersonalCategoryRateEP \
  --RL_val_tasks RLSeqRec,RL+PersonalControlRec,RL-PersonalControlRec,RLPersonalCategoryRateLP_20,RLPersonalCategoryRateMP_30,RLPersonalCategoryRateEP_50,RLItemCount \
  --backup_ip 0.0.0.0 \
  --lr 0.000005 \
  --lora_drop 0.0 \
  --weight_decay 0.0 \
  --kl_coef 0.3 \
  --entropy_weight 0.01 \
  --vf_coef 0.1 \
  --lm_head_full_tune \
  --policy_kl_threshold 0.05 \
  --idx \
  --llama2_chat_template \
  --FA2 \
  --lr_power 2.0 \
  --learn_batch 1 \
  --sample_num 2 \
  --whiten_reward \
  --num_episodes 2 \
  --reward_alpha 0.5 \
  --fine_grain_reward \
  --teacher_port 12621