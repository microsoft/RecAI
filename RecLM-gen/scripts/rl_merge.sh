#!/bin/bash


# --RL_load is the file saved in RL.
# need to keep the setting about model params as same as training, such RL_actor_lora_r, RL_actor_lora_a, RL_critic_lora_r, RL_critic_lora_a and lm_head_full_tune.
CUDA_VISIBLE_DEVICES=8 python main.py \
  --output snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/RL_Total_train_LM-True_VM-False_NR-20.1_SN-2_Q-False_T6_FG-True_LR-5e-06_LDO-0.0_WD-0.0_KLC-0.3_EW-0.01_RS-False_RW-True_VFC-0.1_KLT-0.05_LRP-2.0_GAMMA-0.99_GAS-4_LB-1_RA_0.5_/ \
  --backbone snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch27/ \
  --train_stage RL_Merge \
  --RL_actor_lora_r 4 \
  --RL_actor_lora_a 2 \
  --RL_critic_lora_r 4 \
  --RL_critic_lora_a 2 \
  --RL_load /home/lws/projects/RecLM-gen/snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/RL_Total_train_LM-True_VM-False_NR-20.1_SN-2_Q-False_T6_FG-True_LR-5e-06_LDO-0.0_WD-0.0_KLC-0.3_EW-0.01_RS-False_RW-True_VFC-0.1_KLT-0.05_LRP-2.0_GAMMA-0.99_GAS-4_LB-1_RA_0.5_/4800step_RL \
  --lm_head_full_tune \
  --FA2

