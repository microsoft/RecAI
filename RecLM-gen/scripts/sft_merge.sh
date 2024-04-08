#!/bin/bash


OUTPUT_PATH="snap/ICR_SubMovie/"
BACKBONE="snap/Llama-2-7b-hf-chat/"
SFT_LOAD="Epoch27_SFT"

# --SFT_load is the model parameter file saved in SFT: snap/ICR_SubMovie/Epoch27_SFT.pth
# need to keep the setting about model params as same as training, such as SFT_actor_lora_r and SFT_actor_lora_a.
# You need to ensure all saved parameters in file perfectly cover the trainable parameters of BaseModel.
CUDA_VISIBLE_DEVICES=0 python main.py \
  --backbone $BACKBONE \
  --train_stage SFT_Merge \
  --SFT_actor_lora_r 16 \
  --SFT_actor_lora_a 8 \
  --output_path ${OUTPUT_PATH} \
  --SFT_load ${OUTPUT_PATH}${SFT_LOAD}
