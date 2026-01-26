MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
OUTPUT_PATH="./snap/.../"
LORA_PATH="${OUTPUT_PATH}Epoch20_SFT"


python main.py \
--seed 0 \
--output $OUTPUT_PATH \
--backbone $MODEL_PATH \
--train_stage SFT_Merge \
--SFT_actor_lora_r 16 \
--SFT_actor_lora_a 8 \
--SFT_load $LORA_PATH \
--use_control_symbol \
--gpu cuda:0