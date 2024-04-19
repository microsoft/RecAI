#!/bin/bash

MODEL_NAME=$1
PORT=$2
DATASET=$3

if [ "$DATASET" = "steam" ]; then
  ITEM_INDEX='title'
else
  ITEM_INDEX='title64_t'
fi

GENERAL_LLM=""
OUTPUT_PATH=${MODEL_NAME}
if [ "$MODEL_NAME" = "gpt-3.5-turbo-1106" ]; then
  GENERAL_LLM="--general_llm"
  OUTPUT_PATH=snap/${MODEL_NAME}/${DATASET}/
elif [ "$MODEL_NAME" = "snap/Llama-2-7b-hf-chat/" ]; then
  GENERAL_LLM="--general_llm"
  OUTPUT_PATH=${MODEL_NAME}${DATASET}/
else
  OUTPUT_PATH=${MODEL_NAME}
fi


tasks=(
  "SFTTestSeqRec"
  "SFT+TestPersonalControlRec"
  "SFT-TestPersonalControlRec"
  "SFTTestPersonalCategoryRateLP1_20"
  "SFTTestPersonalCategoryRateEP_30"
  "SFTTestPersonalCategoryRateMP_30"
  "SFTTestItemCount"
)
for task in "${tasks[@]}";
do
  python task_test.py --data_path data/dataset/${DATASET}/ --item_index ${ITEM_INDEX} --SFT_test_task ${task} --model_name ${MODEL_NAME} --output_path ${OUTPUT_PATH} --llama2_chat_template --idx --topk 10 --vllm_port ${PORT} ${GENERAL_LLM}
done
