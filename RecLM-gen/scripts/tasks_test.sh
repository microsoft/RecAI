#!/bin/bash

MODEL_NAME=$1
PORT=$2

CHECK=$(echo "$MODEL_NAME" | grep "Steam")
if [ "$CHECK" != "" ]; then
  ITEM_INDEX='title'
  DATASET='steam'
else
  ITEM_INDEX='title64_t'
  DATASET='sub_movie'
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
for t in "${tasks[@]}";
do
  python task_test.py --data_path data/dataset/${DATASET}/ --item_index ${ITEM_INDEX} --SFT_test_task ${t} --model_name ${MODEL_NAME} --llama2_chat_template --idx --topk 10 --vllm_port ${PORT}
done
