# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

RAW_DATA_DIR=$HOME/RecLM-emb/data/steam/raw_data
EXE_DIR=$HOME/RecLM-emb
TEST_DATA_DIR=$EXE_DIR/data/steam/test
PEFT_MODEL_NAME=castorini/repllama-v1-7b-lora-passage
MODEL_PATH_OR_NAME=meta-llama/Llama-2-7b-hf

TOPK="[1, 5, 10, 20]"
SEED=2023
QUERY_MAX_LEN=512
PASSAGE_MAX_LEN=128
SENTENCE_POOLING_METHOD="last"
torch_dtype="bfloat16"


OUT_DIR="$EXE_DIR/output/steam_infer/repllama-v1-7b-lora-passage"
ALL_METRICS_FILE=$OUT_DIR/all_metrics.jsonl

cd $EXE_DIR

CONFIG_FILE=./shell/infer_case.yaml

echo "infer user2item"
accelerate launch --config_file $CONFIG_FILE infer_metrics.py \
    --in_seq_data $RAW_DATA_DIR/sequential_data.txt \
    --in_meta_data $RAW_DATA_DIR/metadata.json \
    --model_path_or_name $MODEL_PATH_OR_NAME \
    --user_embedding_prompt_path $TEST_DATA_DIR/user2item.jsonl \
    --answer_file $OUT_DIR/answer.jsonl \
    --all_metrics_file $ALL_METRICS_FILE \
    --topk "$TOPK" \
    --seed $SEED \
    --query_max_len $QUERY_MAX_LEN \
    --passage_max_len $PASSAGE_MAX_LEN \
    --per_device_eval_batch_size 16 \
    --task_type "user2item" \
    --sentence_pooling_method $SENTENCE_POOLING_METHOD \
    --normlized \
    --torch_dtype $torch_dtype \
    --has_template \
    --peft_model_name $PEFT_MODEL_NAME

echo "infer query2item"
accelerate launch --config_file $CONFIG_FILE infer_metrics.py \
    --in_seq_data $RAW_DATA_DIR/sequential_data.txt \
    --in_meta_data $RAW_DATA_DIR/metadata.json \
    --model_path_or_name $MODEL_PATH_OR_NAME \
    --user_embedding_prompt_path $TEST_DATA_DIR/query2item.jsonl \
    --answer_file $OUT_DIR/answer.jsonl \
    --all_metrics_file $ALL_METRICS_FILE \
    --topk "$TOPK" \
    --seed $SEED \
    --query_max_len $QUERY_MAX_LEN \
    --passage_max_len $PASSAGE_MAX_LEN \
    --per_device_eval_batch_size 16 \
    --task_type "query2item" \
    --sentence_pooling_method $SENTENCE_POOLING_METHOD \
    --normlized \
    --torch_dtype $torch_dtype \
    --has_template \
    --peft_model_name $PEFT_MODEL_NAME

echo "infer sparse_query2item"
accelerate launch --config_file $CONFIG_FILE infer_metrics.py \
    --in_seq_data $RAW_DATA_DIR/sequential_data.txt \
    --in_meta_data $RAW_DATA_DIR/metadata.json \
    --model_path_or_name $MODEL_PATH_OR_NAME \
    --user_embedding_prompt_path $TEST_DATA_DIR/sparse_query2item.jsonl \
    --answer_file $OUT_DIR/answer.jsonl \
    --all_metrics_file $ALL_METRICS_FILE \
    --topk "$TOPK" \
    --seed $SEED \
    --query_max_len $QUERY_MAX_LEN \
    --passage_max_len $PASSAGE_MAX_LEN \
    --per_device_eval_batch_size 16 \
    --task_type "query2item" \
    --sentence_pooling_method $SENTENCE_POOLING_METHOD \
    --normlized \
    --torch_dtype $torch_dtype \
    --has_template \
    --peft_model_name $PEFT_MODEL_NAME

echo "infer title2item"
accelerate launch --config_file $CONFIG_FILE infer_metrics.py \
    --in_seq_data $RAW_DATA_DIR/sequential_data.txt \
    --in_meta_data $RAW_DATA_DIR/metadata.json \
    --model_path_or_name $MODEL_PATH_OR_NAME \
    --user_embedding_prompt_path $TEST_DATA_DIR/title2item.jsonl \
    --answer_file $OUT_DIR/answer.jsonl \
    --all_metrics_file $ALL_METRICS_FILE \
    --topk "$TOPK" \
    --seed $SEED \
    --query_max_len $QUERY_MAX_LEN \
    --passage_max_len $PASSAGE_MAX_LEN \
    --per_device_eval_batch_size 16 \
    --task_type "title2item" \
    --sentence_pooling_method $SENTENCE_POOLING_METHOD \
    --normlized \
    --torch_dtype $torch_dtype \
    --has_template \
    --peft_model_name $PEFT_MODEL_NAME

echo "infer item2item"
accelerate launch --config_file $CONFIG_FILE infer_metrics.py \
    --in_seq_data $RAW_DATA_DIR/sequential_data.txt \
    --in_meta_data $RAW_DATA_DIR/metadata.json \
    --model_path_or_name $MODEL_PATH_OR_NAME \
    --user_embedding_prompt_path $TEST_DATA_DIR/item2item.jsonl \
    --answer_file $OUT_DIR/answer.jsonl \
    --all_metrics_file $ALL_METRICS_FILE \
    --topk "$TOPK" \
    --seed $SEED \
    --query_max_len $QUERY_MAX_LEN \
    --passage_max_len $PASSAGE_MAX_LEN \
    --per_device_eval_batch_size 16 \
    --task_type "item2item" \
    --sentence_pooling_method $SENTENCE_POOLING_METHOD \
    --normlized \
    --torch_dtype $torch_dtype \
    --has_template \
    --peft_model_name $PEFT_MODEL_NAME

echo "infer queryuser2item"
accelerate launch --config_file $CONFIG_FILE infer_metrics.py \
    --in_seq_data $RAW_DATA_DIR/sequential_data.txt \
    --in_meta_data $RAW_DATA_DIR/metadata.json \
    --model_path_or_name $MODEL_PATH_OR_NAME \
    --user_embedding_prompt_path $TEST_DATA_DIR/queryuser2item.jsonl \
    --answer_file $OUT_DIR/answer.jsonl \
    --all_metrics_file $ALL_METRICS_FILE \
    --topk "$TOPK" \
    --seed $SEED \
    --query_max_len $QUERY_MAX_LEN \
    --passage_max_len $PASSAGE_MAX_LEN \
    --per_device_eval_batch_size 16 \
    --task_type "queryuser2item" \
    --sentence_pooling_method $SENTENCE_POOLING_METHOD \
    --normlized \
    --torch_dtype $torch_dtype \
    --has_template \
    --peft_model_name $PEFT_MODEL_NAME

echo "infer misspell2item"
accelerate launch --config_file $CONFIG_FILE infer_metrics.py \
    --in_seq_data $RAW_DATA_DIR/sequential_data.txt \
    --in_meta_data $RAW_DATA_DIR/metadata.json \
    --model_path_or_name $MODEL_PATH_OR_NAME \
    --user_embedding_prompt_path $TEST_DATA_DIR/misspell2item.jsonl \
    --answer_file $OUT_DIR/answer.jsonl \
    --all_metrics_file $ALL_METRICS_FILE \
    --topk "$TOPK" \
    --seed $SEED \
    --query_max_len $QUERY_MAX_LEN \
    --passage_max_len $PASSAGE_MAX_LEN \
    --per_device_eval_batch_size 16 \
    --task_type "misspell2item" \
    --sentence_pooling_method $SENTENCE_POOLING_METHOD \
    --normlized \
    --torch_dtype $torch_dtype \
    --has_template \
    --peft_model_name $PEFT_MODEL_NAME

echo "infer gpt_misspell"
accelerate launch --config_file $CONFIG_FILE infer_metrics.py \
    --in_seq_data $RAW_DATA_DIR/sequential_data.txt \
    --in_meta_data $RAW_DATA_DIR/metadata.json \
    --model_path_or_name $MODEL_PATH_OR_NAME \
    --user_embedding_prompt_path $TEST_DATA_DIR/gpt_misspell.jsonl \
    --answer_file $OUT_DIR/answer.jsonl \
    --all_metrics_file $ALL_METRICS_FILE \
    --topk "$TOPK" \
    --seed $SEED \
    --query_max_len $QUERY_MAX_LEN \
    --passage_max_len $PASSAGE_MAX_LEN \
    --per_device_eval_batch_size 16 \
    --task_type "misspell2item" \
    --sentence_pooling_method $SENTENCE_POOLING_METHOD \
    --normlized \
    --torch_dtype $torch_dtype \
    --has_template \
    --peft_model_name $PEFT_MODEL_NAME

echo "infer gpt_summary"
accelerate launch --config_file $CONFIG_FILE infer_metrics.py \
    --in_seq_data $RAW_DATA_DIR/sequential_data.txt \
    --in_meta_data $RAW_DATA_DIR/metadata.json \
    --model_path_or_name $MODEL_PATH_OR_NAME \
    --user_embedding_prompt_path $TEST_DATA_DIR/gpt_summary.jsonl \
    --answer_file $OUT_DIR/answer.jsonl \
    --all_metrics_file $ALL_METRICS_FILE \
    --topk "$TOPK" \
    --seed $SEED \
    --query_max_len $QUERY_MAX_LEN \
    --passage_max_len $PASSAGE_MAX_LEN \
    --per_device_eval_batch_size 16 \
    --task_type "user2item" \
    --sentence_pooling_method $SENTENCE_POOLING_METHOD \
    --normlized \
    --torch_dtype $torch_dtype \
    --has_template \
    --peft_model_name $PEFT_MODEL_NAME

echo "infer gpt_summary_query"
accelerate launch --config_file $CONFIG_FILE infer_metrics.py \
    --in_seq_data $RAW_DATA_DIR/sequential_data.txt \
    --in_meta_data $RAW_DATA_DIR/metadata.json \
    --model_path_or_name $MODEL_PATH_OR_NAME \
    --user_embedding_prompt_path $TEST_DATA_DIR/gpt_summary_query.jsonl \
    --answer_file $OUT_DIR/answer.jsonl \
    --all_metrics_file $ALL_METRICS_FILE \
    --topk "$TOPK" \
    --seed $SEED \
    --query_max_len $QUERY_MAX_LEN \
    --passage_max_len $PASSAGE_MAX_LEN \
    --per_device_eval_batch_size 16 \
    --task_type "queryuser2item" \
    --sentence_pooling_method $SENTENCE_POOLING_METHOD \
    --normlized \
    --torch_dtype $torch_dtype \
    --has_template \
    --peft_model_name $PEFT_MODEL_NAME

echo "infer gpt_query"
accelerate launch --config_file $CONFIG_FILE infer_metrics.py \
    --in_seq_data $RAW_DATA_DIR/sequential_data.txt \
    --in_meta_data $RAW_DATA_DIR/metadata.json \
    --model_path_or_name $MODEL_PATH_OR_NAME \
    --user_embedding_prompt_path $TEST_DATA_DIR/gpt_query.jsonl \
    --answer_file $OUT_DIR/answer.jsonl \
    --all_metrics_file $ALL_METRICS_FILE \
    --topk "$TOPK" \
    --seed $SEED \
    --query_max_len $QUERY_MAX_LEN \
    --passage_max_len $PASSAGE_MAX_LEN \
    --per_device_eval_batch_size 16 \
    --task_type "title2item" \
    --sentence_pooling_method $SENTENCE_POOLING_METHOD \
    --normlized \
    --torch_dtype $torch_dtype \
    --has_template \
    --peft_model_name $PEFT_MODEL_NAME

echo "infer vaguequery2item"
accelerate launch --config_file $CONFIG_FILE infer_metrics.py \
    --in_seq_data $RAW_DATA_DIR/sequential_data.txt \
    --in_meta_data $RAW_DATA_DIR/metadata.json \
    --model_path_or_name $MODEL_PATH_OR_NAME \
    --user_embedding_prompt_path $TEST_DATA_DIR/vaguequery2item.jsonl \
    --answer_file $OUT_DIR/answer.jsonl \
    --all_metrics_file $ALL_METRICS_FILE \
    --topk "$TOPK" \
    --seed $SEED \
    --query_max_len $QUERY_MAX_LEN \
    --passage_max_len $PASSAGE_MAX_LEN \
    --per_device_eval_batch_size 16 \
    --task_type "vaguequery2item" \
    --sentence_pooling_method $SENTENCE_POOLING_METHOD \
    --normlized \
    --torch_dtype $torch_dtype \
    --has_template \
    --peft_model_name $PEFT_MODEL_NAME

echo "infer relativequery2item"
accelerate launch --config_file $CONFIG_FILE infer_metrics.py \
    --in_seq_data $RAW_DATA_DIR/sequential_data.txt \
    --in_meta_data $RAW_DATA_DIR/metadata.json \
    --model_path_or_name $MODEL_PATH_OR_NAME \
    --user_embedding_prompt_path $TEST_DATA_DIR/relativequery2item.jsonl \
    --answer_file $OUT_DIR/answer.jsonl \
    --all_metrics_file $ALL_METRICS_FILE \
    --topk "$TOPK" \
    --seed $SEED \
    --query_max_len $QUERY_MAX_LEN \
    --passage_max_len $PASSAGE_MAX_LEN \
    --per_device_eval_batch_size 16 \
    --task_type "title2item" \
    --sentence_pooling_method $SENTENCE_POOLING_METHOD \
    --normlized \
    --torch_dtype $torch_dtype \
    --has_template \
    --peft_model_name $PEFT_MODEL_NAME

echo "infer negquery2item"
accelerate launch --config_file $CONFIG_FILE infer_metrics.py \
    --in_seq_data $RAW_DATA_DIR/sequential_data.txt \
    --in_meta_data $RAW_DATA_DIR/metadata.json \
    --model_path_or_name $MODEL_PATH_OR_NAME \
    --user_embedding_prompt_path $TEST_DATA_DIR/negquery2item.jsonl \
    --answer_file $OUT_DIR/answer.jsonl \
    --all_metrics_file $ALL_METRICS_FILE \
    --topk "$TOPK" \
    --seed $SEED \
    --query_max_len $QUERY_MAX_LEN \
    --passage_max_len $PASSAGE_MAX_LEN \
    --per_device_eval_batch_size 16 \
    --task_type "title2item" \
    --sentence_pooling_method $SENTENCE_POOLING_METHOD \
    --normlized \
    --torch_dtype $torch_dtype \
    --has_template \
    --peft_model_name $PEFT_MODEL_NAME






