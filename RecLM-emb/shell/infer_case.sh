# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

RAW_DATA_DIR=$HOME/RecLM-emb/data/steam/raw_data
EXE_DIR=$HOME/RecLM-emb

cd $EXE_DIR

accelerate launch --config_file ./shell/infer_case.yaml infer_case.py \
    --in_meta_data $RAW_DATA_DIR/metadata.json \
    --user_embedding_prompt_path $EXE_DIR/output/steam_infer_case/user_embedding_prompt.jsonl \
    --model_path_or_name "intfloat/e5-large-v2" \
    --answer_file $EXE_DIR/output/steam_infer_case/intfloat-e5-large-v2/answer.jsonl \
    --topk 10 \
    --seed 2023 \
    --query_max_len 512 \
    --passage_max_len 128 \
    --per_device_eval_batch_size 2 \
    --sentence_pooling_method "mean" \
    --normlized