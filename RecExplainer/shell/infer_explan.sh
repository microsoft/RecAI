# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

DATA_DIR="$HOME/RecExplainer/data/amazon_video_games_v3/process_data/"
UNIREC_DATA_DIR="$HOME/UniRec/output/amazon_video_games_v3/SASRec/RecExplainer/xxx"
DATA_NAMES="explan_both"
TASK_TYPE="both"
max_hist_len=9
llm_model_name_or_path="lmsys/vicuna-7b-v1.3"
llm_max_length=1024

cd $HOME/RecExplainer

accelerate launch --config_file ./shell/config/infer.yaml ./src/inference.py \
    --data_dir $DATA_DIR \
    --data_names $DATA_NAMES \
    --task_type $TASK_TYPE \
    --sequential_file $DATA_DIR"sequential_data.txt" \
    --cache_dir $HOME/.cache \
    --max_hist_len $max_hist_len \
    --max_example_num_per_dataset 400000 \
    --llm_model_name_or_path $llm_model_name_or_path \
    --rec_model_name_or_path $UNIREC_DATA_DIR/SASRec.pth \
    --per_device_eval_batch_size 4 \
    --llm_max_length $llm_max_length \
    --output_dir $HOME/RecExplainer/output/amazon_video_games_v3/explan/recexplainer-H_response.csv \
    --preprocessing_num_workers=4 \
    --use_slow_tokenizer \
    --use_lora \
    --llm_model_ckpt_path xxx/pytorch_model.bin \
    --max_new_tokens 500 \
    --min_new_tokens 150 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --inference_mode 'case study' \
    --metadata_file $DATA_DIR"metadata.json" \
    --test_top_file $UNIREC_DATA_DIR/test_top.txt


# accelerate launch --config_file ./shell/config/infer.yaml ./src/inference.py \
#     --data_dir $DATA_DIR \
#     --data_names explan_behaviour \
#     --task_type none \
#     --sequential_file $DATA_DIR"sequential_data.txt" \
#     --cache_dir $HOME/.cache \
#     --max_hist_len $max_hist_len \
#     --max_example_num_per_dataset 400000 \
#     --llm_model_name_or_path $llm_model_name_or_path \
#     --rec_model_name_or_path $UNIREC_DATA_DIR/SASRec.pth \
#     --per_device_eval_batch_size 4 \
#     --llm_max_length $llm_max_length \
#     --output_dir $HOME/RecExplainer/output/amazon_video_games_v3/explan/vicuna1.3_response.csv \
#     --preprocessing_num_workers=4 \
#     --use_slow_tokenizer \
#     --max_new_tokens 500 \
#     --min_new_tokens 150 \
#     --num_beams 1 \
#     --num_return_sequences 1 \
#     --inference_mode 'case study' \
#     --metadata_file $DATA_DIR"metadata.json" \
#     --test_top_file $UNIREC_DATA_DIR/test_top.txt
# # --use_lora \
# # --llm_model_ckpt_path xxx/pytorch_model.bin \