# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

DATA_DIR="$HOME/RecExplainer/data/amazon_video_games_v3/process_data/"
UNIREC_DATA_DIR="$HOME/UniRec/output/amazon_video_games_v3/SASRec/RecExplainer/xxx"
DATA_NAMES="both"
TASK_TYPE="both"
max_hist_len=9
llm_model_name_or_path="lmsys/vicuna-7b-v1.3"
llm_max_length=1024

cd $HOME/RecExplainer

## infer for interest classification task
accelerate launch --config_file ./shell/config/infer_single_node.yaml ./src/inference.py \
    --data_dir $DATA_DIR \
    --data_names $DATA_NAMES \
    --task_type $TASK_TYPE \
    --sequential_file $DATA_DIR"sequential_data.txt" \
    --cache_dir $HOME/.cache \
    --max_hist_len $max_hist_len \
    --max_example_num_per_dataset 50000 \
    --llm_model_name_or_path $llm_model_name_or_path \
    --rec_model_name_or_path $UNIREC_DATA_DIR/SASRec.pth \
    --per_device_eval_batch_size 4 \
    --llm_max_length $llm_max_length \
    --output_dir $HOME/RecExplainer/output \
    --preprocessing_num_workers=4 \
    --use_slow_tokenizer \
    --use_lora \
    --llm_model_ckpt_path xxx/pytorch_model.bin \
    --max_new_tokens 100 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --inference_mode 'uidiid2binary' \
    --metadata_file $DATA_DIR"metadata.json" \
    --test_top_file $UNIREC_DATA_DIR/test_top.txt

## infer for item ranking task
accelerate launch --config_file ./shell/config/infer_single_node.yaml ./src/inference.py \
    --data_dir $DATA_DIR \
    --data_names $DATA_NAMES \
    --task_type $TASK_TYPE \
    --sequential_file $DATA_DIR"sequential_data.txt" \
    --cache_dir $HOME/.cache \
    --max_hist_len $max_hist_len \
    --max_example_num_per_dataset 50000 \
    --llm_model_name_or_path $llm_model_name_or_path \
    --rec_model_name_or_path $UNIREC_DATA_DIR/SASRec.pth \
    --per_device_eval_batch_size 4 \
    --llm_max_length $llm_max_length \
    --output_dir $HOME/RecExplainer/output \
    --preprocessing_num_workers=4 \
    --use_slow_tokenizer \
    --use_lora \
    --llm_model_ckpt_path xxx/pytorch_model.bin \
    --max_new_tokens 500 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --inference_mode 'uidiid2rank' \
    --metadata_file $DATA_DIR"metadata.json" \
    --test_top_file $UNIREC_DATA_DIR/test_top.txt

## infer for history reconstruction task
accelerate launch --config_file ./shell/config/infer_single_node.yaml ./src/inference.py \
    --data_dir $DATA_DIR \
    --data_names $DATA_NAMES \
    --task_type $TASK_TYPE \
    --sequential_file $DATA_DIR"sequential_data.txt" \
    --cache_dir $HOME/.cache \
    --max_hist_len $max_hist_len \
    --max_example_num_per_dataset 50000 \
    --llm_model_name_or_path $llm_model_name_or_path \
    --rec_model_name_or_path $UNIREC_DATA_DIR/SASRec.pth \
    --per_device_eval_batch_size 4 \
    --llm_max_length $llm_max_length \
    --output_dir $HOME/RecExplainer/output \
    --preprocessing_num_workers=4 \
    --use_slow_tokenizer \
    --use_lora \
    --llm_model_ckpt_path xxx/pytorch_model.bin \
    --max_new_tokens 500 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --inference_mode 'uid2hist' \
    --metadata_file $DATA_DIR"metadata.json" \
    --test_top_file $UNIREC_DATA_DIR/test_top.txt

## infer for next item retrieval task
accelerate launch --config_file ./shell/config/infer_single_node.yaml ./src/inference.py \
    --data_dir $DATA_DIR \
    --data_names $DATA_NAMES \
    --task_type $TASK_TYPE \
    --sequential_file $DATA_DIR"sequential_data.txt" \
    --cache_dir $HOME/.cache \
    --max_hist_len $max_hist_len \
    --max_example_num_per_dataset 50000 \
    --llm_model_name_or_path $llm_model_name_or_path \
    --rec_model_name_or_path $UNIREC_DATA_DIR/SASRec.pth \
    --per_device_eval_batch_size 1 \
    --llm_max_length $llm_max_length \
    --output_dir $HOME/RecExplainer/output \
    --preprocessing_num_workers=4 \
    --use_slow_tokenizer \
    --use_lora \
    --llm_model_ckpt_path xxx/pytorch_model.bin \
    --max_new_tokens 150 \
    --num_beams 5 \
    --num_return_sequences 5 \
    --inference_mode 'uid2next' \
    --metadata_file $DATA_DIR"metadata.json" \
    --test_top_file $UNIREC_DATA_DIR/test_top.txt