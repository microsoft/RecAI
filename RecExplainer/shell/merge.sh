# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

cd $HOME/RecAI/RecExplainer

### --model_name_or_path: the path to the original LLM

python ./src/merge.py \
    --output_dir path/to/your/merged/model \
    --cache_dir $HOME/.cache \
    --peft_model_name path/to/your/training/checkpoint \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --rec_model_name_or_path $HOME/RecAI/RecExplainer/data/amazon_video_games_v3/SASRec.pth \
    --task_type both \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --rec_model_type "SASRec"