# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

DATA_DIR=$HOME/RecAI/RecExplainer/output/amazon_video_games_v3/explan

cd $HOME/RecAI/RecExplainer/preprocess

python eval_explan.py --model_names "recexplainer-B,recexplainer-I,recexplainer-H,llama3,chatgpt" \
    --model_response_files "$DATA_DIR/recexplainer-B_response.csv,$DATA_DIR/recexplainer-I_response.csv,$DATA_DIR/recexplainer-H_response.csv,$DATA_DIR/llama3_response.csv,$DATA_DIR/chatgpt_response.csv" \
    --judge_query_file="$DATA_DIR/judge_query.csv"


python gpt_api.py --input_file $DATA_DIR/judge_query.csv --output_file $DATA_DIR/judge_response.csv \
    --input_columns "model,label,history,target item,question"


python eval_explan.py --model_names "recexplainer-B,recexplainer-I,recexplainer-H,llama3,chatgpt" \
    --judge_response_file="$DATA_DIR/judge_response.csv"