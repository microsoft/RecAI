# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

DATA_DIR=$HOME/RecExplainer/output/amazon_video_games_v3/explan

cd $HOME/RecExplainer/preprocess

python eval_explan.py --model_names "recexplainer-B,recexplainer-I,recexplainer-H,vicuna,chatgpt" \
    --model_response_files "$DATA_DIR/recexplainer-B_response.csv,$DATA_DIR/recexplainer-I_response.csv,$DATA_DIR/recexplainer-H_response.csv,$DATA_DIR/vicuna_response.csv,$DATA_DIR/chatgpt_response.csv" \
    --judge_query_file="$DATA_DIR/judge_query.csv"


#### you'd better use different keys to simultaneously query the api
export OPENAI_API_VERSION="2023-03-15-preview"
export OPENAI_API_TYPE="azure"
ENGINE=$ENGINE_01 OPENAI_API_KEY=$OPENAI_API_KEY_01 OPENAI_API_BASE=$OPENAI_API_BASE_01 python gpt_api.py --query_file $DATA_DIR/judge_query.csv --response_file $DATA_DIR/judge_response.csv


python eval_explan.py --model_names "recexplainer-B,recexplainer-I,recexplainer-H,vicuna,chatgpt" \
    --judge_response_file="$DATA_DIR/judge_response.csv"