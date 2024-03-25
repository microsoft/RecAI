# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

RAW_DATA_DIR="$HOME/RecExplainer/data/amazon_video_games_v3/raw_data"
PROCESS_DATA_DIR="$HOME/RecExplainer/data/amazon_video_games_v3/process_data"
UNIREC_DATA_DIR="$HOME/UniRec/output/amazon_video_games_v3/SASRec/RecExplainer/xxx/"

gpt_response_file="$PROCESS_DATA_DIR/gpt4_data/test_response"
max_seq_len = 9
model_name = 'lmsys/vicuna-7b-v1.3'
model_max_length=1024

EXE_DIR="$HOME/RecExplainer/preprocess"
cd $EXE_DIR


### generate user history summary query
python amazon_generate_v3.py --gpt_query_file $PROCESS_DATA_DIR/gpt4_data/test_query

### gpt4 data generation for user history summary
if [[ -e $gpt_response_file'_1.csv' && -e $gpt_response_file'_2.csv' ]]; then  
    echo "All files exist. jump this step." 
else
    echo "generate gpt_response_file"
    #### you'd better use different keys to simultaneously query the api
    export OPENAI_API_VERSION="2023-03-15-preview"
    export OPENAI_API_TYPE="azure"

    ENGINE=$ENGINE_01 OPENAI_API_KEY=$OPENAI_API_KEY_01 OPENAI_API_BASE=$OPENAI_API_BASE_01 python gpt_api.py --query_file $gpt_query_file'_1.csv' --response_file $gpt_response_file'_1.csv' &
    PID_01=$! 

    ENGINE=$ENGINE_02 OPENAI_API_KEY=$OPENAI_API_KEY_02 OPENAI_API_BASE=$OPENAI_API_BASE_02 python gpt_api.py --query_file $gpt_query_file'_2.csv' --response_file $gpt_response_file'_2.csv' &
    PID_02=$!

    touch $PROCESS_DATA_DIR/gpt4_data/pid.txt
    echo 'PID_01=' $PID_01 >> $PROCESS_DATA_DIR/gpt4_data/pid.txt
    echo 'PID_02=' $PID_02 >> $PROCESS_DATA_DIR/gpt4_data/pid.txt
    wait 
fi


### generate training and testing data for alignment tasks
python amazon_generate_v3.py --sharegpt_file $RAW_DATA_DIR/ShareGPT_V3_unfiltered_cleaned_split.json \
    --seqdata_file $PROCESS_DATA_DIR/sequential_data.txt --metadata_file $PROCESS_DATA_DIR/metadata.json \
    --sim_item_file $UNIREC_DATA_DIR/sim_item.txt --train_top_file $UNIREC_DATA_DIR/train_top.txt --test_top_file $UNIREC_DATA_DIR/test_top.txt \
    --gpt_response_file1 $gpt_response_file'_1.csv' --gpt_response_file2 $gpt_response_file'_2.csv' \
    --save_intention_file $PROCESS_DATA_DIR/intention --save_behavior_file $PROCESS_DATA_DIR/behaviour --save_both_file $PROCESS_DATA_DIR/both \
    --max_seq_len $max_seq_len --model_name $model_name --model_max_length $model_max_length

### generate testing data for explanation task
python explan_data_gen.py --data_dir $PROCESS_DATA_DIR --seqdata_file $PROCESS_DATA_DIR/sequential_data.txt --metadata_file $PROCESS_DATA_DIR/metadata.json \
    --test_top_file $UNIREC_DATA_DIR/test_top.txt --max_seq_len $max_seq_len --max_samples 500

### generate training data for explanation task, used to train classifier and score predictor
python explan_data_gen.py --data_dir $PROCESS_DATA_DIR/discriminator/explan_train --seqdata_file $PROCESS_DATA_DIR/sequential_data.txt --metadata_file $PROCESS_DATA_DIR/metadata.json \
    --test_top_file $UNIREC_DATA_DIR/train_top.txt --max_seq_len $max_seq_len --max_samples 2000 --split "train"