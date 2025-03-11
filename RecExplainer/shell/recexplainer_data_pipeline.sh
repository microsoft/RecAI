# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

RAW_DATA_DIR="$HOME/RecAI/RecExplainer/data/amazon_video_games_v3/raw_data"
PROCESS_DATA_DIR="$HOME/RecAI/RecExplainer/data/amazon_video_games_v3"

MF_PROCESS_DATA_DIR="$HOME/RecAI/RecExplainer/data/mf_amazon_video_games_v3"

gpt_response_file="$PROCESS_DATA_DIR/gpt4_data/test_response.csv"
max_seq_len=9
model_name="meta-llama/Meta-Llama-3-8B-Instruct"
model_max_length=1024

EXE_DIR="$HOME/RecAI/RecExplainer/preprocess"
cd $EXE_DIR


## generate user history summary query
python amazon_generate_v3.py --seqdata_file $PROCESS_DATA_DIR/sequential_data.txt --metadata_file $PROCESS_DATA_DIR/metadata.json \
    --gpt_query_file $PROCESS_DATA_DIR/gpt4_data/test_query.csv --max_seq_len $max_seq_len \
    --model_name $model_name --model_max_length $model_max_length

### gpt4 data generation for user history summary
if [[ -e $gpt_response_file ]]; then  
    echo "All files exist. jump this step." 
else
    echo "generate gpt_response_file"
    
    python gpt_api.py --input_file $PROCESS_DATA_DIR/gpt4_data/test_query.csv --output_file $gpt_response_file

fi

### For SASRec model:

### generate training and testing data for alignment tasks
python amazon_generate_v3.py --sharegpt_file $RAW_DATA_DIR/ShareGPT_V3_unfiltered_cleaned_split.json \
    --seqdata_file $PROCESS_DATA_DIR/sequential_data.txt --metadata_file $PROCESS_DATA_DIR/metadata.json \
    --sim_item_file $PROCESS_DATA_DIR/sim_item.txt --train_top_file $PROCESS_DATA_DIR/train_top.txt --test_top_file $PROCESS_DATA_DIR/test_top.txt \
    --gpt_response_file $gpt_response_file \
    --save_intention_file $PROCESS_DATA_DIR/intention --save_behavior_file $PROCESS_DATA_DIR/behaviour --save_both_file $PROCESS_DATA_DIR/both \
    --max_seq_len $max_seq_len --model_name $model_name --model_max_length $model_max_length

### generate testing data for explanation task
python explan_data_gen.py --data_dir $PROCESS_DATA_DIR --seqdata_file $PROCESS_DATA_DIR/sequential_data.txt --metadata_file $PROCESS_DATA_DIR/metadata.json \
    --test_top_file $PROCESS_DATA_DIR/test_top.txt --max_seq_len $max_seq_len --max_samples 500

### generate training data for explanation task, used to train classifier and score predictor
python explan_data_gen.py --data_dir $PROCESS_DATA_DIR --seqdata_file $PROCESS_DATA_DIR/sequential_data.txt --metadata_file $PROCESS_DATA_DIR/metadata.json \
    --test_top_file $PROCESS_DATA_DIR/train_top.txt --max_seq_len $max_seq_len --max_samples 2000 --split "train"


######################################################
### For MF model:
### We can reuse gpt_response_file/sharegpt_file/seqdata_file/metadata_file from SASRec model.

###  generate training and testing data for alignment tasks
python mf_amazon_video_games_generate.py --sharegpt_file $RAW_DATA_DIR/ShareGPT_V3_unfiltered_cleaned_split.json \
    --seqdata_file $MF_PROCESS_DATA_DIR/sequential_data.txt --metadata_file $MF_PROCESS_DATA_DIR/metadata.json \
    --sim_item_file $MF_PROCESS_DATA_DIR/sim_item.txt --test_top_file $MF_PROCESS_DATA_DIR/test_top.txt \
    --gpt_response_file $gpt_response_file \
    --save_intention_file $MF_PROCESS_DATA_DIR/intention --save_behavior_file $MF_PROCESS_DATA_DIR/behaviour --save_both_file $MF_PROCESS_DATA_DIR/both \
    --max_seq_len $max_seq_len --model_name $model_name --model_max_length $model_max_length

### generate testing data for explanation task
python explan_data_gen.py --data_dir $MF_PROCESS_DATA_DIR --seqdata_file $MF_PROCESS_DATA_DIR/sequential_data.txt --metadata_file $MF_PROCESS_DATA_DIR/metadata.json \
    --test_top_file $MF_PROCESS_DATA_DIR/test_top.txt --max_seq_len $max_seq_len --max_samples 500 --rec_model_type "MF"

### generate training data for explanation task, used to train classifier and score predictor
python explan_data_gen.py --data_dir $MF_PROCESS_DATA_DIR --seqdata_file $MF_PROCESS_DATA_DIR/sequential_data.txt --metadata_file $MF_PROCESS_DATA_DIR/metadata.json \
    --test_top_file $MF_PROCESS_DATA_DIR/test_top.txt --max_seq_len $max_seq_len --max_samples 2000 --split "train" --rec_model_type "MF"
