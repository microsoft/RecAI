# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

RAW_DATA_DIR="$HOME/RecAI/RecExplainer/data/amazon_video_games_v3/raw_data"
full_data_name="Video_Games"
meta_file="$RAW_DATA_DIR/meta_Video_Games.json.gz"
review_file="$RAW_DATA_DIR/Video_Games_5.json.gz"
raw_save_data_file="$RAW_DATA_DIR/sequential_data.txt"
raw_save_metadata_file="$RAW_DATA_DIR/metadata.json"
raw_save_datamaps_file="$RAW_DATA_DIR/datamaps.json"

PROCESS_DATA_DIR="$HOME/RecAI/RecExplainer/data/amazon_video_games_v3"
process_save_data_file="$PROCESS_DATA_DIR/sequential_data.txt"
process_save_metadata_file="$PROCESS_DATA_DIR/metadata.json"
process_save_datamaps_file="$PROCESS_DATA_DIR/datamaps.json"
item_thred=2000
user_thred=4000

UNIREC_RAW_DATA_DIR="$HOME/RecAI/RecExplainer/data/unirec_raw_data/amazon_video_games_v3"

EXE_DIR="$HOME/RecAI/RecExplainer/preprocess"
cd $EXE_DIR

python data_preprocess_amazon.py --full_data_name $full_data_name --meta_file $meta_file --review_file $review_file \
    --save_data_file $raw_save_data_file --save_metadata_file $raw_save_metadata_file --save_datamaps_file $raw_save_datamaps_file

python filter_v3.py --seqdata_file $raw_save_data_file --metadata_file $raw_save_metadata_file \
    --save_data_file $process_save_data_file --save_metadata_file $process_save_metadata_file --save_datamaps_file $process_save_datamaps_file \
    --item_thred $item_thred --user_thred $user_thred

python unirec_utils/unirec_split.py --in_seq_data $process_save_data_file --train_file $UNIREC_RAW_DATA_DIR/train.tsv \
    --valid_file $UNIREC_RAW_DATA_DIR/valid.tsv --test_file $UNIREC_RAW_DATA_DIR/test.tsv --hist_file $UNIREC_RAW_DATA_DIR/user_history.tsv \
    --out_train_data $UNIREC_RAW_DATA_DIR/train_ids.csv --out_test_data $UNIREC_RAW_DATA_DIR/test_ids.csv
