# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

PROCESS_DATA_DIR="$HOME/blob/RecExplainer/amazon_video_games_v3"
UNIREC_DATA_DIR="$HOME/blob/RecExplainer/amazon_video_games_v3"
DISCRIMINATOR_DATA_DIR="$PROCESS_DATA_DIR/discriminator"
EXPLAN_DIR="$HOME/RecExplainer/output/amazon_video_games_v3/explan"

cd $HOME/RecExplainer/discriminator

python data_process.py --top_file $UNIREC_DATA_DIR/train_top.txt --seqdata_file $PROCESS_DATA_DIR/sequential_data.txt --in_gpt_file $EXPLAN_DIR/discriminator_train/chatgpt_response.csv \
    --in_vicuna_file $EXPLAN_DIR/discriminator_train/llama3_response.csv --in_recexplainer_file $EXPLAN_DIR/discriminator_train/recexplainer-H_response.csv \
    --out_cls_file $DISCRIMINATOR_DATA_DIR/classification_train.csv --out_reg_gpt_file $DISCRIMINATOR_DATA_DIR/regression_chatgpt_train.csv \
    --out_reg_vicuna_file $DISCRIMINATOR_DATA_DIR/regression_llama3_train.csv --out_reg_recexplainer_file $DISCRIMINATOR_DATA_DIR/regression_recexplainer_train.csv \
    --split "train" --max_samples 2000

python data_process.py --top_file $UNIREC_DATA_DIR/test_top.txt --seqdata_file $PROCESS_DATA_DIR/sequential_data.txt --in_gpt_file $EXPLAN_DIR/chatgpt_response.csv \
    --in_vicuna_file $EXPLAN_DIR/llama3_response.csv --in_recexplainer_file $EXPLAN_DIR/recexplainer-H_response.csv \
    --out_cls_file $DISCRIMINATOR_DATA_DIR/classification_test.csv --out_reg_gpt_file $DISCRIMINATOR_DATA_DIR/regression_chatgpt_test.csv \
    --out_reg_vicuna_file $DISCRIMINATOR_DATA_DIR/regression_llama3_test.csv --out_reg_recexplainer_file $DISCRIMINATOR_DATA_DIR/regression_recexplainer_test.csv \
    --split "valid" --max_samples 500