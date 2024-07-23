# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

RAW_DATA_DIR="$HOME/RecLM-emb/data/steam/raw_data"
EXE_DIR="$HOME/RecLM-emb"
TRAIN_FLAG="data/steam/train"
OUTPUT_FLAG="data/steam/test"

model_path_or_name="intfloat/e5-large-v2"
max_samples_per_task=50000

in_seq_data="$RAW_DATA_DIR/sequential_data.txt"
in_meta_data="$RAW_DATA_DIR/metadata.json"

out_user2item="$EXE_DIR/$OUTPUT_FLAG/user2item.jsonl"
out_query2item="$EXE_DIR/$OUTPUT_FLAG/query2item.jsonl"
out_title2item="$EXE_DIR/$OUTPUT_FLAG/title2item.jsonl"
out_item2item="$EXE_DIR/$OUTPUT_FLAG/item2item.jsonl"
out_queryuser2item="$EXE_DIR/$OUTPUT_FLAG/queryuser2item.jsonl"
out_misspell2item="$EXE_DIR/$OUTPUT_FLAG/misspell2item.jsonl"
out_sparse_query2item="$EXE_DIR/$OUTPUT_FLAG/sparse_query2item.jsonl"
out_vaguequery2item="$EXE_DIR/$OUTPUT_FLAG/vaguequery2item.jsonl"
out_relativequery2item="$EXE_DIR/$OUTPUT_FLAG/relativequery2item.jsonl"
out_negquery2item="$EXE_DIR/$OUTPUT_FLAG/negquery2item.jsonl"

out_u2i_file="$EXE_DIR/$OUTPUT_FLAG/gpt4/u2i_gpt4.jsonl"
out_q2i_file="$EXE_DIR/$OUTPUT_FLAG/gpt4/q2i_gpt4.jsonl"
out_q2i_misspell_file="$EXE_DIR/$OUTPUT_FLAG/gpt4/q2i_misspell_gpt4.jsonl"
gpt_query_file="$EXE_DIR/$OUTPUT_FLAG/gpt4/query_gpt4"
gpt_response_file="$EXE_DIR/$OUTPUT_FLAG/gpt4/response_gpt4"
out_gpt_summary="$EXE_DIR/$OUTPUT_FLAG/gpt_summary.jsonl"
out_gpt_query="$EXE_DIR/$OUTPUT_FLAG/gpt_query.jsonl"
out_gpt_misspell="$EXE_DIR/$OUTPUT_FLAG/gpt_misspell.jsonl"
out_gpt_summary_query="$EXE_DIR/$OUTPUT_FLAG/gpt_summary_query.jsonl"

cd $EXE_DIR 

echo "generate test data"
python preprocess/gen_test_data.py --in_seq_data $in_seq_data --in_meta_data $in_meta_data --out_user2item $out_user2item \
    --out_query2item $out_query2item --out_title2item $out_title2item --out_item2item $out_item2item --out_queryuser2item $out_queryuser2item \
    --out_misspell2item $out_misspell2item --out_sparse_query2item $out_sparse_query2item --out_vaguequery2item $out_vaguequery2item \
    --out_relativequery2item $out_relativequery2item --out_negquery2item $out_negquery2item \
    --model_path_or_name $model_path_or_name --max_samples_per_task $max_samples_per_task

if [[ -e $gpt_response_file'.csv' ]]; then  
    echo "All files exist. jump to excute test_merge.py"  
else  
    echo "At least one file does not exist." 
    echo "generate gpt_query_file"
    python preprocess/genera_query_file.py --in_seq_data $in_seq_data --in_meta_data $in_meta_data \
        --out_u2i_file $out_u2i_file --out_q2i_file $out_q2i_file --out_q2i_misspell_file $out_q2i_misspell_file \
        --out_query_file $gpt_query_file --task_type "test"

    echo "generate gpt_response_file"

    python preprocess/gpt_api/api.py --input_file $gpt_query_file'.csv' --output_file $gpt_response_file'.csv' 
fi

echo "generate gpt_data_file"
python preprocess/test_merge.py --in_seq_data $in_seq_data --in_meta_data $in_meta_data \
    --in_u2i $out_u2i_file --in_q2i $out_q2i_file --in_q2i_misspell $out_q2i_misspell_file \
    --gpt_path $gpt_response_file --out_gpt_summary $out_gpt_summary --out_gpt_query $out_gpt_query \
    --out_gpt_misspell $out_gpt_misspell --out_gpt_summary_query $out_gpt_summary_query

echo "remove duplicate data"
python preprocess/test_filter.py --train_dir $EXE_DIR/$TRAIN_FLAG --test_dir $EXE_DIR/$OUTPUT_FLAG