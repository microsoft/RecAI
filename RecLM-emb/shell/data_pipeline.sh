# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


RAW_DATA_DIR="$HOME/RecLM-emb/data/steam/raw_data"
EXE_DIR="$HOME/RecLM-emb"
OUTPUT_FLAG="data/steam/train"

in_seq_data="$RAW_DATA_DIR/sequential_data.txt "
in_meta_data="$RAW_DATA_DIR/metadata.json"
# in_search2item="$RAW_DATA_DIR/gpt4_item_info_search/response_gpt4_search_item.jsonl"

out_user2item="$EXE_DIR/$OUTPUT_FLAG/user2item.jsonl"
out_query2item="$EXE_DIR/$OUTPUT_FLAG/query2item.jsonl"
# out_search2item="$EXE_DIR/$OUTPUT_FLAG/searchquery2item.jsonl"
out_title2item="$EXE_DIR/$OUTPUT_FLAG/title2item.jsonl"
out_item2item="$EXE_DIR/$OUTPUT_FLAG/item2item.jsonl"
out_queryuser2item="$EXE_DIR/$OUTPUT_FLAG/queryuser2item.jsonl"
out_misspell2item="$EXE_DIR/$OUTPUT_FLAG/misspell2item.jsonl"
out_vaguequery2item="$EXE_DIR/$OUTPUT_FLAG/vaguequery2item.jsonl"
out_relativequery2item="$EXE_DIR/$OUTPUT_FLAG/relativequery2item.jsonl"
out_negquery2item="$EXE_DIR/$OUTPUT_FLAG/negquery2item.jsonl"
neg_num=7
model_path_or_name="intfloat/e5-large-v2"
out_u2i_file="$EXE_DIR/$OUTPUT_FLAG/gpt4/u2i_gpt4.jsonl"
out_q2i_file="$EXE_DIR/$OUTPUT_FLAG/gpt4/q2i_gpt4.jsonl"
out_q2i_misspell_file="$EXE_DIR/$OUTPUT_FLAG/gpt4/q2i_misspell_gpt4.jsonl"
gpt_query_file="$EXE_DIR/$OUTPUT_FLAG/gpt4/query_gpt4"
gpt_response_file="$EXE_DIR/$OUTPUT_FLAG/gpt4/response_gpt4"
out_gpt="$EXE_DIR/$OUTPUT_FLAG/gpt4_data.jsonl"

cd $EXE_DIR

echo "data processing"
python preprocess/data_process.py --in_seq_data $in_seq_data --in_meta_data $in_meta_data --out_user2item $out_user2item \
    --out_query2item $out_query2item --out_title2item $out_title2item --out_item2item $out_item2item --out_queryuser2item $out_queryuser2item \
    --out_misspell2item $out_misspell2item --out_vaguequery2item $out_vaguequery2item --out_relativequery2item $out_relativequery2item \
    --out_negquery2item $out_negquery2item --neg_num $neg_num --model_path_or_name $model_path_or_name
    # --in_search2item=$in_search2item --out_search2item=$out_search2item 


if [[ -e $gpt_response_file'.csv' ]]; then  
    echo "All files exist. jump to excute merge.py"  
else  
    echo "At least one file does not exist."  
    echo "generate gpt_query_file"
    python preprocess/genera_query_file.py --in_seq_data $in_seq_data --in_meta_data $in_meta_data \
        --out_u2i_file $out_u2i_file --out_q2i_file $out_q2i_file --out_q2i_misspell_file $out_q2i_misspell_file \
        --out_query_file $gpt_query_file

    echo "generate gpt_response_file"

    python preprocess/gpt_api/api.py --input_file $gpt_query_file'.csv' --output_file $gpt_response_file'.csv' 
fi

echo "generate gpt_data_file"
python preprocess/merge.py --in_seq_data $in_seq_data --in_meta_data $in_meta_data \
    --in_u2i $out_u2i_file --in_q2i $out_q2i_file --in_q2i_misspell $out_q2i_misspell_file \
    --gpt_path $gpt_response_file --out_gpt $out_gpt --neg_num $neg_num


out_conv=$EXE_DIR/$OUTPUT_FLAG/gpt4_v2/conv_data.json
out_gpt_conv=$EXE_DIR/$OUTPUT_FLAG/gpt4_v2/gpt_conv_data
out_user_sum=$EXE_DIR/$OUTPUT_FLAG/gpt4_v2/user_sum_data.json
out_gpt_user_sum=$EXE_DIR/$OUTPUT_FLAG/gpt4_v2/gpt_user_sum_data
out_query=$EXE_DIR/$OUTPUT_FLAG/gpt4_v2/query_data.json
out_gpt_query=$EXE_DIR/$OUTPUT_FLAG/gpt4_v2/gpt_query_data
out_neg_query=$EXE_DIR/$OUTPUT_FLAG/gpt4_v2/neg_query_data.json
out_gpt_neg_query=$EXE_DIR/$OUTPUT_FLAG/gpt4_v2/gpt_neg_query_data
out_gpt_v2="$EXE_DIR/$OUTPUT_FLAG/gpt4_data_v2.jsonl"

echo "generate gpt_data_file v2"
python preprocess/data_process_v2.py --in_seq_data $in_seq_data --in_meta_data $in_meta_data \
    --out_conv $out_conv --out_gpt_conv $out_gpt_conv --out_user_sum $out_user_sum --out_gpt_user_sum $out_gpt_user_sum \
    --out_query $out_query --out_gpt_query $out_gpt_query --out_neg_query $out_neg_query --out_gpt_neg_query $out_gpt_neg_query

python preprocess/gpt_api/api.py --input_file $out_gpt_conv'.csv' --output_file $out_gpt_conv'_response.csv' 
python preprocess/gpt_api/api.py --input_file $out_gpt_user_sum'.csv' --output_file $out_gpt_user_sum'_response.csv'
python preprocess/gpt_api/api.py --input_file $out_gpt_query'.csv' --output_file $out_gpt_query'_response.csv'
python preprocess/gpt_api/api.py --input_file $out_gpt_neg_query'.csv' --output_file $out_gpt_neg_query'_response.csv'

python preprocess/merge_v2.py --in_seq_data $in_seq_data --in_meta_data $in_meta_data \
    --out_conv $out_conv --out_gpt_conv $out_gpt_conv'_response.csv' --out_user_sum $out_user_sum --out_gpt_user_sum $out_gpt_user_sum'_response.csv' \
    --out_query $out_query --out_gpt_query $out_gpt_query'_response.csv' --out_neg_query $out_neg_query --out_gpt_neg_query $out_gpt_neg_query'_response.csv' \
    --out_new_gpt $out_gpt_v2 --neg_num $neg_num