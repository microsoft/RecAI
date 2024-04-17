

DATA_PATH=$1

python preprocess/data_preprocess_amazon.py \
  --full_data_name Movies_TV \
  --meta_file ${DATA_PATH}meta_Movies_and_TV.json.gz \
  --review_file ${DATA_PATH}Movies_and_TV_5.json.gz \
  --save_sequential_file ${DATA_PATH}sequential.pickle \
  --save_meta_file ${DATA_PATH}meta.pickle \
  --save_map_file ${DATA_PATH}map.pickle \
  --save_category_file ${DATA_PATH}category.pickle
