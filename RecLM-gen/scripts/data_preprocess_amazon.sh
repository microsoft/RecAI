

DATA_PATH="data/dataset/sub_movie/"
UNIREC_DATA_PATH="unirec/data/sub_movie/"

python preprocess/data_preprocess_amazon.py \
  --full_data_name Movies_TV \
  --meta_file ${DATA_PATH}meta_Movies_and_TV.json.gz \
  --review_file ${DATA_PATH}Movies_and_TV_5.json.gz \
  --data_path ${DATA_PATH} \
  --unirec_data_path ${UNIREC_DATA_PATH} \

