TOKENIZER_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
DATASET_FULL_NAME="Movies_and_TV"
DATASET_NAME="movies"
DATA_PATH="./data/dataset/${DATASET_NAME}/"
UNIREC_DATA_PATH="./unirec/data/${DATASET_NAME}/"
UNIREC_CONFIG_PATH="./unirec/config/dataset/${DATASET_NAME}.yaml"

python preprocess/data_preprocess_amazon.py \
  --full_data_name ${DATASET_FULL_NAME} \
  --meta_file ${DATA_PATH}meta_${DATASET_FULL_NAME}.json.gz \
  --review_file ${DATA_PATH}${DATASET_FULL_NAME}_5.json.gz \
  --data_path ${DATA_PATH} \
  --unirec_data_path ${UNIREC_DATA_PATH} \
  --tokenizer_path ${TOKENIZER_PATH} \
  --unirec_config_path ${UNIREC_CONFIG_PATH} \

