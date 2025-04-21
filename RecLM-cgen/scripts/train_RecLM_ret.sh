
MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
DATASET=$1
TRAIN_ARGS=""
DATA_PATH="data/dataset/$DATASET/"
USE_CONTROL_SYMBOL="1"    # in [0, 1]. whether to use control symbol? must be '1' for RecLM-ret and RecLM-cgen.
IDX="1"
EMB_MODEL="BAAI/bge-m3"
MULTI_ROUND_RATIO="0.1"   # between [0, 1]. the ratio of multi round data in train dataset.
EMB_ALPHA="1"             # a float point number, the weight of Loss_ret.
ITEM_EMB="1"              # in {0, 1}. whether to train item emb?
ONLY_ITEM_PROJ="0"        # in [0, 1}. only train project layer?
OUTPUT_PATH="snap/EMB$EMB_ALPHA-$(date "+%m%d")-$DATASET"

if [ "$USE_CONTROL_SYMBOL" = "1" ]; then
  TRAIN_ARGS=$TRAIN_ARGS" --use_control_symbol"
  OUTPUT_PATH=$OUTPUT_PATH"-CS"
fi

if [ "$ITEM_EMB" = "1" ]; then
  TRAIN_ARGS=$TRAIN_ARGS" --item_emb"
  OUTPUT_PATH=$OUTPUT_PATH"-item"
fi

if [ "$ONLY_ITEM_PROJ" = "1" ]; then
  TRAIN_ARGS=$TRAIN_ARGS" --only_item_emb_proj"
  OUTPUT_PATH=$OUTPUT_PATH"-OP"
fi

if [ "$IDX" = "1" ]; then
  TRAIN_ARGS=$TRAIN_ARGS" --idx"
  OUTPUT_PATH=$OUTPUT_PATH"-IDX"
fi

OUTPUT_PATH=$OUTPUT_PATH"-MRA$MULTI_ROUND_RATIO/"
mkdir -p "$OUTPUT_PATH"
echo "$OUTPUT_PATH"

nohup accelerate launch --num_processes=2 --gpu_ids=8,9 --main_process_port 13336 --config_file accelerate.yaml main.py \
--seed 0 \
--data_path $DATA_PATH \
--output $OUTPUT_PATH \
--backbone $MODEL_PATH \
--item_index title_t \
--batch_size 1 \
--topk 10 \
--clip_grad_norm 1.0 \
--epoch 20 \
--gen_max_length 512 \
--max_token_length 512 \
--lr 0.0001 \
--gradient_accumulation_steps 32 \
--train_stage SFT_Embedding \
--SFT_actor_lora_r 16 \
--SFT_actor_lora_a 8 \
--warmup_ratio 0.0125 \
--val_batch_size 8 \
--SFT_train_tasks SFTSeqRec-CS-MR \
--SFT_val_tasks SFTTestSeqRec-MR \
--backup_ip 0.0.0.0 \
--val_epoch 0 \
--multi_round_ratio $MULTI_ROUND_RATIO \
--chat_template llama-3 \
--embedding_model $EMB_MODEL \
--emb_alpha $EMB_ALPHA \
--teacher_port 2068 \
$TRAIN_ARGS > "$OUTPUT_PATH"output_dot.log 2>&1 &
