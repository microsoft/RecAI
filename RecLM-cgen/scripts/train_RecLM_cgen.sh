
MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
SFT_TRAIN_TASKS="SFTSeqRec-MR"
DATASET=$1
TRAIN_ARGS=""
DATA_PATH="data/dataset/$DATASET/"
USE_CONTROL_SYMBOL="1"  # in {0, 1}. whether to use control symbol? must be '1' for RecLM-ret and RecLM-cgen.
USE_SCOPE_MASK="1"    # in {0, 1}. whether to use scope mask for RecLM-cgen?
IDX="1"
MULTI_ROUND_RATIO="0.1"     # between [0, 1]. the ratio of multi round data in train dataset.

OUTPUT_PATH="snap/$(date "+%m%d")-$DATASET"

if [ "$USE_CONTROL_SYMBOL" = "1" ]; then
  SFT_TRAIN_TASKS="SFTSeqRec-CS-MR"
  TRAIN_ARGS=$TRAIN_ARGS" --use_control_symbol --use_CBS"
  OUTPUT_PATH=$OUTPUT_PATH"-CS"
fi
OUTPUT_PATH=$OUTPUT_PATH"-MR1"

if [ "$USE_SCOPE_MASK" = "1" ]; then
  TRAIN_ARGS=$TRAIN_ARGS" --use_scope_mask"
  OUTPUT_PATH=$OUTPUT_PATH"-SM$SCOPE_MASK_TYPE"
fi

if [ "$IDX" = "1" ]; then
  TRAIN_ARGS=$TRAIN_ARGS" --idx"
  OUTPUT_PATH=$OUTPUT_PATH"-IDX"
fi

OUTPUT_PATH=$OUTPUT_PATH"-MRA$MULTI_ROUND_RATIO/"
mkdir -p "$OUTPUT_PATH"
echo "$OUTPUT_PATH"

nohup accelerate launch --num_processes=2 --gpu_ids=4,5 --main_process_port 13335 --config_file accelerate.yaml main.py \
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
--train_stage SFT \
--SFT_actor_lora_r 16 \
--SFT_actor_lora_a 8 \
--warmup_ratio 0.0125 \
--val_batch_size 12 \
--SFT_train_tasks $SFT_TRAIN_TASKS \
--SFT_val_tasks SFTTestSeqRec-MR \
--backup_ip 0.0.0.0 \
--val_epoch 0 \
--multi_round_ratio $MULTI_ROUND_RATIO \
--chat_template llama-3 \
--CBS_type 2 \
--teacher_port 2068 \
$TRAIN_ARGS > "$OUTPUT_PATH"output.log 2>&1 &
