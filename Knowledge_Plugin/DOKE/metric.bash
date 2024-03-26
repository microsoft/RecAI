RESULT_PATH=$1
DATASET=$2
if [ "$DATASET" == "" ]; then
    DATASET="ml1m"
fi

python ../evaluate/_evaluate.py \
    --data_dir ../data/$DATASET \
    --mode "title_ranking" \
    --output_dir $RESULT_PATH \
    > $RESULT_PATH/metric.log
# python ../evaluate/evaluate_error.py --result_path $RESULT_PATH --dataset $DATASET
cat $RESULT_PATH/metric.log