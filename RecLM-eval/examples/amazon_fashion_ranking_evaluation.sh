# ----------------------------------------------------------
# ---------------- DATA PREPARATION ----------------
# You can generate samples for one or multiple datasets in a single run.
# Simply list target dataset names in the DATASETS array (they must correspond
# to folder names under the data/ directory). The script will loop through them.
# ----------------------------------------------------------

# ---------------- GPU SELECTION ----------------
# Specify GPU ids here. Use comma-separated values such as "0,1" or "2,3,4,5".
# Leave empty (or comment out) to use all visible GPUs.
GPU_IDS="0"        # ← adjust for your machine
export CUDA_VISIBLE_DEVICES=${GPU_IDS}
# ---------------- TOP-K SETTING ----------------
TOP_K=20        # length of recommendation list
# ------------------------------------------
DATASETS=("Amazon_Fashion")

# Generate sample data for each task
for ds in "${DATASETS[@]}"; do
    echo "[DataGen] Generating samples for dataset: $ds"
    python ./preprocess/generate_data.py \
        --tasks ranking,cf_ranking_mc,seq_ranking_mc \
        --sample_num 1000 \
        --dataset "$ds"
done


# task:  ranking, cf_ranking_mc, seq_ranking_mc
tasks=("ranking" "cf_ranking_mc" "seq_ranking_mc")
for task in "${tasks[@]}"
    do
        echo "Running task: $task"
        python eval.py --task-names $task \
            --bench-name "${DATASETS[@]}" \
            --model_path_or_name Qwen/qwen3-8B \
            --batch_size 256 \
            --top_k ${TOP_K}
    done