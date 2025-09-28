export ALLOW_REGENERATE="true"
conda init
conda create -n receval python==3.9
conda activate receval
pip install -r requirements.txt
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
# Control the length of recommendation list (e.g. Recall@K). vllm_models.py and
# other scripts will read this value via --top_k.
TOP_K=20            # ← change this to 10 / 50 … as needed
# ------------------------------------------

# Which dataset(s) to evaluate – list one or more names below.
# Example: ("steam" "Books" "Movies_and_TV")
DATASETS=("steam")

# Generate sample data for each task
for ds in "${DATASETS[@]}"; do
    echo "[DataGen] Generating samples for dataset: $ds"
    python ./preprocess/generate_data.py \
        --tasks ranking,retrieval,cf_ranking_mc,seq_ranking_mc \
        --sample_num 1000 \
        --dataset "$ds"
done


# task:  ranking, retrieval, cf_ranking_mc, seq_ranking_mc, explanation, conversation, embedding_ranking, embedding_retrieval, chatbot  
tasks=("ranking" "retrieval" "cf_ranking_mc" "seq_ranking_mc")
for task in "${tasks[@]}"
    do
        echo "Running task: $task"
        python eval.py --task-names $task \
            --bench-name "${DATASETS[@]}" \
            --model_path_or_name Qwen/Qwen3-8B \
            --batch_size 256 \
            --top_k ${TOP_K}
    done


tasks=("embedding_ranking" )
for task in "${tasks[@]}"
    do
        echo "Running task: $task"
        python eval.py --task-names $task \
            --model_path_or_name text-embedding-ada-002 \
            --bench-name steam \
            --user_emb_type summary \
            --summary-model gpt-35-turbo \
            --item_emb_type title \
            --top_k ${TOP_K}
    done


tasks=("chatbot")
for task in "${tasks[@]}"
    do
        echo "Running task: $task"
        python eval.py --task-names $task \
                --model_path_or_name Qwen/Qwen3-8B \
                --judge-model gpt-4o \
                --baseline-model gpt-35-turbo \
                --top_k ${TOP_K}
    done

tasks=("explanation")
for task in "${tasks[@]}"
    do
        echo "Running task: $task"
        python eval.py --task-names $task \
               --bench-name steam \
                --model_path_or_name Qwen/Qwen3-8B \
                --judge-model gpt-4o \
                --baseline-model gpt-35-turbo \
                --top_k ${TOP_K}
    done

tasks=("conversation")
for task in "${tasks[@]}"
    do
        echo "Running task: $task"
        python eval.py --task-names $task \
            --bench-name steam \
            --model_path_or_name Qwen/Qwen3-8B \
            --simulator-model gpt-35-turbo \
            --max_turn 5 \
            --top_k ${TOP_K}
    done

