export ALLOW_REGENERATE="true"
conda init
conda create -n receval python==3.9
conda activate receval
pip install -r requirements.txt
# set up env

python ./preprocess/generate_data.py --tasks retrieval,ranking,explanation,conversation,chatbot --sample_num 100 --dataset steam
# set up data


# task:  ranking, retrieval, explanation, conversation, embedding_ranking, embedding_retrieval, chatbot  
tasks=("ranking" "retrieval")
for task in "${tasks[@]}"
    do
        echo "Running task: $task"
        python eval.py --task-names $task \
            --bench-name steam \
            --model_path_or_name NousResearch/Hermes-3-Llama-3.1-8B \
            --batch_size 256
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
            --item_emb_type title
    done


tasks=("chatbot")
for task in "${tasks[@]}"
    do
        echo "Running task: $task"
        python eval.py --task-names $task \
                --model_path_or_name Qwen/Qwen2.5-7B-Instruct \
                --judge-model gpt-4o \
                --baseline-model gpt-35-turbo
    done

tasks=("explanation")
for task in "${tasks[@]}"
    do
        echo "Running task: $task"
        python eval.py --task-names $task \
               --bench-name steam \
                --model_path_or_name Qwen/Qwen2.5-7B-Instruct \
                --judge-model gpt-4o \
                --baseline-model gpt-35-turbo
    done

tasks=("conversation")
for task in "${tasks[@]}"
    do
        echo "Running task: $task"
        python eval.py --task-names $task \
            --bench-name steam \
            --model_path_or_name Qwen/Qwen2.5-7B-Instruct \
            --simulator-model gpt-35-turbo \
            --max_turn 5
    done

