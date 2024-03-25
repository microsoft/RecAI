# Overview
This is a project to evaluate how various LLMs perform on recommendation tasks, including retrieval, ranking, explanation, conversation, and chatbot ability. The whole workflow is depicted as the following:
![Figure Caption](evaluation_framework.jpg)

# Usage

## Environment
```bash
conda create -n receval python==3.8
conda activate receval
pip install -r requirements
```

## Set OpenAI API Environment
If you want to use OpenAI API, you need to fill the content in `openai_api_config.yaml`.

## Prepare your test data
For data preparation details, please refer to [[preprocess]](preprocess/data-preparation.md).
For you convenience, there is a toy example dataset derived from the Steam dataset (A simple combination of https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data, https://github.com/kang205/SASRec/blob/master/data/Steam.txt and https://www.kaggle.com/datasets/trolukovich/steam-games-complete-dataset). Please download it from (https://drive.google.com/file/d/1oliigNX_ACRZupf1maFEkJh_uzl2ZUKm/view?usp=sharing) and unzip it to the ./data/ folder.

## Evaluate
You can specify the evaluation tasks through the `task-names` parameter. These values are avaliable: `ranking`, `retrieval`, `explanation`, `conversation`, `embedding_ranking`, `embedding_retrieval`, `chatbot`.

### Ranking/Retrieval
Parameters:
- `--bench-name`: The name of the dataset
- `--model_path_or_name`: The path or name of the evaluated model.

example:
```bash
python eval.py --task-names ranking retrieval \
    --bench-name steam \
    --model_path_or_name facebook/opt-1.3b
```
optional parameters (only for huggingface model):
-  `--nodes NODES`: The number of nodes for distributed inference
-  `--gpus GPUS`: The number gpus per node.
-  `--nr NR`: Then ranking within the nodes.
-  `--master_port MASTER_PORT`: The port of the master node.
-  `--max_new_tokens MAX_NEW_TOKENS`: The maximum number of tokens to generate, prompt+max_new_tokens should be less than your model's max length.
-  `--batch_size BATCH_SIZE`: The batch size during inference.

### Embedding ranking/retrieval
Parameters:
- `--bench-name`: The name of the dataset
- `--model_path_or_name`: The path or name of the evaluated model.
- `--user_emb_type`: The prompt type for user embedding(title or summary).
- `--item_emb_type`: The prompt type for item embedding(title or description).
- `--summary-model`: The name of the model used to summary user preference.

Example:
```bash
python eval.py --task-names embedding_ranking embedding_retrieval \
    --model_path_or_name text-embedding-ada-002 \
    --bench-name steam \
    --user_emb_type title \
    --item_emb_type title

python eval.py --task-names embedding_ranking embedding_retrieval \
    --model_path_or_name text-embedding-ada-002 \
    --bench-name steam \
    --user_emb_type summary \
    --summary-model gpt-3.5-turbo \
    --item_emb_type title
```

###  chatbot ability
Parameters:
- `--model_path_or_name`: The path or name of the evaluated model.
- `--baseline-model`: The path or name of the model acts as a baseline during pairwise evaluation.
- `--judge-model`: The path or name of the model used to perform judge during pairwise evaluation.

example:
```bash
python eval.py --task-names chatbot \
    --model_path_or_name facebook/opt-1.3b \
    --judge-model gpt-3.5-turbo \
    --baseline-model gpt-3.5-turbo
```

### Explanation
Parameters:
- `--bench-name`: The name of the dataset
- `--model_path_or_name`: The path or name of the evaluated model.
- `--baseline-model`: The path or name of the model acts as a baseline during pairwise evaluation.
- `--judge-model`: The path or name of the model used to perform judge during pairwise evaluation.
```bash
python eval.py --task-names explanation \
    --bench-name steam \
    --model_path_or_name facebook/opt-1.3b \
    --judge-model gpt-3.5-turbo \
    --baseline-model gpt-3.5-turbo
```

### Conversation
Parameters:
- `--bench-name`: The name of the dataset
- `--model_path_or_name`: The path or name of the evaluated model.
- `--simulator-model`: The path or name of the model acts as a user simulator during conversation.
- `--max_turn`: The max turns of the conversation.
example:
```bash
python eval.py --task-names conversation \
    --bench-name steam \
    --model_path_or_name facebook/opt-1.3b \
    --simulator-model gpt-3.5-turbo \
    --max_turn 5
```