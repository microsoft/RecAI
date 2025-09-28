# Overview
This is a project to evaluate how various LLMs perform on recommendation tasks, including retrieval, ranking, explanation, conversation, and chatbot ability. The whole workflow is depicted as the following:
![Figure Caption](evaluation_framework.jpg)

# Quick start

1. **Download Steam Data**:  
   [Download link](https://drive.google.com/file/d/1745XoSvkSG2C_1WOFM6PV6DjezrlXa8z/view?usp=drive_link)  
   Unzip to `./data/` folder.

    ```bash
    unzip path_to_downloaded_file.zip -d ./data/
    ```

2. **Navigate to Project**:  
    ```bash
    cd RecLM-eval
    ```

3. **Configure API**:  
   Edit `openai_api_config.yaml`, add your API key:

    ```yaml
    API_BASE: "if-you-have-different-api-url"
    API_KEY: "your-api-key"
    ```

4. **Run**:  
    ```bash
    bash main.sh
    ```

# Usage

## Environment
```bash
conda create -n receval python==3.9
conda activate receval
pip install -r requirements.txt
```

## Set OpenAI API Environment
* If you want to use the OpenAI API, you need to fill in your API key in the openai_api_config.yaml file.
* If you are using models not pre-defined in the project, add their cost information to the api_cost.jsonl file.
## Prepare your test data
* For data preparation details, please refer to [[preprocess]](preprocess/data-preparation.md).
* For you convenience, there is a toy example dataset derived from the Steam dataset (A simple combination of https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data, https://github.com/kang205/SASRec/blob/master/data/Steam.txt and https://www.kaggle.com/datasets/trolukovich/steam-games-complete-dataset). 
* Please download it from (https://drive.google.com/file/d/1745XoSvkSG2C_1WOFM6PV6DjezrlXa8z/view?usp=drive_link) and unzip it to the ./data/ folder.

## Evaluate
* You can specify the evaluation tasks through the `task-names` parameter. 
* These values are avaliable: `ranking`, `retrieval`, `cf_ranking_mc`, `seq_ranking_mc`, `explanation`, `conversation`, `embedding_ranking`, `embedding_retrieval`, `chatbot`.


### Ranking/Retrieval
Parameters:
- `--bench-name`: The name of the dataset
- `--model_path_or_name`: The path or name of the evaluated model.

example:
```bash
python eval.py --task-names ranking retrieval \
    --bench-name steam \
    --model_path_or_name Qwen/Qwen2.5-7B-Instruct
```
optional parameters (only for vllm model):
-  `--max_new_tokens MAX_NEW_TOKENS`: The maximum number of tokens to generate, prompt+max_new_tokens should be less than your model's max length.
-  `--batch_size BATCH_SIZE`: The batch size during inference.

### CF ranking
Parameters:
- `--bench-name`: The name of the dataset
- `--model_path_or_name`: The path or name of the evaluated model.

example:
```bash
python eval.py --task-names cf_ranking_mc \
    --bench-name steam \
    --model_path_or_name Qwen/Qwen2.5-7B-Instruct
```

### Sequential ranking
Parameters:
- `--bench-name`: The name of the dataset
- `--model_path_or_name`: The path or name of the evaluated model.

example:
```bash
python eval.py --task-names seq_ranking_mc \
    --bench-name steam \
    --model_path_or_name Qwen/Qwen2.5-7B-Instruct
```

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
    --model_path_or_name text-embedding-3-small \
    --bench-name steam \
    --user_emb_type title \
    --item_emb_type title

python eval.py --task-names embedding_ranking embedding_retrieval \
    --model_path_or_name text-embedding-3-small \
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
    --model_path_or_name Qwen/Qwen2.5-7B-Instruct \
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
    --model_path_or_name Qwen/Qwen2.5-7B-Instruct \
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
    --model_path_or_name Qwen/Qwen2.5-7B-Instruct \
    --simulator-model gpt-3.5-turbo \
    --max_turn 5
```

Several mainstream open source models were used to evaluate the following tasks:
## Amazon_Fashion · Ranking Evaluation (1000 samples)
Each test instance contains 1 positive item and 19 negatives (20 candidates in total).  The tables report the main top-k ranking metrics as well as four error indicators defined in *RecLM-eval*.

### k = 5

|       Model      | NDCG@5 |  Rec@5 | Hits@5 | Prec@5 |  MAP@5 |  MRR@5 | candidate_error_rate | copy_error | duplicate_error_rate | history_error_rate |
|------------------|--------|--------|--------|--------|--------|--------|----------------------|------------|----------------------|--------------------|
|   Random guess   | 0.1474 | 0.2500 | 0.2500 | 0.0500 | 0.1142 | 0.1142 |            –         |      –     |          –           |         –          |
|   Qwen2.5-3B-it  | 0.1519 | 0.2600 | 0.2600 | 0.0520 | 0.1166 | 0.1166 |          0.001       |    9.085   |        0.001         |       0.016        |
|   gemma3-4B-it   | 0.2257 | 0.3330 | 0.3330 | 0.0666 | 0.1908 | 0.1908 |          0.001       |    8.489   |        0.001         |       0.017        |
|   Llama-3.1-8B-it| 0.2131 | 0.3260 | 0.3260 | 0.0652 | 0.1761 | 0.1761 |          0.001       |    8.935   |        0.001         |       0.016        |
|   Qwen2.5-7B-it  | 0.2180 | 0.3210 | 0.3210 | 0.0642 | 0.1845 | 0.1845 |          0.000       |   10.183   |        0.001         |       0.016        |
|   Qwen3-4B-it    | 0.2427 | 0.3440 | 0.3440 | 0.0688 | 0.2098 | 0.2098 |          0.000       |   16.642   |        0.001         |       0.016        |
|   Qwen2.5-14B-it | 0.3427 | 0.4530 | 0.4530 | 0.0906 | 0.3067 | 0.3067 |          0.001       |    7.438   |        0.001         |       0.017        |

### k = 10

|       Model      |NDCG@10 | Rec@10 | Hits@10 | Prec@10 | MAP@10 | MRR@10 | candidate_error_rate | copy_error | duplicate_error_rate | history_error_rate |
|------------------|--------|--------|---------|---------|--------|--------|----------------------|------------|----------------------|--------------------|
|   Random guess   | 0.2272 | 0.5000 |  0.5000 |  0.0500 | 0.1464 | 0.1464 |            –         |      –     |           –          |           –        |
|   Qwen2.5-3B-it  | 0.2251 | 0.4900 |  0.4900 |  0.0490 | 0.1461 | 0.1461 |          0.001       |    9.085   |        0.001         |       0.016        |
|   gemma3-4B-it   | 0.2992 | 0.5620 |  0.5620 |  0.0562 | 0.2208 | 0.2208 |          0.001       |    8.489   |        0.001         |       0.017        |
|   Llama-3.1-8B-it| 0.2835 | 0.5450 |  0.5450 |  0.0545 | 0.2049 | 0.2049 |          0.001       |    8.935   |        0.001         |       0.016        |
|   Qwen2.5-7B-it  | 0.2841 | 0.5290 |  0.5290 |  0.0529 | 0.2111 | 0.2111 |          0.000       |   10.183   |        0.001         |       0.016        |
|   Qwen3-4B-it    | 0.3100 | 0.5550 |  0.5550 |  0.0555 | 0.2370 | 0.2370 |          0.000       |   16.642   |        0.001         |       0.016        |
|   Qwen2.5-14B-it | 0.4068 | 0.6530 |  0.6530 |  0.0653 | 0.3328 | 0.3328 |          0.001       |    7.438   |        0.001         |       0.017        |

## Amazon_Fashion · cf_ranking_mc Evaluation (1000 samples)
Each test instance contains 1 positive item and 9 negatives (10 candidates in total).  The tables report the metrics acc@1 and one error indicators none_ratio defined in *RecLM-eval*.


|       Model      |      acc@1      |        none_ratio        |
|------------------|-----------------|--------------------------|
|   Random guess   |      0.100      |                          |
|   Qwen2.5-3B-it  |      0.255      |          0.014           |
|   gemma3-4B-it   |      0.265      |          0.000           |
|   Llama-3.1-8B-it|      0.310      |          0.000           |
|   Qwen2.5-7B-it  |      0.333      |          0.014           |
|   Qwen3-4B-it    |      0.340      |          0.000           |
|   Qwen2.5-14B-it |      0.376      |          0.000           |

## Amazon_Fashion · seq_ranking_mc Evaluation (1000 samples)
Each test instance contains 1 positive item and 9 negatives (10 candidates in total).  The tables report the metrics acc@1 and one error indicators none_ratio defined in *RecLM-eval*.

|       Model      |      acc@1      |        none_ratio        |
|------------------|-----------------|--------------------------|
|   Random guess   |      0.100      |                          |
|   Qwen2.5-3B-it  |      0.250      |          0.002           |
|   gemma3-4B-it   |      0.269      |          0.000           |
|   Llama-3.1-8B-it|      0.322      |          0.000           |
|   Qwen2.5-7B-it  |      0.330      |          0.015           |
|   Qwen3-4B-it    |      0.329      |          0.000           |
|   Qwen2.5-14B-it |      0.378      |          0.004           |
