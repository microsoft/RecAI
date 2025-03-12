
# RecLM-cgen
## Introduction
xxx
   
Our implementation leverages the [`transformers`](https://github.com/huggingface/transformers) library by Hugging Face.  
   

## Intermediate dataset format

To use this repo, you'll need an intermediate dataset comprising at least three files located in `data_path`: `category.jsonl`, `meta.pickle`, and `sequential.jsonl`.

**A volunteer has prepared a copy of data for reproducing the experiments. You can download it from [Google Drive link](https://drive.google.com/file/d/1jZMa0Sx-zVccCpkep5KiY6VXoOdl6PCl/view?usp=drive_link). Thanks [Luuuk12321](https://github.com/Luuuk12321)!**

### category.jsonl
This file contains a dictionary where the keys are category names, and the values are lists of item IDs belonging to those categories.
```json
{
  "category_1": ["item_id_1", "..."], 
  "category_2": ["item_id_i", "..."], 
  "...": "...",
  "category_k": ["item_id_j", "..."]
}
```
### meta.pickle
This file contains a dictionary where the keys are item IDs, and the values are dictionaries with at least one type of item index (such as `title_t`). 
```json
{
  "item_id_1": {"title": "...", "title_t": "...", "description": "..."},
  "item_id_2": {"title": "...", "title_t": "...", "description": "..."},
  "...": "...",
  "item_id_n": {"title": "...", "title_t": "...", "description": "..."}
}
```

### sequential.jsonl
This file contains a dictionary where the keys are user IDs, and the values are lists of item IDs that represent the user's historical interactions in a time-dependent order.  
   
```json
{
  "user_id_1": ["item_id_1", "...", "item_id_x"],
  "...": "...",
  "user_id_m": ["item_id_1", "...", "item_id_y"]
}
```


### Raw dataset preprocess
We provide the code in `preprocess/data_preprocess_amazon.py` to automatically generate the intermediate dataset with above format from the downloaded raw dataset. 

Firstly, download `Movies_and_TV_5.json.gz` and `meta_Movies_and_TV.json.gz` from [Amazon](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/), then place them in `data/dataset/movies/` and run the next command.

Then, change the data path and dataset full name in [./scripts/data_preprocess_amazon.sh](https://github.com/Luuuk12321/RecLM-cgen/blob/main/scripts/data_preprocess_amazon.sh).
```shell
TOKENIZER_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
DATASET_FULL_NAME="Movies_and_TV"
DATASET_NAME="movies"     # used for selecting dataset in subsequent experiments.
DATA_PATH="./data/dataset/${DATASET_NAME}/"
UNIREC_DATA_PATH="./unirec/data/${DATASET_NAME}/"
UNIREC_CONFIG_PATH="./unirec/config/dataset/${DATASET_NAME}.yaml"
```
After that, run the command `./scripts/data_preprocess_amazon.sh`.


## 1. SASRec Server
We utilize the [UniRec](https://github.com/microsoft/UniRec) library to implement the SASRec teacher model and deploy as a server.  

### 1.1. Install UniRec

Clone the UniRec repository and install the necessary packages: 

```shell
git clone https://github.com/microsoft/UniRec.git
pip install --user --upgrade setuptools wheel twine
```

Modify the `unirec/setup.py` file to update the `torch` dependency:  
   
```python  
install_requires = [  
    "torch>=1.10.0,<=1.13.1"  # Change this line to the one below  
    # "torch>=1.10.0,<=2.1.2",  
    "..."  
]  
```  

Continue with the installation:  
   
```shell  
cd UniRec  
python setup.py sdist bdist_wheel  
pip install dist/unirec-*.whl   
```  

### 1.2. SASRec dataset and model
Model parameters and weights are saved in `unirec/output/`.

The dataset files `train.pkl`, `valid.pkl`, `test.pkl`, `user_history.pkl`, `map.pkl`, and `category.jsonl` (as described in the intermediate dataset format) should be placed in `unirec/data/movies/`. 

Use these files to train the SASRec model with the UniRec library.

### 1.3. SASRec model training

Train the model by specifying the dataset name (e.g., `movies`):  

```shell
./scripts/unirec_train.sh movies
```

### 1.4. SASRec Server start

Update the `model_path` in `unirec/async_server.py` to point to the model files:  

```python
model_path = {
    'movies': "unirec/output/movies/SASRec/train/checkpoint_.../SASRec-SASRec-movies.pth",
}
```

Start the server by specifying the dataset name (`movies`), port (`2068`), and number of workers (`1`):  

```shell
./scripts/unirec_serve.sh movies 2068 1
```


## 2. SFT stage

### 2.1. Train dataset format

For the SFT stage, the dataset should be formatted as a `List[List[Dict]]`.  

- Each inner `List[Dict]` represents the training data for a specific epoch.  
- Each `Dict` within the list is an individual training sample containing the keys `"input_text"` and `"output_text"`, which are essential for traditional SFT. 
- Additional keys such as `"task"` and `"input_field_data"` are used to calculate metrics for the domain in question.  
   
```js
[
  [ //Epoch 1
    {"input_text": "...", "output_text": "...", "task": "...", "input_field_data": {"...": "..."}},
    "...",
    {"input_text": "...", "output_text": "...", "task": "...", "input_field_data": {"...": "..."}}
  ],
  [ //Epoch 2
    "..."
  ]
]
```

### 2.2. SFT train

The training dataset is dynamically generated during the `__getitem__` function call of the dataset class. An example script for training can be found at [./scripts/train_RecLM_cgen.sh](https://github.com/Luuuk12321/RecLM-cgen/blob/main/scripts/train_RecLM_cgen.sh) for **RecLM-cgen** and [./scripts/train_RecLM_ret.sh](https://github.com/Luuuk12321/RecLM-cgen/blob/main/scripts/train_RecLM_ret.sh) for **RecLM-ret**.
```shell
./scripts/train_RecLM_cgen.sh movies  # RecLM-cgen
./scripts/train_RecLM_ret.sh movies   # RecLM-ret
```

### 2.3. SFT model merge

Merge the trained models using the script found at [./scripts/run_SFT_merge.sh](https://github.com/Luuuk12321/RecLM-cgen/blob/main/scripts/run_SFT_merge.sh). The merged model will be saved to `snap/.../SFT_Epoch20/`.
```shell
./scripts/run_SFT_merge.sh
```

## 3. RecLM-cgen testing

### 3.1. Recommendation testing
```shell
python task_test.py \
--data_path data/dataset/movies/ \
--SFT_test_task SFTTestSeqRec-MR \
--model_name snap/.../SFT_Epoch20/ \
--gpu cuda:0 \
--use_control_symbol \
--batch_size 16 \
--use_CBS \
--CBS_type 2 \
--topk 10 \
--idx

# setting --data_path to `data/dataset/toys/` for cross-domain evaluation.
```

### 3.2. Multi-round conversation testing
```shell
python task_MR_test.py \
--data_path data/dataset/movies/ \
--SFT_test_task SFTTestSeqRec-CS-MR \
--model_name snap/.../SFT_Epoch20/ \
--gpu cuda:0 \
--use_control_symbol \
--batch_size 8 \
--use_CBS \
--CBS_type 2 \
--topk 10 \
--idx
```

### 3.3. SFT model deploying
```shell
python cli_serve.py \
--model_name snap/.../SFT_Epoch20/ \
--gpu cuda:0
```

## 4. RecLM-ret testing

### 4.1. Recommendation testing
```shell
python main.py \
--seed 0 \
--data_path data/dataset/movies/ \
--SFT_test_task SFTTestSeqRec-MR \
--gpu cuda:0 \
--use_control_symbol \
--test_batch_size 8 \
--topk 10 \
--item_index title_t \
--idx \
--gen_max_length 512 \
--max_token_length 1024 \
--train_stage SFT_Embedding_Test \
--SFT_actor_lora_r 16 \
--SFT_actor_lora_a 8 \
--chat_template llama-3 \
--FA2 \
--backbone meta-llama/Meta-Llama-3-8B-Instruct \
--embedding_model BAAI/bge-m3 \
--SFT_load snap/.../Epoch20_SFT_Embedding
```

### 4.2. Multi-round conversation testing
```shell
python main.py \
--seed 0 \
--data_path data/dataset/movies/ \
--SFT_test_task SFTTestSeqRec-CS-MR \
--gpu cuda:0 \
--use_control_symbol \
--test_batch_size 8 \
--topk 10 \
--item_index title_t \
--idx \
--gen_max_length 512 \
--max_token_length 1024 \
--train_stage SFT_Embedding_Test \
--SFT_actor_lora_r 16 \
--SFT_actor_lora_a 8 \
--chat_template llama-3 \
--FA2 \
--backbone meta-llama/Meta-Llama-3-8B-Instruct \
--embedding_model BAAI/bge-m3 \
--SFT_load snap/.../Epoch20_SFT_Embedding
```

## 5. Customized recommendation domain
You can customize the recommendation domain following the next code.
```python
from train_utils.processor import FastPrefixConstrainedLogitsProcessor, Trie_link
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(...)
model = AutoModelForCausalLM.from_pretrained(...)

in_domain_titles: list[str] = [...]     # customizer domain titles
item_ids = tokenizer.batch_encode_plus(item_list).data['input_ids']

num_beams = 1
# create prefix tree
item_prefix_tree = Trie_link(input_ids, tokenizer)
# create logit processor base on prefix tree
processor = FastPrefixConstrainedLogitsProcessor(
    item_prefix_tree.constrain_search_list, 
    num_beams
)

output = model.generate(
    ...,
    logits_processor=[processor],
    num_beams=num_beams
)
```

## Citation
If you find this project useful in your research, please cite our research paper:

```
xxx
```
