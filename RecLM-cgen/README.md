
# RecLM-cgen
## Introduction
This project introduces methods for avoid recommending out-of-domain items in LLM-based recsys. It contains the code for implementing two methods in (arXiv preprint arXiv:2505.03336), i.e., RecLM-cgen and RecLM-ret.

**RecLM-cgen** is a generative recommendation framework in the native structure of LLMs. This framework divides the output space of LLMs into item generation and general text generation parts by introducing item control tokens, and simultaneously employs a decoding strategy with prefix tree constraints to prevent the generation of out-of-domain items. RecLM-cgen enables LLMs to acquire the ability to recommend products without sacrificing their original general capabilities. 

The RecLM-cgen framework seamlessly integrates LLMs with recommendation scenarios. Interacting with RecLM-cgen is just like interacting with general LLMs, enabling users to complete recommendation tasks and other general tasks in multi-round conversations.

The pipeline of RecLM-cgen has 4 steps:
1. Preprocessing raw dataset (Section 1)
2. Training teacher model (Section 2.3)
3. Deploying teacher model service (Section 2.4)
4. Training RecLM-cgen (Section 3.1)

This project is mainly contributed by College of Computer Science and Software Engineering, Shenzhen University.

Our implementation leverages the [`transformers`](https://github.com/huggingface/transformers) library by Hugging Face.  

## 1. Raw dataset preprocess
We provide the code in `preprocess/data_preprocess_amazon.py` to automatically generate the intermediate dataset with above format from the downloaded raw dataset. 

Firstly, download `Movies_and_TV_5.json.gz` and `meta_Movies_and_TV.json.gz` from [Amazon](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/), then place them in `data/dataset/movies/` and run the next command.

Then, change the data path and dataset full name in [./scripts/data_preprocess_amazon.sh](scripts/data_preprocess_amazon.sh).
```shell
TOKENIZER_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
DATASET_FULL_NAME="Movies_and_TV"
DATASET_NAME="movies"     # used for selecting dataset in subsequent experiments.
DATA_PATH="./data/dataset/${DATASET_NAME}/"
UNIREC_DATA_PATH="./unirec/data/${DATASET_NAME}/"
UNIREC_CONFIG_PATH="./unirec/config/dataset/${DATASET_NAME}.yaml"
```
After that, run the command `./scripts/data_preprocess_amazon.sh` to generate the intermediate dataset.


### Intermediate dataset format

To use this repo, you'll need an intermediate dataset comprising at least three files located in data_path: `category.jsonl`, `metas.jsonl`, and `sequential.jsonl`.
You can prepare your own dataset in this format to train the model.

**A volunteer has prepared a copy of data for reproducing the experiments. You can download it from [Google Drive link](https://drive.google.com/file/d/1jZMa0Sx-zVccCpkep5KiY6VXoOdl6PCl/view?usp=drive_link), and place each file of it in the respective path. Thanks [Luuuk12321](https://github.com/Luuuk12321)!**

#### category.jsonl
This file contains a dictionary where the keys are category names, and the values are lists of item IDs belonging to those categories.
```json
{
  "category_1": ["item_id_1", "..."], 
  "category_2": ["item_id_i", "..."], 
  "...": "...",
  "category_k": ["item_id_j", "..."]
}
```
#### metas.jsonl
This file contains a dictionary where the keys are item IDs, and the values are dictionaries with at least one field of item index. This field is used for prefix tree construction (such as `title` or `title_t`). 
```json
{
  "item_id_1": {"title": "...", "title_t": "...", "description": "..."},
  "item_id_2": {"title": "...", "title_t": "...", "description": "..."},
  "...": "...",
  "item_id_n": {"title": "...", "title_t": "...", "description": "..."}
}
```

#### sequential.jsonl
This file contains a dictionary where the keys are user IDs, and the values are lists of item IDs that represent the user's historical interactions in a time-dependent order.  
   
```json
{
  "user_id_1": ["item_id_1", "...", "item_id_x"],
  "...": "...",
  "user_id_m": ["item_id_1", "...", "item_id_y"]
}
```


## 2. SASRec Server
We utilize the [UniRec](https://github.com/microsoft/UniRec) library to implement the SASRec teacher model and deploy as a server.  

### 2.1. Install UniRec

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

### 2.2. Unirec dataset for SASRec model training
You need the dataset files `train.pkl`, `valid.pkl`, `test.pkl`, `user_history.pkl`, `map.pkl`, and `category.jsonl` to train SASRec model with UniRec library.

1. After running of `./scripts/data_preprocess_amazon.sh`, these files will be placed in `./unirec/data/movies/`. 

2. If you had prepared the intermediate dataset, these files will be automatically generated according to the intermediate dataset in `./data/dataset/movies/`.


### 2.3. SASRec model training

Train the model by specifying the dataset name (e.g., `movies`):  

```shell
./scripts/unirec_train.sh movies
```
Model parameters and weights are saved in `./unirec/output/`.

### 2.4. SASRec service deploying

Update the `MODEL_PATH` and `DATASET_NAME` in [./scripts/unirec_serve.sh](./scripts/unirec_serve.sh) to point to the model files:  

```python
DATASET_NAME="movies"
MODEL_PATH="./unirec/output/movies/SASRec/train/checkpoint_.../SASRec-SASRec-movies.pth"
```

Start the server by specifying the serve port(`2068`):  

```shell
./scripts/unirec_serve.sh 2068
```


## 3. SFT stage

### 3.1. SFT train

The training dataset is dynamically generated during the `__getitem__` function call of the dataset class. An example script for training can be found at [./scripts/train_RecLM_cgen.sh](scripts/train_RecLM_cgen.sh) for **RecLM-cgen** and [./scripts/train_RecLM_ret.sh](scripts/train_RecLM_ret.sh) for **RecLM-ret**.
```shell
./scripts/train_RecLM_cgen.sh movies  # RecLM-cgen
./scripts/train_RecLM_ret.sh movies   # RecLM-ret
```

### 3.2. SFT model merge

Merge the trained models using the script found at [./scripts/run_SFT_merge.sh](scripts/run_SFT_merge.sh). The merged model will be saved to `snap/.../SFT_Epoch20/`.
```shell
./scripts/run_SFT_merge.sh
```

## 4. RecLM-cgen testing

### 4.1. Recommendation testing
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

### 4.2. Multi-round conversation testing
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

### 4.3. SFT model deploying
```shell
python cli_serve.py \
--model_name snap/.../SFT_Epoch20/ \
--gpu cuda:0
```

## 5. RecLM-ret testing

### 5.1. Recommendation testing
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

### 5.2. Multi-round conversation testing
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

## 6. Build domain item prefix tree for enabling constrained generation
You can customize the recommendation domain and build the domain item prefix tree for enabling constrained generation following the next code.
```python
from train_utils.processor import FastPrefixConstrainedLogitsProcessor, Trie_link
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(...)
tokenizer.soi_token_id = xxx    # specific a <SOI> token
tokenizer.eoi_token_id = xxx    # specific a <EOI> token
model = AutoModelForCausalLM.from_pretrained(...)

in_domain_titles: list[str] = [...]     # customized domain titles
item_ids = tokenizer.batch_encode_plus(in_domain_titles).data['input_ids']

num_beams = 1
# create prefix tree
item_prefix_tree = Trie_link(item_ids, tokenizer)
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
@article{liao2025avoid,
  title={Avoid Recommending Out-of-Domain Items: Constrained Generative Recommendation with LLMs}, 
  author={Liao, Hao and Lu, Wensheng and Lian, Jianxun and Wu, Mingqi and Wang, Shuo and Zhang, Yong and Huang, Yitian and Zhou, Mingyang and Xie, Xing},
  journal={arXiv preprint arXiv:2505.03336}
  year={2025},
}
```
