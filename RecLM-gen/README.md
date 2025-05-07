# RecLM-gen
## Introduction
Welcome to the repository for  [Aligning Large Language Models for Controllable Recommendations](https://arxiv.org/abs/2403.05063). This project aims to finetune a Large Language Model (LLM) using domain-specific item and user behavior data, enabling the LLM to function as a standalone recommender system. The process consists of two main alignment stages:  
   
1. Supervised Fine-tuning (SFT)  
2. Reinforcement Learning (RL)  
   
Our implementation leverages the [`transformers`](https://github.com/huggingface/transformers) library by Hugging Face.  
   

## Intermediate dataset format

To use this repo, you'll need an intermediate dataset comprising at least three files located in `data_path`: `category.pickle`, `meta.pickle`, and `sequential.pickle`. Additionally, `ranking_candidate.pickle` is required for reranking task tests.  

**A volunteer has prepared a copy of data for reproducing the experiments. You can download it from [Google Drive link](https://drive.google.com/file/d/1cfw-KSqEwGF0eB_hm1PUWhUTdloT04Le/view?usp=drive_link). Thanks [Luuuk12321](https://github.com/Luuuk12321)!**

### category.pickle
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
This file contains a dictionary where the keys are item IDs, and the values are dictionaries with at least one type of item index (such as `title`). 
```json
{
  "item_id_1": {"title": "..."},
  "item_id_2": {"title": "..."},
  "...": "...",
  "item_id_n": {"title": "..."}
}
```

### sequential.pickle
This file contains a dictionary where the keys are user IDs, and the values are lists of item IDs that represent the user's historical interactions in a time-dependent order.  
   
```json
{
  "user_id_1": ["item_id_1", "...", "item_id_x"],
  "...": "...",
  "user_id_m": ["item_id_1", "...", "item_id_y"]
}
```

### ranking_candidate.pickle (needed for testing reranking task)
This file contains a dictionary where the keys are user IDs, and the values are lists of 100 randomly chosen negative samples.  
   
```json
{
  "user_id_1": ["item_id_1", "...", "item_id_100"],
  "...": "...",
  "user_id_m": ["item_id_1", "...", "item_id_100"]
}
```

### Raw dataset preprocess
We provide the code in `preprocess/data_preprocess_amazon.py` to automatically generate the intermediate dataset with above format from the downloaded raw dataset. 

Firstly, download `Movies_and_TV_5.json.gz` and `meta_Movies_and_TV.json.gz` from [Amazon](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/), then place them in `data/dataset/sub_movie/` and run the next command.
```shell
./scripts/data_preprocess_amazon.sh data/dataset/sub_movie/
```

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

The dataset files `train.pkl`, `valid.pkl`, `test.pkl`, `user_history.pkl`, `map.pkl`, and `category.pickle` (as described in the intermediate dataset format) should be placed in `unirec/data/sub_movie/`. 

Use these files to train the SASRec model with the UniRec library.

### 1.3. SASRec model training

Train the model by specifying the dataset name (e.g., `sub_movie`):  

```shell
./scripts/unirec_train.sh sub_movie
```

### 1.4. SASRec Server start

Update the `model_path` in `unirec/async_server.py` to point to the model files:  

```python
model_path = {
    'sub_movie': "unirec/output/sub_movie/SASRec/train/checkpoint_2024-03-17_014803_35/SASRec-SASRec-sub_movie.pth",
    'steam': "unirec/output/steam/SASRec/train/checkpoint_2024-03-17_014033_93/SASRec-SASRec-steam.pth",
}
```

Start the server by specifying the dataset name (`sub_movie`), port (`12621`), and number of workers (`1`):  

```shell
./scripts/unirec_serve.sh sub_movie 12621 1
```

To expedite dataset preparation, increase the number of workers (e.g., `4`).  

## 2. SFT stage

### 2.1. Dataset format

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

### 2.2. Dataset prepare
   
Prepare the dataset and save it to `{data_path}/SFT_dataset_train.pickle` for training and `{data_path}/SFT_dataset_val.pickle` for validation:  
   
```shell
python data_process.py 
--seed 0 
--data_path data/dataset/sub_movie/ 
--item_index title64_t 
--topk 10 
--epoch 10 
--train_stage SFT 
--SFT_train_tasks SFTSeqRec,SFTPersonalControlRec,SFTControlRec_re,SFTPersonalCategoryRate,ShareChatGPT 
--SFT_val_tasks SFTTestSeqRec,SFT+TestPersonalControlRec,SFT-TestPersonalControlRec,SFTTestPersonalCategoryRateEP_50 
--backup_ip 0.0.0.0 
--share_chat_gpt_ratio 0.5 
--val_num_per_task 320 
--llama2_chat_template 
--idx 
--teacher_port 12621 
```

### 2.3. SFT train

The training dataset is dynamically generated during the `__getitem__` function call of the dataset class. An example script for training can be found at [scripts/sft_train.sh](https://github.com/Luuuk12321/RecLM-gen/blob/main/scripts/sft_train.sh).

To use a static dataset instead, specify the `--train_data_file` and `--val_data_file` parameters:  

```shell
  --train_data_file data/dataset/sub_movie/SFT_dataset_train.pickle 
  --val_data_file data/dataset/sub_movie/SFT_dataset_val.pickle 
```
`RecLM-gen` is compatible with single-GPU training during the SFT stage. For an example, refer to [scripts/single_gpu_sft_train.sh](https://github.com/Luuuk12321/RecLM-gen/blob/main/scripts/single_gpu_sft_train.sh).

### 2.4. SFT model merge

Merge the trained models using the script found at [scripts/sft_merge.sh](https://github.com/Luuuk12321/RecLM-gen/blob/main/scripts/sft_merge.sh). The merged model will be saved to `snap/ICR_SubMovie/SFT_Epoch27/`.  
   
**Note: Use `CUDA_VISIBLE_DEVICES=x` to select a GPU. Do not set the `--gpu` command parameter.**  



## 3. RL stage

### 3.1. Dataset format

The RL stage also utilizes a dataset of type `List[List[Dict]]`.  
   
- Each inner `List[Dict]` represents the training data for a specific episode.  
- Each `Dict` is a training sample that must contain the key `'input_text'` for RL.  
- Keys such as `task` and `input_field_data` are used to calculate metrics and rewards for the domain.  
   
```js
[
  [ //Episode 1
    {"input_text": "...", "task": "...", "input_field_data": {"...": "..."}},
    "...",
    {"input_text": "...", "task": "...", "input_field_data": {"...": "..."}}
  ],
  [ //Episode 2
    "..."
  ]
]
```

### 3.2. Dataset prepare
The dataset file is saved to `{data_path}/RL_dataset_train.pickle` and `{data_path}/RL_dataset_val.pickle`.
```shell
python data_process.py 
--seed 0 
--data_path data/dataset/sub_movie/ 
--item_index title64_t 
--topk 10 
--num_episodes 2 
--train_stage RL 
--RL_train_tasks RLSeqRec,RL+PersonalControlRec,RL-PersonalControlRec,RLPersonalCategoryRateLP,RLPersonalCategoryRateMP,RLPersonalCategoryRateEP 
--RL_val_tasks RLSeqRec,RL+PersonalControlRec,RL-PersonalControlRec,RLPersonalCategoryRateLP_20,RLPersonalCategoryRateMP_30,RLPersonalCategoryRateEP_50,RLItemCount 
--backup_ip 0.0.0.0 
--val_num_per_task 320 
--llama2_chat_template 
--idx 
--teacher_port 12621 
```


### 3.3. RL train

As with the SFT stage, the RL training dataset is dynamically generated. An example training script is available at [scripts/rl_train.sh](https://github.com/Luuuk12321/RecLM-gen/blob/main/scripts/rl_train.sh).

To use a static dataset, specify the `--train_data_file` and `--val_data_file`:  

```shell
  --train_data_file data/dataset/sub_movie/RL_dataset_train.pickle 
  --val_data_file data/dataset/sub_movie/RL_dataset_val.pickle 
```

Single-GPU training is supported for the RL stage as well. See [scripts/single_gpu_rl_train.sh](https://github.com/Luuuk12321/RecLM-gen/blob/main/scripts/single_gpu_rl_train.sh) for an example.  


### 3.4. RL model merge
Merge the RL-trained models using the script provided at [scripts/rl_merge.sh](https://github.com/Luuuk12321/RecLM-gen/blob/main/scripts/rl_merge.sh). The merged model will be saved in an appropriately named directory within the `snap/` folder, such as `snap/ICR_SubMovie/SFT_Epoch27/RL/RLHF_Step3000/`


## 4. Test stage

### 4.1. Llama2 deploy and test
```shell
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --port 13579 --model snap/Llama-2-7b-hf-chat/
./scripts/tasks_test.sh snap/Llama-2-7b-hf-chat/ 13579 sub_movie
```

### 4.2. SFT model deploy and test
```shell
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --port 13579 --model snap/ICR_SubMovie/SFT_Epoch27/
./scripts/tasks_test.sh snap/ICR_SubMovie/SFT_Epoch27/ 13579 sub_movie
```

### 4.3. RL model deploy and test
```shell
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --port 13579 --model snap/ICR_SubMovie/SFT_Epoch27/RL/RLHF_Step3000/
./scripts/tasks_test.sh snap/ICR_SubMovie/SFT_Epoch27/RL/RLHF_Step3000/ 13579 sub_movie
```

### 4.4. ChatGPT test
If you want to test the capability of ChatGPT, you need to firstly set these environment variables.  If it is not Azure OpenAI API (OPENAI_API_TYPE is not "azure"), you only need to specify OPENAI_API_KEY and ENGINE.

```shell
export OPENAI_API_KEY=xxx
export OPENAI_API_BASE=https://xxx.openai.azure.com/
export OPENAI_API_VERSION=2023-03-15-preview
export OPENAI_API_TYPE=azure
export ENGINE=gpt-3.5-turbo-1106

./scripts/tasks_test.sh gpt-3.5-turbo-1106 0 sub_movie
```


## Citation
If you find this project useful in your research, please cite our research paper:

```
@inproceedings{lu-etal-2024-aligning,
    title = "Aligning Large Language Models for Controllable Recommendations",
    author = "Lu, Wensheng  and Lian, Jianxun  and Zhang, Wei  and Li, Guanghua  and Zhou, Mingyang  and Liao, Hao  and Xie, Xing",
    editor = "Ku, Lun-Wei  and Martins, Andre  and Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.443/",
    doi = "10.18653/v1/2024.acl-long.443",
    pages = "8159--8172",
}
```
