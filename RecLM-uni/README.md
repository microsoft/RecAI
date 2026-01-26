
# RecLM-uni
## Introduction
This project introduces methods for avoiding recommending out-of-domain items in LLM-based recsys. It contains the code for implementing three methods, i.e., RecLM-cgen, RecLM-ret and RecLM-token.

**RecLM-uni** is a generative recommendation framework in the native structure of LLMs. This framework divides the output space of LLMs into item generation and general text generation parts by introducing item control tokens, and simultaneously employs a decoding strategy with prefix tree constraints to prevent the generation of out-of-domain items. RecLM-uni enables LLMs to acquire the ability to recommend products without sacrificing their original general capabilities.

The RecLM-uni framework seamlessly integrates LLMs with recommendation scenarios. Interacting with RecLM-uni is just like interacting with general LLMs, enabling users to complete recommendation tasks and other general tasks in multi-round conversations.

The pipeline of RecLM-uni has 4 steps:
1. Preprocessing raw dataset
2. Training teacher model
3. Deploying teacher model service
4. Training RecLM-uni

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

The training dataset is dynamically generated during the `__getitem__` function call of the dataset class. An example script for training can be found at [./scripts/train_RecLM_cgen.sh](scripts/train_RecLM_cgen.sh) for **RecLM-cgen** and [./scripts/train_RecLM_ret.sh](scripts/train_RecLM_ret.sh) for **RecLM-ret**. The training script for **RecLM-token** is shown in Section 4.4.
```shell
./scripts/train_RecLM_cgen.sh movies  # RecLM-cgen
./scripts/train_RecLM_ret.sh movies   # RecLM-ret
```

### 3.2. SFT model merge

Merge the trained models using the script found at [./scripts/run_SFT_merge.sh](scripts/run_SFT_merge.sh). The merged model will be saved to `snap/.../SFT_Epoch20/`.
```shell
./scripts/run_SFT_merge.sh
```

## 4. RecLM-token

**RecLM-token** uses RQ-VAE to encode items as sequences of special tokens (codebook), enabling semantic item representation while maintaining generation constraints.

### 4.1. Step 1: Prepare Item Embeddings

Generate item description embeddings by extracting the last hidden layer from LLaMA-3-8B and applying attention-masked weighted average pooling. Each item's text (title + description from `metas.jsonl`) should be encoded into a fixed-dimensional embedding vector.

The embeddings should be saved as a numpy array with shape `(num_items, hidden_dim)`, where `hidden_dim` is the LLaMA model's hidden size. The output file should be named `{dataset}.emb-llama-td.npy` and placed in the data directory (e.g., `data/dataset/movies/movies.emb-llama-td.npy`).

### 4.2. Step 2: Train RQ-VAE Model

Train the Residual Quantized Variational AutoEncoder(RQ-VAE) to learn item codebook mappings using the `index/` module from RecLM-uni or RecLM-cgen:

```bash
cd index

python main.py \
  --lr 1e-3 \
  --epochs 10000 \
  --batch_size 1024 \
  --weight_decay 1e-4 \
  --lr_scheduler_type linear \
  --dropout_prob 0.0 \
  --bn False \
  --e_dim 32 \
  --quant_loss_weight 1.0 \
  --contrastive_loss_weight 0.1 \
  --beta 0.25 \
  --num_emb_list 256 256 256 256 \
  --sk_epsilons 0.0 0.0 0.0 0.003 \
  --layers 2048 1024 512 256 128 64 \
  --device cuda:0 \
  --data_path ../../RecAI/RecLM-uni/data/dataset/movies/movies.emb-llama-td.npy \
  --ckpt_dir ./ckpt/movies/
```

**Key Parameters:**
- `--num_emb_list`: Codebook sizes per layer (e.g., `256 256 256 256` = 4 layers, 256 codes each)
- `--sk_epsilons`: Sinkhorn algorithm epsilon for collision avoidance (enable on last layer)
- `--e_dim`: Codebook embedding dimension

**Output:** Model checkpoint saved to `./ckpt/movies/{timestamp}/best_collision_model.pth`

### 4.3. Step 3: Generate Item Indices

Generate `.index.json` and `.item2id` files from the trained RQ-VAE model:

```bash
python generate_indices.py \
  --dataset movies \
  --ckpt_path ./ckpt/movies/best_collision_model.pth \
  --output_dir ../../RecAI/RecLM-uni/data/dataset/movies/ \
  --metas_file ../../RecAI/RecLM-uni/data/dataset/movies/metas.jsonl \
  --device cuda:0
```

**Output Files:**
- `movies.index.json`: Maps indices to token sequences (e.g., `"0": ["<a_137>", "<b_180>", "<c_11>", "<d_76>"]`)
- `movies.item2id`: Maps item IDs to indices (e.g., `B00001234 0`)

Return to RecLM-uni directory after generating the files:

### 4.4. Step 4: Train SFT Model with Codebook

> **Prerequisite**: Before training the SFT model, ensure that the SASRec teacher model has been trained and deployed as a service (follow the steps in Sections 2.2-2.4). The training script will connect to the teacher service at the port specified by `--teacher_port`.

Train the recommendation model using codebook representation. The codebase will **automatically detect** and use the codebook files if they exist in the data directory.

```bash
python main.py \
  --seed 0 \
  --data_path data/dataset/movies/ \
  --backbone meta-llama/Meta-Llama-3-8B-Instruct \
  --item_index rq_token_seq \
  --train_stage SFT \
  --SFT_train_tasks SFTSeqRec-CS-MR \
  --SFT_val_tasks SFTTestSeqRec-MR \
  --multi_round_ratio 0.1 \
  --use_control_symbol \
  --use_CBS \
  --CBS_type 2 \
  --batch_size 2 \
  --topk 10 \
  --epoch 20 \
  --lr 0.0001 \
  --gradient_accumulation_steps 32 \
  --SFT_actor_lora_r 16 \
  --SFT_actor_lora_a 8 \
  --chat_template llama-3 \
  --teacher_port 2609 \
  --FA2 \
  --backup_ip 0.0.0.0 \
  --loss_type 3 \
  --scope_mask_type 3 \
  --fl_gamma 2 \
  --token_emb \
  --output snap/movies-token/
```

**Key Configuration:**
- `--item_index rq_token_seq`: Use codebook representation (optional, will auto-detect)
- `--use_control_symbol`: Enable `<SOI>` and `<EOI>` markers
- `--use_CBS`: Enable Constrained Beam Search for valid token sequences

**Note:** The codebook files (`{dataset}.index.json` and `{dataset}.item2id`) must be present in the `data_path` directory. If they are missing, the model will automatically fall back to using text-based item representation (`title_t`).

### 4.5. Step 5: GRPO Training

After SFT training, you can apply GRPO to further improve recommendation quality using reinforcement learning. This step uses the SFT model as the initial policy and optimizes ranking-based rewards.

**Prerequisite**: Complete SFT model merge (Section 3.2) to obtain the merged model checkpoint.

Update the configuration in [./GRPO/run_GRPO.sh](./GRPO/run_GRPO.sh):

```bash
INITIAL_MODEL="snap/movies-token/SFT_Epoch20"  # Path to merged SFT model
OUTPUT_BASE=./output/$(date "+%m%d")-${DATASET}-RL
DATASET="movies"
DATA_PATH="data/${DATASET}/"
```

Then launch GRPO training:

```bash
./GRPO/run_GRPO.sh
```

**Key GRPO Parameters:**
- `--num_generations 16`: Number of generations per prompt for advantage estimation
- `--reward_type ranking`: Use ranking-based reward (hit rate, ndcg)
- `--beta 1e-3`: KL divergence coefficient for policy constraint
- `--use_lora`: Apply LoRA for efficient parameter updates
- `--rl_max_samples 10000`: Maximum number of RL training samples

The GRPO-trained model will be saved to `${OUTPUT_BASE}` and can be used for evaluation in the next step.

### 4.6. Step 6: RecLM-token Evaluation

#### Recommendation Testing

```bash
python task_test_tokenizer.py \
  --data_path data/dataset/movies/ \
  --SFT_test_task SFTTestSeqRec-MR \
  --model_name snap/movies-token/SFT_Epoch20/ \
  --gpu cuda:0 \
  --item_index rq_token_seq \
  --use_control_symbol \
  --batch_size 72 \
  --use_CBS \
  --CBS_type 2 \
  --idx \
  --topk 10
```

**Key Difference from RecLM-cgen:**
- **RecLM-cgen**: Items represented as text (e.g., `"The Dark Knight"`)
- **RecLM-token**: Items represented as token sequences (e.g., `"<a_137><b_180><c_11><d_76>"`)

## 5. RecLM-cgen (Text-based Item Generation)

**RecLM-cgen** uses item title text directly as the generation target, enabling flexible and interpretable item recommendations.

> **Prerequisite**: Before training RecLM-cgen, you need to train an item title rewrite model using MS-Swift (Section 8). The rewrite model helps improve recommendation quality by generating paraphrased item titles that maintain semantic meaning while varying surface form.

### 5.1. Recommendation testing
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

### 5.2. Multi-round conversation testing
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

### 5.3. SFT model deploying
```shell
python cli_serve.py \
--model_name snap/.../SFT_Epoch20/ \
--gpu cuda:0
```

## 6. RecLM-ret testing

### 6.1. Recommendation testing
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

### 6.2. Multi-round conversation testing
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

## 7. Build domain item prefix tree for enabling constrained generation
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

## 8. Item Title Rewrite Training with ms-swift

The item title rewrite model is a crucial component for RecLM-cgen that paraphrases item titles to improve recommendation robustness and diversity. This model is trained using reinforcement learning via  [[ms-swift](https://github.com/modelscope/ms-swift)] framework with custom reward functions.

The rewrite model learns to generate paraphrased versions of item titles while preserving their semantic meaning. This is achieved through GRPO (Group Relative Policy Optimization) training with multiple reward signals:

- **ConditionalPPL**: Measures language model quality of the rewritten title in context
- **LengthReward**: Encourages appropriate title length relative to the original
- **DiscriminativeReward**: Tests if the rewrite can be correctly matched to the original item
- **Item2ItemReward**: Evaluates semantic similarity using embedding-based retrieval
- **User2ItemReward**: Assesses if the rewrite maintains user-relevance in recommendation context

All reward functions are implemented in [./plugin.py](./plugin.py). 

The paraphrased titles generated by the rewrite model are then used as additional training samples for RecLM-cgen (Section 3.1), improving the model's robustness to variations in item title phrasing and enhancing recommendation diversity.

## 9. Citation
If you find this project useful in your research, please cite our research paper:

```
@misc{liao2026eliminatingoutofdomainrecommendationsllmbased,
      title={Eliminating Out-of-Domain Recommendations in LLM-based Recommender Systems: A Unified View}, 
      author={Hao Liao and Jiwei Zhang and Jianxun Lian and Wensheng Lu and Mingqi Wu and Shuo Wang and Yong Zhang and Yitian Huang and Mingyang Zhou and Rui Mao},
      year={2026},
      eprint={2505.03336},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2505.03336}, 
}
```
