# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from tqdm import tqdm
import json

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader

from transformers import pipeline, GenerationConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel

DEFAULT_SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
class ChatDataset(Dataset):
    def __init__(self, test_dataset, tokenizer, max_seq_len, world_size, rank, system_prompt) -> None:
        super().__init__()
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.world_size = world_size
        self.rank = rank
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = DEFAULT_SYSTEM_PROMPT
            
    def __len__(self):
        length = (len(self.test_dataset)+self.world_size-1) // self.world_size
        return length
    
    def __getitem__(self, idx):
        if idx * self.world_size + self.rank < len(self.test_dataset):
            data = self.test_dataset[idx * self.world_size + self.rank]
        else:
            data = self.test_dataset[-1]
        
        if isinstance(data["prompt"], str):
            inputs = f"USER: {data['prompt'].strip()} "
        else:
            inputs = ""
            for text in data["prompt"]:
                if text["role"] == "assistant":
                    inputs += "ASSISTANT: " + text["content"] + ' '
                else:
                    inputs += "USER: " + text["content"] + ' '
        tokens = self.tokenizer.tokenize(f"{self.system_prompt} {inputs}")

        tokens = tokens[:self.max_seq_len-len(self.tokenizer.tokenize("ASSISTANT:"))]
        truncated_prompt = self.tokenizer.convert_tokens_to_string(tokens) + "ASSISTANT:"

        return truncated_prompt.strip()
    
class EmbDataset(Dataset):
    def __init__(self, test_dataset, tokenizer, max_seq_len, world_size, rank) -> None:
        super().__init__()
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.world_size = world_size
        self.rank = rank
            
    def __len__(self):
        length = (len(self.test_dataset)+self.world_size-1) // self.world_size
        return length
    
    def __getitem__(self, idx):
        if idx * self.world_size + self.rank < len(self.test_dataset):
            data = self.test_dataset[idx * self.world_size + self.rank]
        else:
            data = self.test_dataset[-1]
        
        tokens = self.tokenizer.tokenize(data['prompt'].strip())[:self.max_seq_len]
        truncated_prompt = self.tokenizer.convert_tokens_to_string(tokens)

        return truncated_prompt.strip()
    
def run_chat(local_gpu_rank, model_path_or_name, question_file, answer_file, args, system_prompt):
    args.rank = args.nr * args.gpus + local_gpu_rank
    args.device = torch.device("cuda", local_gpu_rank)
    torch.cuda.set_device(args.device)
    dist.init_process_group(backend='nccl', init_method="env://", world_size=args.world_size, rank=args.rank)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path_or_name, fast_tokenizer=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # load model
    model_config = AutoConfig.from_pretrained(args.model_path_or_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path_or_name,
        from_tf=bool(".ckpt" in args.model_path_or_name),
        config=model_config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        # use_safetensors=False
    )

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=args.device)
    greedy_config = GenerationConfig(return_full_text=False, max_new_tokens=args.max_new_tokens)

    # load test dataset
    test_data = []
    for line in open(question_file):
        test_data.append(json.loads(line))
    test_dataset = ChatDataset(test_data, tokenizer, model_config.max_position_embeddings - args.max_new_tokens - 100, args.world_size, args.rank, system_prompt) # 1700 < 2048 - max new tokens
    
    if args.rank == 0:
        pbar = tqdm(total=len(test_dataset), desc="inferencing")
        result_lists = []
    
    with torch.no_grad():
        for raw_result in generator(test_dataset, generation_config=greedy_config, batch_size=args.batch_size):
            result = raw_result[0]['generated_text'].split("ASSISTANT:")[-1].strip()
            if "<|endoftext|>" in result:
                result = result.split('<|endoftext|>')[0].strip()
            if "</s>" in result:
                result = result.split('</s>')[0].strip()
            result = (args.rank, result)
            gather_data = [None for _ in range(args.world_size)]
            dist.all_gather_object(gather_data, result)
            if args.rank == 0:
                gather_data = sorted(gather_data, key=lambda x:x[0])
                for result in gather_data:
                    result_lists.append(result[1])
                pbar.update(1)

    if args.rank == 0:
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        fd = open(answer_file, "w", encoding='utf-8')
        for data, result in zip(test_data, result_lists):
            data["answer"] = result
            fd.write(json.dumps(data, ensure_ascii=False) + '\n')

def run_embedding(local_gpu_rank, model_path_or_name, question_file, answer_file, args):
    args.rank = args.nr * args.gpus + local_gpu_rank
    args.device = torch.device("cuda", local_gpu_rank)
    torch.cuda.set_device(args.device)
    dist.init_process_group(backend='nccl', init_method="env://", world_size=args.world_size, rank=args.rank)

    model_config = AutoConfig.from_pretrained(args.model_path_or_name)
    if "CausalLM" in model_config.architectures[0]:
        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_name, use_fast=False, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        # load model
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path_or_name,
            from_tf=bool(".ckpt" in args.model_path_or_name),
            config=model_config,
            low_cpu_mem_usage=True
        ).to(args.device)
        model.config.end_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
        model_type = "CausalLM"
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
        model = AutoModel.from_pretrained(model_path_or_name).to(args.device)
        model_type = "Other"

    # load test dataset
    test_data = []
    for line in open(question_file):
        test_data.append(json.loads(line))
    test_dataset = EmbDataset(test_data, tokenizer, model_config.max_position_embeddings - 100, args.world_size, args.rank) # 1700 < 2048 - max new tokens
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    if args.rank == 0:
        pbar = tqdm(total=len(dataloader), desc="inferencing")
        result_lists = []
    
    with torch.no_grad():
        for data in dataloader:
            t_input = tokenizer(data, padding=True, truncation=True, return_tensors="pt").to(args.device)
            if model_type == "CausalLM":
                last_hidden_state = model(**t_input, output_hidden_states=True).hidden_states[-1]
                sum_embeddings = torch.sum(last_hidden_state * t_input.attention_mask.unsqueeze(-1), dim=1)
                num_of_none_padding_tokens = torch.sum(t_input.attention_mask, dim=-1).unsqueeze(-1)
                sentence_embeddings = sum_embeddings / num_of_none_padding_tokens
            else:
                model_output = model(**t_input)
                # Perform pooling. In this case, cls pooling.
                sentence_embeddings = model_output[0][:, 0]
            # normalize embeddings
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1).tolist()
            sentence_embeddings = (args.rank, sentence_embeddings)
            gather_data = [None for _ in range(args.world_size)]
            dist.all_gather_object(gather_data, sentence_embeddings)
            if args.rank == 0:
                gather_data = sorted(gather_data, key=lambda x:x[0])
                for idx in range(len(gather_data[0][1])):
                    for data in gather_data:
                        result_lists.append(data[1][idx])
                pbar.update(1)

    if args.rank == 0:
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        fd = open(answer_file, "w", encoding='utf-8')
        for data, result in zip(test_data, result_lists):
            data["answer"] = result
            fd.write(json.dumps(data, ensure_ascii=False) + '\n')

def gen_model_chat_answer(model_path_or_name, question_file, answer_file, args, system_prompt):
    if args.gpus < 0:
        args.gpus = torch.cuda.device_count()
    args.world_size = args.nodes * args.gpus
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.master_port
    mp.spawn(run_chat, nprocs=args.gpus, args=(model_path_or_name, question_file, answer_file, args, system_prompt))

def gen_model_embedding_answer(model_path_or_name, question_file, answer_file, args):
    if args.gpus < 0:
        args.gpus = torch.cuda.device_count()
    args.world_size = args.nodes * args.gpus
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.master_port
    mp.spawn(run_embedding, nprocs=args.gpus, args=(model_path_or_name, question_file, answer_file, args))