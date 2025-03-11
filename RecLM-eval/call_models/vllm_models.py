# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from tqdm import tqdm
import json
from transformers import AutoConfig, AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import vllm
import functools

def map_gpu_count(m):
    n_values = [1, 2, 4, 8]
    for n in reversed(n_values):
        if m >= n:
            return n
    return None  
def env(var_name):
    return os.getenv(var_name)
if not env("VLLM_TENSOR_PARALLEL_SIZE"):
    gpu_count = torch.cuda.device_count()
    VLLM_TENSOR_PARALLEL_SIZE = map_gpu_count(gpu_count)
    print(f"set VLLM_TENSOR_PARALLEL_SIZE == {VLLM_TENSOR_PARALLEL_SIZE}")
else:
    VLLM_TENSOR_PARALLEL_SIZE = env("VLLM_TENSOR_PARALLEL_SIZE")

if not env("VLLM_GPU_MEMORY_UTILIZATION"):
    VLLM_GPU_MEMORY_UTILIZATION = 0.9
    print(f"set VLLM_GPU_MEMORY_UTILIZATION == {VLLM_GPU_MEMORY_UTILIZATION}")
else:
    VLLM_GPU_MEMORY_UTILIZATION = env("VLLM_GPU_MEMORY_UTILIZATION")
    
DEFAULT_SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."

class vllmModel:
    def __init__(self, model_name: str, max_model_len: int = 4096, dtype: str = "bfloat16"):
        self.model_name = model_name
        self.llm = vllm.LLM(
            self.model_name,
            max_model_len=max_model_len,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
            trust_remote_code=True,
            dtype=dtype,
            enforce_eager=False
        )
        self.tokenizer = self.llm.get_tokenizer()

    def batch_predict(self, prompts: list[str], max_new_tokens: int = 1000) -> list[str]:
        responses = self.llm.generate(
            prompts,
            vllm.SamplingParams(
                n=1,
                top_p=0.9,
                temperature=0.3,
                skip_special_tokens=True,
                max_tokens=max_new_tokens,
            ),
            use_tqdm=False
        )

        batch_response = [response.outputs[0].text.strip() for response in responses]
        return batch_response

class ChatDataset(Dataset):
    def __init__(self, test_dataset, tokenizer, max_seq_len, system_prompt) -> None:
        super().__init__()
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.system_prompt = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT
            
    def __len__(self):
        return len(self.test_dataset)
    
    def __getitem__(self, idx):
        data = self.test_dataset[idx]
        
        if isinstance(data["prompt"], str):
            conv = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": data["prompt"]}]
            inputs = self.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        else:
            data["prompt"].insert(0, {"role": "system", "content": self.system_prompt})
            inputs = self.tokenizer.apply_chat_template(data["prompt"], tokenize=False, add_generation_prompt=True)
        inputs = inputs[-self.max_seq_len:]
        return inputs

def get_batches(data, batch_size):

    batches = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batches.append(batch)
    return batches

@functools.lru_cache(maxsize=1)  
def get_cached_model(model_path_or_name):
    return vllmModel(model_path_or_name)



def run_chat(model_path_or_name, question_file, answer_file, args, system_prompt):
    # load tokenizer
    model = get_cached_model(model_path_or_name)
    # load test dataset
    test_data = []
    for line in open(question_file):
        test_data.append(json.loads(line))

    model_config = AutoConfig.from_pretrained(args.model_path_or_name, trust_remote_code=True)
    max_position_embeddings = getattr(model_config, 'max_position_embeddings', 2048)

    test_dataset = ChatDataset(
        test_data, 
        model.tokenizer, 
        max_position_embeddings - args.max_new_tokens - 100, 
        system_prompt
    )

    dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    result_lists = []
    
    with torch.no_grad():
        for batch in dataloader:
            responses = model.batch_predict(
                batch,
                max_new_tokens=args.max_new_tokens
            )
            result_lists.extend(responses)

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "w", encoding='utf-8') as fd:
        for data, result in zip(test_data, result_lists):
            data["answer"] = result
            fd.write(json.dumps(data, ensure_ascii=False) + '\n')

def gen_model_chat_answer(model_path_or_name, question_file, answer_file, args, system_prompt):
    run_chat(model_path_or_name, question_file, answer_file, args, system_prompt)
