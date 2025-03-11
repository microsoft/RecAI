# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pickle
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel
from peft import PeftModel
from dataclasses import dataclass

class EmbDataset(Dataset):
    def __init__(self, test_dataset, tokenizer, max_seq_len, has_template, qorp) -> None:
        super().__init__()
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.has_template = has_template
        self.qorp = qorp
            
    def __len__(self):
        return len(self.test_dataset)
    
    def __getitem__(self, idx):
        data = self.test_dataset[idx]
        if self.has_template:
            data = f'{self.qorp}: {data}'
        return data

@dataclass
class InferCollator(DataCollatorWithPadding):
    truncation: bool = True
    add_eos: bool = False
    def __call__(self, features):
        if not self.add_eos:
            t_input = self.tokenizer(features, padding=True, truncation=self.truncation, max_length=self.max_length, return_tensors="pt")
        else:
            t_input = self.tokenizer(features, padding=True, truncation=self.truncation, max_length=self.max_length-1, return_tensors="pt")
            t_batch = len(t_input['input_ids'])
            eos = torch.full((t_batch, 1), self.tokenizer.eos_token_id, dtype=torch.long)
            eos_attn = torch.ones((t_batch, 1), dtype=torch.long)
            t_input['input_ids'] = torch.cat((t_input['input_ids'], eos), dim=1)
            t_input['attention_mask'] = torch.cat((t_input['attention_mask'], eos_attn), dim=1)
        return t_input

def sentence_embedding(hidden_state, mask, sentence_pooling_method, normlized):
    if sentence_pooling_method == 'mean':
        s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
        d = mask.sum(axis=1, keepdim=True).float()
        sentence_embeddings =  s / d
    elif sentence_pooling_method == 'cls':
        sentence_embeddings = hidden_state[:, 0]
    elif sentence_pooling_method == 'last':
        sentence_embeddings = hidden_state[:, -1]
    else:
        raise ValueError("sentence_pooling_method should be mean or cls")

    if normlized:
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings

def run_model_embedding(model_path_or_name, max_seq_len, batch_size, prompt_path, emb_path, accelerator, args, qorp):
    model_config = AutoConfig.from_pretrained(model_path_or_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name, use_fast=False,
        padding_side='left' if "Llama" in model_path_or_name else 'right',
        truncation_side='right',)
    if "Llama" in model_path_or_name:
        tokenizer.pad_token = tokenizer.unk_token
    torch_dtype = (
            args.torch_dtype
            if args.torch_dtype in ["auto", None]
            else getattr(torch, args.torch_dtype)
        )
    model = AutoModel.from_pretrained(model_path_or_name, config=model_config, torch_dtype=torch_dtype)
    if args.peft_model_name:
        model = PeftModel.from_pretrained(model, args.peft_model_name)
        # model = model.merge_and_unload()

    accelerator.print(f'loading file {prompt_path}')
    test_data = pd.read_json(prompt_path, lines=True)
    test_data = test_data['text'].tolist()

    test_dataset = EmbDataset(test_data, tokenizer, max_seq_len, args.has_template, qorp)
    dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, collate_fn=InferCollator(tokenizer, max_length=max_seq_len, truncation=True, add_eos="Llama" in model_path_or_name))
    model, dataloader = accelerator.prepare(model, dataloader)
    model.eval()

    pbar = tqdm(total=len(dataloader), desc="inferencing", disable=not accelerator.is_main_process)
    result_lists = []
    
    with torch.no_grad():
        for data in dataloader:
            model_output = model(**data, return_dict=True)
            # Perform pooling and normalization
            sentence_embeddings = sentence_embedding(model_output.last_hidden_state, data['attention_mask'], args.sentence_pooling_method, args.normlized)
            
            sentence_embeddings = accelerator.gather_for_metrics(sentence_embeddings).tolist()
            result_lists.extend(sentence_embeddings)
            pbar.update(1)
    if accelerator.is_main_process:
        print("shape of result lists: ", len(result_lists))
        pickle.dump(result_lists, open(emb_path, "wb"))