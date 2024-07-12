"""
The following code is modified from
https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/finetune/data.py
"""

import math
import os.path
import random
from dataclasses import dataclass
from typing import List, Tuple, Union
import torch

import datasets
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizer, BatchEncoding

from .arguments import DataArguments


class TrainDatasetForEmbedding(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer
    ):
        if os.path.isdir(args.train_data):
            train_datasets = []
            for file in os.listdir(args.train_data):
                temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_data, file),
                                                     split='train', cache_dir=args.data_cache_dir)
                if len(temp_dataset) > args.max_example_num_per_dataset:
                    temp_dataset = temp_dataset.select(
                        random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset))
                column_names = temp_dataset.column_names
                remove_columns = ['user_id', 'item_id', 'neg_ids', 'pos_id']
                remove_columns = [c for c in remove_columns if c in column_names]
                temp_dataset = temp_dataset.remove_columns(remove_columns)
                train_datasets.append(temp_dataset)
            self.dataset = datasets.concatenate_datasets(train_datasets)
        elif len(args.train_data.split(',')) > 1:
            train_datasets = []
            for file in args.train_data.split(','):
                temp_dataset = datasets.load_dataset('json', data_files=file, split='train',
                                                     cache_dir=args.data_cache_dir)
                if len(temp_dataset) > args.max_example_num_per_dataset:
                    temp_dataset = temp_dataset.select(
                        random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset))
                column_names = temp_dataset.column_names
                remove_columns = ['user_id', 'item_id', 'neg_ids', 'pos_id']
                remove_columns = [c for c in remove_columns if c in column_names]
                temp_dataset = temp_dataset.remove_columns(remove_columns)
                train_datasets.append(temp_dataset)
            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train',
                                                 cache_dir=args.data_cache_dir)
        ## shuffle self.dataset 
        self.dataset = self.dataset.shuffle()
        
        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        query = self.dataset[item]['query']
        passages = []
        pos = random.choice(self.dataset[item]['pos'])
        passages.append(pos)

        if len(self.dataset[item]['neg']) < self.args.train_group_size - 1:
            num = math.ceil((self.args.train_group_size - 1) / len(self.dataset[item]['neg']))
            negs = random.sample(self.dataset[item]['neg'] * num, self.args.train_group_size - 1)
        else:
            negs = random.sample(self.dataset[item]['neg'], self.args.train_group_size - 1)
        passages.extend(negs)
        if self.args.has_template:
            query = f'query: {query}'
            for i in range(len(passages)):
                passages[i] = f'passage: {passages[i]}'

        return query, passages


@dataclass
class EmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 32
    passage_max_len: int = 128
    truncation_strategy: Union[bool, str] = True
    padding_strategy: Union[bool, str] = 'max_length'
    add_eos: bool = False

    def padding_score(self, teacher_score):
        group_size = None
        for scores in teacher_score:
            if scores is not None:
                group_size = len(scores)
                break
        if group_size is None:
            return None

        padding_scores = [100.0] + [0.0] * (group_size - 1)
        new_teacher_score = []
        for scores in teacher_score:
            if scores is None:
                new_teacher_score.append(padding_scores)
            else:
                new_teacher_score.append(scores)
        return new_teacher_score

    def __call__(self, features):
        query = [f[0] for f in features]
        passage = [f[1] for f in features]

        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])

        if not self.add_eos:
            q_collated = self.tokenizer(
                query,
                padding=self.padding_strategy,
                truncation=self.truncation_strategy,
                max_length=self.query_max_len,
                return_tensors="pt",
            )
            d_collated = self.tokenizer(
                passage,
                padding=self.padding_strategy,
                truncation=self.truncation_strategy,
                max_length=self.passage_max_len,
                return_tensors="pt",
            )
        else:
            q_collated = self.tokenizer(
                query,
                padding=self.padding_strategy,
                truncation=self.truncation_strategy,
                max_length=self.query_max_len-1,
                return_tensors="pt",
            )
            d_collated = self.tokenizer(
                passage,
                padding=self.padding_strategy,
                truncation=self.truncation_strategy,
                max_length=self.passage_max_len-1,
                return_tensors="pt",
            )
            q_batch = len(q_collated['input_ids'])
            eos = torch.full((q_batch, 1), self.tokenizer.eos_token_id, dtype=torch.long)
            eos_attn = torch.ones((q_batch, 1), dtype=torch.long)
            q_collated['input_ids'] = torch.cat((q_collated['input_ids'], eos), dim=1)
            q_collated['attention_mask'] = torch.cat((q_collated['attention_mask'], eos_attn), dim=1)
            
            d_batch = len(d_collated['input_ids'])
            eos = torch.full((d_batch, 1), self.tokenizer.eos_token_id, dtype=torch.long)
            eos_attn = torch.ones((d_batch, 1), dtype=torch.long)
            d_collated['input_ids'] = torch.cat((d_collated['input_ids'], eos), dim=1)
            d_collated['attention_mask'] = torch.cat((d_collated['attention_mask'], eos_attn), dim=1)

        return {"query": q_collated, "passage": d_collated}
