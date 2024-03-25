# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import random
import json
import torch
from torch.utils.data import Dataset
import os

from dataset.conversation import template_dict

logger = logging.getLogger(__name__)
IGNORE_TOKEN_ID=-100


class TrainingDataset4Exp(Dataset):
    def __init__(self, args, tokenizer, split='train'):
        self.split = split
        self.function_map = {
            'sharegpt': self.process_sharegpt,
            'iid2title': self.process_iid2title,
            'iid2feature': self.process_iid2title,
            'iid2description': self.process_iid2title,
            'iid2brand': self.process_iid2title,
            'iid2tags': self.process_iid2title,
            'iid2sim': self.process_iid2title,
            'feature2iid': self.process_iid2title,
            'description2iid': self.process_iid2title,
            'uid2hist': self.process_uid2hist,
            'uid2summary': self.process_uid2hist,
            'uid2next': self.process_uid2hist,
            'uidiid2review': self.process_uidiid2review,
            'uidiid2rank': self.process_uidiid2review,
            'uidiid2binary': self.process_uidiid2review,
        }
        self.tokenizer = tokenizer
        self.args = args
        self.user_history = self.load_user_history(args.sequential_file)
        self.max_hist_len = args.max_hist_len

        data_names = args.data_names.split(',')
        self.dataset = []
        for data_name in data_names:
            if os.path.exists(args.data_dir+data_name+f'_{split}.json') == False:
                continue
            temp_dataset = json.load(open(args.data_dir+data_name+f'_{split}.json', "r"))
            if len(temp_dataset) > args.max_example_num_per_dataset:
                temp_dataset = random.sample(temp_dataset, args.max_example_num_per_dataset)
            self.dataset.extend(temp_dataset)

        if args.data_type_filter != None:
            data_type_filter = set(args.data_type_filter.split(','))
            filter_dataset = []
            for data_item in self.dataset:
                if data_item['type'] not in data_type_filter:
                    filter_dataset.append(data_item)
            self.dataset = filter_dataset
            logger.info(f"Remain {len(filter_dataset)} examples with data_type_filter: {data_type_filter}")

        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        data_item = self.dataset[idx]
        func = self.function_map[data_item['type']]
        data = func([data_item])
 
        while torch.all(torch.eq(data['labels'][0], IGNORE_TOKEN_ID)):
            idx = random.randint(0, self.total_len-1)
            data_item = self.dataset[idx]
            func = self.function_map[data_item['type']]
            data = func([data_item])

        return {
            'input_ids': data['input_ids'][0],
            'attention_mask': data['attention_mask'][0],
            'labels': data['labels'][0],
            'user_pos': data['user_pos'][0],
            'item_seq': data['item_seq'][0],
            'item_pos': data['item_pos'][0],
            'item_ids': data['item_ids'][0],
            'type': data_item['type'],
        }

    def load_user_history(self, sequential_file):
        user_history = {}
        with open(sequential_file, 'r') as f:
            for line in f:
                ids = line.strip().split(' ')
                user_id = int(ids[0])
                item_id_list = [int(i) for i in ids[1:]]
                user_history[user_id] = item_id_list
        return user_history

    def pad_seq(self, x):
        len_seq = len(x)
        k = self.max_hist_len
        res = [0]*k
        if len_seq < k:
            res[(k-len_seq):] = x[:]
        else:
            res[:] = x[len_seq-k:]
        return res
    
    def process_sharegpt(self, dataset):
        sources = [example["conversations"] for example in dataset]
        
        conv = template_dict['sharegpt']
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            # if len(source) == 0 or any([messa["from"] not in roles for messa in source]):
            #     continue
            # if roles[source[0]["from"]] != conv.roles[0]:
            #     # Skip the first one if it is not from human
            #     source = source[1:]

            conv.messages = []
            # flag = 0
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2] and len(sentence["value"]) > 0, f"i: {i}, j: {j}, role: {role}, conv.roles: {conv.roles}, sentence: {sentence}"
                # if role != conv.roles[j % 2] or len(sentence["value"]) == 0:
                #     flag = 1
                #     break
                conv.append_message(role, sentence["value"])
            # if flag == 1:
            #     continue
            conversations.append(conv.get_prompt())

        # Tokenize conversations
        input_ids = self.tokenizer(
            conversations,
            return_tensors="pt",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        targets = input_ids.clone()

        # Mask targets. Only compute loss on the assistant outputs.
        sep = conv.sep + conv.roles[1] + ": "
        # valid = 0
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(self.tokenizer.pad_token_id).sum())

            turns = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_TOKEN_ID
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(self.tokenizer(turn).input_ids) #there will be <s> at the beginning, so cur_len += turn_len can cover the artificially added </s>

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
                instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2

                # Ignore the user instructions
                target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
                cur_len += turn_len

            target[cur_len:] = IGNORE_TOKEN_ID

            if False:  # Inspect and check the correctness of masking
                z = target.clone()
                z = torch.where(z == IGNORE_TOKEN_ID, self.tokenizer.unk_token_id, z)
                print(self.tokenizer.decode(z))
            # valid += 1
            if cur_len < self.tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_TOKEN_ID
                    logger.info( f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)")
                    # valid -= 1
        # logger.info(f"Valid examples: {valid} / {len(conversations)}")

        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            labels=targets,
            user_pos=[[0] for _ in range(len(input_ids))],
            item_seq=[[self.pad_seq([])] for _ in range(len(input_ids))],
            item_pos=[[0] for _ in range(len(input_ids))],
            item_ids=[[0] for _ in range(len(input_ids))],
        )

    def process_iid2title(self, dataset):
        conv = template_dict['iid2title']
        conversations = []
        item_ids = []
        for example in dataset:
            conv.messages = []
            conv.append_message(conv.roles[0], example['question'])
            conv.append_message(conv.roles[1], example['answer'])
            conversations.append(conv.get_prompt(example['template']))

            if self.args.task_type!="behaviour":
                item_ids.append([example['iid']])
        
        input_ids = self.tokenizer(
            conversations,
            return_tensors="pt",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        targets = input_ids.clone()
        sep = conv.sep + conv.roles[1] + ": "
        for conversation, target in zip(conversations, targets):
            cur_len = 1
            target[:cur_len] = IGNORE_TOKEN_ID
            parts = conversation.split(sep)
            assert len(parts) == 2
            parts[0] += sep
            # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
            instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID

            if False:  # Inspect and check the correctness of masking
                z = target.clone()
                z = torch.where(z == IGNORE_TOKEN_ID, self.tokenizer.unk_token_id, z)
                print(self.tokenizer.decode(z))

        targets = torch.where(targets==self.tokenizer.pad_token_id, IGNORE_TOKEN_ID, targets)
        if self.args.task_type!="behaviour":
            pre_item_pos = (input_ids==self.tokenizer.additional_special_tokens_ids[1]).nonzero()
            item_pos = [[] for _ in range(len(input_ids))]
            for i, pos in enumerate(pre_item_pos):
                item_pos[pos[0].item()].append(pos[1].item())
        else:
            item_pos = [[0] for _ in range(len(input_ids))]
            item_ids = [[0] for _ in range(len(input_ids))]

        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            labels=targets,
            user_pos=[[0] for _ in range(len(input_ids))],
            item_seq=[[self.pad_seq([])] for _ in range(len(input_ids))],
            item_pos=item_pos,
            item_ids=item_ids,
        )

    def process_uid2hist(self, dataset):
        conv = template_dict['uid2hist']
        conversations = []
        item_seq = []
        for example in dataset:
            conv.messages = []
            conv.append_message(conv.roles[0], example['question'])
            conv.append_message(conv.roles[1], example['answer'])
            conversations.append(conv.get_prompt(example['template']))

            if self.args.task_type!="behaviour":
                hist = self.user_history[example['uid']]
                hist = hist[:hist.index(example['iid'])] if example['iid']!=-1 else hist
                hist = self.pad_seq(hist)
                item_seq.append([hist])
        
        input_ids = self.tokenizer(
            conversations,
            return_tensors="pt",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        targets = input_ids.clone()
        sep = conv.sep + conv.roles[1] + ": "
        for conversation, target in zip(conversations, targets):
            cur_len = 1
            target[:cur_len] = IGNORE_TOKEN_ID
            parts = conversation.split(sep)
            assert len(parts) == 2
            parts[0] += sep
            # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
            instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            if False:  # Inspect and check the correctness of masking
                z = target.clone()
                z = torch.where(z == IGNORE_TOKEN_ID, self.tokenizer.unk_token_id, z)
                print(self.tokenizer.decode(z))
        targets = torch.where(targets==self.tokenizer.pad_token_id, IGNORE_TOKEN_ID, targets)
        if self.args.task_type!="behaviour":
            pre_user_pos = (input_ids==self.tokenizer.additional_special_tokens_ids[0]).nonzero()
            user_pos = [[] for _ in range(len(input_ids))]
            for i, pos in enumerate(pre_user_pos):
                user_pos[pos[0].item()].append(pos[1].item())
        else:
            user_pos = [[0] for _ in range(len(input_ids))]
            item_seq = [[self.pad_seq([])] for _ in range(len(input_ids))]

        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            labels=targets,
            user_pos=user_pos,
            item_seq=item_seq,
            item_pos=[[0] for _ in range(len(input_ids))],
            item_ids=[[0] for _ in range(len(input_ids))],
        )

    def process_uidiid2review(self, dataset):
        conv = template_dict['uidiid2review']
        conversations = []
        item_seq = []
        item_ids = []
        for example in dataset:
            conv.messages = []
            conv.append_message(conv.roles[0], example['question'])
            conv.append_message(conv.roles[1], example['answer'])
            conversations.append(conv.get_prompt(example['template']))
            if self.args.task_type!="behaviour":
                hist = self.user_history[example['uid']]
                hist = hist[:hist.index(example['iid'])] if example['iid']!=-1 else hist
                hist = self.pad_seq(hist)
                item_seq.append([hist])

                if example['type'] == 'uidiid2review':
                    item_ids.append([example['iid']])
                elif example['type'] == 'uidiid2rank':
                    item_ids.append(example['topk'])
                elif example['type'] == 'uidiid2binary':
                    item_ids.append([example['target']])
        
        input_ids = self.tokenizer(
            conversations,
            return_tensors="pt",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        targets = input_ids.clone()
        sep = conv.sep + conv.roles[1] + ": "
        for conversation, target in zip(conversations, targets):
            cur_len = 1
            target[:cur_len] = IGNORE_TOKEN_ID
            parts = conversation.split(sep)
            assert len(parts) == 2
            parts[0] += sep
            # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
            instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            if False:  # Inspect and check the correctness of masking
                z = target.clone()
                z = torch.where(z == IGNORE_TOKEN_ID, self.tokenizer.unk_token_id, z)
                print(self.tokenizer.decode(z))

        targets = torch.where(targets==self.tokenizer.pad_token_id, IGNORE_TOKEN_ID, targets)
        if self.args.task_type!="behaviour":
            pre_user_pos = (input_ids==self.tokenizer.additional_special_tokens_ids[0]).nonzero()
            user_pos = [[] for _ in range(len(input_ids))]
            for i, pos in enumerate(pre_user_pos):
                user_pos[pos[0].item()].append(pos[1].item())

            pre_item_pos = (input_ids==self.tokenizer.additional_special_tokens_ids[1]).nonzero()
            item_pos = [[] for _ in range(len(input_ids))]
            for i, pos in enumerate(pre_item_pos):
                item_pos[pos[0].item()].append(pos[1].item())
        else:
            user_pos = [[0] for _ in range(len(input_ids))]
            item_seq = [[self.pad_seq([])] for _ in range(len(input_ids))]
            item_pos = [[0] for _ in range(len(input_ids))]
            item_ids = [[0] for _ in range(len(input_ids))]

        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            labels=targets,
            user_pos=user_pos,
            item_seq=item_seq,
            item_pos=item_pos,
            item_ids=item_ids,
        )