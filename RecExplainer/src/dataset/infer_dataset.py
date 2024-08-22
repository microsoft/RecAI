# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import random
import json
import torch
import os
from torch.utils.data import Dataset

from dataset.conversation import template_dict

logger = logging.getLogger(__name__)

class InferDataset4Exp(Dataset):
    def __init__(self, args, tokenizer):
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
            'uidiid2explan': self.process_uidiid2review,
            "demo": self.process_demo,
        }
        self.tokenizer = tokenizer
        self.args = args
        self.user_history = self.load_user_history(args.sequential_file)
        self.max_hist_len = args.max_hist_len
        
        self.dataset = json.load(open(args.validation_file, "r"))

        if args.inference_mode!='case study':
            filter_dataset = []
            for data_item in self.dataset:
                if data_item['type'] == args.inference_mode:
                    filter_dataset.append(data_item)
            self.dataset = filter_dataset
            logger.info(f"Remain {len(filter_dataset)} examples with inference_mode: {args.inference_mode}")
        
        if len(self.dataset) > args.max_example_num:
            self.dataset = self.dataset[:args.max_example_num]
            logger.info(f"Remain {len(self.dataset)} examples with max_example_num: {args.max_example_num}")

        self.meta_infos = {}
        with open(args.metadata_file, 'r') as f:
            for i, line in enumerate(f):
                line = json.loads(line)
                self.meta_infos[i+1] = line
            
        self.uidiid2topk = {}
        with open(args.test_top_file, 'r') as f:
            for line in f:
                uid, iid, top1, topk, pos, neg, _, _ = line.strip().split('\t') 
                uid, iid, top1, topk, pos, neg = int(uid), int(iid), int(top1), [int(x) for x in topk.split(',')], int(pos), int(neg)
                self.uidiid2topk[(uid, iid)] = topk

        self.total_len = len(self.dataset)
        

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        data_item = self.dataset[idx]
        func = self.function_map[data_item['type']]
        data = func([data_item])

        return {
            'input_ids': data['input_ids'][0],
            'attention_mask': data['attention_mask'][0],
            'user_pos': data['user_pos'][0],
            'user_ids': data['user_ids'][0],
            'item_seq': data['item_seq'][0],
            'item_pos': data['item_pos'][0],
            'item_ids': data['item_ids'][0],
            'answers': data['answers'][0] if 'answers' in data else None,
        }

    def get_title(self, meta_infos, iid):
        if 'title' in meta_infos[iid]:
            return meta_infos[iid]['title']
        elif 'TitleName' in meta_infos[iid]:
            return meta_infos[iid]['TitleName']
        else:
            raise ValueError(f"Cannot find title for iid: {iid}")

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
        
        conv = template_dict[self.args.template_name]
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
                if j % 2 == 0:
                    conv.append_message(role, sentence["value"])
                else:
                    conv.append_message(role, None)
                    break
            # if flag == 1:
            #     continue
            conversations.append(conv.get_prompt(name='sharegpt'))

        # Tokenize conversations
        inputs = self.tokenizer(
            conversations,
            return_tensors="pt",
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )

        return dict(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            user_pos=[[0] for _ in range(len(inputs.input_ids))],
            user_ids=[[0] for _ in range(len(inputs.input_ids))],
            item_seq=[[self.pad_seq([])] for _ in range(len(inputs.input_ids))],
            item_pos=[[0] for _ in range(len(inputs.input_ids))],
            item_ids=[[0] for _ in range(len(inputs.input_ids))],
        )

    def process_iid2title(self, dataset):
        conv = template_dict[self.args.template_name]
        conversations = []
        item_ids = []
        answers = []
        for example in dataset:
            conv.messages = []
            conv.append_message(conv.roles[0], example['question'])
            conv.append_message(conv.roles[1], None)
            conversations.append(conv.get_prompt(name=example['type'], template=example['template']))

            if self.args.task_type=="intention" or self.args.task_type=="both":
                item_ids.append([example['iid']])
            answers.append(example['answer'].strip().lower())
            
        inputs = self.tokenizer(
            conversations,
            return_tensors="pt",
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )

        if self.args.task_type=="intention" or self.args.task_type=="both":
            pre_item_pos = (inputs.input_ids==self.tokenizer.additional_special_tokens_ids[1]).nonzero()
            item_pos = [[] for _ in range(len(inputs.input_ids))]
            for i, pos in enumerate(pre_item_pos):
                item_pos[pos[0].item()].append(pos[1].item())
        else:
            item_pos = [[0] for _ in range(len(inputs.input_ids))]
            item_ids = [[0] for _ in range(len(inputs.input_ids))]

        return dict(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            user_pos=[[0] for _ in range(len(inputs.input_ids))],
            user_ids=[[0] for _ in range(len(inputs.input_ids))],
            item_seq=[[self.pad_seq([])] for _ in range(len(inputs.input_ids))],
            item_pos=item_pos,
            item_ids=item_ids,
            answers=answers,
        )

    def process_uid2hist(self, dataset):
        conv = template_dict[self.args.template_name]
        conversations = []
        item_seq = []
        user_ids = []
        answers = []
        for example in dataset:
            conv.messages = []
            conv.append_message(conv.roles[0], example['question'])
            conv.append_message(conv.roles[1], None)
            conversations.append(conv.get_prompt(name=example['type'], template=example['template']))

            hist = self.user_history[example['uid']]
            hist = hist[:hist.index(example['iid'])] if example['iid']!=-1 else hist
            hist = self.pad_seq(hist)

            if self.args.task_type=="intention" or self.args.task_type=="both":
                item_seq.append([hist])
                user_ids.append([example['uid']])

            if example['type'] == 'uid2hist':
                answers.append([self.get_title(self.meta_infos, iid).strip().lower() for iid in hist if iid!=0])
            else:
                answers.append(example['answer'].strip().lower())
        
        inputs = self.tokenizer(
            conversations,
            return_tensors="pt",
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        
        if self.args.task_type=="intention" or self.args.task_type=="both":
            pre_user_pos = (inputs.input_ids==self.tokenizer.additional_special_tokens_ids[0]).nonzero()
            user_pos = [[] for _ in range(len(inputs.input_ids))]
            for i, pos in enumerate(pre_user_pos):
                user_pos[pos[0].item()].append(pos[1].item())
        else:
            user_pos = [[0] for _ in range(len(inputs.input_ids))]
            item_seq = [[self.pad_seq([])] for _ in range(len(inputs.input_ids))]
            user_ids = [[0] for _ in range(len(inputs.input_ids))]

        return dict(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            user_pos=user_pos,
            user_ids=user_ids,
            item_seq=item_seq,
            item_pos=[[0] for _ in range(len(inputs.input_ids))],
            item_ids=[[0] for _ in range(len(inputs.input_ids))],
            answers=answers,
        )

    def process_uidiid2review(self, dataset):
        conv = template_dict[self.args.template_name]
        conversations = []
        item_seq = []
        item_ids = []
        user_ids = []
        answers = []
        for example in dataset:
            conv.messages = []
            conv.append_message(conv.roles[0], example['question'])
            conv.append_message(conv.roles[1], None)
            conversations.append(conv.get_prompt(name=example['type'], template=example['template']))

            if self.args.task_type=="intention" or self.args.task_type=="both":
                user_ids.append([])
                item_seq.append([])
                item_ids.append([])
                if 'template_uidiidtargets' in example and example['type'] == 'uidiid2explan':
                    for uid, iid, target in example['template_uidiidtargets']:
                        hist = self.user_history[uid]
                        hist = hist[:hist.index(iid)] if iid!=-1 else hist
                        hist = self.pad_seq(hist)
                        item_seq[-1].append(hist)
                        item_ids[-1].append(target)
                        user_ids[-1].append(uid)

                hist = self.user_history[example['uid']]
                hist = hist[:hist.index(example['iid'])] if example['iid']!=-1 else hist
                hist = self.pad_seq(hist)
                item_seq[-1].append(hist)
                user_ids[-1].append(example['uid'])

                if example['type'] == 'uidiid2review':
                    item_ids[-1].append(example['iid'])
                elif example['type'] == 'uidiid2rank':
                    item_ids[-1].extend(example['topk'])
                elif example['type'] == 'uidiid2binary' or example['type'] == 'uidiid2explan':
                    item_ids[-1].append(example['target'])
            
            if example['type'] == 'uidiid2rank':
                answers.append([self.get_title(self.meta_infos, iid).strip().lower() for iid in self.uidiid2topk[(example['uid'], example['iid'])]])
            elif example['type'] == 'uidiid2explan':
                answers.append(example['answer'])
            else:
                answers.append(example['answer'].strip().lower())
        
        inputs = self.tokenizer(
            conversations,
            return_tensors="pt",
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )

        if self.args.task_type=="intention" or self.args.task_type=="both":
            pre_user_pos = (inputs.input_ids==self.tokenizer.additional_special_tokens_ids[0]).nonzero()
            user_pos = [[] for _ in range(len(inputs.input_ids))]
            for i, pos in enumerate(pre_user_pos):
                user_pos[pos[0].item()].append(pos[1].item())

            pre_item_pos = (inputs.input_ids==self.tokenizer.additional_special_tokens_ids[1]).nonzero()
            item_pos = [[] for _ in range(len(inputs.input_ids))]
            for i, pos in enumerate(pre_item_pos):
                item_pos[pos[0].item()].append(pos[1].item())
        else:
            user_pos = [[0] for _ in range(len(inputs.input_ids))]
            item_seq = [[self.pad_seq([])] for _ in range(len(inputs.input_ids))]
            item_pos = [[0] for _ in range(len(inputs.input_ids))]
            item_ids = [[0] for _ in range(len(inputs.input_ids))]
            user_ids = [[0] for _ in range(len(inputs.input_ids))]


        return dict(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            user_pos=user_pos,
            user_ids=user_ids,
            item_seq=item_seq,
            item_pos=item_pos,
            item_ids=item_ids,
            answers=answers,
        )

    def process_demo(self, dataset):
        conv = template_dict[self.args.template_name]
        conversations = []
        item_seq = []
        item_ids = []
        answers = []
        for example in dataset:
            conv.messages = []
            conv.append_message(conv.roles[0], example['question'])
            conv.append_message(conv.roles[1], None)
            conversations.append(conv.get_prompt(name=example['type'], template=example['template']))

            if self.args.task_type=="intention" or self.args.task_type=="both":
                item_seq.append([])
                item_ids.append([])

                hist = example['hist_ids']
                hist = self.pad_seq(hist)
                item_seq[-1].append(hist)
                item_ids[-1].append(example['target'])
        
        inputs = self.tokenizer(
            conversations,
            return_tensors="pt",
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )

        if self.args.task_type=="intention" or self.args.task_type=="both":
            pre_user_pos = (inputs.input_ids==self.tokenizer.additional_special_tokens_ids[0]).nonzero()
            user_pos = [[] for _ in range(len(inputs.input_ids))]
            for i, pos in enumerate(pre_user_pos):
                user_pos[pos[0].item()].append(pos[1].item())

            pre_item_pos = (inputs.input_ids==self.tokenizer.additional_special_tokens_ids[1]).nonzero()
            item_pos = [[] for _ in range(len(inputs.input_ids))]
            for i, pos in enumerate(pre_item_pos):
                item_pos[pos[0].item()].append(pos[1].item())
        else:
            user_pos = [[0] for _ in range(len(inputs.input_ids))]
            item_seq = [[self.pad_seq([])] for _ in range(len(inputs.input_ids))]
            item_pos = [[0] for _ in range(len(inputs.input_ids))]
            item_ids = [[0] for _ in range(len(inputs.input_ids))]


        return dict(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            user_pos=user_pos,
            user_ids=[[0] for _ in range(len(inputs.input_ids))], # currently not supported MF model
            item_seq=item_seq,
            item_pos=item_pos,
            item_ids=item_ids,
        )