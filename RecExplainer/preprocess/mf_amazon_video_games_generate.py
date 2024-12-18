# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import re
import json
import gzip
import torch
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import copy
from transformers import AutoTokenizer
import argparse

from utils import set_seed, read_seq_and_meta, gen_iid2text, gen_sharegpt, prompt_templates, gen_uid2summary

set_seed(2023)

def parse_args():
    parser = argparse.ArgumentParser(description="data process")
    parser.add_argument(
        "--sharegpt_file", type=str, help=""
    )
    parser.add_argument(
        "--seqdata_file", type=str, help=""
    )
    parser.add_argument(
        "--metadata_file", type=str, help=""
    )
    parser.add_argument(
        "--sim_item_file", type=str, default='', help=""
    )
    parser.add_argument(
        "--train_top_file", type=str, help="", default=None
    )
    parser.add_argument(
        "--test_top_file", type=str, help=""
    )
    parser.add_argument(
        "--gpt_response_file", type=str, help=""
    )
    parser.add_argument(
        "--gpt_query_file", type=str, help="", default=None
    )
    parser.add_argument(
        "--save_intention_file", type=str, help=""
    )
    parser.add_argument(
        "--save_behavior_file", type=str, help=""
    )
    parser.add_argument(
        "--save_both_file", type=str, help=""
    )
    parser.add_argument(
        "--max_seq_len", type=int, help=""
    )
    parser.add_argument(
        "--model_max_length", type=int, help=""
    )
    parser.add_argument(
        "--model_name", type=str, help=""
    )
    
    args = parser.parse_args()
    return args

def gen_uid2text(user_items, meta_infos, args):
    ## uid2hist
    train_uid2hist_intention = []
    train_uid2hist_both = []
    valid_uid2hist_intention = []
    valid_uid2hist_both = []
    for uid, items in user_items.items():
        if uid>len(user_items)*0.9:
            uid2hist_template = random.choice(prompt_templates['uid2hist'])
            valid_uid2hist_intention.append({'uid': uid, 'iid': items[-1], 'question': '<user>', 'answer': '; '.join([meta_infos[x]['title'] for x in items[-args.max_seq_len-1:-1]]), 'template': uid2hist_template, 'type': 'uid2hist'})
            valid_uid2hist_both.append(valid_uid2hist_intention[-1])
        else:
            train_uid2hist_intention.append({'uid': uid, 'iid': items[-1], 'question': '<user>', 'answer': '; '.join([meta_infos[x]['title'] for x in items[-args.max_seq_len-1:-1]]), 'template': random.choice(prompt_templates['uid2hist']), 'type': 'uid2hist'})
            train_uid2hist_both.append(train_uid2hist_intention[-1])
            train_uid2hist_intention.append({'uid': uid, 'iid': items[-1], 'question': '<user>', 'answer': '; '.join([meta_infos[x]['title'] for x in items[-args.max_seq_len-1:-1]]), 'template': random.choice(prompt_templates['uid2hist']), 'type': 'uid2hist'})
            train_uid2hist_both.append(train_uid2hist_intention[-1])
            train_uid2hist_intention.append({'uid': uid, 'iid': items[-1], 'question': '<user>', 'answer': '; '.join([meta_infos[x]['title'] for x in items[-args.max_seq_len-1:-1]]), 'template': random.choice(prompt_templates['uid2hist']), 'type': 'uid2hist'})
            train_uid2hist_both.append(train_uid2hist_intention[-1])
    print(f"train_uid2hist_intention: {len(train_uid2hist_intention)}, valid_uid2hist_intention: {len(valid_uid2hist_intention)}, train_uid2hist_both: {len(train_uid2hist_both)}, valid_uid2hist_both: {len(valid_uid2hist_both)}")
    train_intention.extend(train_uid2hist_intention)
    train_both.extend(train_uid2hist_both)
    valid_intention.extend(valid_uid2hist_intention)
    valid_both.extend(valid_uid2hist_both)

    ## uid2summary
    train_uid2summary_intention = []
    train_uid2summary_behavior = []
    train_uid2summary_both = []
    query_profile = pd.read_csv(args.gpt_response_file, header=None, sep=',', names=['question', 'target'])
    query_profile = query_profile.iloc[1:].reset_index(drop=True)

    for uid, items in user_items.items():
        if uid<=len(user_items)*0.9:
            answer = query_profile.loc[uid-1, 'target']
            uid2summary_template = random.choice(prompt_templates['uid2summary'])
            history = '; '.join([meta_infos[x]['title'] for x in items[-args.max_seq_len-1:-1]])
            train_uid2summary_intention.append({'uid': uid, 'iid': items[-1], 'question': '<user>', 'answer': answer, 'template': uid2summary_template, 'type': 'uid2summary'})
            train_uid2summary_behavior.append({'uid': uid, 'iid': items[-1], 'question': history, 'answer': answer, 'template': uid2summary_template, 'type': 'uid2summary'})
            train_uid2summary_both.append({'uid': uid, 'iid': items[-1], 'question': '<user>'+history, 'answer': answer, 'template': uid2summary_template, 'type': 'uid2summary'})
    print(f"train_uid2summary_intention: {len(train_uid2summary_intention)}, train_uid2summary_behavior: {len(train_uid2summary_behavior)}, train_uid2summary_both: {len(train_uid2summary_both)}")
    train_intention.extend(train_uid2summary_intention)
    train_behavior.extend(train_uid2summary_behavior)
    train_both.extend(train_uid2summary_both)

def gen_uid2next_rank(user_items, args):
    with open(args.test_top_file, 'r') as f:
        for idx, line in enumerate(f):
            uid2next_intention = []
            uid2next_behavior = []
            uid2next_both = []
            uidiid2rank_intention = []
            uidiid2rank_behavior = []
            uidiid2rank_both = []
            uidiid2binary_intention = []
            uidiid2binary_behavior = []
            uidiid2binary_both = []

            uid, iid, top1, topk, pos, neg, pos_score, neg_score = line.strip().split('\t') 
            uid, iid, top1, topk, pos, neg = int(uid), int(iid), int(top1), [int(x) for x in topk.split(',')], int(pos), int(neg)
            history = user_items[uid]
            history = history[:history.index(iid)][-args.max_seq_len:]
            history = '; '.join([meta_infos[x]['title'] for x in history])
            
            uid2next_template = random.choice(prompt_templates['uid2next'])
            uid2next_intention.append({'uid': uid, 'iid': iid, 'question': '<user>', 'answer': meta_infos[top1]['title'], 'template': uid2next_template, 'type': 'uid2next'})
            uid2next_behavior.append({'uid': uid, 'iid': iid, 'question': history, 'answer': meta_infos[top1]['title'], 'template': uid2next_template, 'type': 'uid2next'})
            uid2next_both.append({'uid': uid, 'iid': iid, 'question': '<user>'+history, 'answer': meta_infos[top1]['title'], 'template': uid2next_template, 'type': 'uid2next'})
            
            uidiid2template = random.choice(prompt_templates['uidiid2rank'])
            ranks_intention = {'uid': uid, 'iid': iid, 'template': uidiid2template, 'type': 'uidiid2rank'}
            ranks_intention['question'] = ('<user>', '<item>'*len(topk))
            ranks_intention['answer'] = '; '.join([meta_infos[x]['title'] for x in topk])
            random.shuffle(topk)
            ranks_intention['topk'] = topk
            ranks_behavior = {'uid': uid, 'iid': iid, 'template': uidiid2template, 'type': 'uidiid2rank', 'question': (history, '; '.join([meta_infos[x]['title'] for x in topk])), 'answer': ranks_intention['answer'], 'topk': topk}
            ranks_both = {'uid': uid, 'iid': iid, 'template': uidiid2template, 'type': 'uidiid2rank', 'question': ('<user>'+history, '; '.join(['<item>'+meta_infos[x]['title'] for x in topk])), 'answer': ranks_intention['answer'], 'topk': topk}
            uidiid2rank_intention.append(ranks_intention)
            uidiid2rank_behavior.append(ranks_behavior)
            uidiid2rank_both.append(ranks_both)

            pos_template = random.choice(prompt_templates['uidiid2binary'])
            neg_template = random.choice(prompt_templates['uidiid2binary'])
            uidiid2binary_intention.append({'uid': uid, 'iid': iid, 'target': pos, 'question': ('<user>', '<item>'), 'answer': 'Yes', 'template': pos_template, 'type': 'uidiid2binary'})
            uidiid2binary_intention.append({'uid': uid, 'iid': iid, 'target': neg, 'question': ('<user>', '<item>'), 'answer': 'No', 'template': neg_template, 'type': 'uidiid2binary'})
            uidiid2binary_behavior.append({'uid': uid, 'iid': iid, 'target': pos, 'question': (history, meta_infos[pos]['title']), 'answer': 'Yes', 'template': pos_template, 'type': 'uidiid2binary'})
            uidiid2binary_behavior.append({'uid': uid, 'iid': iid, 'target': neg, 'question': (history, meta_infos[neg]['title']), 'answer': 'No', 'template': neg_template, 'type': 'uidiid2binary'})
            uidiid2binary_both.append({'uid': uid, 'iid': iid, 'target': pos, 'question': ('<user>'+history, '<item>'+meta_infos[pos]['title']), 'answer': 'Yes', 'template': pos_template, 'type': 'uidiid2binary'})
            uidiid2binary_both.append({'uid': uid, 'iid': iid, 'target': neg, 'question': ('<user>'+history, '<item>'+meta_infos[neg]['title']), 'answer': 'No', 'template': neg_template, 'type': 'uidiid2binary'})

            if idx%6<5:
                if uid>len(user_items)*0.9:
                    valid_intention.extend(uid2next_intention)
                    valid_behavior.extend(uid2next_behavior)
                    valid_both.extend(uid2next_both)
                else:
                    train_intention.extend(uid2next_intention)
                    train_behavior.extend(uid2next_behavior)
                    train_both.extend(uid2next_both)
            if idx%6<5:
                train_intention.extend(uidiid2rank_intention)
                train_behavior.extend(uidiid2rank_behavior)
                train_both.extend(uidiid2rank_both)
            elif idx%6==5:
                valid_intention.extend(uidiid2rank_intention)
                valid_behavior.extend(uidiid2rank_behavior)
                valid_both.extend(uidiid2rank_both)

            if idx%6<3:
                train_intention.extend(uidiid2binary_intention)
                train_behavior.extend(uidiid2binary_behavior)
                train_both.extend(uidiid2binary_both)
            elif idx%6==3 and random.random()<0.5:
                valid_intention.extend(uidiid2binary_intention)
                valid_behavior.extend(uidiid2binary_behavior)
                valid_both.extend(uidiid2binary_both)
            

if __name__ == '__main__':
    args = parse_args()
    user_items, meta_infos = read_seq_and_meta(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=args.model_max_length, use_fast=True)

    if args.gpt_query_file is not None:
        os.makedirs(os.path.dirname(args.gpt_query_file), exist_ok=True)
        gen_uid2summary(user_items, meta_infos, args)
    else:
        os.makedirs(os.path.dirname(args.save_intention_file), exist_ok=True)
        train_intention = []
        train_behavior = []
        train_both = []
        valid_intention = []
        valid_behavior = []
        valid_both = []
        
        train_iid2text_intention, train_iid2text_behavior, train_iid2text_both, valid_iid2text_intention, valid_iid2text_both = gen_iid2text(meta_infos, args, tokenizer, prompt_templates)
        train_intention.extend(train_iid2text_intention)
        train_behavior.extend(train_iid2text_behavior)
        train_both.extend(train_iid2text_both)
        valid_intention.extend(valid_iid2text_intention)
        valid_both.extend(valid_iid2text_both)

        gen_uid2text(user_items, meta_infos, args)
        gen_uid2next_rank(user_items, args)

        train_sharegpt, valid_sharegpt = gen_sharegpt(args, tokenizer, 9000, 900)
        train_intention.extend(train_sharegpt)
        train_behavior.extend(train_sharegpt)
        train_both.extend(train_sharegpt)
        valid_intention.extend(valid_sharegpt)
        valid_behavior.extend(valid_sharegpt)
        valid_both.extend(valid_sharegpt)


        print(f"train_intention: {len(train_intention)}, train_behavior: {len(train_behavior)}, train_both: {len(train_both)}")
        print(f"valid_intention: {len(valid_intention)}, valid_behavior: {len(valid_behavior)}, valid_both: {len(valid_both)}")

        json.dump(train_intention, open(args.save_intention_file+'_train.json', 'w'))
        json.dump(train_behavior, open(args.save_behavior_file+'_train.json', 'w'))
        json.dump(train_both, open(args.save_both_file+'_train.json', 'w'))
        json.dump(valid_intention, open(args.save_intention_file+'_valid.json', 'w'))
        json.dump(valid_behavior, open(args.save_behavior_file+'_valid.json', 'w'))
        json.dump(valid_both, open(args.save_both_file+'_valid.json', 'w'))