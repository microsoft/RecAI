# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import random
import pandas as pd
import math
from transformers import AutoTokenizer
from collections import defaultdict
from tqdm import tqdm
import argparse
import os
from test_template import user2item_template, query2item_template, title2item_template, item2item_template, queryuser2item_template, vaguequery2item_template, relativequery2item_template, negquery2item_template

from utils import get_item_text, text4query2item, cal_item2pos, text4item2item, random_replace, vaguequery, get_item_stats, get_feature2itemid, text4negquery, get_price_date_stats
random.seed(2023)


def parse_args():
    parser = argparse.ArgumentParser(description="data process")
    parser.add_argument(
        "--in_seq_data", type=str, help=""
    )
    parser.add_argument(
        "--in_meta_data", type=str, help=""
    )
    parser.add_argument(
        "--out_user2item", type=str, help=""
    )
    parser.add_argument(
        "--out_query2item", type=str, help=""
    )
    parser.add_argument(
        "--out_title2item", type=str, help=""
    )
    parser.add_argument(
        "--out_item2item", type=str, help=""
    )
    parser.add_argument(
        "--out_queryuser2item", type=str, help=""
    )
    parser.add_argument(
        "--out_misspell2item", type=str, help=""
    )
    parser.add_argument(
        "--out_sparse_query2item", type=str, help=""
    )
    parser.add_argument(
        "--out_vaguequery2item", type=str, help=""
    )
    parser.add_argument(
        "--out_relativequery2item", type=str, help=""
    )
    parser.add_argument(
        "--out_negquery2item", type=str, help=""
    )
    parser.add_argument(
        "--model_path_or_name", type=str, help=""
    )
    parser.add_argument(
        "--max_samples_per_task", type=int, default=100000000000, help=""
    )
    args = parser.parse_args()
    return args

def gen_user2item(itemid2title, args, has_prefix=False):
    with open(args.in_seq_data, 'r') as rd:
        all_samples = rd.readlines()
    if len(all_samples) > args.max_samples_per_task:
        all_samples = random.sample(all_samples, args.max_samples_per_task)
        
    dataset=[]
    for idx, line in tqdm(enumerate(all_samples), desc='gen_user2item'):
        userid, itemids = line.strip().split(' ', 1)
        itemids = itemids.split(' ')
        data = {}
        data["user_id"] = int(userid)
        item_titles = ', '.join([itemid2title[int(x)][0]+itemid2title[int(x)][1] if has_prefix else itemid2title[int(x)][1] for x in itemids[:-1][::-1][:20]])
        if random.random() < 0.5:
            template = "{}"
        else:
            template = random.choice(user2item_template)
        
        data["text"] = template.format(item_titles)
        data["ground_truth"] = int(itemids[-1])
        data["history"] = [int(x) for x in itemids[:-1]]
        dataset.append(data)

    print('gen_user2item total samples: ', len(dataset))
    with open(args.out_user2item, "w", encoding='utf-8') as fd:
        for d in dataset:
            fd.write(json.dumps(d, ensure_ascii=False) + '\n')

def gen_query2item(itemid2title, itemid2features, args):
    dataset=[]
    for idx, cont in tqdm(enumerate(itemid2features[1:]), desc='gen_query2item', total=len(itemid2features)-1):
        for _ in range(3):
            target_item_title = itemid2title[idx+1][1]
            target_features = [itemid2title[idx+1]] + cont if random.random() < 0.5 else cont
            query, sampled_features, ground_truth = text4query2item(target_features, target_item_title, 1, len(target_features), 1, math.inf)
            template = random.choice(query2item_template)

            template_length = len(tokenizer.tokenize(template))
            tokens = tokenizer.tokenize(query)[:args.max_seq_len-template_length]
            truncated_query = tokenizer.convert_tokens_to_string(tokens).strip()

            query = template.format(truncated_query)
            data = {'item_id': idx+1, 'text': query, 'ground_truth': ground_truth}
            dataset.append(data)
    if len(dataset) > args.max_samples_per_task:
        dataset = random.sample(dataset, args.max_samples_per_task)
    print('gen_query2item total samples: ', len(dataset))
    with open(args.out_query2item, "w", encoding='utf-8') as fd:
        for d in dataset:
            fd.write(json.dumps(d, ensure_ascii=False) + '\n')

def gen_title2item(itemid2title, title2itemid, args):
    dataset=[]
    for idx, cont in tqdm(enumerate(itemid2title[1:]), desc='gen_title2item', total=len(itemid2title)-1):
        target_item_title = cont[1]
        for _ in range(1):
            query = target_item_title
            template = random.choice(title2item_template)

            template_length = len(tokenizer.tokenize(template))
            tokens = tokenizer.tokenize(query)[:args.max_seq_len-template_length]
            truncated_query = tokenizer.convert_tokens_to_string(tokens).strip()

            query = template.format(truncated_query)
            data = {'item_id': idx+1, 'text': query, 'ground_truth': title2itemid[target_item_title]}
            dataset.append(data)
    if len(dataset) > args.max_samples_per_task:
        dataset = random.sample(dataset, args.max_samples_per_task)
    print('gen_title2item total samples: ', len(dataset))
    with open(args.out_title2item, 'w', encoding='utf-8') as fd:
        for d in dataset:
            fd.write(json.dumps(d, ensure_ascii=False) + '\n')

def gen_item2item(itemid2title, itemid2features, args):
    item2pos = cal_item2pos(args.in_seq_data)
    dataset=[]
    for item, pos_set in tqdm(item2pos.items(), desc='gen_item2item', total=len(item2pos)):
        source_item_features = itemid2features[item]
        source_item_title = itemid2title[item][1]
        for _ in range(2):
            query = text4item2item(source_item_features, source_item_title)
            
            template = random.choice(item2item_template)
            template_length = len(tokenizer.tokenize(template))
            tokens = tokenizer.tokenize(query)[:args.max_seq_len-template_length]
            truncated_query = tokenizer.convert_tokens_to_string(tokens).strip()

            query = template.format(truncated_query)
            data = {'item_id': item, 'text': query, 'ground_truth': list(pos_set)}
            dataset.append(data)
    if len(dataset) > args.max_samples_per_task:
        dataset = random.sample(dataset, args.max_samples_per_task)
    print('gen_item2item total samples: ', len(dataset))
    with open(args.out_item2item, 'w', encoding='utf-8') as fd:
        for d in dataset:
            fd.write(json.dumps(d, ensure_ascii=False) + '\n')

def gen_queryuser2item(itemid2title, itemid2features, args):
    with open(args.in_seq_data, 'r') as rd:
        all_samples = rd.readlines()
    if len(all_samples) > args.max_samples_per_task//2:
        all_samples = random.sample(all_samples, args.max_samples_per_task//2)
        
    dataset=[]
    for idx, line in tqdm(enumerate(all_samples), desc='gen_queryuser2item'):
        userid, itemids = line.strip().split(' ', 1)
        itemids = itemids.split(' ')
        target_item = int(itemids[-1])
        user_hist = [int(x) for x in itemids[:-1]][::-1]
        template = random.choice(queryuser2item_template)
        query = ''
        for x in user_hist[:20]:
            query += itemid2title[int(x)][1] + ', '
        query = query.strip().strip(',')

        target_item_title = itemid2title[target_item]

        if random.random() < 0.6:
            target_features = [target_item_title] + itemid2features[target_item] if random.random() < 0.5 else itemid2features[target_item]
            target_query, _, _ = text4query2item(target_features, target_item_title[1], 1, min(4, len(target_features)), 1, math.inf)## don't provide too much info
        else: #sparse case
            target_features = [x for x in itemid2features[target_item] if x[0] not in ['description: ']]
            target_query, _, _ = text4query2item(target_features, target_item_title[1], 1, min(3, len(target_features)), 1, 2)## don't provide too much info
        

        template_length = len(tokenizer.tokenize(template))
        tokens = tokenizer.tokenize(query)
        query_length = min((args.max_seq_len-template_length)*4//5, len(tokens))
        target_query_length = args.max_seq_len-template_length-query_length

        tokens = tokens[:query_length]
        truncated_query = tokenizer.convert_tokens_to_string(tokens).strip().strip(',')
        target_tokens = tokenizer.tokenize(target_query)[:target_query_length]
        truncated_target_query = tokenizer.convert_tokens_to_string(target_tokens).strip().strip(',')
        
        query = template.format(truncated_query, truncated_target_query)
        data = {'user_id': int(userid), 'text': query, 'ground_truth': target_item, 'history': user_hist[::-1]}
        dataset.append(data)
    print('gen_queryuser2item total samples: ', len(dataset))
    with open(args.out_queryuser2item, 'w') as fd:
        for d in dataset:
            fd.write(json.dumps(d, ensure_ascii=False) + '\n')  

def gen_misspell2item(itemid2title, title2itemid, args):
    dataset=[]
    for idx, cont in tqdm(enumerate(itemid2title[1:]), desc='gen_misspell2item', total=len(itemid2title)-1):
        target_item_title = cont[1]
        query = random_replace(target_item_title)
        while query == target_item_title:
            query = random_replace(target_item_title)
        if random.random() < 0.5:
            template = "{}"
        else:
            template = random.choice(title2item_template)
        query = template.format(query)
        data = {'item_id': idx+1, 'text': query, 'ground_truth': title2itemid[target_item_title]}
        dataset.append(data)
    if len(dataset) > args.max_samples_per_task:
        dataset = random.sample(dataset, args.max_samples_per_task)
    print('gen_misspell2item total samples: ', len(dataset))
    with open(args.out_misspell2item, 'w', encoding='utf-8') as fd:
        for d in dataset:
            fd.write(json.dumps(d, ensure_ascii=False) + '\n')

def gen_sparse_query2item(itemid2title, itemid2features, args): # query with #conditions <= 3
    dataset=[]
    for idx, cont in tqdm(enumerate(itemid2features[1:]), desc='gen_sparse_query2item', total=len(itemid2features)-1):
        for _ in range(3):
            target_item_title = itemid2title[idx+1][1]
            target_features = [x for x in cont if x[0] not in ['description: ']]
            query, _, ground_truth = text4query2item(target_features, target_item_title, 1, min(3, len(target_features)), 1, 2)
            
            template = random.choice(query2item_template)

            template_length = len(tokenizer.tokenize(template))
            tokens = tokenizer.tokenize(query)[:args.max_seq_len-template_length]
            truncated_query = tokenizer.convert_tokens_to_string(tokens).strip()

            query = template.format(truncated_query)
            data = {'item_id': idx+1, 'text': query, 'ground_truth': ground_truth}
            dataset.append(data)
    if len(dataset) > args.max_samples_per_task:
        dataset = random.sample(dataset, args.max_samples_per_task)
    print('gen_sparse_query2item total samples: ', len(dataset))
    with open(args.out_sparse_query2item, "w", encoding='utf-8') as fd:
        for d in dataset:
            fd.write(json.dumps(d, ensure_ascii=False) + '\n')

def gen_vaguequery2item(itemid2price_date_map, args):
    dataset=[]
    price_date_stats = get_price_date_stats(itemid2price_date_map)
    for idx, cont in tqdm(enumerate(itemid2price_date_map[1:]), desc='gen_vaguequery2item', total=len(itemid2price_date_map)-1):
        price, date, next_month, last_month = cont['price'], cont['release date'], cont['next month'], cont['last month']
        if not price or not date or not next_month or not last_month:
            continue
        for _ in range(3):
            query, combine_flag, price_flag, month_flag, year_flag = vaguequery(price, date, next_month, last_month, price_date_stats)
            if month_flag:
                year_flag = (year_flag[0], year_flag[1].strftime("%B %d, %Y")) # datetime to str for json saving
            template = random.choice(vaguequery2item_template)
            data = {'item_id': idx+1, 'text': template.format(query), 'ground_truth': [combine_flag, price_flag, month_flag, year_flag]}
            dataset.append(data)
    if len(dataset) > args.max_samples_per_task:
        dataset = random.sample(dataset, args.max_samples_per_task)
    print('gen_vaguequery2item total samples: ', len(dataset))
    with open(args.out_vaguequery2item, 'w', encoding='utf-8') as fd:
        for d in dataset:
            fd.write(json.dumps(d) + '\n')

def gen_relativequery2item(args):
    recent_itemset, cheap_itemset, expensive_itemset, popular_itemset, total_num = get_item_stats(args.in_seq_data, args.in_meta_data)
    dataset=[]
    for task, itemset in zip(['recent', 'cheap', 'expensive', 'popular'], [recent_itemset, cheap_itemset, expensive_itemset, popular_itemset]):
        for target_item in itemset:
            for _ in range(2):
                query = random.choice(relativequery2item_template[task])
                data = {'item_id': target_item, 'text': query, 'ground_truth': list(itemset)}
                dataset.append(data)
    if len(dataset) > args.max_samples_per_task:
        dataset = random.sample(dataset, args.max_samples_per_task)
    print('gen_relativequery2item total samples: ', len(dataset))
    with open(args.out_relativequery2item, 'w', encoding='utf-8') as fd:
        for d in dataset:
            fd.write(json.dumps(d) + '\n')

def gen_negquery2item(itemid2text, args):
    features2itemids = get_feature2itemid(args.in_meta_data)
    sample_names_l1 = list(features2itemids.keys())
    sample_names_l2 = {x: list(features2itemids[x].keys()) for x in sample_names_l1}
    dataset=[]
    for _ in tqdm(range(7000), desc='gen_negquery2item'):
        query, pos_set, neg_set = text4negquery(sample_names_l1, sample_names_l2, itemid2text, features2itemids)
        if len(pos_set) == 0 or len(neg_set) <= 3:
            continue
        template = random.choice(negquery2item_template)
        data = {'text': template.format(query), 'ground_truth': list(pos_set)}
        dataset.append(data)
        
    if len(dataset) > args.max_samples_per_task:
        dataset = random.sample(dataset, args.max_samples_per_task)
    print('gen_negquery2item total samples: ', len(dataset))
    with open(args.out_negquery2item, 'w', encoding='utf-8') as fd:
        for d in dataset:
            fd.write(json.dumps(d) + '\n')

if __name__ == "__main__":
    args = parse_args() 
    os.makedirs(os.path.dirname(args.out_user2item), exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_name, use_fast=True)
    args.max_seq_len = tokenizer.model_max_length
    itemid2text, itemid2title, itemid2features, itemid2price_date_map = get_item_text(args.in_meta_data)
    title2itemid = defaultdict(list)
    for idx, v in enumerate(itemid2title):
        if v[1] is not None:
            title2itemid[v[1]].append(idx)
    
    gen_user2item(itemid2title, args)
    gen_query2item(itemid2title, itemid2features, args)
    gen_title2item(itemid2title, title2itemid, args)
    gen_item2item(itemid2title, itemid2features, args)
    gen_queryuser2item(itemid2title, itemid2features, args)
    gen_misspell2item(itemid2title, title2itemid, args)
    gen_sparse_query2item(itemid2title, itemid2features, args)
    gen_vaguequery2item(itemid2price_date_map, args)
    gen_relativequery2item(args)
    gen_negquery2item(itemid2text, args)
    