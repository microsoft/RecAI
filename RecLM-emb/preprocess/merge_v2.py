# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import random
import pandas as pd
from tqdm import tqdm
import argparse
import os
import math

from utils import get_item_text, transform_date_format, get_feature2itemid
random.seed(2024)

def parse_args():
    parser = argparse.ArgumentParser(description="data process")
    parser.add_argument(
        "--in_seq_data", type=str, help="", default=""
    )
    parser.add_argument(
        "--in_meta_data", type=str, help="", default=""
    )
    parser.add_argument(
        "--out_conv", type=str, help="", default=""
    )
    parser.add_argument(
        "--out_gpt_conv", type=str, help="", default=""
    )
    parser.add_argument(
        "--out_user_sum", type=str, help="", default=""
    )
    parser.add_argument(
        "--out_gpt_user_sum", type=str, help="", default=""
    )
    parser.add_argument(
        "--out_query", type=str, help="", default=""
    )
    parser.add_argument(
        "--out_gpt_query", type=str, help="", default=""
    )
    parser.add_argument(
        "--out_neg_query", type=str, help="", default=""
    )
    parser.add_argument(
        "--out_gpt_neg_query", type=str, help="", default=""
    )
    parser.add_argument(
        "--out_new_gpt", type=str, help="", default=""
    )
    parser.add_argument(
        "--neg_num", type=int, help="", default=7
    )
    args = parser.parse_args()
    return args

def sample_conv(args, itemid2text, out_text):
    query_profile = pd.read_csv(args.out_gpt_conv, header=None, sep=',', names=['question', 'response'])
    query_profile = query_profile.iloc[1:]
    query_profile = query_profile.reset_index(drop=True)
    n_rows = len(query_profile)
    with open(args.out_conv, 'r') as f:
        for idx, line in tqdm(enumerate(f), desc='sample_conv'):
            line = json.loads(line)
            userid = int(line['user_id'])
            target_item = int(line['target_id'])
            query = query_profile.loc[idx, 'response']
            if not isinstance(query, str):
                continue
            if random.random() < 0.25:
                try:
                    temp_q = json.loads(query)
                    if temp_q[-1]["role"] == "Assistant":
                        query = temp_q[:-1]
                        query = json.dumps(query)
                except:
                    pass
                
            neg_items = []
            while len(neg_items) < args.neg_num:
                neg_item = random.randint(1, len(itemid2text)-1)
                if neg_item != target_item:
                    neg_items.append(neg_item)
            output = {
                'query': query,
                'pos': [itemid2text[target_item]],
                'neg': [itemid2text[x] for x in neg_items]
            }
            out_text.append(output)
            if idx == n_rows-1:
                break
    
    return out_text

def sample_user_sum(args, itemid2text, out_text):
    user2seq = {}
    with open(args.in_seq_data, 'r') as f:
        for line in f:
            line = line.strip().split()
            userid = int(line[0])
            seq = [int(x) for x in line[1:]]
            user2seq[userid] = seq

    query_profile = pd.read_csv(args.out_gpt_user_sum, header=None, sep=',', names=['question', 'response'])
    query_profile = query_profile.iloc[1:]
    query_profile = query_profile.reset_index(drop=True)
    n_rows = len(query_profile)
    with open(args.out_user_sum, 'r') as f:
        for idx, line in tqdm(enumerate(f), desc='sample_user_sum'):
            line = json.loads(line)
            userid = int(line['user_id'])
            target_item = int(line['target_id'])
            query = query_profile.loc[idx, 'response']
            if not isinstance(query, str):
                continue
            
            target_index = user2seq[userid].index(target_item)
            ground_set = set(user2seq[userid][target_index:])
                
            neg_items = []
            while len(neg_items) < args.neg_num:
                neg_item = random.randint(1, len(itemid2text)-1)
                if neg_item not in ground_set:
                    neg_items.append(neg_item)
            output = {
                'query': query,
                'pos': [itemid2text[target_item]],
                'neg': [itemid2text[x] for x in neg_items]
            }
            out_text.append(output)
            if idx == n_rows-1:
                break
    
    return out_text
            
def sample_query(args, itemid2text, itemid2features, out_text):
    query_profile = pd.read_csv(args.out_gpt_query, header=None, sep=',', names=['question', 'response'])
    query_profile = query_profile.iloc[1:]
    query_profile = query_profile.reset_index(drop=True)
    n_rows = len(query_profile)
    with open(args.out_query, 'r') as f:
        for idx, line in tqdm(enumerate(f), desc='sample_query'):
            line = json.loads(line)
            target_id = int(line['target_id'])
            target_info = line['target_info']
            del target_info['title']
            if 'tags' in target_info:
                target_info['tags'] = target_info['tags'].split(', ')
            if 'game details' in target_info:
                target_info['game details'] = target_info['game details'].split(', ')
            
            query = query_profile.loc[idx, 'response']
            if not isinstance(query, str):
                continue
                
            neg_items = []
            while len(neg_items) < args.neg_num:
                neg_item = random.randint(1, len(itemid2text)-1)
                neg_features = {}
                for x in itemid2features[neg_item]:
                    neg_features[x[0][:-2]] = x[1]
                unsim_count=0
                for key, value in target_info.items():
                    if key in neg_features:
                        if isinstance(value, list):
                            for x in value:
                                if x not in neg_features[key]:
                                    unsim_count+=1
                        else:
                            if key=='price' and float(value) + 10 < float(neg_features[key]):
                                unsim_count+=1
                            elif key=='release date':
                                cur_date = transform_date_format(value)
                                neg_date = transform_date_format(neg_features[key])
                                if cur_date.year+3<= neg_date.year or cur_date.year-3>=neg_date.year:
                                    unsim_count+=1
                            elif value != neg_features[key]:
                                unsim_count+=1
                    else:
                        unsim_count+=1
                if unsim_count >= min(2, len(target_info)) and neg_item != target_id:
                    neg_items.append(neg_item)

            output = {
                'query': query,
                'pos': [itemid2text[target_id]],
                'neg': [itemid2text[x] for x in neg_items]
            }
            out_text.append(output)
            if idx == n_rows-1:
                break
    
    return out_text
            
def sample_neg_query(args, itemid2text, out_text):
    features2itemids = get_feature2itemid(args.in_meta_data)
    query_profile = pd.read_csv(args.out_gpt_neg_query, header=None, sep=',', names=['question', 'response'])
    query_profile = query_profile.iloc[1:]
    query_profile = query_profile.reset_index(drop=True)
    n_rows = len(query_profile)
    count_jump=0
    with open(args.out_neg_query, 'r') as f:
        for idx, line in tqdm(enumerate(f), desc='sample_neg_query'):
            line = json.loads(line)
            target_id = int(line['target_id'])
            target_info = line['target_info']
            del target_info['title']
            if 'tags' in target_info:
                target_info['tags'] = target_info['tags'].split(', ')
            if 'game details' in target_info:
                target_info['game details'] = target_info['game details'].split(', ')
            
            query = query_profile.loc[idx, 'response']
            if not isinstance(query, str):
                continue

            neg_set = set()
            for key, value in target_info.items():
                if isinstance(value, list):
                    for v in value:
                        neg_set.update(features2itemids[key+": "][v])
                else:
                    neg_set.update(features2itemids[key+": "][value])
        
            pos_set = set(range(1, len(itemid2text)))
            pos_set = pos_set - neg_set
            if len(pos_set) == 0 or len(neg_set) <= args.neg_num//2:
                count_jump+=1
                continue
            target_item = random.sample(list(pos_set), min(2, len(pos_set)))
            neg_items = random.sample(list(neg_set), min(args.neg_num, len(neg_set)))
            output = {
                'query': query,
                'pos': [itemid2text[x] for x in target_item],
                'neg': [itemid2text[x] for x in neg_items]
            }
            out_text.append(output)
            if idx == n_rows-1:
                break
    
    print("count_jump: ", count_jump)
    return out_text
            
if __name__ == "__main__":
    args = parse_args() 
    os.makedirs(os.path.dirname(args.out_conv), exist_ok=True)
    itemid2text, itemid2title, itemid2features, itemid2price_date_map = get_item_text(args.in_meta_data)
    out_text = []
    out_text = sample_conv(args, itemid2text, out_text)
    out_text = sample_user_sum(args, itemid2text, out_text)
    out_text = sample_query(args, itemid2text, itemid2features, out_text)
    out_text = sample_neg_query(args, itemid2text, out_text)
    print("len(out_text): ", len(out_text))

    with open(args.out_new_gpt, 'w') as f:
        for line in out_text:
            f.write(json.dumps(line)+'\n')