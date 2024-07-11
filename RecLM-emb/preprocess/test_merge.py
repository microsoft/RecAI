# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import random
import math
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import argparse

from utils import get_item_text
from test_template import querysummary2item_template, title2item_template

random.seed(2023)

def parse_args():
    parser = argparse.ArgumentParser(description="merge")
    parser.add_argument(
        "--in_seq_data", type=str, help=""
    )
    parser.add_argument(
        "--in_meta_data", type=str, help=""
    )
    parser.add_argument(
        "--in_u2i", type=str, help=""
    )
    parser.add_argument(
        "--in_q2i", type=str, help=""
    )
    parser.add_argument(
        "--in_q2i_misspell", type=str, help=""
    )
    parser.add_argument(
        "--gpt_path", type=str, help=""
    )
    parser.add_argument(
        "--out_gpt_summary", type=str, help=""
    )
    parser.add_argument(
        "--out_gpt_query", type=str, help=""
    )
    parser.add_argument(
        "--out_gpt_misspell", type=str, help=""
    )
    parser.add_argument(
        "--out_gpt_summary_query", type=str, help=""
    )
    args = parser.parse_args()
    return args

def merge(itemid2title, title2itemid, args):
    query_profile = pd.read_csv(args.gpt_path+'.csv', header=None, sep=',', names=['question', 'target'])
    query_profile = query_profile.iloc[1:]
    query_profile = query_profile.reset_index(drop=True)
    # query_profile1 = pd.read_csv(args.gpt_path+'_1.csv', header=None, sep=',', names=['question', 'target'])
    # query_profile1 = query_profile1.iloc[1:]
    # query_profile2 = pd.read_csv(args.gpt_path+'_2.csv', header=None, sep=',', names=['question', 'target'])
    # query_profile2 = query_profile2.iloc[1:]
    # query_profile = pd.concat([query_profile1, query_profile2], axis=0).reset_index(drop=True)
    id2queries = defaultdict(list)
    uidiid2query = {}
    idx = 0
    with open(args.out_gpt_summary, 'w') as w, open(args.in_u2i, 'r') as f:
        count = 0
        for line in tqdm(f):
            line = json.loads(line)
            userid = int(line['userid'])
            target_item = int(line['target_id'])
            history= line['history']
            query = query_profile.loc[idx, 'target']
            uidiid2query[(userid, target_item)] = [query, history]
            
            data = {'user_id': userid, 'ground_truth': target_item, 'history': history, 'text': query}
            w.write(json.dumps(data, ensure_ascii=False) + '\n')
            idx += 1
            count += 1
        print('out_gpt_summary num: ', count)

    with open(args.out_gpt_query, 'w') as w, open(args.in_q2i, 'r') as f:
        count = 0
        for line in tqdm(f):
            line = json.loads(line)
            target_item = int(line['item_id'])
            
            query = query_profile.loc[idx, 'target']
            query = query.split('#SEP#')
            id2queries[target_item] = query
            for q in query:
                data = {'item_id': target_item, 'text': q, 'ground_truth': target_item}
                w.write(json.dumps(data, ensure_ascii=False) + '\n')
                count += 1
            idx += 1
        print('out_gpt_query num: ', count)
        
    with open(args.out_gpt_misspell, 'w') as w, open(args.in_q2i_misspell, 'r') as f:
        count = 0
        for line in tqdm(f):
            line = json.loads(line)
            target_item = int(line['item_id'])
            
            query = query_profile.loc[idx, 'target']
            query = query.split('#SEP#')
            for q in query:
                if random.random() < 0.5:
                    template = "{}"
                else:
                    template = random.choice(title2item_template)
                q = template.format(q)
                data = {'item_id': target_item, 'text': q, 'ground_truth': title2itemid[itemid2title[target_item][1]]}
                w.write(json.dumps(data, ensure_ascii=False) + '\n')
                count += 1
            idx += 1
        print('out_gpt_misspell num: ', count)
        
    with open(args.out_gpt_summary_query, 'w') as w:
        count = 0
        for (userid, target_item), user in tqdm(uidiid2query.items()):
            user_queries = random.sample(id2queries[target_item], min(2, len(id2queries[target_item])))

            for user_query in user_queries:
                template = random.choice(querysummary2item_template)
                query = template.format(user[0], user_query)
                data = {'user_id': int(userid), 'text': query, 'ground_truth': target_item, 'history': user[1]}
                w.write(json.dumps(data, ensure_ascii=False) + '\n')
                count += 1
        print('out_gpt_summary_query num: ', count)
                

    
if __name__ == "__main__":
    args = parse_args()
    itemid2text, itemid2title, itemid2features, _ = get_item_text(args.in_meta_data)
    title2itemid = defaultdict(list)
    for idx, v in enumerate(itemid2title):
        if v[1] is not None:
            title2itemid[v[1]].append(idx)
    merge(itemid2title, title2itemid, args)