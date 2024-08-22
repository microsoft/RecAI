# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="data process")
    parser.add_argument(
        "--top_file", type=str, help=""
    )
    parser.add_argument(
        "--seqdata_file", type=str, help=""
    )
    parser.add_argument(
        "--in_gpt_file", type=str, help=""
    )
    parser.add_argument(
        "--in_vicuna_file", type=str, help=""
    )
    parser.add_argument(
        "--in_recexplainer_file", type=str, help=""
    )
    parser.add_argument(
        "--out_cls_file", type=str, help=""
    )
    parser.add_argument(
        "--out_reg_gpt_file", type=str, help=""
    )
    parser.add_argument(
        "--out_reg_vicuna_file", type=str, help=""
    )
    parser.add_argument(
        "--out_reg_recexplainer_file", type=str, help=""
    )
    parser.add_argument(
        "--split", type=str, help="", choices=['train', 'valid'], default='valid'
    )
    parser.add_argument(
        "--max_samples", type=int, help=""
    )
    args = parser.parse_args()
    return args

args = parse_args()

os.makedirs(os.path.dirname(args.out_cls_file), exist_ok=True)

user_items = {}
with open(args.seqdata_file, 'r') as f:
    for idx, line in enumerate(f):
        line = line.strip().split(' ')
        user = int(line[0])
        items = [int(x) for x in line[1:]]
        user_items[user] = items

gpt_df = pd.read_csv(args.in_gpt_file)
gpt_df = gpt_df.drop(['question'], axis=1)
gpt_df = gpt_df.rename(columns={'answer':'explan'})
gpt_df['label'] = 0.0
gpt_df['user_id'] = 0
gpt_df['item_id'] = 0
gpt_df['target_id'] = 0
gpt_df = gpt_df[['user_id', 'item_id', 'target_id', 'explan', 'label']]
vicuna_df = pd.read_csv(args.in_vicuna_file)
vicuna_df = vicuna_df.drop(['label','history','target item'], axis=1)
vicuna_df = vicuna_df.rename(columns={'answer':'explan'})
vicuna_df['label'] = 0.0
vicuna_df['user_id'] = 0
vicuna_df['item_id'] = 0
vicuna_df['target_id'] = 0
vicuna_df = vicuna_df[['user_id', 'item_id', 'target_id', 'explan', 'label']]
recexplainer_df = pd.read_csv(args.in_recexplainer_file)
recexplainer_df = recexplainer_df.drop(['label','history','target item'], axis=1)
recexplainer_df = recexplainer_df.rename(columns={'answer':'explan'})
recexplainer_df['label'] = 0.0
recexplainer_df['user_id'] = 0
recexplainer_df['item_id'] = 0
recexplainer_df['target_id'] = 0
recexplainer_df = recexplainer_df[['user_id', 'item_id', 'target_id', 'explan', 'label']]

count=0
with open(args.top_file, 'r') as f:
    for i, line in enumerate(f):
        uid, iid, top1, topk, pos, neg, pos_score, neg_score = line.strip().split('\t') 
        uid, iid, top1, topk, pos, neg, pos_score, neg_score = int(uid), int(iid), int(top1), [int(x) for x in topk.split(',')], int(pos), int(neg), float(pos_score), float(neg_score)
        hist = user_items[uid]
        if args.split=='valid' or (args.split=='train' and iid==hist[-2]):
            gpt_df.loc[count, 'user_id'] = uid
            gpt_df.loc[count, 'item_id'] = iid
            gpt_df.loc[count, 'target_id'] = pos
            gpt_df.loc[count, 'label'] = pos_score
            gpt_df.loc[count+1, 'user_id'] = uid
            gpt_df.loc[count+1, 'item_id'] = iid
            gpt_df.loc[count+1, 'target_id'] = neg
            gpt_df.loc[count+1, 'label'] = neg_score
            vicuna_df.loc[count, 'user_id'] = uid
            vicuna_df.loc[count, 'item_id'] = iid
            vicuna_df.loc[count, 'target_id'] = pos
            vicuna_df.loc[count, 'label'] = pos_score
            vicuna_df.loc[count+1, 'user_id'] = uid
            vicuna_df.loc[count+1, 'item_id'] = iid
            vicuna_df.loc[count+1, 'target_id'] = neg
            vicuna_df.loc[count+1, 'label'] = neg_score
            recexplainer_df.loc[count, 'user_id'] = uid
            recexplainer_df.loc[count, 'item_id'] = iid
            recexplainer_df.loc[count, 'target_id'] = pos
            recexplainer_df.loc[count, 'label'] = pos_score
            recexplainer_df.loc[count+1, 'user_id'] = uid
            recexplainer_df.loc[count+1, 'item_id'] = iid
            recexplainer_df.loc[count+1, 'target_id'] = neg
            recexplainer_df.loc[count+1, 'label'] = neg_score
            count+=2

        if count>=args.max_samples:
            break



gpt_df.to_csv(args.out_reg_gpt_file, index=False)
vicuna_df.to_csv(args.out_reg_vicuna_file, index=False)
recexplainer_df.to_csv(args.out_reg_recexplainer_file, index=False)


gpt_df = gpt_df.drop(['label'], axis=1)
vicuna_df = vicuna_df.drop(['label'], axis=1)
recexplainer_df = recexplainer_df.drop(['label'], axis=1)

gpt_df['label']=2
vicuna_df['label']=1
recexplainer_df['label']=0

all_df = pd.concat([recexplainer_df, vicuna_df, gpt_df], axis=0)
all_df = all_df[['user_id', 'item_id', 'target_id', 'explan', 'label']]
all_df.to_csv(args.out_cls_file, index=False)

            

