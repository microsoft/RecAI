# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import pickle
import gzip
from tqdm import tqdm
import math
import random
import torch

from collections import defaultdict

import sys
sys.path.append('')
import argparse

random.seed(43)

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def ReadLineFromFile(path):
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def parse(path):
    if path.endswith('gz'):
        g = gzip.open(path, 'r')
    else:
        g = open(path, 'r')
    nan_default = {'NaN': "", 'false': "", 'true': ""}
    for l in g:
        yield eval(l, nan_default)

def prepare_global_CF(sequential_data):
    item_click_dict = defaultdict(int) # the total number of clicks each item was clicked by all users
    itempair_click_dict = defaultdict(lambda: defaultdict(int)) # the number of co-clicks for a pair of items, (item1,item2) and (item2,item1) are the same

    for line in tqdm(ReadLineFromFile(sequential_data), ncols=80):
        user, items = line.strip().split(' ', 1)
        items = [int(x) for x in items.split(' ')]
        for item in items:
            item_click_dict[item] += 1
        
        for idx1, item1 in enumerate(items[:-1]):
            for idx2, item2 in enumerate(items[:-1]):
                if item1 < item2: # avoid repeating counting
                    itempair_click_dict[item1][item2] += 1
                    # itempair_click_dict[item2][item1] += 1
    candidate_pairs = []
    for item1 in itempair_click_dict:
        item2freq = itempair_click_dict[item1]
        item2freq = {x: item2freq[x]/math.sqrt(item_click_dict[item1]*item_click_dict[x]) for x in item2freq}
        for item2, freq in sorted(item2freq.items(), key=lambda x:-x[1])[:30]:
            candidate = {
                "item1": item1,
                "item2": item2,
                "co-click": itempair_click_dict[item1][item2],
                "item1_click": item_click_dict[item1], 
                "item2_click": item_click_dict[item2], 
                "score": freq
            }
            if item_click_dict[item1] >= 5 and item_click_dict[item2] >= 5:
                candidate_pairs.append(candidate)
        candidate_pairs = sorted(candidate_pairs, key=lambda x:-x["score"])[:100]
    return candidate_pairs

def prepare_CF_base_on_statistic(sequential_data, candidate_data):
    item_click_dict = defaultdict(int) # the total number of clicks each item was clicked by all users
    itempair_click_dict = defaultdict(lambda: defaultdict(int)) # the number of co-clicks for a pair of items, (item1,item2) and (item2,item1) are the same

    for line in tqdm(ReadLineFromFile(sequential_data), ncols=80):
        user, items = line.strip().split(' ', 1)
        items = [int(x) for x in items.split(' ')]
        for item in items:
            item_click_dict[item] += 1
        
        for idx1, item1 in enumerate(items[:-1]):
            for idx2, item2 in enumerate(items[:-1]):
                if item1 < item2: # avoid repeating counting
                    itempair_click_dict[item1][item2] += 1
                    itempair_click_dict[item2][item1] += 1

    top_count = defaultdict(int)
    for item1 in itempair_click_dict:
        item2freq = itempair_click_dict[item1]
        top_item2 = sorted(list(item2freq.items()), key=lambda x:-x[1])[:5]
        for item2 in top_item2:
            top_count[item2[0]] += 1

    naive_cf_dict = defaultdict(dict)
    normalized_cf_dict = defaultdict(dict)
    naive_cf_candidate_dict = defaultdict(dict)
    normalized_cf_candidate_dict = defaultdict(dict)

    for line1, line2 in tqdm(zip(ReadLineFromFile(sequential_data)[:200], ReadLineFromFile(candidate_data)[:200]), ncols=80):
        user, items = line1.strip().split(' ', 1)
        sequence = [int(x) for x in items.split(' ')]
        user, items = line2.strip().split(' ', 1)
        candidate = [int(x) for x in items.split(' ')]
        history = sequence[:-1]
        candidates = [sequence[-1]] + candidate[:19]
        
        for idx, item1 in enumerate(history):
            item2freq = itempair_click_dict[item1]
            naive_cf_dict[user][item1] = sorted(item2freq.items(), key=lambda x:-x[1])[:10]
            naive_cf_candidate_dict[user][item1] = [x for x in sorted(item2freq.items(), key=lambda x:-x[1]) if x[0] in candidates][:10]
            item2freq = {x: item2freq[x]/math.sqrt(item_click_dict[item1]*item_click_dict[x]) for x in item2freq}
            normalized_cf_dict[user][item1] = sorted(item2freq.items(), key=lambda x:-x[1])[:10]
            normalized_cf_candidate_dict[user][item1] = [x for x in sorted(item2freq.items(), key=lambda x:-x[1]) if x[0] in candidates][:10]

        for idx, item1 in enumerate(candidates):
            item2freq = itempair_click_dict[item1]
            naive_cf_dict[user][item1] = sorted(item2freq.items(), key=lambda x:-x[1])[:10]
            item2freq = {x: item2freq[x]/math.sqrt(item_click_dict[item1]*item_click_dict[x]) for x in item2freq}
            normalized_cf_dict[user][item1] = sorted(item2freq.items(), key=lambda x:-x[1])[:10]

    return naive_cf_dict, naive_cf_candidate_dict, \
        normalized_cf_dict, normalized_cf_candidate_dict

def prepare_CF_base_on_embedding(sequential_data, candidate_data, embedding):
    item_embeddings = torch.tensor(embedding[0])
    print("shape of item embeddings: ", item_embeddings.shape)
    CF_dict = {}
    CF_candidate_dict = {}
    item_item_score = torch.softmax(torch.matmul(item_embeddings, item_embeddings.T)+torch.diag_embed(torch.ones(len(item_embeddings))*float("-inf")), -1)

    for line1, line2 in tqdm(zip(ReadLineFromFile(sequential_data)[:200], ReadLineFromFile(candidate_data)[:200]), ncols=80):
        user, items = line1.strip().split(' ', 1)
        sequence = [int(x) for x in items.split(' ')]
        user, items = line2.strip().split(' ', 1)
        candidate = [int(x) for x in items.split(' ')]
        history = sequence[:-1]
        candidates = [sequence[-1]] + candidate[:19]
        all_scores = item_item_score[history].tolist()
        candi_scores = torch.softmax(torch.matmul(item_embeddings[history], item_embeddings[candidates].T), -1).tolist()
        CF_dict[user] = {}
        CF_candidate_dict[user] = {}
        for idx, scores in enumerate(all_scores):
            scores = [(item2_id, score) for item2_id, score in enumerate(scores) if item2_id != history[idx]]
            CF_dict[user][history[idx]] = sorted(scores, key=lambda x:-x[1])[:10]
        for idx, scores in enumerate(candi_scores):
            scores = [(candidates[index], score) for index, score in enumerate(scores) if candidates[index] != history[idx]]
            CF_candidate_dict[user][history[idx]] = sorted(scores, key=lambda x:-x[1])[:10]
    
        all_scores = item_item_score[candidates].tolist()
        for idx, scores in enumerate(all_scores):
            scores = [(item2_id, score) for item2_id, score in enumerate(scores) if item2_id != candidates[idx]]
            CF_dict[user][candidates[idx]] = sorted(scores, key=lambda x:-x[1])[:10]

    return CF_dict, CF_candidate_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='extract CF information')
    parser.add_argument('--dataset', type=str, default='ml1m', help='dataset')
    parser.add_argument('--negative_type', type=str, help='negative_type')
    args = parser.parse_args()

    metadata = f"../data/{args.dataset}/metadata.json"
    sequential_data = f"../data/{args.dataset}/sequential_data.txt"
    if args.negative_type == "pop":
        candidate_data = f"../data/{args.dataset}/negative_samples_pop.txt"
    else:
        candidate_data = f"../data/{args.dataset}/negative_samples.txt"

    itemid2title = ['padding_title']
    for line in open(metadata):
        line = json.loads(line)
        if "app_name" in line:
            itemid2title.append(line['app_name'])
        elif "title" in line:
            title = line['title']
        else:
            if "description" in line:
                itemid2title.append(line['description'][:100])
            elif "categories" in line:
                itemid2title.append(line['categories'][0][-1])


    os.makedirs(f"../data/{args.dataset}", exist_ok=True)

    # obtain global CF based on statistics
    global_pairs = prepare_global_CF(sequential_data)
    with open(f"../data/{args.dataset}/global_CF.json", "w") as fw:
        fw.write(json.dumps(global_pairs) + "\n")

    # obtain native CF information based on statistics
    print("preparing naive CF dict")
    naive_cf_dict, naive_cf_candidate_dict, \
    normalized_cf_dict, normalized_cf_candidate_dict \
        = prepare_CF_base_on_statistic(sequential_data, candidate_data)
    with open(f"../data/{args.dataset}/naive_CF.json", "w") as fw:
        fw.write(json.dumps(naive_cf_dict) + "\n")
    with open(f"../data/{args.dataset}/naive_CF_candidate_{args.negative_type}.json", "w") as fw:
        fw.write(json.dumps(naive_cf_candidate_dict) + "\n")
    with open(f"../data/{args.dataset}/normalized_CF.json", "w") as fw:
        fw.write(json.dumps(normalized_cf_dict) + "\n")
    with open(f"../data/{args.dataset}/normalized_CF_candidate_{args.negative_type}.json", "w") as fw:
        fw.write(json.dumps(normalized_cf_candidate_dict) + "\n")

    # obtain CF information based on BPR-MF Algorithm
    print("preparing MF CF dict")
    embedding = pickle.load(open(f"../data/{args.dataset}/MF_embeddings_{args.negative_type}.pkl", "rb"))

    MF_cf_dict, MF_candidate_cf_dict = prepare_CF_base_on_embedding(sequential_data, candidate_data, embedding)
    with open(f"../data/{args.dataset}/MF_CF_{args.negative_type}.json", "w") as fw:
        fw.write(json.dumps(MF_cf_dict) + "\n")
    with open(f"../data/{args.dataset}/MF_CF_candidate_{args.negative_type}.json", "w") as fw:
        fw.write(json.dumps(MF_candidate_cf_dict) + "\n")