# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import pickle
import gzip
from tqdm import tqdm
import pandas as pd
import math
import random
import networkx as nx
import torch

import re
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

def prepare_U2I_dict(embedding, sequential_data, candidate_data):
    item_embedding = torch.tensor(embedding[0])
    user_embedding = torch.tensor(embedding[1])
    print("shape of user embeddings: ", user_embedding.shape)
    print("shape of item embeddings: ", item_embedding.shape)
    U2I_dict = {}
    U2I_candidate_dict = {}
    user_item_score = torch.softmax(torch.matmul(user_embedding, item_embedding.T), -1).tolist()

    for line1, line2 in tqdm(zip(ReadLineFromFile(sequential_data)[:200], ReadLineFromFile(candidate_data)[:200]), ncols=80):
        user, items = line1.strip().split(' ', 1)
        sequence = [int(x) for x in items.split(' ')]
        user, items = line2.strip().split(' ', 1)
        candidate = [int(x) for x in items.split(' ')]
        history = sequence[:-1]
        candidates = [sequence[-1]] + candidate[:19]
        idx = int(user)-1
        scores = [(item2_id, score) for item2_id, score in enumerate(user_item_score[idx])]
        U2I_dict[user] = sorted(scores, key=lambda x:-x[1])[:20]
        candidate_scores = [(item2_id, score) for item2_id, score in enumerate(user_item_score[idx]) if item2_id in candidates]
        U2I_candidate_dict[user] = sorted(candidate_scores, key=lambda x:-x[1])[:20]
    return U2I_dict, U2I_candidate_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='extract CF information')
    parser.add_argument('--dataset', type=str, default='steam', help='dataset')
    parser.add_argument('--negative_type', type=str, help='negative_type')
    args = parser.parse_args()

    os.makedirs(f"../data/{args.dataset}", exist_ok=True)
    sequential_data = f"../data/{args.dataset}/sequential_data.txt"
    if args.negative_type == "pop":
        candidate_data = f"../data/{args.dataset}/negative_samples_pop.txt"
    else:
        candidate_data = f"../data/{args.dataset}/negative_samples.txt"

    print("preparing SASRec U2I dict")
    embedding = pickle.load(open(f"../data/{args.dataset}/SASRec_embeddings_{args.negative_type}.pkl", "rb"))
    SASRec_U2I_dict, SASRec_U2I_candidate_dict = prepare_U2I_dict(embedding, sequential_data, candidate_data)
    with open(f"../data/{args.dataset}/SASRec_U2I_{args.negative_type}.json", "w") as fw:
        fw.write(json.dumps(SASRec_U2I_dict) + "\n")
    with open(f"../data/{args.dataset}/SASRec_U2I_candidate_{args.negative_type}.json", "w") as fw:
        fw.write(json.dumps(SASRec_U2I_candidate_dict) + "\n")