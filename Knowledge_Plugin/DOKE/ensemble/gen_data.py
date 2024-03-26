import os
import re
import json
import gzip
import math
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from multiprocessing import Pool
from collections import defaultdict
import importlib

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str)
    parser.add_argument('--sample_num', type=int, default=200, help='sample number for each task.')  
    parser.add_argument('--dataset', type=str, default='ml1m', help='the dataset to be evaluated, steam/beauty/sports')

    return parser.parse_args()

def get_gen_data_func(method_name):
    if importlib.util.find_spec(f'method.{method_name}', __name__):
        model_module = importlib.import_module(f'method.{method_name}', __name__)
        method_func = getattr(model_module, "gen_data")
        return method_func
    else:
        raise NotImplementedError(f"'{method_name}' not exist!")

if __name__ == '__main__':
    args = parse_args()

    sequential_data = f"../../data/{args.dataset}/sequential_data.txt"
    candidate_data = f"../../data/{args.dataset}/negative_samples_pop.txt"
    metadata = f"../../data/{args.dataset}/metadata.json"
    item2title = ['padding']
    for line in open(metadata):
        line = json.loads(line)
        title = "No title"
        if "app_name" in line:
            title = line['app_name']
        elif "title" in line:
            title = line['title']
        else:
            if "description" in line:
                title = line['description'][:50]
            elif "categories" in line:
                title = line['categories'][0][-1]
        title = f"\"{title}\""
        item2title.append(title)
        
    gen_data = get_gen_data_func(args.method)
    question_file = f"outputs/{args.dataset}/{args.method}/question.jsonl"
    os.makedirs(os.path.dirname(question_file), exist_ok=True)
    fd = open(question_file, "w", encoding="utf-8")
    for sequential, candidate in zip(ReadLineFromFile(sequential_data)[:args.sample_num], ReadLineFromFile(candidate_data)[:args.sample_num]):
        user, sequential = sequential.split(' ', 1)
        sequential = [int(x) for x in sequential.split(' ')]
        target = sequential[-1]
        sequential = sequential[:-1][-50:]
        _, candidate = candidate.split(' ', 1)
        candidate = [int(x) for x in candidate.split(' ')][:19] + [target]
        random.shuffle(candidate)
        data = gen_data(user, sequential, candidate, target, item2title, args.dataset)
        fd.write(json.dumps(data, ensure_ascii=False)+'\n')