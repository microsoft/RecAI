# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import pickle
import gzip
import random

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
        
def extract_cf_information(sequential_items, cf_dict, n_item=3, n_len=10):
    text = ""
    all_pairs = set()
    for idx, item in enumerate(sequential_items[-n_len:]):
        most_co_click_ids = [x[0] for x in cf_dict[str(item)]][:n_item]
        all_pairs |= set([(item, x) for x in most_co_click_ids])
        most_co_click_titles = ", ".join([itemid2title[x] for x in most_co_click_ids])
        if args.dataset == "steam":
            if 'd1' in args.config:
                for x in most_co_click_ids:
                    text += f"Users who played {itemid2title[item]}, also frequently played game: {itemid2title[x]}.\n"
            else:
                text += f"Users who played {itemid2title[item]}, their most frequently played games in descending order are: {most_co_click_titles}.\n"
        elif args.dataset == "ml1m":
            if 'd1' in args.config:
                for x in most_co_click_ids:
                    text += f"Users who watched {itemid2title[item]}, also frequently watched movie: {itemid2title[x]}.\n"
            else:
                text += f"Users who watched {itemid2title[item]}, their most frequently watched movies in descending order are: {most_co_click_titles}.\n"
        elif args.dataset in ["beauty", "sports", "online_retail"]:
            if 'd1' in args.config:
                for x in most_co_click_ids:
                    text += f"Users who purchased {itemid2title[item]}, also frequently purchased product: {itemid2title[x]}.\n"
            else:
                text += f"Users who purchased {itemid2title[item]}, their most frequently purchased products in descending order are: {most_co_click_titles}.\n"
    return text, list(all_pairs)

def extract_reasoning_path(item_pairs, reasoning_path_text_dict):
    all_path = []
    for item1, item2 in item_pairs:
        if f"{item1},{item2}" in reasoning_path_text_dict:
            all_path.append(reasoning_path_text_dict[f"{item1},{item2}"][0])
    if args.dataset == "steam":
        prefix = "I will also provide a reasoning path between several freqently co-played game pairs to let you know why they are always played by same user. And use these relations to make further recommendation.\n"
    if args.dataset in ["beauty", "sports", "online_retail"]:
        prefix = "I will also provide a reasoning path between several freqently co-purchased product pairs to let you know why they are always purchased by same user. And use these relations to make further recommendation.\n"
    if args.dataset == "ml1m":
        prefix = "I will also provide a reasoning path between several freqently co-watched movie pairs to explain why they are always watched by same user.\n"
    if len(all_path) > 0:
        return prefix + "\n".join(all_path)+'\n' + 'You can follow these reasoning path to check whether there is also same pattern between my history and each following candidate item to make recommendation. \n'
    else:
        return ""

def char_idx(idx):
    return chr(ord('a') + idx)

def parse_U2I_result(result):
    text = ""
    if 'd1' in args.config:
        for x in result[:20]:
            if args.dataset == "ml1m":
                text += f"This user like to watch {itemid2title[x[0]]}.\n"
            elif args.dataset in ["beauty", "sports", "online_retail"]:
                text += f"This user like to purchase {itemid2title[x[0]]}.\n"
    else:
        text = ", ".join([itemid2title[x[0]] for x in result[:20]])
    return f"{text}.\n"

def extract_cf_with_path(items, cf_dict, path_dict, n_len=10, n_item=3):
    text = ""
    for idx, item1 in enumerate(items[-n_len:]):
        most_co_click_ids = [x[0] for x in cf_dict[str(item1)]][:n_item]

        for item2 in most_co_click_ids:
            if args.dataset == "steam":
                text += f"Users who played {itemid2title[item1]}, most frequently play {itemid2title[item2]}."
            elif args.dataset == "ml1m":
                text += f"Users who watched {itemid2title[item1]}, most frequently watch {itemid2title[item2]}."
            elif args.dataset in ["beauty", "sports", "online_retail"]:
                text += f"Users who purchased {itemid2title[item1]}, most frequently purchase {itemid2title[item2]}."
            if f"{item1},{item2}" in path_dict:
                path_text = path_dict[f"{item1},{item2}"][0]
                text += f" You should learn the recommendation pattern from the reasoning path: {path_text}."
            text += '\n'
    return text

def prepare_global_CF_text(global_CF_list):
    text = ""
    for data in global_CF_list[:20]:
        item1 = data["item1"]
        item2 = data["item2"]
        if args.dataset == "steam":
            text += f"Users who played {itemid2title[item1]}, also frequently play {itemid2title[item2]}.\n"
        elif args.dataset == "ml1m":
            text += f"Users who watched {itemid2title[item1]}, also frequently watch {itemid2title[item2]}.\n"
        elif args.dataset in ["beauty", "sports", "online_retail"]:
            text += f"Users who purchased {itemid2title[item1]}, also frequently purchase {itemid2title[item2]}.\n"
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='prompt construction')
    parser.add_argument('-c', '--config', default="./config/prompt_config.json", type=str, help='config file path (default: None)')
    parser.add_argument('-d', '--dataset', default="ml1m", type=str, help='dataset name (default: ml1m)')
    args = parser.parse_args()
    config = json.load(open(args.config)) # refer to ./config/data_config.json

    sequential_data = config['sequential_data_path']
    meta_data = config['meta_data_path']
    candidate_data = config['candidate_data_path']
    topk = config['topk'] # default: 100, #samples for testing
    max_his_len = config['max_his_len']  # default: 50

    itemid2title = ["paddind_title"]
    itemid2title_with_feature = ["padding_title"]
    for line in parse(config['meta_data_path']):
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
        feature = []
        if "popular_tags" in line:
            feature.append('tags: ' + ', '.join(line['popular_tags'][:3]))
        if "genre" in line:
            feature.append('genre: ' + line['genre'])
        if "year" in line:
            feature.append('publish year: ' + line["year"])
        if "brand" in line:
            feature.append(f"brand: {line['brand']}")
        if "categories" in line:
            feature.append('categories: ' + line['categories'][0][-1])
        feature = '; '.join(feature)
        title = f"\"{title}\""
        if len(feature) != 0:
            title_with_feature = f"{title}({feature})"
        else:
            title_with_feature = title
        itemid2title.append(title)
        itemid2title_with_feature.append(title_with_feature)

    if "reasoning_path_data_path" in config:
        reasoning_path_text_dict = json.loads(open(config['reasoning_path_data_path']).readline())
    else:
        reasoning_path_text_dict = defaultdict(list)

    if "u2i_data_path" in config:
        U2I_dict = json.loads(open(config['u2i_data_path']).readline())
    else:
        U2I_dict = defaultdict(list)
        
    if "global_cf_data_path" in config:
        global_CF_list = json.loads(open(config['global_cf_data_path']).readline())
    else:
        global_CF_list = []
    global_CF_text = prepare_global_CF_text(global_CF_list)
    
    if "cf_data_path" in config:
        CF_dict = json.loads(open(config['cf_data_path']).readline())
    else:
        CF_dict = defaultdict(lambda:defaultdict(list))
    os.makedirs(f"out/prompts/{args.dataset}", exist_ok=True)
    fd = open(f"out/prompts/{args.dataset}/{args.config.split('/')[-1].split('.json')[0]}.json", "w")

    total_path_cnt = 0

    for index, (sequential, candidate) in enumerate(zip(ReadLineFromFile(sequential_data)[:topk], ReadLineFromFile(candidate_data)[:topk])):
        _, sequential = sequential.split(' ', 1)
        sequential = [int(x) for x in sequential.split(' ')]
        target = sequential[-1]
        sequential = sequential[:-1][-max_his_len:]
        _, candidate = candidate.split(' ', 1)
        candidate = [int(x) for x in candidate.split(' ')][:19] + [target]
        random.shuffle(candidate)
        sequential_title = '[' + ', '.join([f"{idx}. {itemid2title[x]}" for idx, x in enumerate(sequential)])+']\n'
        sequential_title_with_feature = '[' + ', '.join([f"{idx}. {itemid2title_with_feature[x]}" for idx, x in enumerate(sequential)])+']\n'
        candiadte_title = '[' + ', '.join([f"{idx}. {itemid2title[x]}" for idx, x in enumerate(candidate)])+']\n'
        candiadte_title_with_feature = '[' + ', '.join([f"{idx}. {itemid2title_with_feature[x]}" for idx, x in enumerate(candidate)])+']\n'

        CF_text, CF_pairs = extract_cf_information(sequential, CF_dict[str(index+1)])

        path_text = extract_reasoning_path(CF_pairs, reasoning_path_text_dict)
        CF_path_text = extract_cf_with_path(sequential, CF_dict[str(index+1)], reasoning_path_text_dict)

        total_path_cnt += len(path_text.split('\n'))
        
        U2I_text = parse_U2I_result(U2I_dict[str(index+1)])

        info = {
            "sequential_title": sequential_title,
            "candidate_title": candiadte_title,
            "sequential_title_with_feature": sequential_title_with_feature,
            "candidate_title_with_feature": candiadte_title_with_feature,
            "last_title": f"{itemid2title[sequential[-1]]}.\n",
            "global_CF_text": global_CF_text,
            "CF_text": CF_text,
            "path_text": path_text,
            "CF_path_text": CF_path_text,
            "U2I_text": U2I_text,
        }

        prompt = ""
        for block in config['template']:
            if block[0] == '{' and block[-1] == '}':
                prompt += info[block[1:-1]]
            else:
                prompt += block
        
        data = {
            'prompt': prompt,
            'candidate': [itemid2title[x][1:-1] for x in candidate],
            'ground_truth': itemid2title[target][1:-1]
        }
        fd.write(json.dumps(data, ensure_ascii=False)+'\n')
    print(total_path_cnt/config['topk'])