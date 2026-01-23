"""
The following code is modified from
https://github.com/aHuiWang/CIKM2020-S3Rec/blob/master/data/data_process.py
"""

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
import argparse

'''
Set seeds
'''
seed = 2022
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="data process")
    parser.add_argument("--data_path", type=str, help="")
    parser.add_argument("--unirec_data_path", type=str, help="")
    parser.add_argument("--unirec_config_path", type=str, help="")
    parser.add_argument("--tokenizer_path", type=str, help="")
    return parser.parse_args()


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def save_json(data, filename):
    if filename is None:
        return
    with open(filename, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def id_map(user_items):  # user_items dict
    user2id = {'[PAD]': 0}  # raw 2 uid
    item2id = {'[PAD]': 0}  # raw 2 iid
    id2user = ['[PAD]']  # uid 2 raw
    id2item = ['[PAD]']  # iid 2 raw
    user_id = 1  # start from 1
    item_id = 1
    random_user_list = list(user_items.keys())
    # random.shuffle(random_user_list)  # user is shuffled and re-encoded
    for user in random_user_list:
        items = user_items[user]
        if user not in user2id:
            user2id[user] = user_id
            id2user.append(user)
            user_id += 1
        for item in items:
            if item not in item2id:
                item2id[item] = item_id
                id2item.append(item)
                item_id += 1
    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item
    }
    # return: final_data: {uid: [iid1, iid2, ...], ...}, user_num, item_num, data_maps
    return user_id - 1, item_id - 1, data_maps


default_dataset_config = f"""group_size: -1
n_neg_test_from_sampling: 0
n_neg_train_from_sampling: 0
n_neg_valid_from_sampling: 0
test_file_format: user-item
train_file_format: user-item
user_history_file_format: user-item_seq
valid_file_format: user-item
"""


def main_process(args):
    user_items = load_json(os.path.join(args.data_path, 'sequential.jsonl'))
    meta_infos = load_json(os.path.join(args.data_path, 'metas.jsonl'))
    category_infos = load_json(os.path.join(args.data_path, 'category.jsonl'))
    user_num, item_num, datamaps = id_map(user_items)  # get mapping dicts

    train_data = {'user_id': [], 'item_id': []}
    valid_data = {'user_id': [], 'item_id': []}
    test_data = {'user_id': [], 'item_id': []}
    user_history = {'user_id': [], 'item_seq': []}
    for u in user_items:
        user_history['user_id'].append(datamaps['user2id'][u])
        user_history['item_seq'].append(np.array([datamaps['item2id'][_] for _ in user_items[u][:-1]], dtype=np.int32))
        for i in user_items[u][:-2]:
            train_data['user_id'].append(datamaps['user2id'][u])
            train_data['item_id'].append(datamaps['item2id'][i])

        valid_item, test_item = user_items[u][-2], user_items[u][-1]
        valid_data['user_id'].append(datamaps['user2id'][u])
        valid_data['item_id'].append(datamaps['item2id'][valid_item])
        test_data['user_id'].append(datamaps['user2id'][u])
        test_data['item_id'].append(datamaps['item2id'][test_item])

    if not os.path.exists(args.unirec_data_path):
        os.makedirs(args.unirec_data_path)
    save_pickle(pd.DataFrame(train_data), os.path.join(args.unirec_data_path, 'train.pkl'))
    save_pickle(pd.DataFrame(valid_data), os.path.join(args.unirec_data_path, 'valid.pkl'))
    save_pickle(pd.DataFrame(test_data), os.path.join(args.unirec_data_path, 'test.pkl'))
    save_pickle(pd.DataFrame(user_history), os.path.join(args.unirec_data_path, 'user_history.pkl'))
    save_pickle(datamaps, os.path.join(args.unirec_data_path, 'map.pkl'))
    save_json(category_infos, os.path.join(args.unirec_data_path, 'category.jsonl'))

    os.makedirs(os.path.dirname(args.unirec_config_path), exist_ok=True)
    with open(args.unirec_config_path, 'w') as f:
        f.write(default_dataset_config)
        f.write(f"n_items: {len(meta_infos)+1}\n")
        f.write(f"n_users: {len(user_items)+1}\n")

    print('Transform Done!!!')


if __name__ == '__main__':
    _args = parse_args()
    main_process(args=_args)
