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
    parser.add_argument("--full_data_name", type=str, help="")
    parser.add_argument("--meta_file", type=str, help="")
    parser.add_argument("--review_file", type=str, help="")
    parser.add_argument("--data_path", type=str, help="")
    parser.add_argument("--unirec_data_path", type=str, help="")
    parser.add_argument("--unirec_config_path", type=str, help="")
    parser.add_argument("--tokenizer_path", type=str, help="")
    return parser.parse_args()


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_json(data, filename):
    if filename is None:
        return
    with open(filename, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def Amazon(rating_score=-1.0, args=None):
    '''
    return (user, item, timestamp) sort in get_interaction

    reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
    asin - ID of the product, e.g. 0000013714
    reviewerName - name of the reviewer
    helpful - helpfulness rating of the review, e.g. 2/3
    reviewText - text of the review
    overall - rating of the product
    summary - summary of the review
    unixReviewTime - time of the review (unix time)
    reviewTime - time of the review (raw)
    '''
    items_with_title = {}
    with gzip.open(args.meta_file, "r") as fr:
        for line in tqdm(fr, desc="load meta data"):
            line = json.loads(line)
            if "title" not in line or len(line['title']) == 0 or 'category' not in line or len(line['category']) == 0:
                continue
            items_with_title[line['asin']] = 1
    print(f"items with title and category: {add_comma(len(items_with_title))}")  # title: 181781
    datas = []
    data_dict = {}
    with gzip.open(args.review_file, "r") as fr:
        for line in tqdm(fr, desc="load all interactions"):
            # try:
            line = json.loads(line)
            user = line['reviewerID']
            item = line['asin']
            if float(line['overall']) <= rating_score or item not in items_with_title:  # remove low rating
                continue
            if (user, item) in data_dict:
                continue
            time = line['unixReviewTime']
            data_dict[(user, item)] = int(time)  # merge duplicate interactions, keep the first record
            datas.append((user, item, int(time)))
    print(f"total interactions: {add_comma(len(datas))}")
    return datas


def Amazon_meta(datamaps, args):
    '''
    asin - ID of the product, e.g. 0000031852
    title - name of the product  --"title": "Girls Ballet Tutu Zebra Hot Pink",
    description
    price - price in US dollars (at time of crawl) --"price": 3.17,
    imUrl - url of the product image (str) --"imUrl": "http://ecx.images-amazon.com/images/I/51fAmVkTbyL._SY300_.jpg",
    related - related products (also bought, also viewed, bought together, buy after viewing)
    salesRank - sales rank information --"salesRank": {"Toys & Games": 211836}
    brand - brand name --"brand": "Coxlures",
    categories - list of categories the product belongs to --"categories": [["Sports & Outdoors", "Other Sports", "Dance"]]
    '''
    meta_datas = {}
    item_ids = set(datamaps['item2id'].keys())
    with gzip.open(args.meta_file, "r") as fr:
        for line in tqdm(fr, desc="load meta data"):
            line = json.loads(line)
            if line['asin'] not in item_ids:
                continue
            # if "title" in line:
            line['title'] = re.sub(r'\n\t', ' ', line['title']).encode('UTF-8', 'ignore').decode('UTF-8').strip()
            # line['title'] = line['title'].split(",")[0]
            if "description" in line:
                if type(line['description']) == str:
                    line['description'] = re.sub(r'\n\t', ' ', line['description']).encode('UTF-8', 'ignore').decode('UTF-8')
                elif type(line['description']) == list:
                    descs = []
                    for desc in line['description']:
                        desc = re.sub(r'\n\t', ' ', desc).encode('UTF-8', 'ignore').decode('UTF-8')
                        descs.append(desc)
                    line['description'] = '\n'.join(descs)
            if 'related' in line:
                del line['related']
            if 'imUrl' in line:
                del line['imUrl']
            if 'imageURL' in line:
                del line['imageURL']
            if 'imageURLHighRes' in line:
                del line['imageURLHighRes']
            if 'similar_item' in line:
                del line['similar_item']
            meta_datas[line['asin']] = line
    return meta_datas


def reformat_titles(meta_infos, args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    titles_o = [meta_infos[_]['title'] for _ in meta_infos]
    titles_ids = tokenizer.batch_encode_plus(titles_o)
    titles_t = tokenizer.batch_decode(titles_ids['input_ids'])
    for i, t in zip(meta_infos, titles_t):
        meta_infos[i]['title_t'] = t

    return meta_infos


def reformat_repeat_title(meta_infos):
    title_set = set()
    for i in meta_infos:
        origin_title = meta_infos[i]['title_t']
        if origin_title in title_set:
            idx = 1
            while f"{origin_title} ({idx})" in title_set:
                idx += 1

            meta_infos[i]['title_t'] = f"{origin_title} ({idx})"

        title_set.add(meta_infos[i]['title_t'])

    return meta_infos


def Amazon_category(meta_infos):
    category_datas = {}
    for item in meta_infos:
        categories = []
        for c in meta_infos[item]['category'][1:]:  # filter the subset category, such as 'Movies $ TV'.
            categories.append(c)
            temp = ', '.join(categories)
            if temp not in category_datas:
                category_datas[temp] = []
            category_datas[temp].append(item)

    return category_datas


def add_comma(num):  # 1000000 -> 1,000,000
    str_num = str(num)
    res_num = ''
    for i in range(len(str_num)):
        res_num += str_num[i]
        if (len(str_num) - i - 1) % 3 == 0:
            res_num += ','
    return res_num[:-1]


def get_interaction(datas):
    # get user interaction sequence for sequential recommendation
    user_seq = {}
    for data in datas:
        user, item, time = data
        if user in user_seq:
            user_seq[user].append((item, time))
        else:
            user_seq[user] = []
            user_seq[user].append((item, time))

    for user, item_time in user_seq.items():
        item_time.sort(key=lambda x: x[1])
        items = []
        for t in item_time:
            items.append(t[0])
        user_seq[user] = items
    return user_seq


def check_Kcore(user_items, user_core, item_core):
    # K-core user_core item_core, return False if any user/item < core
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, items in user_items.items():
        user_count[user] += 0
        for item in items:
            user_count[user] += 1
            item_count[item] += 1

    for user, num in user_count.items():
        if num < user_core:
            return user_count, item_count, False
    for item, num in item_count.items():
        if num < item_core:
            return user_count, item_count, False
    return user_count, item_count, True  # Already guaranteed Kcore


def filter_Kcore(user_items, user_core, item_core):
    # Loop filter K-core, filter out users and items that do not meet K-core
    user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    while not isKcore:
        cur_user_items = copy.deepcopy(user_items)
        for user, num in user_count.items():
            if user_count[user] < user_core:  # Delete the user
                cur_user_items.pop(user)
            else:
                for item in user_items[user]:
                    if item_count[item] < item_core:
                        cur_user_items[user].remove(item)
        user_items = cur_user_items
        user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    total_interactions = 0
    for user, items in user_items.items():
        total_interactions += len(items)
    print("interactions: {0} after k-core filter".format(add_comma(total_interactions)))
    return user_items


def sample_inter(user_items, user_num, item_len):
    total_interactions = 0
    cur_user_items = copy.deepcopy(user_items)
    valid_users = {user: items[:item_len] for user, items in cur_user_items.items() if len(items) >= item_len}

    random.seed(0)
    sampled_users = random.choices(list(valid_users.keys()), k=user_num)
    user_items = {user: valid_users[user] for user in sampled_users}
    for user, items in user_items.items():
        total_interactions += len(items)
    print("interactions: {0} after sampling".format(add_comma(total_interactions)))
    return user_items


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


def main_process(data_name, args, data_type='Amazon'):
    assert data_type in {'Amazon', 'Yelp', 'Steam'}
    rating_score = -0.1  # rating score smaller than this score would be deleted

    datas = Amazon(rating_score, args)  # list of [user, item, timestamp]

    user_items = get_interaction(datas)  # dict of {user: interaction list sorted by time}
    print(f'{data_name} Raw data has been processed! Lower than {rating_score} are deleted!')
    print(f'User Num: {len(user_items)}')
    # raw_id user: [item1, item2, item3...]

    # user 25-core item 10-core
    # user_core, item_core = 25, 10
    # user_items = filter_Kcore(user_items, user_core=user_core, item_core=item_core)
    # print(f'User {user_core}-core complete! Item {item_core}-core complete!')
    # user_num, item_num, datamaps = id_map(user_items)  # get mapping dicts
    # user_count, item_count, isKcore = check_Kcore(user_items, user_core=user_core, item_core=item_core)
    # assert isKcore is True

    # sample 10000 users, item max len is 17
    user_num, item_len = 10000, 17
    user_items = sample_inter(user_items, user_num=user_num, item_len=item_len)
    user_num, item_num, datamaps = id_map(user_items)  # get mapping dicts
    user_count, item_count, _ = check_Kcore(user_items, user_core=1, item_core=1)

    user_count_list = list(user_count.values())  # user click count
    user_avg, user_min, user_max = np.mean(user_count_list), np.min(user_count_list), np.max(user_count_list)
    item_count_list = list(item_count.values())  # item click count
    item_avg, item_min, item_max = np.mean(item_count_list), np.min(item_count_list), np.max(item_count_list)
    interact_num = np.sum([x for x in user_count_list])
    sparsity = (1 - interact_num / (user_num * item_num)) * 100
    show_info = f'Total User: {user_num}, Avg User: {user_avg:.4f}, Min Len: {user_min}, Max Len: {user_max}\n' + \
                f'Total Item: {item_num}, Avg Item: {item_avg:.4f}, Min Inter: {item_min}, Max Inter: {item_max}\n' + \
                f'Interaction Num: {interact_num}, Sparsity: {sparsity:.2f}%'
    print(show_info)

    print('Begin extracting meta infos...')

    meta_infos = Amazon_meta(datamaps, args)
    meta_infos = reformat_titles(meta_infos, args)
    meta_infos = reformat_repeat_title(meta_infos)

    print(f'{data_name} & {add_comma(user_num)} & {add_comma(item_num)} & {user_avg:.1f}'
          f'& {item_avg:.1f} & {add_comma(interact_num)} & {sparsity:.2f}%')

    category_infos = Amazon_category(meta_infos)

    # -------------- Save Data ---------------
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)
    save_json(user_items, os.path.join(args.data_path, 'sequential.jsonl'))
    save_json(meta_infos, os.path.join(args.data_path, 'metas.jsonl'))
    save_json(category_infos, os.path.join(args.data_path, 'category.jsonl'))

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

    print('Done!!!')


if __name__ == '__main__':
    _args = parse_args()
    main_process(_args.full_data_name, args=_args, data_type='Amazon')
