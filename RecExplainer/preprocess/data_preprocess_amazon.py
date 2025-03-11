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
    parser.add_argument(
        "--full_data_name", type=str, help=""
    )
    parser.add_argument(
        "--meta_file", type=str, help=""
    )
    parser.add_argument(
        "--review_file", type=str, help=""
    )
    parser.add_argument(
        "--save_data_file", type=str, default='', help=""
    )
    parser.add_argument(
        "--save_metadata_file", type=str, default='', help=""
    )
    parser.add_argument(
        "--save_datamaps_file", type=str, help=""
    )
    parser.add_argument(
        "--save_review_file", type=str, help=""
    )
    args = parser.parse_args()
    return args


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def Amazon(rating_score=-1, args=None):
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
            if "title" not in line or len(line['title'])==0:
                continue
            items_with_title[line['asin']] = 1 
    print(f"items with title: {add_comma(len(items_with_title))}")
    datas = []
    data_dict = {}
    with gzip.open(args.review_file, "r") as fr:
        for line in tqdm(fr, desc="load all interactions"):
            # try:
            line = json.loads(line)
            user = line['reviewerID']
            item = line['asin']
            if float(line['overall']) <= rating_score or item not in items_with_title: # remove low rating
                continue
            if (user, item) in data_dict:
                continue
            time = line['unixReviewTime']
            data_dict[(user, item)] = int(time) # merge duplicate interactions, keep the first record
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
            line['title'] = re.sub(r'\n\t', ' ', line['title']).encode('UTF-8', 'ignore').decode('UTF-8')
                # line['title'] = line['title'].split(",")[0]
            if "description" in line:
                if type(line['description']) == str:
                    line['description'] = re.sub(r'\n\t', ' ', line['description']).encode('UTF-8', 'ignore').decode('UTF-8')
                elif type(line['description']) == list:
                    descs = []
                    for desc in line['description']:
                        desc = re.sub(r'\n\t', ' ', desc).encode('UTF-8', 'ignore').decode('UTF-8')
                        descs.append(desc)
                    line['description'] = descs
            if 'related' in line:
                del line['related']
            if 'imUrl' in line:
                del line['imUrl']
            if 'imageURL' in line:
                del line['imageURL']
            if 'imageURLHighRes' in line:
                del line['imageURLHighRes']
            mapped_id = datamaps['item2id'][line['asin']]
            meta_datas[mapped_id] = line
    return meta_datas

def Amazon_Review(user2id, item2id, rating_score=-1, args=None):
    '''
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
    review_data = {}
    # aspect_explanations = None
    # if os.path.exists(f"../../P5/raw_data/reviews_{dataset_name}.pkl"):
    #     aspect_explanations = load_pickle(f"../../P5/raw_data/reviews_{dataset_name}.pickle")
    # no_sentence = 0
    text_count = 0
    with gzip.open(args.review_file, "r") as fr:
        for lidx, line in tqdm(enumerate(fr)):
            line = json.loads(line)
            if float(line['overall']) <= rating_score: # remove low rating
                continue
            user = line['reviewerID']
            item = line['asin']
            if (user, item) in review_data or user not in user2id or item not in item2id:
                continue

            if 'reviewText' in line:
                # exp_ = line
                # if aspect_explanations is not None:
                #     exp_ = aspect_explanations[lidx]
                #     assert exp_['user'] == user and exp_['item'] == item
                # if 'sentence' in exp_:
                #     selected_idx = random.randint(0, len(exp_['sentence'])-1)  # randomly sample review of only one feature
                #     line['explanation'] = exp_['sentence'][selected_idx][2]
                #     line['feature'] = exp_['sentence'][selected_idx][0]
                # else:
                #     no_sentence += 1
                line['reviewText'] = re.sub(r'\n\t', ' ', line['reviewText']).encode('UTF-8', 'ignore').decode('UTF-8')
            if 'summary' in line:
                line['summary'] = re.sub(r'\n\t', ' ', line['summary']).encode('UTF-8', 'ignore').decode('UTF-8')
            
            if 'reviewText' in line or 'summary' in line:
                text_count += 1
            review_data[(user, item)] = line

    print(f"total review: {add_comma(len(review_data))}, text review: {add_comma(text_count)}")
    # how to obtain better review data?
    # print(f"No sentence: {no_sentence}/{len(review_data)}.")
    return review_data
        
def add_comma(num): # 1000000 -> 1,000,000
    str_num = str(num)
    res_num = ''
    for i in range(len(str_num)):
        res_num += str_num[i]
        if (len(str_num)-i-1) % 3 == 0:
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
        for item in items:
            user_count[user] += 1
            item_count[item] += 1

    for user, num in user_count.items():
        if num < user_core:
            return user_count, item_count, False
    for item, num in item_count.items():
        if num < item_core:
            return user_count, item_count, False
    return user_count, item_count, True # Already guaranteed Kcore

def filter_Kcore(user_items, user_core, item_core):
    # Loop filter K-core, filter out users and items that do not meet K-core
    user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    while not isKcore:
        cur_user_items = copy.deepcopy(user_items)
        for user, num in user_count.items():
            if user_count[user] < user_core: # Delete the user
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

def id_map(user_items): # user_items dict
    user2id = {} # raw 2 uid
    item2id = {} # raw 2 iid
    id2user = {} # uid 2 raw
    id2item = {} # iid 2 raw
    user_id = 1  # start from 1
    item_id = 1
    final_data = {}
    random_user_list = list(user_items.keys())
    random.shuffle(random_user_list)  # user is shuffled and re-encoded
    for user in random_user_list:
        items = user_items[user]
        if user not in user2id:
            user2id[user] = str(user_id)
            id2user[str(user_id)] = user
            user_id += 1
        iids = [] # item id lists
        for item in items:
            if item not in item2id:
                item2id[item] = str(item_id)
                id2item[str(item_id)] = item
                item_id += 1
            iids.append(item2id[item])
        uid = user2id[user]
        final_data[uid] = iids
    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item
    }
    #return: final_data: {uid: [iid1, iid2, ...], ...}, user_num, item_num, data_maps
    return final_data, user_id-1, item_id-1, data_maps

def main_process(data_name, args, data_type='Amazon'):
    assert data_type in {'Amazon', 'Yelp', 'Steam'}
    rating_score = -0.1  # rating score smaller than this score would be deleted
    # user 5-core item 5-core
    user_core = 5
    item_core = 5
    attribute_core = 0

    datas = Amazon(rating_score, args)  # list of [user, item, timestamp]

    user_items = get_interaction(datas) # dict of {user: interaction list sorted by time} 
    print(f'{data_name} Raw data has been processed! Lower than {rating_score} are deleted!')
    print(f'User Num: {len(user_items)}')
    # raw_id user: [item1, item2, item3...]
    
    user_items = filter_Kcore(user_items, user_core=user_core, item_core=item_core)
    print(f'User {user_core}-core complete! Item {item_core}-core complete!')
    
    user_items, user_num, item_num, datamaps = id_map(user_items) # get mapping dicts, randomly shuffle
    user_count, item_count, isKcore = check_Kcore(user_items, user_core=user_core, item_core=item_core)
    assert isKcore==True
    user_count_list = list(user_count.values()) # user click count
    user_avg, user_min, user_max = np.mean(user_count_list), np.min(user_count_list), np.max(user_count_list)
    item_count_list = list(item_count.values()) # item click count
    item_avg, item_min, item_max = np.mean(item_count_list), np.min(item_count_list), np.max(item_count_list)
    interact_num = np.sum([x for x in user_count_list])
    sparsity = (1 - interact_num / (user_num * item_num)) * 100
    show_info = f'Total User: {user_num}, Avg User: {user_avg:.4f}, Min Len: {user_min}, Max Len: {user_max}\n' + \
                f'Total Item: {item_num}, Avg Item: {item_avg:.4f}, Min Inter: {item_min}, Max Inter: {item_max}\n' + \
                f'Iteraction Num: {interact_num}, Sparsity: {sparsity:.2f}%'
    print(show_info)


    print('Begin extracting meta infos...')
    
    meta_infos = Amazon_meta(datamaps, args)

    print(f'{data_name} & {add_comma(user_num)} & {add_comma(item_num)} & {user_avg:.1f}'
          f'& {item_avg:.1f} & {add_comma(interact_num)} & {sparsity:.2f}\% \\')

    # -------------- Save Data ---------------
    with open(args.save_data_file, 'w') as out:
        for user, items in user_items.items():
            out.write(user + ' ' + ' '.join(items) + '\n')

    item_keys = sorted(meta_infos.keys(), key=lambda x: int(x))
    print(f"item2id: {len(datamaps['item2id'])}, meta_infos: {len(meta_infos)}, item_keys: {item_keys[:100]}")
    with open(args.save_metadata_file, 'w') as out:
        for key in item_keys:
            out.write(json.dumps(meta_infos[key]) + '\n')

    json_str = json.dumps(datamaps)
    with open(args.save_datamaps_file, 'w') as out:
        out.write(json_str)

    # -------------- Split Train/Valid/Test for Item Import & Tagging ---------------
    # all_items = [item for item in datamaps['item2id'].keys()]
    # random.shuffle(all_items)
    # train_split = int(len(all_items) * 0.8)
    # valid_split = int(len(all_items) * 0.1)
    # train_items = all_items[:train_split]
    # valid_items = all_items[train_split:train_split+valid_split]
    # test_items = all_items[train_split+valid_split:]
    # outputs = {'train': train_items, 'val': valid_items, 'test': test_items}
    # save_pickle(outputs, '../data/{}/item_splits.pkl'.format(short_data_name))


    # -------------- Create Train/Valid/Test for Review ---------------
    # review_data = Amazon_Review(datamaps['user2id'], datamaps['item2id'], rating_score)
    # train_exp_data, valid_exp_data, test_exp_data = [], [], []
    # train_review_data, valid_review_data, test_review_data = [], [], []
    # id2user, id2item = datamaps['id2user'], datamaps['id2item']
    # for user, items in user_items.items():
    #     user = id2user[user]
    #     test_item = id2item[items[-1]]
    #     valid_item = id2item[items[-2]]
    #     test_review_data.append(review_data[(user, test_item)])
    #     # if 'explanation' in review_data[(user, test_item)]:
    #     #     test_exp_data.append(review_data[(user, test_item)])
    #     valid_review_data.append(review_data[(user, valid_item)])
    #     # if 'explanation' in review_data[(user, valid_item)]:
    #     #     valid_exp_data.append(review_data[(user, valid_item)])
    #     for item in items[1:-2]:
    #         train_review_data.append(review_data[(user, id2item[item])])
    #         # if 'explanation' in review_data[(user, id2item[item])]:
    #         #     train_exp_data.append(review_data[(user, id2item[item])])
    # review_outputs = {'train': train_review_data, 'val': valid_review_data, 'test': test_review_data}
    # save_pickle(review_outputs, save_review_file)
    # # exp_outputs = {'train': train_exp_data, 'val': valid_exp_data, 'test': test_exp_data}
    # # save_pickle(exp_outputs, '../data/{}/exp_splits.pkl'.format(short_data_name))

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_data_file), exist_ok=True)
    main_process(args.full_data_name, args=args, data_type='Amazon')