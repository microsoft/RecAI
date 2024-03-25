# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import numpy as np
import copy
import json
from collections import defaultdict
import argparse

seed = 2023
random.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="data process")
    parser.add_argument(
        "--seqdata_file", type=str, help=""
    )
    parser.add_argument(
        "--metadata_file", type=str, help=""
    )
    parser.add_argument(
        "--save_data_file", type=str, help=""
    )
    parser.add_argument(
        "--save_metadata_file", type=str, default='', help=""
    )
    parser.add_argument(
        "--save_datamaps_file", type=str, help=""
    )
    parser.add_argument(
        "--item_thred", type=int, help=""
    )
    parser.add_argument(
        "--user_thred", type=int, help=""
    )
    args = parser.parse_args()
    return args

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

def check_Kcore(user_items, user_core, item_core):
    # K-core user_core item_core, return False if any user/item < core
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, items in user_items.items():
        user_count[user] = 0
        for item in items:
            user_count[user] += 1
            item_count[item] += 1

    for user, num in user_count.items():
        if num < user_core:
            return user_count, item_count, False
    for item, num in item_count.items():
        if num < item_core:
            return user_count, item_count, False
    return user_count, item_count, True # Kcore has been guaranteed

def filter_Kcore(user_items, user_core, item_core):
    # Filter out users and items that do not meet K-core requirements
    user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    while not isKcore:
        cur_user_items = copy.deepcopy(user_items)
        for user, num in user_count.items():
            if user_count[user] < user_core: # Delete the user directly
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
    print("interactions: {0} after k-core filter".format(total_interactions))
    return user_items

def filter(args):
    item_thred = args.item_thred
    item_count = {}
    user_items = {}
    with open(args.seqdata_file, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip().split(' ')
            user = int(line[0])
            items = [int(x) for x in line[1:]]
            user_items[user] = items
            for item in items:
                if item not in item_count:
                    item_count[item] = 0
                item_count[item] += 1

    sorted_item_count = sorted(item_count.items(), key=lambda x: x[1], reverse=True)
    assert sorted_item_count[item_thred][1] > 5
    all_items = set(x[0] for x in sorted_item_count[:item_thred])

    cur_user_items = {}
    for user, items in user_items.items():
        cur_seq = [x for x in items if x in all_items]
        if len(cur_seq) >= 5:
            cur_user_items[user] = cur_seq
    sorted_user_items = sorted(cur_user_items.items(), key=lambda x: len(x[1]), reverse=True)
    sorted_user_items = random.sample(sorted_user_items, args.user_thred)

    user_items = {}
    for user, items in sorted_user_items:
        user_items[user] = items

    user_items = filter_Kcore(user_items, 5, 5)
    user_items, user_num, item_num, datamaps = id_map(user_items)
    print("user_num: {0}, item_num: {1}".format(user_num, item_num))

    meta_infos = {}
    with open(args.metadata_file, 'r') as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            meta_infos[i+1] = line
        
    filtered_meta_infos = {}
    for old_id,iid in datamaps['item2id'].items():
        filtered_meta_infos[int(iid)] = meta_infos[old_id]

    with open(args.save_data_file, 'w') as out:
        for user, items in user_items.items():
            out.write(user + ' ' + ' '.join(items) + '\n')

    item_keys = sorted(filtered_meta_infos.keys(), key=lambda x: int(x))
    print(f"item2id: {len(datamaps['item2id'])}, meta_infos: {len(filtered_meta_infos)}, item_keys: {item_keys[:100]}")
    with open(args.save_metadata_file, 'w') as out:
        for key in item_keys:
            out.write(json.dumps(filtered_meta_infos[key]) + '\n')
    
    json_str = json.dumps(datamaps)
    with open(args.save_datamaps_file, 'w') as out:
        out.write(json_str)

def metadata_filter(args):
    import re
    punctuation = ' \t\n!.,;?/:'
    meta_infos = {}
    with open(args.save_metadata_file, 'r') as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            meta_infos[i+1] = line
        
    for iid, info in meta_infos.items():
        new_title = info['title'].strip(punctuation)
        assert len(new_title) > 0
        meta_infos[iid]['title'] = new_title
        if 'feature' in info and len(info['feature'])>0:
            new_features = []
            features = info['feature']
            for feature in features:
                result = feature.strip(punctuation)
                if len(result) > 0:
                    new_features.append(result)
            meta_infos[iid]['feature'] = new_features
        if 'description' in info and len(info['description'])>0:
            new_desc = []
            desc = info['description']
            for d in desc:
                result = d.strip(punctuation)
                if len(result) > 0:
                    new_desc.append(result)
            meta_infos[iid]['description'] = new_desc
        if 'brand' in info and len(info['brand'])>0:
            new_brand = re.sub(r'\n    \n    ', ' ', info['brand'])
            new_brand = new_brand.strip(punctuation)
            if len(new_brand) > 0:
                meta_infos[iid]['brand'] = new_brand

    item_keys = sorted(meta_infos.keys(), key=lambda x: int(x))
    print(f"item_keys: {item_keys[:100]}")
    with open(args.save_metadata_file, 'w') as out:
        for key in item_keys:
            out.write(json.dumps(meta_infos[key]) + '\n')

if __name__ == '__main__':
    args = parse_args()
    filter(args)
    metadata_filter(args)