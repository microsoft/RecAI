# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import random
import pandas as pd
import math
from transformers import AutoTokenizer
from collections import defaultdict
from tqdm import tqdm
import argparse
import os
from template import user2item_template, query2item_template, title2item_template, item2item_template, queryuser2item_template, vaguequery2item_template, relativequery2item_template, negquery2item_template

from utils import get_item_text, load_titleid_2_index, text4query2item, cal_item2pos, text4item2item, random_replace, vaguequery, vaguequery_neg_sample, get_item_stats, get_feature2itemid, text4negquery, get_price_date_stats
random.seed(2023)


def parse_args():
    parser = argparse.ArgumentParser(description="data process")
    parser.add_argument(
        "--in_seq_data", type=str, help=""
    )
    parser.add_argument(
        "--in_meta_data", type=str, help=""
    )
    parser.add_argument(
        "--in_search2item", type=str, default='', help=""
    )
    parser.add_argument(
        "--out_search2item", type=str, default='', help=""
    )
    parser.add_argument(
        "--out_user2item", type=str, help=""
    )
    parser.add_argument(
        "--out_query2item", type=str, help=""
    )
    parser.add_argument(
        "--out_title2item", type=str, help=""
    )
    parser.add_argument(
        "--out_item2item", type=str, help=""
    )
    parser.add_argument(
        "--out_queryuser2item", type=str, help=""
    )
    parser.add_argument(
        "--out_misspell2item", type=str, help=""
    )
    parser.add_argument(
        "--out_vaguequery2item", type=str, help=""
    )
    parser.add_argument(
        "--out_relativequery2item", type=str, help=""
    )
    parser.add_argument(
        "--out_negquery2item", type=str, help=""
    )
    parser.add_argument(
        "--neg_num", type=int, default=7, help=""
    )
    parser.add_argument(
        "--model_path_or_name", type=str, help=""
    )
    args = parser.parse_args()
    return args


def gen_user2item(itemid2text, itemid2title, itemid2features, args):
    count=0
    total_q_len = 0
    max_q_len = 0
    min_q_len = 100000
    
    max_sample_num = 30000
    with open(args.in_seq_data, 'r') as rd:
        all_samples = rd.readlines()
    if len(all_samples) > max_sample_num:
        all_samples = random.sample(all_samples, max_sample_num)
        
        
    with open(args.out_user2item, 'w') as f:
        for line in tqdm(all_samples, desc='gen_user2item', total=len(all_samples)):
            userid, itemids = line.strip().split(' ', 1)
            itemids = itemids.split(' ')
            ground_set = set([int(x) for x in itemids])
            
            select_prob = 2.0 / (len(itemids) - 1) ## for each users, we sample about 2 data samples
            for target_index in range(1, len(itemids)-1):
                if random.random() > select_prob:
                    continue
                query_items = itemids[:target_index][::-1]
                query_items = query_items[:20] # truncate to 20
                if random.random() < 0.5:
                    template = "{}"
                else:
                    template = random.choice(user2item_template)

                query = ''
                has_prefix = False #if random.random() < 0.5 else True
                #if random.random() >= 0.2:
                for x in query_items:
                    if has_prefix:
                        query += 'title: ' + itemid2title[int(x)][1] + ', '
                    else:
                        query += itemid2title[int(x)][1] + ', '
                # else:
                #     for x in query_items:
                #         if has_prefix:
                #             query += 'title: ' + itemid2title[int(x)][1] + ', '
                #         else:
                #             query += itemid2title[int(x)][1] + ', '
                #         features = itemid2features[int(x)]
                #         sampled_fea = random.sample(features, random.randint(1, len(features)))
                #         for key, value in sampled_fea:
                #             if 'game details: '==key or 'tags: '==key:
                #                 features_value = ','.join(random.sample(value, random.randint(1, len(value))))
                #                 if has_prefix:
                #                     query += key + features_value + ', '
                #                 else:
                #                     query += features_value + ', '
                #             else:
                #                 if has_prefix:
                #                     query += key + value + ', '
                #                 else:
                #                     query += value + ', '
                query = query.strip().strip(',')

                template_length = len(tokenizer.tokenize(template))
                tokens = tokenizer.tokenize(query)[:args.max_seq_len-template_length]
                truncated_query = tokenizer.convert_tokens_to_string(tokens).strip().strip(',')

                query = template.format(truncated_query)

                q_len = template_length + len(tokens)
                total_q_len += q_len
                max_q_len = max(max_q_len, q_len)
                min_q_len = min(min_q_len, q_len)

                target_item = int(itemids[target_index])
                neg_items = []
                while len(neg_items) < args.neg_num:
                    neg_item = random.randint(1, len(itemid2title)-1)
                    if neg_item not in ground_set:
                        neg_items.append(neg_item)
                output = {
                    'user_id': userid,
                    'item_id': target_item,
                    'neg_ids': neg_items,
                    'query': query,
                    'pos': [itemid2text[target_item]],
                    'neg': [itemid2text[x] for x in neg_items]
                }
                f.write(json.dumps(output) + '\n')
                count += 1
    print('gen_user2item total samples: ', count)
    print('avg q len: ', total_q_len/count)
    print('max q len: ', max_q_len)
    print('min q len: ', min_q_len)


def gen_searchquery2item(itemid2text, itemid2title, itemid2features, args, titleid2idx):
    infile = args.in_search2item
    outfile = args.out_search2item
    search_data = pd.read_json(infile, lines=True)
    count=0
    with open(outfile, 'w') as f:
        n = len(search_data)
        for i in range(n):
            row = search_data.iloc[i]
            queries = row['response'].split('#TAB#')
            titleid = str(row['TitleId'])
            if len(queries) > 1 and titleid in titleid2idx:
                for query in queries:
                    target_item = titleid2idx[titleid]
                    neg_items = []
                    while len(neg_items) < args.neg_num:
                        neg_item = random.randint(1, len(itemid2title)-1)
                        if neg_item != target_item:
                            neg_items.append(neg_item)
                    output = {
                        'item_id': target_item,
                        'neg_ids': neg_items,
                        'query': query,
                        'pos': [itemid2text[target_item]],
                        'neg': [itemid2text[x] for x in neg_items]
                    }
                    f.write(json.dumps(output) + '\n') 
                    count += 1
    print('gen_searchquery2item total samples: ', count)
                

def gen_query2item(itemid2text, itemid2title, itemid2features, args):
    count=0
    total_q_len = 0
    max_q_len = 0
    min_q_len = 100000
    with open(args.out_query2item, 'w') as f:
        for idx, cont in tqdm(enumerate(itemid2features[1:]), desc='gen_query2item', total=len(itemid2features)-1):
            #cont[:-1] if cont[-1].startswith('description: ') else cont
            for _ in range(3):
                target_item_title = itemid2title[idx+1][1]
                if random.random() < 0.6:
                    target_features = [itemid2title[idx+1]] + cont if random.random() < 0.5 else cont
                    query, sampled_features, ground_truth = text4query2item(target_features, target_item_title, 1, len(target_features), 1, math.inf)
                else: # sparsequery2item
                    target_features = [x for x in cont if x[0] not in ['description: ']]
                    query, _, ground_truth = text4query2item(target_features, target_item_title, 1, min(3, len(target_features)), 1, 2)
                
                template = random.choice(query2item_template)

                template_length = len(tokenizer.tokenize(template))
                tokens = tokenizer.tokenize(query)[:args.max_seq_len-template_length]
                truncated_query = tokenizer.convert_tokens_to_string(tokens).strip()

                query = template.format(truncated_query)

                q_len = template_length + len(tokens)
                total_q_len += q_len
                max_q_len = max(max_q_len, q_len)
                min_q_len = min(min_q_len, q_len)

                target_item = int(idx+1)
                neg_items = []
                while len(neg_items) < args.neg_num:
                    neg_item = random.randint(1, len(itemid2title)-1)
                    neg_features = {itemid2title[neg_item][0]: itemid2title[neg_item][1]}
                    for x in itemid2features[neg_item]:
                        neg_features[x[0]] = x[1]
                    for key, value in ground_truth:
                        if key in neg_features:
                            if isinstance(value, list):
                                if any([x not in neg_features[key] for x in value]):
                                    neg_items.append(neg_item)
                                    break
                            else:
                                if value != neg_features[key]:
                                    neg_items.append(neg_item)
                                    break
                        else:
                            neg_items.append(neg_item)
                            break
                    random.shuffle(ground_truth) # to sample negative items based on different features
                output = {
                    'item_id': target_item,
                    'neg_ids': neg_items,
                    'query': query,
                    'pos': [itemid2text[target_item]],
                    'neg': [itemid2text[x] for x in neg_items]
                }
                f.write(json.dumps(output) + '\n')
                count += 1
    print('gen_query2item total samples: ', count)
    print('avg q len: ', total_q_len/count)
    print('max q len: ', max_q_len)
    print('min q len: ', min_q_len)

def gen_title2item(itemid2text, itemid2title, args):
    count=0
    total_q_len = 0
    max_q_len = 0
    min_q_len = 100000
    with open(args.out_title2item, 'w') as f:
        for idx, cont in tqdm(enumerate(itemid2title[1:]), desc='gen_title2item', total=len(itemid2title)-1):
            target_item_title = cont[1]
            for _ in range(1):
                if random.random() < 0.6:
                    continue
                query = target_item_title
                template = random.choice(title2item_template)

                template_length = len(tokenizer.tokenize(template))
                tokens = tokenizer.tokenize(query)[:args.max_seq_len-template_length]
                truncated_query = tokenizer.convert_tokens_to_string(tokens).strip()

                query = template.format(truncated_query)

                q_len = template_length + len(tokens)
                total_q_len += q_len
                max_q_len = max(max_q_len, q_len)
                min_q_len = min(min_q_len, q_len)

                target_item = int(idx+1)
                neg_items = []
                while len(neg_items) < args.neg_num:
                    neg_item = random.randint(1, len(itemid2title)-1)
                    if neg_item != target_item and target_item_title != itemid2title[neg_item][1]:
                        neg_items.append(neg_item)
                output = {
                    'item_id': target_item,
                    'neg_ids': neg_items,
                    'query': query,
                    'pos': [itemid2text[target_item]],
                    'neg': [itemid2text[x] for x in neg_items]
                }
                f.write(json.dumps(output) + '\n')
                count += 1
    print('gen_title2item total samples: ', count)
    print('avg q len: ', total_q_len/count)
    print('max q len: ', max_q_len)
    print('min q len: ', min_q_len)   

def gen_item2item(itemid2text, itemid2title, itemid2features, args):
    item2pos = cal_item2pos(args.in_seq_data)

    count=0
    total_q_len = 0
    max_q_len = 0
    min_q_len = 100000
    with open(args.out_item2item, 'w') as f:
        for item, pos_set in tqdm(item2pos.items(), desc='gen_item2item', total=len(item2pos)):
            source_item_features = itemid2features[item]
            source_item_title = itemid2title[item][1]
            for _ in range(1):
                for target_item in pos_set:
                    if random.random() > 0.12:
                        continue
                    query = text4item2item(source_item_features, source_item_title)

                    template = random.choice(item2item_template)
                    template_length = len(tokenizer.tokenize(template))
                    tokens = tokenizer.tokenize(query)[:args.max_seq_len-template_length]
                    truncated_query = tokenizer.convert_tokens_to_string(tokens).strip()

                    query = template.format(truncated_query)

                    q_len = template_length + len(tokens)
                    total_q_len += q_len
                    max_q_len = max(max_q_len, q_len)
                    min_q_len = min(min_q_len, q_len)
                    
                    neg_items = []
                    while len(neg_items) < args.neg_num:
                        neg_item = random.randint(1, len(itemid2title)-1)
                        if neg_item not in pos_set and neg_item != item:
                            neg_items.append(neg_item)
                    output = {
                        'item_id': item,
                        'pos_id': target_item,
                        'neg_ids': neg_items,
                        'query': query,
                        'pos': [itemid2text[target_item]],
                        'neg': [itemid2text[x] for x in neg_items]
                    }
                    f.write(json.dumps(output) + '\n')
                    count += 1
    print('gen_item2item total samples: ', count)
    print('avg q len: ', total_q_len/count)
    print('max q len: ', max_q_len)
    print('min q len: ', min_q_len)

def gen_queryuser2item(itemid2text, itemid2title, itemid2features, args):
    count=0
    total_q_len = 0
    max_q_len = 0
    min_q_len = 100000
    max_sample_num = 12000
    with open(args.in_seq_data, 'r') as rd:
        all_samples = rd.readlines()
    if len(all_samples) > max_sample_num:
        all_samples = random.sample(all_samples, max_sample_num)
        
    with open(args.out_queryuser2item, 'w') as f:
        for line in tqdm(all_samples, desc='gen_queryuser2item', total=len(all_samples)):
            userid, itemids = line.strip().split(' ', 1)
            itemids = itemids.split(' ')
            ground_set = set([int(x) for x in itemids])
            
            query_items = itemids[:-2][::-1]
            query_items = query_items[:20] # truncate to 20
            template = random.choice(queryuser2item_template)

            query = ''
            has_prefix = False #if random.random() < 0.5 else True
            # if random.random() >= 0.6:
            for x in query_items:
                if has_prefix:
                    query += 'title: ' + itemid2title[int(x)][1] + ', '
                else:
                    query += itemid2title[int(x)][1] + ', '
            # else:
            #     for x in query_items:
            #         if has_prefix:
            #             query += 'title: ' + itemid2title[int(x)][1] + ', '
            #         else:
            #             query += itemid2title[int(x)][1] + ', '
            #         features = itemid2features[int(x)]
            #         sampled_fea = random.sample(features, random.randint(1, len(features)))
            #         for key, value in sampled_fea:
            #             if 'game details: '==key or 'tags: '==key:
            #                 features_value = ','.join(random.sample(value, random.randint(1, len(value))))
            #                 if has_prefix:
            #                     query += key + features_value + ', '
            #                 else:
            #                     query += features_value + ', '
            #             else:
            #                 if has_prefix:
            #                     query += key + value + ', '
            #                 else:
            #                     query += value + ', '
            query = query.strip().strip(',')

            target_item_title = itemid2title[int(itemids[-2])]
            
            if random.random() < 0.6:
                target_features = [target_item_title] + itemid2features[int(itemids[-2])] if random.random() < 0.5 else itemid2features[int(itemids[-2])]
                target_query, _, _ = text4query2item(target_features, target_item_title[1], 1, min(4, len(target_features)), 1, math.inf)## don't provide too much info
            else: #sparse case
                target_features = [x for x in itemid2features[int(itemids[-2])] if x[0] not in ['description: ']]
                target_query, _, _ = text4query2item(target_features, target_item_title[1], 1, min(3, len(target_features)), 1, 2)## don't provide too much info
          

            template_length = len(tokenizer.tokenize(template))
            tokens = tokenizer.tokenize(query)
            query_length = min((args.max_seq_len-template_length)*4//5, len(tokens))
            target_query_length = args.max_seq_len-template_length-query_length

            tokens = tokens[:query_length]
            truncated_query = tokenizer.convert_tokens_to_string(tokens).strip().strip(',')
            target_tokens = tokenizer.tokenize(target_query)[:target_query_length]
            truncated_target_query = tokenizer.convert_tokens_to_string(target_tokens).strip().strip(',')
            
            query = template.format(truncated_query, truncated_target_query)            

            q_len = template_length + len(tokens) + len(target_tokens)
            total_q_len += q_len
            max_q_len = max(max_q_len, q_len)
            min_q_len = min(min_q_len, q_len)

            target_item = int(itemids[-2])
            neg_items = []
            while len(neg_items) < args.neg_num:
                neg_item = random.randint(1, len(itemid2title)-1)
                # neg_features = set(itemid2features[neg_item])
                if neg_item != target_item:
                    neg_items.append(neg_item)
            output = {
                'user_id': userid,
                'item_id': target_item,
                'neg_ids': neg_items,
                'query': query,
                'pos': [itemid2text[target_item]],
                'neg': [itemid2text[x] for x in neg_items]
            }
            f.write(json.dumps(output) + '\n')
            count += 1
    print('gen_queryuser2item total samples: ', count)
    print('avg q len: ', total_q_len/count)
    print('max q len: ', max_q_len)
    print('min q len: ', min_q_len)

def gen_misspell2item(itemid2text, itemid2title, args):
    dataset=[]
    for idx, cont in tqdm(enumerate(itemid2title[1:]), desc='gen_misspell2item', total=len(itemid2title)-1):
        target_item_title = cont[1]
        for _ in range(2):
            query = random_replace(target_item_title)
            while query == target_item_title:
                query = random_replace(target_item_title)
            if random.random() < 0.5:
                template = "{}"
            else:
                template = random.choice(title2item_template)
            query = template.format(query)
            target_item = int(idx+1)
            neg_items = []
            while len(neg_items) < args.neg_num:
                neg_item = random.randint(1, len(itemid2title)-1)
                if neg_item != target_item and target_item_title != itemid2title[neg_item][1]:
                    neg_items.append(neg_item)
            output = {
                'item_id': target_item,
                'neg_ids': neg_items,
                'query': query,
                'pos': [itemid2text[target_item]],
                'neg': [itemid2text[x] for x in neg_items]
            }
            dataset.append(output)
    print('gen_misspell2item total samples: ', len(dataset))
    with open(args.out_misspell2item, 'w') as f:
        for d in dataset:
            f.write(json.dumps(d) + '\n')

def gen_vaguequery2item(itemid2text, itemid2price_date_map, args):
    dataset=[]
    price_date_stats = get_price_date_stats(itemid2price_date_map)
    for idx, cont in tqdm(enumerate(itemid2price_date_map[1:]), desc='gen_vaguequery2item', total=len(itemid2price_date_map)-1):
        price, date, next_month, last_month = cont['price'], cont['release date'], cont['next month'], cont['last month']
        if not price or not date or not next_month or not last_month:
            continue
        for _ in range(1):
            query, combine_flag, price_flag, month_flag, year_flag = vaguequery(price, date, next_month, last_month, price_date_stats)
            template = random.choice(vaguequery2item_template)
            target_item = int(idx+1)
            neg_items = vaguequery_neg_sample(combine_flag, price_flag, month_flag, year_flag, itemid2price_date_map, args)
            if len(neg_items) <= args.neg_num//2:
                continue
            output = {
                    'item_id': target_item,
                    'neg_ids': neg_items,
                    'query': template.format(query),
                    'pos': [itemid2text[target_item]],
                    'neg': [itemid2text[x] for x in neg_items]
                }
            dataset.append(output)
    print('gen_vaguequery2item total samples: ', len(dataset))
    with open(args.out_vaguequery2item, 'w') as f:
        for d in dataset:
            f.write(json.dumps(d) + '\n')

def gen_relativequery2item(itemid2text, args):
    recent_itemset, cheap_itemset, expensive_itemset, popular_itemset, total_num = get_item_stats(args.in_seq_data, args.in_meta_data)
    count = 0
    with open(args.out_relativequery2item, 'w') as f:
        for task, itemset in zip(['recent', 'cheap', 'expensive', 'popular'], [recent_itemset, cheap_itemset, expensive_itemset, popular_itemset]):
            for target_item in itemset:
                for _ in range(1):
                    query = random.choice(relativequery2item_template[task])
                    neg_items = []
                    while len(neg_items) < args.neg_num:
                        neg_item = random.randint(1, len(itemid2text)-1)
                        if neg_item not in itemset:
                            neg_items.append(neg_item)
                    output = {
                        'item_id': target_item,
                        'neg_ids': neg_items,
                        'query': query,
                        'pos': [itemid2text[target_item]],
                        'neg': [itemid2text[x] for x in neg_items]
                    }
                    f.write(json.dumps(output) + '\n')
                    count += 1
    print('gen_relativequery2item total samples: ', count)

def gen_negquery2item(itemid2text, args):
    features2itemids = get_feature2itemid(args.in_meta_data)
    sample_names_l1 = list(features2itemids.keys())
    sample_names_l2 = {x: list(features2itemids[x].keys()) for x in sample_names_l1}
    count=0
    with open(args.out_negquery2item, 'w') as f:
        for _ in tqdm(range(30000), desc='gen_negquery2item'):
            query, pos_set, neg_set = text4negquery(sample_names_l1, sample_names_l2, itemid2text, features2itemids)
            if len(pos_set) == 0 or len(neg_set) <= args.neg_num//2:
                continue
            target_item = random.sample(list(pos_set), min(2, len(pos_set)))
            neg_items = random.sample(list(neg_set), min(args.neg_num, len(neg_set)))
            template = random.choice(negquery2item_template)
            output = {
                'item_id': target_item,
                'neg_ids': neg_items,
                'query': template.format(query),
                'pos': [itemid2text[x] for x in target_item],
                'neg': [itemid2text[x] for x in neg_items]
            }
            f.write(json.dumps(output) + '\n')
            count += 1
    print('gen_negquery2item total samples: ', count)
                

if __name__ == "__main__":
    args = parse_args() 
    os.makedirs(os.path.dirname(args.out_user2item ), exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_name, use_fast=True)
    args.max_seq_len = tokenizer.model_max_length
    itemid2text, itemid2title, itemid2features, itemid2price_date_map = get_item_text(args.in_meta_data)
    gen_query2item(itemid2text, itemid2title, itemid2features, args)
    gen_title2item(itemid2text, itemid2title, args)
    gen_item2item(itemid2text, itemid2title, itemid2features, args)
    gen_queryuser2item(itemid2text, itemid2title, itemid2features, args)
    gen_user2item(itemid2text, itemid2title, itemid2features, args)
    gen_misspell2item(itemid2text, itemid2title, args)
    gen_vaguequery2item(itemid2text, itemid2price_date_map, args)
    gen_relativequery2item(itemid2text, args)
    gen_negquery2item(itemid2text, args)
    
    if args.in_search2item and os.path.exists(args.in_search2item):
        titleid2idx = load_titleid_2_index(args.in_meta_data)
        gen_searchquery2item(itemid2text, itemid2title, itemid2features, args, titleid2idx)