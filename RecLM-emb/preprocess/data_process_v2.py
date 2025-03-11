# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import random
import pandas as pd
from tqdm import tqdm
import argparse
import os
from template import dialog_template, usersummary_template, query_template, neg_query_template

from utils import get_item_text
random.seed(2024)

def parse_args():
    parser = argparse.ArgumentParser(description="data process")
    parser.add_argument(
        "--in_seq_data", type=str, help="", default=""
    )
    parser.add_argument(
        "--in_meta_data", type=str, help="", default=""
    )
    parser.add_argument(
        "--out_conv", type=str, help="", default=""
    )
    parser.add_argument(
        "--out_gpt_conv", type=str, help="", default=""
    )
    parser.add_argument(
        "--out_user_sum", type=str, help="", default=""
    )
    parser.add_argument(
        "--out_gpt_user_sum", type=str, help="", default=""
    )
    parser.add_argument(
        "--out_query", type=str, help="", default=""
    )
    parser.add_argument(
        "--out_gpt_query", type=str, help="", default=""
    )
    parser.add_argument(
        "--out_neg_query", type=str, help="", default=""
    )
    parser.add_argument(
        "--out_gpt_neg_query", type=str, help="", default=""
    )
    args = parser.parse_args()
    return args


def gen_conv(args, itemid2title, itemid2features):
    max_sample_num = 10000
    with open(args.in_seq_data, 'r') as rd:
        all_samples = rd.readlines()
    if len(all_samples) > max_sample_num:
        all_samples = random.sample(all_samples, max_sample_num)

    df = pd.DataFrame(columns=['question'])
    seq_datas = []

    for line in tqdm(all_samples, desc='gen_conv', total=len(all_samples)):
        userid, itemids = line.strip().split(' ', 1)
        itemids = itemids.split(' ')
        assert len(itemids) > 3
        max_index = len(itemids)-2
        min_index = min(1, max_index)
        target_index = random.randint(min_index, max_index)
        target_item = int(itemids[target_index])
        history = [int(itemid) for itemid in itemids[:target_index]]
        history = history[-20:]
        if random.random() > 0.5:
            history = history[::-1]
        
        history_titles = ', '.join([itemid2title[int(x)][1] for x in history])
        num_round = random.randint(2, 5)
        difficulty = random.choice(['elementary school', 'high school', 'college', 'PhD'])

        target_info = {'title': itemid2title[target_item][1]}
        sampled_features = random.sample(itemid2features[target_item], random.randint(min(2, len(itemid2features[target_item])), len(itemid2features[target_item])))
        for feature in sampled_features:
            if feature[0] in ['tags: ', 'game details: ']:
                target_info[feature[0][:-2]] = ", ".join(random.sample(feature[1], random.randint(1, len(feature[1]))))
            else:
                target_info[feature[0][:-2]] = feature[1]
        
        seq_datas.append({
            'user_id': int(userid),
            'target_id': target_item,
            'history_ids': history,
            'num_round': num_round,
            'difficulty': difficulty,
            'target_info': target_info,
        })

        query = dialog_template.replace('{{history}}', history_titles).replace('{{target_info}}', json.dumps(target_info)).replace('{{num_round}}', str(num_round)).replace('{{difficulty}}', difficulty)
        if random.random() > 0.7: #use neg requirment
            query = query.replace('{{neg_requirement}}', '\n    8. The conversation should include some wrong guesses about user preferences from the Assistant.')
        else:
            query = query.replace('{{neg_requirement}}', '')
        
        df.loc[len(df)] = [query]

    with open(args.out_conv, 'w') as wt:
        for sample in seq_datas:
            wt.write(json.dumps(sample) + '\n')
    # split_index = len(df) // 3   
    # first_half = df.iloc[:split_index, :]  
    # second_half = df.iloc[split_index:2*split_index, :]
    # third_half = df.iloc[2*split_index:, :]
    df.to_csv(args.out_gpt_conv+".csv", index=False)


def gen_summary(args, itemid2title):
    max_sample_num = 10000
    with open(args.in_seq_data, 'r') as rd:
        all_samples = rd.readlines()
    if len(all_samples) > max_sample_num:
        all_samples = random.sample(all_samples, max_sample_num)

    df = pd.DataFrame(columns=['question'])
    seq_datas = []

    for line in tqdm(all_samples, desc='gen_user_summary', total=len(all_samples)):
        userid, itemids = line.strip().split(' ', 1)
        itemids = itemids.split(' ')
        assert len(itemids) > 3
        max_index = len(itemids)-2
        min_index = min(1, max_index)
        target_index = random.randint(min_index, max_index)
        target_item = int(itemids[target_index])
        history = [int(itemid) for itemid in itemids[:target_index]]
        history = history[-20:]
        if random.random() > 0.5:
            history = history[::-1]
        
        history_titles = ', '.join([itemid2title[int(x)][1] for x in history])

        difficulty = random.choice(['elementary school', 'high school', 'college', 'PhD'])
        num_words = random.choice([50, 100, 150, 200])
        clarity = random.choice(['clear', 'understandable with some effort', 'ambiguous'])
        writing_style = random.choice(['first-person', 'third-person', 'non-narrative'])

        seq_datas.append({
            'user_id': int(userid),
            'target_id': target_item,
            'history_ids': history,
            'difficulty': difficulty,
            'num_words': num_words,
            'clarity': clarity,
            'writing_style': writing_style,
        })
        query = usersummary_template.format(history=history_titles, difficulty=difficulty, num_words=num_words, clarity=clarity, writing_style=writing_style)
        df.loc[len(df)] = [query]
    
    with open(args.out_user_sum, 'w') as wt:
        for sample in seq_datas:
            wt.write(json.dumps(sample) + '\n')
    # split_index = len(df) // 3   
    # first_half = df.iloc[:split_index, :]  
    # second_half = df.iloc[split_index:2*split_index, :]
    # third_half = df.iloc[2*split_index:, :]
    df.to_csv(args.out_gpt_user_sum+".csv", index=False)

def gen_query(args, itemid2title, itemid2features):
    df = pd.DataFrame(columns=['question'])
    item_datas = []
    for id, title in tqdm(enumerate(itemid2title), desc='gen_query', total=len(itemid2title)):
        if id==0:
            continue
        for _ in range(1):
            target_info = {'title': title[1]}
            features = itemid2features[id] if 'description: ' not in itemid2features[id][-1][0] else itemid2features[id][:-1]

            if random.random() > 0.7 and 'tags: ' in itemid2features[id][0][0]:
                target_info['tags'] = ", ".join(random.sample(itemid2features[id][0][1], random.randint(1, min(5, len(itemid2features[id][0][1])))))
                features = features[1:]
                sampled_features = random.sample(features, random.randint(1, min(3, len(features))))
            else:
                sampled_features = random.sample(features, random.randint(1, min(5, len(features))))
            for feature in sampled_features:
                if feature[0] in ['tags: ', 'game details: ']:
                    target_info[feature[0][:-2]] = ", ".join(random.sample(feature[1], random.randint(1, min(4, len(feature[1])))))
                else:
                    target_info[feature[0][:-2]] = feature[1]
            
            difficulty = random.choice(['elementary school', 'high school', 'college', 'PhD'])
            num_words = random.choice([30, 50, 70, 100])
            clarity = random.choice(['clear', 'understandable with some effort', 'ambiguous'])
            writing_style = random.choice(['first-person', 'third-person', 'non-narrative'])
            query = query_template.format(target_info=json.dumps(target_info), difficulty=difficulty, num_words=num_words, clarity=clarity, writing_style=writing_style)
            df.loc[len(df)] = [query]
            item_datas.append({
                'target_id': id,
                'target_info': target_info,
                'difficulty': difficulty,
                'num_words': num_words,
                'clarity': clarity,
                'writing_style': writing_style,
            })
    with open(args.out_query, 'w') as wt:
        for sample in item_datas:
            wt.write(json.dumps(sample) + '\n')
    df.to_csv(args.out_gpt_query+".csv", index=False)

def gen_neg_query(args, itemid2title, itemid2features):
    df = pd.DataFrame(columns=['question'])
    item_datas = []
    for id, title in tqdm(enumerate(itemid2title), desc='gen_neg_query', total=len(itemid2title)):
        if id==0:
            continue
        for _ in range(1):
            target_info = {'title': title[1]}
            features = []
            for feature in itemid2features[id]:
                if feature[0] not in ['description: ', 'price: ', 'release date: ']:
                    features.append(feature)

            if random.random() > 0.5 and 'tags: ' in itemid2features[id][0][0]:
                target_info['tags'] = ", ".join(random.sample(itemid2features[id][0][1], random.randint(1, min(4, len(itemid2features[id][0][1])))))
                features = features[1:]
                sampled_features = random.sample(features, random.randint(0, min(2, len(features))))
            else:
                sampled_features = random.sample(features, random.randint(1, min(3, len(features))))
            for feature in sampled_features:
                if feature[0] in ['tags: ', 'game details: ']:
                    target_info[feature[0][:-2]] = ", ".join(random.sample(feature[1], random.randint(1, min(3, len(feature[1])))))
                else:
                    target_info[feature[0][:-2]] = feature[1]
            
            difficulty = random.choice(['elementary school', 'high school', 'college', 'PhD'])
            num_words = random.choice([30, 50, 70, 100])
            clarity = random.choice(['clear', 'understandable with some effort', 'ambiguous'])
            writing_style = random.choice(['first-person', 'third-person', 'non-narrative'])
            query = neg_query_template.format(target_info=json.dumps(target_info), difficulty=difficulty, num_words=num_words, clarity=clarity, writing_style=writing_style)
            df.loc[len(df)] = [query]
            item_datas.append({
                'target_id': id,
                'target_info': target_info,
                'difficulty': difficulty,
                'num_words': num_words,
                'clarity': clarity,
                'writing_style': writing_style,
            })
    with open(args.out_neg_query, 'w') as wt:
        for sample in item_datas:
            wt.write(json.dumps(sample) + '\n')
    df.to_csv(args.out_gpt_neg_query+".csv", index=False)


if __name__ == "__main__":
    args = parse_args() 
    os.makedirs(os.path.dirname(args.out_conv), exist_ok=True)
    itemid2text, itemid2title, itemid2features, itemid2price_date_map = get_item_text(args.in_meta_data)
    gen_conv(args, itemid2title, itemid2features)
    gen_summary(args, itemid2title)
    gen_query(args, itemid2title, itemid2features)
    gen_neg_query(args, itemid2title, itemid2features)