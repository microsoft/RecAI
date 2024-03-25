# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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
from transformers import AutoTokenizer
import argparse

seed = 2023
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


prompt_templates={
    "iid2title": [{"USER": "What is the title of the item: {0} ?", "ASSISTANT": "{0}"},
                    {"USER": "What is the name of the item: {0} ?", "ASSISTANT": "{0}"},
                    {"USER": "Given the item: {0} , generate its title.", "ASSISTANT": "{0}"},
                    {"USER": "For the item: {0} , what is its title?", "ASSISTANT": "{0}"},
                    {"USER": "Can you tell me the title of the item: {0} ?", "ASSISTANT": "{0}"},
                    {"USER": "What's the title for the item: {0} ?", "ASSISTANT": "{0}"},
                    {"USER": "Please generate the title of the item: {0} ?", "ASSISTANT": "{0}"},
                    {"USER": "Title of the item: {0} ?", "ASSISTANT": "{0}"},
                    {"USER": "Item: {0} , what is its name?", "ASSISTANT": "{0}"},
                    {"USER": "Could you generate the title for the item: {0} ?", "ASSISTANT": "{0}"},
                    {"USER": "Regarding the item: {0} , what is its title?", "ASSISTANT": "{0}"},
                ],
    "iid2feature": [{"USER": "What are the features of the item: {0} ?", "ASSISTANT": "{0}"},
                    {"USER": "Given the item: {0} , generate its features.", "ASSISTANT": "{0}"},
                    {"USER": "Generate features for the item: {0} .", "ASSISTANT": "{0}"},
                    {"USER": "Can you tell me the features of the item: {0} ?", "ASSISTANT": "{0}"},
                    {"USER": "What are the features for the item: {0} ?", "ASSISTANT": "{0}"},
                    {"USER": "Please generate the features of the item: {0} ?", "ASSISTANT": "{0}"},
                    {"USER": "Features of the item: {0} ?", "ASSISTANT": "{0}"},
                    {"USER": "Item: {0} , what are its features?", "ASSISTANT": "{0}"},
                    {"USER": "Could you generate the features for the item: {0} ?", "ASSISTANT": "{0}"},
                    {"USER": "Regarding the item: {0} , what are its features?", "ASSISTANT": "{0}"},
                ],
    "iid2description": [{"USER": "What is the description of the item: {0} ?", "ASSISTANT": "{0}"},
                        {"USER": "Given the item: {0} , generate its description.", "ASSISTANT": "{0}"},
                        {"USER": "Generate the description for the item: {0} .", "ASSISTANT": "{0}"},
                        {"USER": "Can you tell me the description of the item: {0} ?", "ASSISTANT": "{0}"},
                        {"USER": "What's the description for the item: {0} ?", "ASSISTANT": "{0}"},
                        {"USER": "Please generate the description of the item: {0} ?", "ASSISTANT": "{0}"},
                        {"USER": "Description of the item: {0} ?", "ASSISTANT": "{0}"},
                        {"USER": "Item: {0} , what is its description?", "ASSISTANT": "{0}"},
                        {"USER": "Could you generate the description for the item: {0} ?", "ASSISTANT": "{0}"},
                        {"USER": "Regarding the item: {0} , what is its description?", "ASSISTANT": "{0}"},
                    ],
    "iid2brand": [{"USER": "What is the brand of the item: {0} ?", "ASSISTANT": "{0}"},
                    {"USER": "Given the item: {0} , generate its brand.", "ASSISTANT": "{0}"},
                    {"USER": "Generate the brand for the item: {0} .", "ASSISTANT": "{0}"},
                    {"USER": "Can you tell me the brand of the item: {0} ?", "ASSISTANT": "{0}"},
                    {"USER": "What's the brand for the item: {0} ?", "ASSISTANT": "{0}"},
                    {"USER": "Please generate the brand of the item: {0} ?", "ASSISTANT": "{0}"},
                    {"USER": "Brand of the item: {0} ?", "ASSISTANT": "{0}"},
                    {"USER": "Item: {0} , what is its brand?", "ASSISTANT": "{0}"},
                    {"USER": "Could you generate the brand for the item: {0} ?", "ASSISTANT": "{0}"},
                    {"USER": "Regarding the item: {0} , what is its brand?", "ASSISTANT": "{0}"},
                ],
    "iid2sim": [{"USER": "What is the most similar item to the item: {0} ?", "ASSISTANT": "{0}"},
                {"USER": "Given the item: {0} , generate the most similar item title.", "ASSISTANT": "{0}"},
                {"USER": "Generate the title of the most similar item to the item: {0} .", "ASSISTANT": "{0}"},
                {"USER": "Can you find an item closely related to this item: {0} ?", "ASSISTANT": "{0}"},
                {"USER": "What is the closest match to the item: {0} ?", "ASSISTANT": "{0}"},
                {"USER": "Generate an item with the highest similarity to: {0} .", "ASSISTANT": "{0}"},
                {"USER": "What item has the greatest resemblance to: {0} ?", "ASSISTANT": "{0}"},
                {"USER": "What is the nearest item to the given item: {0} ?", "ASSISTANT": "{0}"},
                {"USER": "What item shares the most similarities with the item: {0} ?", "ASSISTANT": "{0}"},
            ], 
    "feature2iid": [{"USER": "What is the item represented by the features: {0} ?", "ASSISTANT": "{0}"},
                    {"USER": "Given the features: {0} , generate the item title.", "ASSISTANT": "{0}"},
                    {"USER": "Generate the title of the item represented by the features: {0} .", "ASSISTANT": "{0}"},
                    {"USER": "What item corresponds to the features: {0} ?", "ASSISTANT": "{0}"},
                    {"USER": "Generate the item's name based on the features: {0} .", "ASSISTANT": "{0}"},
                    {"USER": "Generate the title for the item described by the features: {0} .", "ASSISTANT": "{0}"},
                    {"USER": "What item is characterized by the following features: {0} ?", "ASSISTANT": "{0}"},
                    {"USER": "Produce the name of the item associated with the features: {0} .", "ASSISTANT": "{0}"},
                    {"USER": "Link the features: {0} to the corresponding item's title.", "ASSISTANT": "{0}"},
                    {"USER": "Using the features: {0} , generate the item title them represent.", "ASSISTANT": "{0}"},
            ],
    "description2iid": [{"USER": "What is the item represented by the description: {0} ?", "ASSISTANT": "{0}"},
                        {"USER": "Given the description: {0} , generate the item title.", "ASSISTANT": "{0}"},
                        {"USER": "Generate the title of the item represented by the description: {0} .", "ASSISTANT": "{0}"},
                        {"USER": "What item corresponds to the description: {0} ?", "ASSISTANT": "{0}"},
                        {"USER": "Generate the item's name based on the description: {0} .", "ASSISTANT": "{0}"},
                        {"USER": "Generate the title for the item described by the description: {0} .", "ASSISTANT": "{0}"},
                        {"USER": "What item is characterized by the following description: {0} ?", "ASSISTANT": "{0}"},
                        {"USER": "Produce the name of the item associated with the description: {0} .", "ASSISTANT": "{0}"},
                        {"USER": "Link the description: {0} to the corresponding item's title.", "ASSISTANT": "{0}"},
                        {"USER": "Using the description: {0} , generate the item title it represents.", "ASSISTANT": "{0}"},
            ],

    "uid2summary": [{"USER": "Based on a user's purchase history, create a concise user summary under 100 words, highlighting patterns and traits from past items so that it is helpful to identify preferences and predict future item choices. Do not mention the future predictions in the summary. Here is the purchase history: {0} .", "ASSISTANT": "{0}"},
                    {"USER": "Given the user purchase history: {0} , please summarize it under 100 words, highlighting patterns and traits from past items so that it is helpful to identify preferences and predict future item choices. Do not mention the future predictions in the summary.", "ASSISTANT": "{0}"},
                    {"USER": "From the provided purchase history: {0} , create a brief summary under 100 words that emphasizes patterns and traits from previous purchases, which can be used to identify preferences and predict future item choices. Do not include future predictions in the summary.", "ASSISTANT": "{0}"},
                    {"USER": "Summarize the user's purchase history: {0} , in no more than 100 words by focusing on patterns and traits from past items. This will help to determine preferences and anticipate future item choices. Do not mention future predictions in the summary.", "ASSISTANT": "{0}"},
                    {"USER": "Using the purchase history provided: {0} , craft a concise summary, limited to 100 words, that outlines patterns and traits from past items. This will be useful in identifying preferences and predicting future item choices. Do not mention future predictions in the summary.", "ASSISTANT": "{0}"},
                    {"USER": "With the given user purchase history: {0} , construct a short summary under 100 words that highlights patterns and traits from past items. This will aid in understanding preferences and foreseeing future item choices. Do not mention future predictions in the summary.", "ASSISTANT": "{0}"},
                    {"USER": "Based on the user's purchase history: {0} , write a succinct summary of no more than 100 words that emphasizes patterns and traits from past items. This information will be helpful in recognizing preferences and predicting future item choices. Do not mention future predictions in the summary.", "ASSISTANT": "{0}"},
                    {"USER": "Considering the user's purchase history: {0} , develop a concise summary, not exceeding 100 words, that focuses on patterns and traits from past items. This will be beneficial in identifying preferences and projecting future item choices. Do not mention future predictions in the summary.", "ASSISTANT": "{0}"},
                    {"USER": "Analyze the provided purchase history and create a brief summary within 100 words that highlights patterns and traits from past items. This will assist in determining preferences and anticipating future item choices. Do not mention future predictions in the summary. Here is the purchase history: {0} .", "ASSISTANT": "{0}"},
                    {"USER": "Please summarize the user's purchase history in a concise manner, limited to 100 words, by emphasizing patterns and traits from past items. This will be useful for identifying preferences and predicting future item choices. Do not mention future predictions in the summary. Here is the purchase history: {0} .", "ASSISTANT": "{0}"},
                    {"USER": "Using the provided user purchase history: {0} , create a short summary of 100 words or less that outlines patterns and traits from past items. This information will help recognize preferences and predict future item choices. Do not mention future predictions in the summary.", "ASSISTANT": "{0}"},
                    {"USER": "Examine the user's purchase history and write a brief summary, not exceeding 100 words, that focuses on patterns and traits from past items. This will aid in discovering preferences and estimating future item choices. Do not mention future predictions in the summary. Here is the purchase history: {0} .", "ASSISTANT": "{0}"},
            ],

    "uid2hist": [{"USER": "What are the history titles of the user: {0} ?", "ASSISTANT": "{0}"},
                 {"USER": "Given the user purchase history: {0} , generate his history titles.", "ASSISTANT": "{0}"},
                 {"USER": "Generate the titles of the user history: {0} .", "ASSISTANT": "{0}"},
                 {"USER": "What are the titles associated with user purchase history: {0} ?", "ASSISTANT": "{0}"},
                 {"USER": "Can you list the titles in the purchase history of the user: {0} ?", "ASSISTANT": "{0}"},
                 {"USER": "Show me the history titles for the user: {0} .", "ASSISTANT": "{0}"},
                 {"USER": "Please generate the titles from the user's purchase history: {0} .", "ASSISTANT": "{0}"},
                 {"USER": "List the titles in the purchase history for user: {0} .", "ASSISTANT": "{0}"},
                 {"USER": "What titles can be found in user's purchase history: {0} ?", "ASSISTANT": "{0}"},
                 {"USER": "Generate the titles connected to the purchase history of user: {0} .", "ASSISTANT": "{0}"},
                 {"USER": "Generate the titles related to the user's purchase history: {0} .", "ASSISTANT": "{0}"},
            ],
    "uid2next": [{"USER": "Given the user purchase history: {0} , generate the next most likely clicked item title.", "ASSISTANT": "{0}"},
                 {"USER": "What is the next most likely clicked item title for the purchase history: {0} ?", "ASSISTANT": "{0}"},
                 {"USER": "Predict the item that the user with this history: {0} might like next.", "ASSISTANT": "{0}"},
                 {"USER": "Considering the purchasing history: {0} , what will be the next item they click on?", "ASSISTANT": "{0}"},
                 {"USER": "Based on the buying history {0} , what item is the user likely to click on next?", "ASSISTANT": "{0}"},
                 {"USER": "With the given purchase records {0} , can you determine the next item the user will click?", "ASSISTANT": "{0}"},
                 {"USER": "What item is expected to be clicked next by a user who has this purchase history: {0} ?", "ASSISTANT": "{0}"},
                 {"USER": "Generate the next probable clicked item for a user with the purchase history: {0} .", "ASSISTANT": "{0}"},
                 {"USER": "For a user with the following purchase background: {0} , which item will he most likely click next?", "ASSISTANT": "{0}"},
                 {"USER": "Can you predict the next item to be clicked by a user who has purchased these items: {0} ?", "ASSISTANT": "{0}"},
                 {"USER": "Given this purchasing data: {0} , what's the next item the user is inclined to click?", "ASSISTANT": "{0}"},
                 {"USER": "From the user's purchase history: {0} , determine the next item he is likely to click on.", "ASSISTANT": "{0}"},
                 {"USER": "Analyze the purchase history: {0} and suggest the next item the user might click.", "ASSISTANT": "{0}"},
            ],
    
    "uidiid2rank": [{"USER": "Given the user history: {0} and next items to be ranked: {1} , generate the sorted item titles from the user's favorite to least favorite.", "ASSISTANT": "{0}"},
                    {"USER": "Considering user: {0} and some items he might like next: {1} , provide a ranking list of them according to the user preference.", "ASSISTANT": "{0}"},
                    {"USER": "Please rank the following items: {1} from what the user likes to dislikes. Here is the user history: {0} .", "ASSISTANT": "{0}"},
                    {"USER": "For user with purchase history: {0} , please arrange these items in order of preference: {1} .", "ASSISTANT": "{0}"},
                    {"USER": "Taking into account user's history: {0} , create a list of the items: {1} ranked by the user's interests.", "ASSISTANT": "{0}"},
                    {"USER": "With the user's purchase history given: {0} , sort the items: {1} based on the user's taste from best to worst.", "ASSISTANT": "{0}"},
                    {"USER": "Based on the purchase history: {0} , please provide a ranking of the following items: {1} according to the user's preferences.", "ASSISTANT": "{0}"},
                    {"USER": "Given user's past history: {0} , rank these items: {1} from most to least appealing.", "ASSISTANT": "{0}"},
                    {"USER": "Using the provided user purchase history: {0} , generate a ranked list of items: {1} in accordance with the user's likes and dislikes.", "ASSISTANT": "{0}"},
                    {"USER": "Based on the user's past interactions: {0} , rank the items: {1} from the most to the least preferred.", "ASSISTANT": "{0}"},
                    {"USER": "In reference to user's history: {0} , sort these items: {1} based on their preference from favorite to least favorite.", "ASSISTANT": "{0}"},
                    {"USER": "Taking user's history: {0} into consideration, arrange the items: {1} in order of the user's liking.", "ASSISTANT": "{0}"},
                    {"USER": "Given the user's past choices: {0} , create a preference-based ranking of the items: {1} from the most liked to the least liked.", "ASSISTANT": "{0}"},
            ],
    # "uidiid2pairwise": [{"USER": "Given the user history:{0} and two items:{1}, which one will the user prefer?", "ASSISTANT": "{0}"},
    #                 {"USER": "Considering user:{0} and two items:{1}, which one will the user prefer?", "ASSISTANT": "{0}"},
    #                 {"USER": "Please select one item from the two:{1} that the user will prefer. Here is the user history:{0}", "ASSISTANT": "{0}"},
    # ],
    "uidiid2binary": [{"USER": "The user has the following purchase history: {0} . Will he like the item: {1} ?", "ASSISTANT": "{0}"},
                    {"USER": "Considering user: {0} and item: {1} , will the user like the item?", "ASSISTANT": "{0}"},
                    {"USER": "Here is the user history: {0} . Do you think he will prefer the item: {1} ?", "ASSISTANT": "{0}"},
                    {"USER": "User's purchase records are: {0} . Can you tell if he will enjoy item: {1} ?", "ASSISTANT": "{0}"},
                    {"USER": "Given the purchase background of the user: {0} , would he appreciate the item: {1} ?", "ASSISTANT": "{0}"},
                    {"USER": "The buyer has this purchase history: {0} . Would he be interested in the product: {1} ?", "ASSISTANT": "{0}"},
                    {"USER": "With the following purchasing history for the user: {0} , can we predict if he'll like item: {1} ?", "ASSISTANT": "{0}"},
                    {"USER": "Here's the customer's buying log: {0} . Would you say he might favor the item: {1} ?", "ASSISTANT": "{0}"},
                    {"USER": "This is the shopper's past transactions: {0} . Can we determine if he'll be fond of the item: {1} ?", "ASSISTANT": "{0}"},
                    {"USER": "Looking at the user's previous purchases: {0} , is it likely he'll enjoy the product: {1} ?", "ASSISTANT": "{0}"},
                    {"USER": "Based on the user's purchase history: {0} , can you assess if he will be attracted to item: {1} ?", "ASSISTANT": "{0}"},
                    {"USER": "Reviewing the customer's purchase patterns: {0} , is it possible they'll appreciate the item: {1} ?", "ASSISTANT": "{0}"},
                    {"USER": "Taking into account the user's shopping background: {0} , do you believe they'll be keen on the product: {1} ?", "ASSISTANT": "{0}"},
            ],
}

def parse_args():
    parser = argparse.ArgumentParser(description="data process")
    parser.add_argument(
        "--sharegpt_file", type=str, help=""
    )
    parser.add_argument(
        "--seqdata_file", type=str, help=""
    )
    parser.add_argument(
        "--metadata_file", type=str, help=""
    )
    parser.add_argument(
        "--sim_item_file", type=str, default='', help=""
    )
    parser.add_argument(
        "--train_top_file", type=str, help=""
    )
    parser.add_argument(
        "--test_top_file", type=str, help=""
    )
    parser.add_argument(
        "--gpt_response_file", type=str, help=""
    )
    parser.add_argument(
        "--gpt_query_file", type=str, help="", default=None
    )
    parser.add_argument(
        "--save_intention_file", type=str, help=""
    )
    parser.add_argument(
        "--save_behavior_file", type=str, help=""
    )
    parser.add_argument(
        "--save_both_file", type=str, help=""
    )
    parser.add_argument(
        "--max_seq_len", type=int, help=""
    )
    parser.add_argument(
        "--model_max_length", type=int, help=""
    )
    parser.add_argument(
        "--model_name", type=str, help=""
    )
    
    args = parser.parse_args()
    return args

def read_seq_and_meta(args):
    user_items = {}
    with open(args.seqdata_file, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip().split(' ')
            user = int(line[0])
            items = [int(x) for x in line[1:]]
            user_items[user] = items

    meta_infos = {}
    with open(args.metadata_file, 'r') as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            meta_infos[i+1] = line
    return user_items, meta_infos

def gen_iid2text(meta_infos, args, tokenizer):
    train_iid2text_intention = []
    train_iid2text_behavior = []
    train_iid2text_both = []
    for iid, info in meta_infos.items():
        train_iid2text_intention.append({'iid': iid, 'question': '<item>', 'answer': info['title'], 'template': random.choice(prompt_templates['iid2title']), 'type': 'iid2title'})
        train_iid2text_both.append(train_iid2text_intention[-1])
        if 'feature' in info and len(info['feature'])>0 and len('; '.join(info['feature']))>0:
            iid2feature_template = random.choice(prompt_templates['iid2feature'])
            feature2iid_template = random.choice(prompt_templates['feature2iid'])
            train_iid2text_intention.append({'iid': iid, 'question': '<item>', 'answer': '; '.join(info['feature']), 'template': iid2feature_template, 'type': 'iid2feature'})
            train_iid2text_behavior.append({'iid': iid, 'question': info['title'], 'answer': '; '.join(info['feature']), 'template': iid2feature_template, 'type': 'iid2feature'})

            tokens = tokenizer.tokenize('; '.join(info['feature']))
            if len(tokens)>args.model_max_length-200:
                print(f"feature truncated: {len(tokens)}")
            tokens = tokens[:args.model_max_length-200]
            truncated_prompt = tokenizer.convert_tokens_to_string(tokens)
            train_iid2text_behavior.append({'iid': iid, 'question': truncated_prompt, 'answer':info['title'], 'template': feature2iid_template, 'type': 'feature2iid'})

            train_iid2text_both.append({'iid': iid, 'question': '<item>'+info['title'], 'answer': '; '.join(info['feature']), 'template': iid2feature_template, 'type': 'iid2feature'})
        
        if 'description' in info and len(info['description'])>0 and len('; '.join(info['description']))>0:
            desc = info['description'][0]
            for d in info['description']:
                if len(d)>50:
                    desc = d
                    break
            assert len(desc)>0
            iid2description_template = random.choice(prompt_templates['iid2description'])
            description2iid_template = random.choice(prompt_templates['description2iid'])
            train_iid2text_intention.append({'iid': iid, 'question': '<item>', 'answer': desc, 'template': iid2description_template, 'type': 'iid2description'})
            train_iid2text_behavior.append({'iid': iid, 'question': info['title'], 'answer': desc, 'template': iid2description_template, 'type': 'iid2description'})

            tokens = tokenizer.tokenize(desc)
            if len(tokens)>args.model_max_length-200:
                print(f"desc truncated: {len(tokens)}")
            tokens = tokens[:args.model_max_length-200]
            truncated_prompt = tokenizer.convert_tokens_to_string(tokens)
            train_iid2text_behavior.append({'iid': iid, 'question': truncated_prompt, 'answer':info['title'], 'template': description2iid_template, 'type': 'description2iid'})

            train_iid2text_both.append({'iid': iid, 'question': '<item>'+info['title'], 'answer': desc, 'template': iid2description_template, 'type': 'iid2description'})
        if 'brand' in info and len(info['brand'])>0:
            iid2brand_template = random.choice(prompt_templates['iid2brand'])
            train_iid2text_intention.append({'iid': iid, 'question': '<item>', 'answer': re.sub(r'\n    \n    ', ' ', info['brand']), 'template': iid2brand_template, 'type': 'iid2brand'})
            train_iid2text_behavior.append({'iid': iid, 'question': info['title'], 'answer': re.sub(r'\n    \n    ', ' ', info['brand']), 'template': iid2brand_template, 'type': 'iid2brand'})
            train_iid2text_both.append({'iid': iid, 'question': '<item>'+info['title'], 'answer': re.sub(r'\n    \n    ', ' ', info['brand']), 'template': iid2brand_template, 'type': 'iid2brand'})

    ## for similar items
    with open(args.sim_item_file, 'r') as f:
        for line in f:
            iid, sim_item = line.strip().split('\t')
            iid2sim_template = random.choice(prompt_templates['iid2sim'])
            train_iid2text_intention.append({'iid': int(iid), 'question': '<item>', 'answer': meta_infos[int(sim_item)]['title'], 'template':iid2sim_template, 'type': 'iid2sim'})
            train_iid2text_behavior.append({'iid': int(iid), 'question': meta_infos[int(iid)]['title'], 'answer': meta_infos[int(sim_item)]['title'], 'template': iid2sim_template, 'type': 'iid2sim'})
            train_iid2text_both.append({'iid': int(iid), 'question': '<item>'+meta_infos[int(iid)]['title'], 'answer': meta_infos[int(sim_item)]['title'], 'template': iid2sim_template, 'type': 'iid2sim'})

    print(f"train_iid2text_intention: {len(train_iid2text_intention)}, train_iid2text_behavior: {len(train_iid2text_behavior)}, train_iid2text_both: {len(train_iid2text_both)}")
    train_intention.extend(train_iid2text_intention)
    train_behavior.extend(train_iid2text_behavior)
    train_both.extend(train_iid2text_both)

def gen_uid2text(user_items, meta_infos, args):
    ## uid2hist
    train_uid2hist_intention = []
    train_uid2hist_both = []
    valid_uid2hist_intention = []
    valid_uid2hist_both = []
    for uid, items in user_items.items():
        uid2hist_template = random.choice(prompt_templates['uid2hist'])
        valid_uid2hist_intention.append({'uid': uid, 'iid': -1, 'question': '<user>', 'answer': ', '.join([meta_infos[x]['title'] for x in items[-args.max_seq_len:]]), 'template': uid2hist_template, 'type': 'uid2hist'})
        valid_uid2hist_both.append(valid_uid2hist_intention[-1])
        for i, item in enumerate(items[1:]):
            train_uid2hist_intention.append({'uid': uid, 'iid': item, 'question': '<user>', 'answer': ', '.join([meta_infos[x]['title'] for x in items[:i+1][-args.max_seq_len:]]), 'template': random.choice(prompt_templates['uid2hist']), 'type': 'uid2hist'})
            train_uid2hist_both.append(train_uid2hist_intention[-1])
    print(f"train_uid2hist_intention: {len(train_uid2hist_intention)}, valid_uid2hist_intention: {len(valid_uid2hist_intention)}, train_uid2hist_both: {len(train_uid2hist_both)}, valid_uid2hist_both: {len(valid_uid2hist_both)}")
    train_intention.extend(train_uid2hist_intention)
    train_both.extend(train_uid2hist_both)
    valid_intention.extend(valid_uid2hist_intention)
    valid_both.extend(valid_uid2hist_both)

    ## uid2summary
    train_uid2summary_intention = []
    train_uid2summary_behavior = []
    train_uid2summary_both = []
    query_profile1 = pd.read_csv(args.gpt_response_file+"_1.csv", header=None, sep=',', names=['q', 'x', 's', 'target'])
    query_profile1 = query_profile1.iloc[1:]
    query_profile2 = pd.read_csv(args.gpt_response_file+"_2.csv", header=None, sep=',', names=['q', 'x', 's', 'target'])
    query_profile2 = query_profile2.iloc[1:]
    query_profile = pd.concat([query_profile1, query_profile2], axis=0).reset_index(drop=True)
    for uid, items in user_items.items():
        answer = query_profile.loc[uid-1, 'target']
        uid2summary_template = random.choice(prompt_templates['uid2summary'])
        history = ', '.join([meta_infos[x]['title'] for x in items[-args.max_seq_len-1:-1]])
        train_uid2summary_intention.append({'uid': uid, 'iid': items[-1], 'question': '<user>', 'answer': answer, 'template': uid2summary_template, 'type': 'uid2summary'})
        train_uid2summary_behavior.append({'uid': uid, 'iid': items[-1], 'question': history, 'answer': answer, 'template': uid2summary_template, 'type': 'uid2summary'})
        train_uid2summary_both.append({'uid': uid, 'iid': items[-1], 'question': '<user>'+history, 'answer': answer, 'template': uid2summary_template, 'type': 'uid2summary'})
    print(f"train_uid2summary_intention: {len(train_uid2summary_intention)}, train_uid2summary_behavior: {len(train_uid2summary_behavior)}, train_uid2summary_both: {len(train_uid2summary_both)}")
    train_intention.extend(train_uid2summary_intention)
    train_behavior.extend(train_uid2summary_behavior)
    train_both.extend(train_uid2summary_both)

def gen_uid2next_rank(user_items, args):
    for split, file in zip(['train', 'valid'], [args.train_top_file, args.test_top_file]):
        uid2next_intention = []
        uid2next_behavior = []
        uid2next_both = []
        uidiid2rank_intention = []
        uidiid2rank_behavior = []
        uidiid2rank_both = []
        uidiid2binary_intention = []
        uidiid2binary_behavior = []
        uidiid2binary_both = []
        with open(file, 'r') as f:
            for line in f:
                uid, iid, top1, topk, pos, neg, pos_score, neg_score = line.strip().split('\t') 
                uid, iid, top1, topk, pos, neg = int(uid), int(iid), int(top1), [int(x) for x in topk.split(',')], int(pos), int(neg)
                history = user_items[uid]
                history = history[:history.index(iid)][-args.max_seq_len:]
                history = ', '.join([meta_infos[x]['title'] for x in history])
                
                uid2next_template = random.choice(prompt_templates['uid2next'])
                uid2next_intention.append({'uid': uid, 'iid': iid, 'question': '<user>', 'answer': meta_infos[top1]['title'], 'template': uid2next_template, 'type': 'uid2next'})
                uid2next_behavior.append({'uid': uid, 'iid': iid, 'question': history, 'answer': meta_infos[top1]['title'], 'template': uid2next_template, 'type': 'uid2next'})
                uid2next_both.append({'uid': uid, 'iid': iid, 'question': '<user>'+history, 'answer': meta_infos[top1]['title'], 'template': uid2next_template, 'type': 'uid2next'})
                
                uidiid2template = random.choice(prompt_templates['uidiid2rank'])
                ranks_intention = {'uid': uid, 'iid': iid, 'template': uidiid2template, 'type': 'uidiid2rank'}
                ranks_intention['question'] = ('<user>', '<item>'*len(topk))
                ranks_intention['answer'] = ', '.join([meta_infos[x]['title'] for x in topk])
                random.shuffle(topk)
                ranks_intention['topk'] = topk
                ranks_behavior = {'uid': uid, 'iid': iid, 'template': uidiid2template, 'type': 'uidiid2rank', 'question': (history, ', '.join([meta_infos[x]['title'] for x in topk])), 'answer': ranks_intention['answer'], 'topk': topk}
                ranks_both = {'uid': uid, 'iid': iid, 'template': uidiid2template, 'type': 'uidiid2rank', 'question': ('<user>'+history, ', '.join(['<item>'+meta_infos[x]['title'] for x in topk])), 'answer': ranks_intention['answer'], 'topk': topk}
                uidiid2rank_intention.append(ranks_intention)
                uidiid2rank_behavior.append(ranks_behavior)
                uidiid2rank_both.append(ranks_both)


                if (split=='train' and iid==user_items[uid][-2]) or (split=='valid' and random.random()<0.3):
                    pos_template = random.choice(prompt_templates['uidiid2binary'])
                    neg_template = random.choice(prompt_templates['uidiid2binary'])
                    uidiid2binary_intention.append({'uid': uid, 'iid': iid, 'target': pos, 'question': ('<user>', '<item>'), 'answer': 'Yes', 'template': pos_template, 'type': 'uidiid2binary'})
                    uidiid2binary_intention.append({'uid': uid, 'iid': iid, 'target': neg, 'question': ('<user>', '<item>'), 'answer': 'No', 'template': neg_template, 'type': 'uidiid2binary'})
                    uidiid2binary_behavior.append({'uid': uid, 'iid': iid, 'target': pos, 'question': (history, meta_infos[pos]['title']), 'answer': 'Yes', 'template': pos_template, 'type': 'uidiid2binary'})
                    uidiid2binary_behavior.append({'uid': uid, 'iid': iid, 'target': neg, 'question': (history, meta_infos[neg]['title']), 'answer': 'No', 'template': neg_template, 'type': 'uidiid2binary'})
                    uidiid2binary_both.append({'uid': uid, 'iid': iid, 'target': pos, 'question': ('<user>'+history, '<item>'+meta_infos[pos]['title']), 'answer': 'Yes', 'template': pos_template, 'type': 'uidiid2binary'})
                    uidiid2binary_both.append({'uid': uid, 'iid': iid, 'target': neg, 'question': ('<user>'+history, '<item>'+meta_infos[neg]['title']), 'answer': 'No', 'template': neg_template, 'type': 'uidiid2binary'})

        print(f"{split}: uid2next_intention: {len(uid2next_intention)}, uidiid2rank_intention: {len(uidiid2rank_intention)}, uidiid2binary_intention: {len(uidiid2binary_intention)}")
        print(f"{split}: uid2next_behavior: {len(uid2next_behavior)}, uidiid2rank_behavior: {len(uidiid2rank_behavior)}, uidiid2binary_behavior: {len(uidiid2binary_behavior)}")
        print(f"{split}: uid2next_both: {len(uid2next_both)}, uidiid2rank_both: {len(uidiid2rank_both)}, uidiid2binary_both: {len(uidiid2binary_both)}")
        if split=='train':
            train_intention.extend(uid2next_intention)
            train_intention.extend(uidiid2rank_intention)
            train_intention.extend(uidiid2binary_intention)
            train_behavior.extend(uid2next_behavior)
            train_behavior.extend(uidiid2rank_behavior)
            train_behavior.extend(uidiid2binary_behavior)
            train_both.extend(uid2next_both)
            train_both.extend(uidiid2rank_both)
            train_both.extend(uidiid2binary_both)
        elif split=='valid':
            valid_intention.extend(uid2next_intention)
            valid_intention.extend(uidiid2rank_intention)
            valid_intention.extend(uidiid2binary_intention)
            valid_behavior.extend(uid2next_behavior)
            valid_behavior.extend(uidiid2rank_behavior)
            valid_behavior.extend(uidiid2binary_behavior)
            valid_both.extend(uid2next_both)
            valid_both.extend(uidiid2rank_both)
            valid_both.extend(uidiid2binary_both)

def gen_sharegpt(args, tokenizer):
    ### save sharegpt
    with open(args.sharegpt_file, 'r') as f:
        data = json.load(f)
    filter_data = []
    conv_roles = ("USER", "ASSISTANT")
    roles = {"human": conv_roles[0], "gpt": conv_roles[1]}
    for line in data:
        source = line['conversations']
        if len(source) == 0 or any([messa["from"] not in roles for messa in source]):
            continue
        if roles[source[0]["from"]] != conv_roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        if len(source) < 2:
            continue
        
        flag = 0
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            if role != conv_roles[j % 2] or len(sentence["value"]) == 0:
                flag = 1
                break
        if flag == 1:
            continue

        if len(tokenizer.tokenize(source[0]["value"])) > args.model_max_length-300:
            continue

        filter_data.append({"conversations": source, "type": "sharegpt"})
    random.shuffle(filter_data)
    print(f"sharegpt: {len(filter_data)}")

    train_sharegpt = filter_data[:10000]
    valid_sharegpt = filter_data[10000:11000]
    print(f"sharegpt: train: {len(train_sharegpt)}")
    print(f"sharegpt: valid: {len(valid_sharegpt)}")
    train_intention.extend(train_sharegpt)
    train_behavior.extend(train_sharegpt)
    train_both.extend(train_sharegpt)
    valid_intention.extend(valid_sharegpt)
    valid_behavior.extend(valid_sharegpt)
    valid_both.extend(valid_sharegpt)

def gen_uid2summary(user_items, meta_infos, args):
    # uid2summary
    u2i_template = "Based on a user's purchase history, create a concise user summary under 100 words, highlighting patterns and traits from past items so that it is helpful to identify preferences and predict future item choices. Randomly select a writing style from third-person and non-narrative. Do not mention the chosen style in the summary. Do not mention the future predictions in the summary. Here is his purchase history: {}"

    ## generate an empty pandas dataframe, with columns: 'question', 'response', "target"
    df = pd.DataFrame(columns=['question', 'response', 'target'])
    ## insert a row to the dataframe with values: 'question': 'What is the meaning of life?', 'response': '42', 'target': '42'
    
    for user, items in user_items.items():
        l_items = items[:-1]
        seq_text = [meta_infos[x]['title'] for x in l_items[-args.max_seq_len:]]
        query = ", ".join(seq_text)
        df.loc[len(df)] = [u2i_template.format(query), '0', '0']

    ## save the dataframe to a csv file
    
    split_index = len(df) // 2   
    first_half = df.iloc[:split_index, :]  
    second_half = df.iloc[split_index:, :] 

    first_half.to_csv(args.gpt_query_file+"_1.csv", index=False)
    second_half.to_csv(args.gpt_query_file+"_2.csv", index=False)

if __name__ == '__main__':
    args = parse_args()
    user_items, meta_infos = read_seq_and_meta(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=args.model_max_length, use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token

    if args.gpt_query_file is not None:
        gen_uid2summary(user_items, meta_infos, args)
    else:
        train_intention = []
        train_behavior = []
        train_both = []
        valid_intention = []
        valid_behavior = []
        valid_both = []
        
        gen_iid2text(meta_infos, args, tokenizer)
        gen_uid2text(user_items, meta_infos, args)
        gen_uid2next_rank(user_items, args)
        gen_sharegpt(args, tokenizer)

        print(f"train_intention: {len(train_intention)}, train_behavior: {len(train_behavior)}, train_both: {len(train_both)}")
        print(f"valid_intention: {len(valid_intention)}, valid_behavior: {len(valid_behavior)}, valid_both: {len(valid_both)}")

        json.dump(train_intention, open(args.save_intention_file+'_train.json', 'w'))
        json.dump(train_behavior, open(args.save_behavior_file+'_train.json', 'w'))
        json.dump(train_both, open(args.save_both_file+'_train.json', 'w'))
        json.dump(valid_intention, open(args.save_intention_file+'_valid.json', 'w'))
        json.dump(valid_behavior, open(args.save_behavior_file+'_valid.json', 'w'))
        json.dump(valid_both, open(args.save_both_file+'_valid.json', 'w'))