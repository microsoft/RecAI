# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import random
import copy
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="data process")
    parser.add_argument(
        "--data_dir", type=str, help=""
    )
    parser.add_argument(
        "--seqdata_file", type=str, help=""
    )
    parser.add_argument(
        "--metadata_file", type=str, help=""
    )
    parser.add_argument(
        "--test_top_file", type=str, help=""
    )
    parser.add_argument(
        "--max_seq_len", type=int, help=""
    )
    parser.add_argument(
        "--max_samples", type=int, help=""
    )
    parser.add_argument(
        "--split", type=str, help="", choices=['train', 'valid'], default='valid'
    )
    parser.add_argument(
        "--rec_model_type", type=str, help="", choices=['SASRec', 'MF'], default='SASRec'
    )
    args = parser.parse_args()
    return args


intention_template = {"USER": "The user has the following purchase history: {0} . Will the user like the item: {1} ? Please give your answer and explain why you make this decision from the perspective of a recommendation model. Your explanation should include the following aspects: summary of patterns and traits from user purchase history, the consistency or inconsistency between user preferences and the item.", "ASSISTANT": "{0}"}
behavior_template = {"USER": "The user has the following purchase history: {0} . Will the user like the item: {1} ? Please give your answer and explain why you make this decision from the perspective of a recommendation model. Your explanation should include the following aspects: summary of patterns and traits from user purchase history, the consistency or inconsistency between user preferences and the item.", "ASSISTANT": "{0}"}
both_template = {"USER": "The user has the following purchase history: {0} . Will the user like the item: {1} ? Please give your answer and explain why you make this decision from the perspective of a recommendation model. Your explanation should include the following aspects: summary of patterns and traits from user purchase history, the consistency or inconsistency between user preferences and the item.", "ASSISTANT": "{0}"}
# both_template = {"USER": "The user has the following purchase history: {0} . Will the user like the item: {1} ? Please give your answer and explain why you make this decision. Your explanation should focus on the gaming consoles or platforms between the item and the user history.", "ASSISTANT": "{0}"}
# both_template2 = {"USER": "The user has the following purchase history: {0} . Will the user like the item: {1} ? Please give your answer and explain why you make this decision. Your explanation should focus on the game types, themes and gameplay between the item and the user history.", "ASSISTANT": "{0}"}

args = parse_args()

os.makedirs(args.data_dir, exist_ok=True)

user_items = {}
with open(args.seqdata_file, 'r') as f:
    for idx, line in enumerate(f):
        line = line.strip().split(' ')
        user = int(line[0])
        items = [int(x) for x in line[1:]]
        user_items[user] = items

metainfo = {}
with open(args.metadata_file, 'r') as f:
    for i, line in enumerate(f):
        line = json.loads(line)
        metainfo[i+1] = line

intention_infer_data = []
behavior_infer_data = []
both_infer_data = []
gpt_df = pd.DataFrame(columns=['question'])
with open(args.test_top_file, 'r') as f:
    for i, line in enumerate(f):
        uid, iid, top1, topk, pos, neg, _, _ = line.strip().split('\t') 
        uid, iid, top1, topk, pos, neg = int(uid), int(iid), int(top1), [int(x) for x in topk.split(',')], int(pos), int(neg)
        hist = user_items[uid]
        if (args.rec_model_type=='SASRec' and (args.split=='valid' or (args.split=='train' and iid==hist[-2]))) or (args.rec_model_type=='MF' and ((args.split=='valid' and i%6==3) or (args.split=='train' and i%6<3))):
            hist = hist[:hist.index(iid)] if iid!=-1 else hist
            hist = hist[-args.max_seq_len:]
            hist_titles = [metainfo[i]['title'] for i in hist]
            hist_text = '; '.join(hist_titles)
            pos_title = metainfo[pos]['title']
            neg_title = metainfo[neg]['title']

            intention_pos_item = {'uid': uid, 'iid': iid, 'target': pos, 'question': ('<user>', '<item>'), 'answer': ['Yes', hist_titles, pos_title], 'template': intention_template, 'type': 'uidiid2explan'}
            intention_neg_item = {'uid': uid, 'iid': iid, 'target': neg, 'question': ('<user>', '<item>'), 'answer': ['No', hist_titles, neg_title], 'template': intention_template, 'type': 'uidiid2explan'}
            
            behavior_pos_item = {'uid': uid, 'iid': iid, 'target': pos, 'question': (hist_text, pos_title), 'answer': ['Yes', hist_titles, pos_title], 'template': behavior_template, 'type': 'uidiid2explan'}
            behavior_neg_item = {'uid': uid, 'iid': iid, 'target': neg, 'question': (hist_text, neg_title), 'answer': ['No', hist_titles, neg_title], 'template': behavior_template, 'type': 'uidiid2explan'}

            both_pos_item = {'uid': uid, 'iid': iid, 'target': pos, 'question': ('<user>'+hist_text, '<item>'+pos_title), 'answer': ['Yes', hist_titles, pos_title], 'template': both_template, 'type': 'uidiid2explan'}
            both_neg_item = {'uid': uid, 'iid': iid, 'target': neg, 'question': ('<user>'+hist_text, '<item>'+neg_title), 'answer': ['No', hist_titles, neg_title], 'template': both_template, 'type': 'uidiid2explan'}

            intention_infer_data.append(intention_pos_item)
            intention_infer_data.append(intention_neg_item)
            behavior_infer_data.append(behavior_pos_item)
            behavior_infer_data.append(behavior_neg_item)
            both_infer_data.append(both_pos_item)
            both_infer_data.append(both_neg_item)
            
            ## insert a row to the dataframe with values: 'question': 'What is the meaning of life?', 'response': '42', 'target': '42'
            pos_query = behavior_template['USER'].format(hist_text, pos_title)
            neg_query = behavior_template['USER'].format(hist_text, neg_title)
            gpt_df.loc[len(gpt_df)] = [pos_query]
            gpt_df.loc[len(gpt_df)] = [neg_query]

        if len(intention_infer_data)>=args.max_samples:
            break

with open(os.path.join(args.data_dir, f'explan_intention_{args.split}.json'), 'w') as f:
    json.dump(intention_infer_data, f)
with open(os.path.join(args.data_dir, f'explan_behaviour_{args.split}.json'), 'w') as f:
    json.dump(behavior_infer_data, f)
with open(os.path.join(args.data_dir, f'explan_both_{args.split}.json'), 'w') as f:
    json.dump(both_infer_data, f)
gpt_df.to_csv(os.path.join(args.data_dir, f'explan_chatgpt_{args.split}.csv'), index=False)
