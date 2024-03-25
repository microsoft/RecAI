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
gpt_df = pd.DataFrame(columns=['question', 'response', 'target'])
with open(args.test_top_file, 'r') as f:
    for i, line in enumerate(f):
        uid, iid, top1, topk, pos, neg, _, _ = line.strip().split('\t') 
        uid, iid, top1, topk, pos, neg = int(uid), int(iid), int(top1), [int(x) for x in topk.split(',')], int(pos), int(neg)
        hist = user_items[uid]
        if args.split=='valid' or (args.split=='train' and iid==hist[-2]):
            hist = hist[:hist.index(iid)] if iid!=-1 else hist
            hist = hist[-args.max_seq_len:]
            hist_titles = [metainfo[i]['title'] for i in hist]
            hist_text = ', '.join(hist_titles)
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
            gpt_df.loc[len(gpt_df)] = [pos_query, '0', '0']
            gpt_df.loc[len(gpt_df)] = [neg_query, '0', '0']

        if len(intention_infer_data)>=args.max_samples:
            break

with open(os.path.join(args.data_dir, 'explan_intention_valid.json'), 'w') as f:
    json.dump(intention_infer_data, f)
with open(os.path.join(args.data_dir, 'explan_behaviour_valid.json'), 'w') as f:
    json.dump(behavior_infer_data, f)
with open(os.path.join(args.data_dir, 'explan_both_valid.json'), 'w') as f:
    json.dump(both_infer_data, f)
gpt_df.to_csv(os.path.join(args.data_dir, 'explan_chatgpt.csv'), index=False)



# intention_template = {"USER": "You are a recommendation explainer. Given the user purchase history and an item, your task is to answer whether you will recommend the item to the user and explain why you make this decision. Your explanation should include the following aspects: summary of patterns and traits from user purchase history, the consistency or inconsistency between user preferences and the item. Here are some examples:\nexample 1:\n user purchase history: <user> , target item: <item> , answer: Yes. We identified your preference for action-adventure games on the PlayStation 4, with immersive storylines and challenging gameplay. Based on this pattern, Senran Kagura Estival Versus is recommended due to: 1. Genre compatibility: Senran Kagura Estival Versus is an action game, similar to your past purchases. 2. Platform: The game is available on PlayStation 4, ensuring compatibility with your system. 3. Gameplay: While not open-world, it offers engaging combat mechanics, which may appeal to fans of Metal Gear Solid V and Dark Souls III. 4. Storyline: Senran Kagura Estival Versus features a narrative-driven campaign, appealing to those who enjoyed rich storylines in your previous purchases.\nexample2:\nuser purchase history: <user> , target item: <item> , answer: No. We analyzed your purchase history, which consists of action and first-person shooter games on the Xbox One platform. Based on this pattern, the model decided not to recommend Ultimate Marvel Vs. Capcom 3 - Xbox 360 due to: 1. Genre mismatch: Your purchase history indicates a preference for action and first-person shooter games, while Ultimate Marvel Vs. Capcom 3 is a fighting game, a different genre that may not align with your interests. 2. Platform discrepancy: Your past purchases are exclusively for the Xbox One console, whereas Ultimate Marvel Vs. Capcom 3 is for the Xbox 360, an older platform. Given the differences between your established preferences and the recommended item, the model decided against suggesting Ultimate Marvel Vs. Capcom 3 - Xbox 360, as it does not align with your past gaming interests and platform choice.\n\nNow here is the user purchase history: {0}, the target item: {1}, Please give your answer. Do not simply answer Yes or No.", "ASSISTANT": "{0}"}
# behavior_template = {"USER": "You are a recommendation explainer. Given the user purchase history and an item, your task is to answer whether you will recommend the item to the user and explain why you make this decision. Your explanation should include the following aspects: summary of patterns and traits from user purchase history, the consistency or inconsistency between user preferences and the item. Here are some examples:\nexample 1:\n user purchase history: Metal Gear Solid V: The Phantom Pain - PlayStation 4, Dark Souls III - PlayStation 4 Standard Edition, The Elder Scrolls V: Skyrim Special Edition - PS4 [Digital Code], Horizon Zero Dawn - PlayStation 4 , target item: Senran Kagura Estival Versus - PlayStation 4 , answer: Yes. We identified your preference for action-adventure games on the PlayStation 4, with immersive storylines and challenging gameplay. Based on this pattern, Senran Kagura Estival Versus is recommended due to: 1. Genre compatibility: Senran Kagura Estival Versus is an action game, similar to your past purchases. 2. Platform: The game is available on PlayStation 4, ensuring compatibility with your system. 3. Gameplay: While not open-world, it offers engaging combat mechanics, which may appeal to fans of Metal Gear Solid V and Dark Souls III. 4. Storyline: Senran Kagura Estival Versus features a narrative-driven campaign, appealing to those who enjoyed rich storylines in your previous purchases.\nexample2:\nuser purchase history: Titanfall - Xbox One, Wolfenstein: The New Order, Call of Duty Advanced Warfare - Day Zero Edition, Halo: The Master Chief Collection, Tom Clancy's The Division - Xbox One , target item: Ultimate Marvel Vs. Capcom 3 - Xbox 360 , answer: No. We analyzed your purchase history, which consists of action and first-person shooter games on the Xbox One platform. Based on this pattern, the model decided not to recommend Ultimate Marvel Vs. Capcom 3 - Xbox 360 due to: 1. Genre mismatch: Your purchase history indicates a preference for action and first-person shooter games, while Ultimate Marvel Vs. Capcom 3 is a fighting game, a different genre that may not align with your interests. 2. Platform discrepancy: Your past purchases are exclusively for the Xbox One console, whereas Ultimate Marvel Vs. Capcom 3 is for the Xbox 360, an older platform. Given the differences between your established preferences and the recommended item, the model decided against suggesting Ultimate Marvel Vs. Capcom 3 - Xbox 360, as it does not align with your past gaming interests and platform choice.\n\nNow here is the user purchase history: {0}, the target item: {1}, Please give your answer. Do not simply answer Yes or No.", "ASSISTANT": "{0}"}
# both_template = {"USER": "You are a recommendation explainer. Given the user purchase history and an item, your task is to answer whether you will recommend the item to the user and explain why you make this decision. Your explanation should include the following aspects: summary of patterns and traits from user purchase history, the consistency or inconsistency between user preferences and the item. Here are some examples:\nexample 1:\n user purchase history: <user>Metal Gear Solid V: The Phantom Pain - PlayStation 4, Dark Souls III - PlayStation 4 Standard Edition, The Elder Scrolls V: Skyrim Special Edition - PS4 [Digital Code], Horizon Zero Dawn - PlayStation 4 , target item: <item>Senran Kagura Estival Versus - PlayStation 4 , answer: Yes. We identified your preference for action-adventure games on the PlayStation 4, with immersive storylines and challenging gameplay. Based on this pattern, Senran Kagura Estival Versus is recommended due to: 1. Genre compatibility: Senran Kagura Estival Versus is an action game, similar to your past purchases. 2. Platform: The game is available on PlayStation 4, ensuring compatibility with your system. 3. Gameplay: While not open-world, it offers engaging combat mechanics, which may appeal to fans of Metal Gear Solid V and Dark Souls III. 4. Storyline: Senran Kagura Estival Versus features a narrative-driven campaign, appealing to those who enjoyed rich storylines in your previous purchases.\nexample2:\nuser purchase history: <user>Titanfall - Xbox One, Wolfenstein: The New Order, Call of Duty Advanced Warfare - Day Zero Edition, Halo: The Master Chief Collection, Tom Clancy's The Division - Xbox One , target item: <item>Ultimate Marvel Vs. Capcom 3 - Xbox 360 , answer: No. We analyzed your purchase history, which consists of action and first-person shooter games on the Xbox One platform. Based on this pattern, the model decided not to recommend Ultimate Marvel Vs. Capcom 3 - Xbox 360 due to: 1. Genre mismatch: Your purchase history indicates a preference for action and first-person shooter games, while Ultimate Marvel Vs. Capcom 3 is a fighting game, a different genre that may not align with your interests. 2. Platform discrepancy: Your past purchases are exclusively for the Xbox One console, whereas Ultimate Marvel Vs. Capcom 3 is for the Xbox 360, an older platform. Given the differences between your established preferences and the recommended item, the model decided against suggesting Ultimate Marvel Vs. Capcom 3 - Xbox 360, as it does not align with your past gaming interests and platform choice.\n\nNow here is the user purchase history: {0}, the target item: {1}, Please give your answer. Do not simply answer Yes or No.", "ASSISTANT": "{0}"}
# template_uidiidtargets = [(21, 94, 1147), (20, 167, 199)]#(uid, iid, target)
# max_seq_len = 9

# user_items = {}
# with open(seqdata_file, 'r') as f:
#     for idx, line in enumerate(f):
#         line = line.strip().split(' ')
#         user = int(line[0])
#         items = [int(x) for x in line[1:]]
#         user_items[user] = items

# metainfo = {}
# with open(metadata_file, 'r') as f:
#     for i, line in enumerate(f):
#         line = json.loads(line)
#         metainfo[i+1] = line

# intention_infer_data = []
# behavior_infer_data = []
# both_infer_data = []
# with open(test_top_file, 'r') as f:
#     for i, line in enumerate(f):
#         uid, iid, top1, topk, pos, neg = line.strip().split('\t') 
#         uid, iid, top1, topk, pos, neg = int(uid), int(iid), int(top1), [int(x) for x in topk.split(',')], int(pos), int(neg)
#         hist = user_items[uid]
#         hist = hist[:hist.index(iid)] if iid!=-1 else hist
#         hist = hist[-max_seq_len:]
#         hist_titles = [metainfo[i]['title'] for i in hist]
#         hist_text = ', '.join(hist_titles)
#         pos_title = metainfo[pos]['title']
#         neg_title = metainfo[neg]['title']

#         intention_pos_item = {'uid': uid, 'iid': iid, 'target': pos, 'question': ('<user>', '<item>'), 'answer': ['Yes', hist_titles, pos_title], 'template': intention_template, 'type': 'uidiid2explan', 'template_uidiidtargets': template_uidiidtargets}
#         intention_neg_item = {'uid': uid, 'iid': iid, 'target': neg, 'question': ('<user>', '<item>'), 'answer': ['No', hist_titles, neg_title], 'template': intention_template, 'type': 'uidiid2explan', 'template_uidiidtargets': template_uidiidtargets}
        
#         behavior_pos_item = {'uid': uid, 'iid': iid, 'target': pos, 'question': (hist_text, pos_title), 'answer': ['Yes', hist_titles, pos_title], 'template': behavior_template, 'type': 'uidiid2explan', 'template_uidiidtargets': template_uidiidtargets}
#         behavior_neg_item = {'uid': uid, 'iid': iid, 'target': neg, 'question': (hist_text, neg_title), 'answer': ['No', hist_titles, neg_title], 'template': behavior_template, 'type': 'uidiid2explan', 'template_uidiidtargets': template_uidiidtargets}

#         both_pos_item = {'uid': uid, 'iid': iid, 'target': pos, 'question': ('<user>'+hist_text, '<item>'+pos_title), 'answer': ['Yes', hist_titles, pos_title], 'template': both_template, 'type': 'uidiid2explan', 'template_uidiidtargets': template_uidiidtargets}
#         both_neg_item = {'uid': uid, 'iid': iid, 'target': neg, 'question': ('<user>'+hist_text, '<item>'+neg_title), 'answer': ['No', hist_titles, neg_title], 'template': both_template, 'type': 'uidiid2explan', 'template_uidiidtargets': template_uidiidtargets}

#         intention_infer_data.append(intention_pos_item)
#         intention_infer_data.append(intention_neg_item)
#         behavior_infer_data.append(behavior_pos_item)
#         behavior_infer_data.append(behavior_neg_item)
#         both_infer_data.append(both_pos_item)
#         both_infer_data.append(both_neg_item)
#         if i==18:
#             break

# with open(os.path.join(data_dir, 'explan_intention_demo_2_valid.json'), 'w') as f:
#     json.dump(intention_infer_data, f)
# with open(os.path.join(data_dir, 'explan_behaviour_demo_2_valid.json'), 'w') as f:
#     json.dump(behavior_infer_data, f)
# with open(os.path.join(data_dir, 'explan_both_demo_2_valid.json'), 'w') as f:
#     json.dump(both_infer_data, f)