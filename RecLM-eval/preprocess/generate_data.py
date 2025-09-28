# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import re
import json
# ---- patch: import universal task templates ----
from all_steam_templates import all_tasks as all_task_templates
import gzip
import math
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from multiprocessing import Pool
from collections import defaultdict

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

class Test_Dataset():
    def __init__(self, all_task_templates, dataset='steam', split='test'):
        self.all_task_templates = all_task_templates
        self.dataset = dataset  # dataset to use
        self.split = split  # train/valid/test

        self.sequential_data = ReadLineFromFile(os.path.join('./data', dataset, 'sequential_data.txt'))
        self.negative_samples = ReadLineFromFile(os.path.join('./data', dataset, 'negative_samples.txt'))
        self.test_items = None
        if os.path.exists(os.path.join('./data', dataset, 'item_datasets.pkl')):
            self.test_items = load_pickle(os.path.join('./data', dataset, 'item_datasets.pkl'))['test']
        self.meta_data = ['padding']  # item_id start from 0
        self.raw_id2meta_id = {}
        for meta in parse(os.path.join('./data', dataset, 'metadata.json')):
            self.meta_data.append(meta)
            meta["title"] = re.sub('[^A-Za-z0-9_.,!?;:<>\n ]', '', meta["title"])
            self.raw_id2meta_id[meta['id']] = len(self.meta_data) - 1
        self.search_data = None
        if os.path.exists(os.path.join('./data', dataset, 'search_data.csv')):
            self.search_data = pd.read_csv(os.path.join('./data', dataset, 'search_data.csv'))

    def _token_to_id(self, token):
        """Robustly map **any** external token to an *internal* item index.

        Strategy:
        1. Always check ``raw_id2meta_id`` first (regardless of whether the
           token is numeric).
        2. If not found **and** the token is numeric, treat it as an already
           internal index *only if* ``0 <= idx < len(meta_data)``.
        3. Otherwise return 0 (padding token).
        """
        token_str = str(token)
        mapped = self.raw_id2meta_id.get(token_str)
        if mapped is not None:
            return mapped
        if token_str.isdigit():
            idx = int(token_str)
            if 0 <= idx < len(self.meta_data):
                return idx
        return 0

    def gen_retrieval_data(self, sample_num):
        datasets = []
        sample_num = min(sample_num, len(self.sequential_data))
        for idx in range(sample_num):
            retrieval_datum = self.sequential_data[idx]
            sequence = [self._token_to_id(x) for x in retrieval_datum.split()]
            click_history = sequence[1:-1]
            target_item = sequence[-1]
            history_titles = []
            for item_id in click_history:
                item_datum = self.meta_data[item_id]
                item_title = 'unknown title'
                if 'title' in item_datum:
                    item_title = item_datum['title']
                history_titles.append(item_title)
            
            target_item_datum = self.meta_data[target_item]
            target_item_title = 'unknown title'
            if 'title' in target_item_datum:
                target_item_title = target_item_datum['title']

            task_template = self.all_task_templates['retrieval']
            source_text = task_template['source'].format(', '.join(history_titles))
            target_text = task_template['target'].format(target_item_title)
            source_text = re.sub('[^A-Za-z0-9_.,!?;:<>\n ]', '', source_text)
            target_text = re.sub('[^A-Za-z0-9_.,!?;:<>\n ]', '', target_text)
            datasets.append({
                "source": source_text,
                "target": target_text,
                "history": history_titles,
                "task": "retrieval"
            })
        return datasets

    def gen_ranking_data(self, sample_num):
        datasets = []
        sample_num = min(sample_num, len(self.sequential_data))
        for idx in range(sample_num):
            ranking_datum = self.sequential_data[idx]
            sequence = [self._token_to_id(x) for x in ranking_datum.split()]
            click_history = sequence[1:-1]
            target_item = sequence[-1]
            history_titles = []
            for item_id in click_history:
                item_datum = self.meta_data[item_id]
                item_title = 'unknown title'
                if 'title' in item_datum:
                    item_title = item_datum['title']
                history_titles.append(item_title)
            
            target_item_datum = self.meta_data[target_item]
            target_item_title = 'unknown title'
            if 'title' in target_item_datum:
                target_item_title = target_item_datum['title']

            user_token = sequence[0]
            candidate_samples = self.negative_samples[idx].split()
            # Remove the first token if it duplicates the user or target item token
            if candidate_samples and candidate_samples[0] == str(user_token):
                candidate_samples = candidate_samples[1:]
            candidate_samples = random.sample(candidate_samples, min(20, len(candidate_samples)))
            candidate_samples.append(target_item)
            random.shuffle(candidate_samples)

            candidate_titles = []
            for item_id in candidate_samples:
                item_datum = self.meta_data[self._token_to_id(item_id)]
                item_title = 'unknown title'
                if 'title' in item_datum:
                    item_title = item_datum['title']
                candidate_titles.append(item_title)

            task_template = self.all_task_templates['ranking']
            # Build wrapped candidate list for prompt (index + delimiters)
            wrapped_candidates = '\n'.join(f'<SOI>{t}<EOI>' for t in candidate_titles)
            source_text = task_template['source'].format(', '.join(history_titles), wrapped_candidates)
            target_text = task_template['target'].format(target_item_title)
            source_text = re.sub('[^A-Za-z0-9_.,!?;:<>\n ]', '', source_text)
            target_text = re.sub('[^A-Za-z0-9_.,!?;:<>\n ]', '', target_text)
            datasets.append({
                "source": source_text,
                "target": target_text,
                "history": history_titles,
                "candidate": candidate_titles,
                "task": "ranking"
            })
        return datasets

    def gen_explanation_data(self, sample_num):
        datasets = []
        sample_num = min(sample_num, len(self.sequential_data))
        for idx in range(sample_num):
            exp_datum = self.sequential_data[idx]
            sequence = [self._token_to_id(x) for x in exp_datum.split()]
            click_history = sequence[1:-1]
            target_item = sequence[-1]
            history_titles = []
            for item_id in click_history:
                item_datum = self.meta_data[item_id]
                item_title = 'unknown title'
                if 'title' in item_datum:
                    item_title = item_datum['title']
                history_titles.append(item_title)
            
            target_item_datum = self.meta_data[target_item]
            target_item_title = 'unknown title'
            if 'title' in target_item_datum:
                target_item_title = target_item_datum['title']

            task_template = self.all_task_templates['explanation']
            source_text = task_template['source'].format(', '.join(history_titles), target_item_title)
            target_text = task_template['target'].format("No ground truth.")
            source_text = re.sub('[^A-Za-z0-9_.,!?;:<>\n ]', '', source_text)
            target_text = re.sub('[^A-Za-z0-9_.,!?;:<>\n ]', '', target_text)
            history_titles = re.sub('[^A-Za-z0-9_.,!?;:<>\n ]', '', ', '.join(history_titles))
            target_item_title = re.sub('[^A-Za-z0-9_.,!?;:<>\n ]', '', target_item_title)
            datasets.append({
                "source": source_text,
                "history": history_titles,
                "target": target_item_title,
                "task": "explanation"
            })
        return datasets
    def gen_conversation_data(self, sample_num):
        datasets = []
        sample_num = min(sample_num, len(self.sequential_data))
        for idx in range(sample_num):
            exp_datum = self.sequential_data[idx]
            sequence = [self._token_to_id(x) for x in exp_datum.split()]
            click_history = sequence[1:-1]
            target_item = sequence[-1]
            history_titles = []
            for item_id in click_history:
                item_datum = self.meta_data[item_id]
                item_title = 'unknown title'
                if 'title' in item_datum:
                    item_title = item_datum['title']
                history_titles.append(item_title)
            
            target_item_datum = self.meta_data[target_item]
            target_item_title = 'unknown title'
            if 'title' in target_item_datum:
                target_item_title = target_item_datum['title']

            history_titles = re.sub('[^A-Za-z0-9_.,!?;:<>\n ]', '', ', '.join(history_titles))
            target_item_title = re.sub('[^A-Za-z0-9_.,!?;:<>\n ]', '', target_item_title)
            datasets.append({
                "history": history_titles,
                "target": target_item_title,
                "task": "conversation"
            })
        return datasets

    def gen_search_data(self, sample_num):
        datasets = []
        sample_num = min(sample_num, len(self.search_data))
        for idx in range(sample_num):
            search_datum = self.search_data.iloc[idx]
            query = search_datum['query']
            target_item = search_datum['target']
            candidates = search_datum['candidates'].split('|')
            
            task_template = self.all_task_templates['search']
            source_text = task_template['source'].format(query, ', '.join(candidates))
            target_text = task_template['target'].format(target_item)
            source_text = re.sub('[^A-Za-z0-9_.,!?;:<>\n ]', '', source_text)
            target_text = re.sub('[^A-Za-z0-9_.,!?;:<>\n ]', '', target_text)
            datasets.append({
                "source": source_text,
                "target": target_text,
                "query": query,
                "candidates": candidates,
                "task": "search"
            })
        return datasets

    def gen_cf_ranking_data(self, sample_num):
        """Generate data for the *collaborative-filtering multiple-choice* task."""
        datasets = []
        sample_num = min(sample_num, len(self.sequential_data))
        for idx in range(sample_num):
            cf_datum = self.sequential_data[idx]
            sequence = [self._token_to_id(x) for x in cf_datum.split()]
            click_history = sequence[1:-1]
            target_item = sequence[-1]
            
            # Shuffle user history order so that it is no longer chronological
            random.shuffle(click_history)
            
            # Randomly select one item from history as the target to predict
            target_item = random.choice(click_history)
            click_history = [x for x in click_history if x != target_item]
            
            history_titles = []
            for item_id in click_history:
                item_datum = self.meta_data[item_id]
                item_title = 'unknown title'
                if 'title' in item_datum:
                    item_title = item_datum['title']
                history_titles.append(item_title)
            
            target_item_datum = self.meta_data[target_item]
            target_item_title = 'unknown title'
            if 'title' in target_item_datum:
                target_item_title = target_item_datum['title']

            # Build candidates: 1 positive + 9 negatives
            candidates = [target_item_title]
            negative_items = self.negative_samples[idx].split()
            for neg_item_id in negative_items[:9]:
                neg_item_datum = self.meta_data[self._token_to_id(neg_item_id)]
                neg_item_title = 'unknown title'
                if 'title' in neg_item_datum:
                    neg_item_title = neg_item_datum['title']
                candidates.append(neg_item_title)

            random.shuffle(candidates)

            # Add letter indices A–J in front of each candidate title
            letters = list("ABCDEFGHIJ")
            lettered_candidates = [f"{letters[i]}. {candidates[i]}" for i in range(len(candidates))]
            correct_letter = letters[candidates.index(target_item_title)]

            task_template = self.all_task_templates['cf_ranking_mc']
            source_text = task_template['source'].format(', '.join(history_titles), ', '.join(lettered_candidates))
            target_text = task_template['target'].format(correct_letter)
            source_text = re.sub('[^A-Za-z0-9_.,!?;:<>\n ]', '', source_text)
            target_text = re.sub('[^A-Za-z0-9_.,!?;:<>\n ]', '', target_text)
            datasets.append({
                "source": source_text,
                "target": target_text,
                "history": history_titles,
                "candidates": lettered_candidates,
                "correct_answer": correct_letter,
                "task": "cf_ranking_mc"
            })
        return datasets

    def gen_sequential_ranking_data(self, sample_num):
        """Generate data for the *sequential recommendation multiple-choice* task."""
        datasets = []
        sample_num = min(sample_num, len(self.sequential_data))
        for idx in range(sample_num):
            seq_datum = self.sequential_data[idx]
            sequence = [self._token_to_id(x) for x in seq_datum.split()]
            click_history = sequence[1:-1]
            target_item = sequence[-1]  # The next (ground-truth) item
            
            history_titles = []
            for item_id in click_history:
                item_datum = self.meta_data[item_id]
                item_title = 'unknown title'
                if 'title' in item_datum:
                    item_title = item_datum['title']
                history_titles.append(item_title)
            
            target_item_datum = self.meta_data[target_item]
            target_item_title = 'unknown title'
            if 'title' in target_item_datum:
                target_item_title = target_item_datum['title']

            # Build candidate set (1 positive + 9 negatives)
            candidates = [target_item_title]
            negative_items = self.negative_samples[idx].split()
            for neg_item_id in negative_items[:9]:  # Add 9 negative samples
                neg_item_datum = self.meta_data[self._token_to_id(neg_item_id)]
                neg_item_title = 'unknown title'
                if 'title' in neg_item_datum:
                    neg_item_title = neg_item_datum['title']
                candidates.append(neg_item_title)

            # Shuffle the candidate order
            random.shuffle(candidates)

            # Add index letters A-J to candidate titles
            letters = list("ABCDEFGHIJ")
            lettered_candidates = [f"{letters[i]}. {candidates[i]}" for i in range(len(candidates))]

            # Find the letter index corresponding to the correct answer
            correct_letter = letters[candidates.index(target_item_title)]

            task_template = self.all_task_templates['seq_ranking_mc']
            source_text = task_template['source'].format(', '.join(history_titles), ', '.join(lettered_candidates))
            target_text = task_template['target'].format(correct_letter)
            source_text = re.sub('[^A-Za-z0-9_.,!?;:<>\n ]', '', source_text)
            target_text = re.sub('[^A-Za-z0-9_.,!?;:<>\n ]', '', target_text)
            datasets.append({
                "source": source_text,
                "target": target_text,
                "history": history_titles,
                "candidates": lettered_candidates,
                "correct_answer": correct_letter,
                "task": "seq_ranking_mc"
            })
        return datasets

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=str, default='ranking,retrieval,explanation,conversation,cf_ranking_mc,seq_ranking_mc', help='tasks for data generation.')
    parser.add_argument('--sample_num', type=int, default=1000, help='sample number for each task.')  
    parser.add_argument('--dataset', type=str, default='steam', help='the dataset to be evaluated, steam/beauty/sports')
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train/val/test)") 

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if args.dataset == 'steam':    
        from all_steam_templates import all_tasks as all_task_templates
    
    dataset = Test_Dataset(all_task_templates, dataset=args.dataset, split=args.split)

    if "retrieval" in args.tasks:
        print(f'generating retrieval data, sample number: {args.sample_num} ...')
        data = dataset.gen_retrieval_data(args.sample_num)
        fd = open(f"data/{args.dataset}/retrieval.jsonl", "w")
        for line in data:
            line = {
                "prompt": line["source"],
                "target": line["target"],
                "history": line["history"],
                "task": line["task"],
            }
            fd.write(json.dumps(line)+'\n')
    if "ranking" in args.tasks:
        print(f'generating ranking data, sample number: {args.sample_num} ...')
        data = dataset.gen_ranking_data(args.sample_num)
        fd = open(f"data/{args.dataset}/ranking.jsonl", "w")
        for line in data:
            line = {
                "prompt": line["source"],
                "target": line["target"],
                "history": line["history"],
                "candidate": line["candidate"],
                "task": line["task"],
            }
            fd.write(json.dumps(line)+'\n')
    if "explanation" in args.tasks:
        print(f'generating explanation data, sample number: {args.sample_num} ...')
        data = dataset.gen_explanation_data(args.sample_num)
        fd = open(f"data/{args.dataset}/explanation.jsonl", "w")
        for line in data:
            line = {
                "prompt": line["source"],
                "history": line["history"],
                "target": line["target"],
                "task": line["task"],
            }
            fd.write(json.dumps(line)+'\n')
    if "conversation" in args.tasks:
        print(f'generating conversation data, sample number: {args.sample_num} ...')
        data = dataset.gen_conversation_data(args.sample_num)
        fd = open(f"data/{args.dataset}/conversation.jsonl", "w")
        for line in data:
            line = {
                "history": line["history"],
                "target": line["target"],
                "task": line["task"],
                "user_simulator_system_prompt": "You are a user chatting with a recommender for recommendation in turn. Your history is {history}. Your target items: {target}.\nYou must follow the instructions below during chat.\nIf the recommender recommends {target}, you should accept.\nIf the recommender recommends other items, you should refuse them and provide the information about {target}. You should never directly tell the target item title.\nIf the recommender asks for your preference, you should provide the information about {target}. You should never directly tell the target item title.\nNow lets start, you first, act as a user.\n Your output is only allowed to be the words from the user you act.".format(
                    history=line["history"],
                    target=line["target"],
                )
            }
            fd.write(json.dumps(line)+'\n')
    if "chatbot" in args.tasks:  # use previous 3 tasks' data
        print(f'generating chatbot data, sample number: {args.sample_num} ...')

        data1 = dataset.gen_explanation_data(args.sample_num // 3)
        data2 = dataset.gen_ranking_data(args.sample_num // 3)
        data3 = dataset.gen_retrieval_data(args.sample_num // 3)

        combined_data = data1 + data2 + data3
        if len(combined_data) > args.sample_num:
            combined_data = combined_data[:args.sample_num]  
        elif len(combined_data) < args.sample_num:
            additional_data = dataset.gen_explanation_data(args.sample_num - len(combined_data))
            combined_data += additional_data

        fd = open(f"data/{args.dataset}/chatbot.jsonl", "w")
        for line in combined_data:
            line_to_write = {
                "prompt": line["source"],
                "task": "chatbot",
            }
            fd.write(json.dumps(line_to_write) + '\n')
        fd.close()

        print(f"Successfully generated {len(combined_data)} samples for chatbot.")
    
    if "cf_ranking_mc" in args.tasks:
        print(f'generating cf_ranking_mc data, sample number: {args.sample_num} ...')
        data = dataset.gen_cf_ranking_data(args.sample_num)
        fd = open(f"data/{args.dataset}/cf_ranking_mc.jsonl", "w")
        for line in data:
            line = {
                "prompt": line["source"],
                "target": line["target"],
                "history": line["history"],
                "candidates": line["candidates"],
                "correct_answer": line["correct_answer"],
                "task": line["task"],
            }
            fd.write(json.dumps(line)+'\n')
        fd.close()
    
    if "seq_ranking_mc" in args.tasks:
        print(f'generating seq_ranking_mc data, sample number: {args.sample_num} ...')
        data = dataset.gen_sequential_ranking_data(args.sample_num)
        fd = open(f"data/{args.dataset}/seq_ranking_mc.jsonl", "w")
        for line in data:
            line = {
                "prompt": line["source"],
                "target": line["target"],
                "history": line["history"],
                "candidates": line["candidates"],
                "correct_answer": line["correct_answer"],
                "task": line["task"],
            }
            fd.write(json.dumps(line)+'\n')
        fd.close()