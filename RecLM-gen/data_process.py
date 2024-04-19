# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
from concurrent.futures.thread import ThreadPoolExecutor
import numpy as np
import torch
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.tools import load_pickle, save_pickle
from param import get_args, Config


if __name__ == '__main__':
    args = get_args()
    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    transformers.set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained("snap/Llama-2-7b-hf-chat/")

    category2item = load_pickle(args.data_path + 'category.pickle')
    metas = load_pickle(args.data_path + 'meta.pickle')
    item2category = {}
    for c in category2item:
        for i in category2item[c]:
            if item2category.get(i) is None:
                item2category[i] = []
            item2category[i].append(c)
    title2item = {}
    for _ in metas:
        if title2item.get(metas[_][args.item_index]) is None:
            title2item[metas[_][args.item_index]] = []
        title2item[metas[_][args.item_index]].append(_)
    data = {
        'category2item': category2item,
        'item2category': item2category,
        'metas': metas,
        'title2item': title2item,
        'sequential': load_pickle(args.data_path + 'sequential.pickle'),
        'ranking_candidate': load_pickle(args.data_path + 'ranking_candidate.pickle'),
        'share_chat_gpt': load_pickle('data/dataset/share_chat_gpt2.pickle'),
    }
    if args.train_stage == 'SFT' and args.share_chat_gpt_ratio > 0.:
        args.SFT_train_tasks = args.SFT_train_tasks + ',ShareChatGPT'

    if args.train_stage in ['SFT']:
        from sft.dataset import SFTDataset, Train_task_group_mapping, Val_task_group_mapping

        TaskTemplate = {_: Train_task_group_mapping[_] for _ in args.SFT_train_tasks.split(',')}
        TaskNum = {_: 1 for _ in args.SFT_train_tasks.split(',')}
        ValTaskTemplate = {_: Val_task_group_mapping[_.split('_')[0]] for _ in args.SFT_val_tasks.split(',')}
        ValTaskNum = {_: 1 for _ in args.SFT_val_tasks.split(',')}
        train_data = SFTDataset(args, TaskTemplate, TaskNum, data, tokenizer, 'train', immediately=False)
        train_datasets = []
        for epoch in range(args.epoch):
            with ThreadPoolExecutor(max_workers=512) as executor:
                results = list(tqdm(executor.map(train_data.getitem, range(len(train_data))), total=len(train_data)))
            train_datasets.append(results)
        if args.epoch > 0:
            save_pickle(train_datasets, args.data_path+'SFT_dataset_train.pickle')

        val_data = SFTDataset(args, ValTaskTemplate, ValTaskNum, data, tokenizer, 'val', immediately=False)
        with ThreadPoolExecutor(max_workers=512) as executor:
            results = list(tqdm(executor.map(val_data.getitem, range(len(val_data))), total=len(val_data)))
        val_datasets = [results]
        save_pickle(val_datasets, args.data_path+'SFT_dataset_val.pickle')

    elif args.train_stage in ['RL']:
        from rl.dataset import RLDataset, Train_task_group_mapping, Val_task_group_mapping

        TaskTemplate = {_: Train_task_group_mapping[_] for _ in args.RL_train_tasks.split(',')}
        TaskNum = {_: 1 for _ in args.RL_train_tasks.split(',')}
        ValTaskTemplate = {_: Val_task_group_mapping[_.split('_')[0]] for _ in args.RL_val_tasks.split(',')}
        ValTaskNum = {_: 1 for _ in args.RL_val_tasks.split(',')}
        train_data = RLDataset(args, TaskTemplate, TaskNum, data, tokenizer, 'train', immediately=False)
        train_datasets = []
        for episode in range(args.num_episodes):
            with ThreadPoolExecutor(max_workers=512) as executor:
                results = list(tqdm(executor.map(train_data.getitem, range(len(train_data))), total=len(train_data)))
            train_datasets.append(results)
        if args.num_episodes > 0:
            save_pickle(train_datasets, args.data_path+'RL_dataset_train.pickle')

        val_data = RLDataset(args, ValTaskTemplate, ValTaskNum, data, tokenizer, 'val', immediately=False)
        with ThreadPoolExecutor(max_workers=512) as executor:
            results = list(tqdm(executor.map(val_data.getitem, range(len(val_data))), total=len(val_data)))
        val_datasets = [results]
        save_pickle(val_datasets, args.data_path+'RL_dataset_val.pickle')

    else:
        raise NotImplementedError
