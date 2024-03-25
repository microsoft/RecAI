# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from template import *
import argparse

task_map = {
    'query2item': query2item_template,
    # 'title2item': title2item_template,
    'item2item': item2item_template,
    'vaguequery2item': vaguequery2item_template,
    'negquery2item': negquery2item_template,
    'sparse_query2item': query2item_template,
    'misspell2item': title2item_template,
    # 'user2item': user2item_template,
    # 'queryuser2item': queryuser2item_template,
    # 'gpt4_data': '',
    # 'relativequery2item': relativequery2item_template,
}

def get_templates(task_name):
    templates = task_map[task_name]
    split_templates = []
    for temp in templates:
        left, right = temp.split('{}')
        if left=='' and right=='':
            continue
        split_templates.append([left, right])
    return split_templates

def read_train_data(task_name, split_templates, train_dir, query_name='query'):
    train_data = set()
    with open(f'{train_dir}/{task_name}.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            query = data[query_name]
            for temp in split_templates:
                if query.startswith(temp[0]) and query.endswith(temp[1]):
                    left_index = len(temp[0]) if temp[0] else 0
                    right_index = -len(temp[1]) if temp[1] else None
                    query = query[left_index: right_index]
                    break
            train_data.add(query.lower())
    return train_data

def check_test_data(task_name, split_templates, train_data, test_dir, query_name='text'):
    test_data = []
    with open(f'{test_dir}/{task_name}.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            query = data[query_name]
            for temp in split_templates:
                if query.startswith(temp[0]) and query.endswith(temp[1]):
                    left_index = len(temp[0]) if temp[0] else 0
                    right_index = -len(temp[1]) if temp[1] else None
                    query = query[left_index: right_index]
                    break
            if query.lower() not in train_data:
                test_data.append(line)
    print(f'{task_name}: {len(test_data)}')
    if len(test_data)>0:
        with open(f'{test_dir}/{task_name}.jsonl', 'w') as f:
            for line in test_data:
                f.write(line)

def deduplicate(task_name, args):
    split_templates = get_templates(task_name)

    name = task_name if task_name!='sparse_query2item' else 'query2item'
    train_data = read_train_data(name, split_templates, args.train_dir)
    
    check_test_data(task_name, split_templates, train_data, args.test_dir)

def gpt_deduplicate(args, task_name='gpt4_data'):
    split_templates = get_templates('misspell2item')
    train_data = read_train_data('misspell2item', split_templates, args.train_dir)
    gpt_train_data = read_train_data(task_name, split_templates, args.train_dir)

    train_data = train_data.union(gpt_train_data)
    check_test_data('misspell2item', split_templates, train_data, args.test_dir)

    check_test_data('gpt_misspell', split_templates, train_data, args.test_dir)

    check_test_data('gpt_query', [], train_data, args.test_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="test filter")
    parser.add_argument(
        "--train_dir", type=str, help=""
    )
    parser.add_argument(
        "--test_dir", type=str, help=""
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    for task_name in task_map.keys():
        if task_name=='misspell2item':
            continue
        deduplicate(task_name, args)
    gpt_deduplicate(args)