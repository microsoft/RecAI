# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import re
import json
import time
import pickle
import argparse

from typing import *


parser = argparse.ArgumentParser()
parser.add_argument("--demo_dir_or_file", type=str, default="./work/gen_demos")
parser.add_argument("--save", type=str, default="./tagged/")
args, _ = parser.parse_known_args()


def read_jsonl(fpath: str) -> List[Dict]:
    res = []
    with open(fpath, 'r') as f:
        for line in f:
            data = json.loads(line)
            res.append(data)
    return res


def write_jsonl(obj: List[Dict], fpath: str) -> None:
    try:
        with open(fpath, 'w') as outfile:
            for entry in obj:
                json.dump(entry, outfile)
                outfile.write('\n')
        print("Sucessfully saved into {}.".format(fpath))
    except Exception as e:
        print(f"Error {e} raised. The temp file would be saved in {fpath}.pkl")
        with open(f"{fpath}.pkl", 'wb') as tempfile:
            pickle.dump(obj, tempfile)
    return


def load_examples(dir: str) -> List[Dict]:
    examples = []
    if os.path.isdir(dir):
        for f in os.listdir(dir):
            if f.endswith("jsonl"):
                fname = os.path.join(dir, f)
                examples.extend(read_jsonl(fname))
    else:
        assert dir.endswith('.jsonl'), "Only jsonl file is supported for demonstration loading"
        examples.extend(read_jsonl(dir))
    return examples

import re  
  
def extract_tags(file_path):  
    tags = []  
    with open(file_path, 'r') as file:  
        content = file.read()  
        pattern = r'Tag whether the plan reasonable \(Y/N\):\s*([ynYN])'  
        tags = re.findall(pattern, content)  
    return tags  
  
# Example usage  
file_path = './work/LLM4CRS/tagged/tag_cache.txt'  
tags = extract_tags(file_path)  
# print(tags)  # Output：['y', 'n', 'Y', ...]  


examples = load_examples(args.demo_dir_or_file)

print(f"Tagging for {args.demo_dir_or_file}...")
result = []
for i, example in enumerate(examples):
    print(f"{i+1} / {len(examples)} ---")
    print("Request: ", example['request'])
    print("Plan: ", example['plan'])
    # tag = input("Tag whether the plan reasonable (Y/N): ")
    tag = tags[i]
    if 'y' in tag.lower():
        example['tag'] = 1
    else:
        example['tag'] = 0
    result.append(example)

print("Completed.")

if not os.path.exists(args.save):
    os.makedirs(args.save)

now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
basename = os.path.basename(args.demo_dir_or_file)
fname = os.path.join(args.save, f"tagged_{now}_{basename}.jsonl")
write_jsonl(result, fname)

print(f"Tagged file saved in {fname}")