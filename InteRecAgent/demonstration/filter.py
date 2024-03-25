# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# The script is used to filter out unreasonable demonstrations 
# Here are some rules:
# 1. tool using order: 'candidate store tool' > 'look up tool' > 'hard candidate filter tool' > 'soft condition filter tool' > 'ranking tool' > 'map tool'
# 2. repetitive examples are not needed

import os
import re
import json
import time
import pickle
import argparse
from typing import *
from functools import partial
from multiprocessing import Pool
from rouge_score import rouge_scorer


def read_jsonl(fpath: str) -> List[Dict]:
    res = []
    with open(fpath, 'r') as f:
        for line in f:
            data = json.loads(line)
            res.append(data)
    return res


def get_all_files(folder):
    all_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return all_files


def load_demos(dir: str) -> List[Dict]:
    """Load all demonstration from the directory"""
    examples = []
    for f in get_all_files(dir):
        if f.endswith("jsonl"):
            # fname = os.path.join(dir, f)
            examples.extend(read_jsonl(f))
    return examples


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


def tool_using_order_rule(demo: Dict) -> bool:
    # rule1: tool using order; rule2: map tool is required
    pattern = r"(candidate store tool)?.*(look up tool)?.*(hard candidate filter tool)?.*(soft condition filter tool)?.*(ranking tool)?(map tool)?.*"
    if re.match(pattern, demo['plan']):
        return True
    else:
        return False


class RepetitiveRequestRule:
    def __init__(self, seed_demos: List[Dict], thres: float=0.8):
        # Use RougeL score, refer to Self-Instruct
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        self.thres = thres
        self.selected_examples = [demo['request'] for demo in seed_demos]


    def __call__(self, demo: Dict) -> bool:
        if len(self.selected_examples) <= 0:
            return True
        with Pool(4) as p:
            scores = p.map(partial(self.scorer.score, demo['request']), self.selected_examples)
        scores = [score['rougeL'].fmeasure for score in scores]
        if max(scores) >= self.thres:   # repetitive example
            return False
        else:
            self.selected_examples.append(demo['request'])
            return True



def main():
    parser = argparse.ArgumentParser(prog="Demonstration Filter")
    parser.add_argument("--demo_dir", type=str, help="directory of demonstration files")
    parser.add_argument("--seed_demo_file", type=str, help="file path of seed demonstration file")
    parser.add_argument("--rougeL_thres", type=float, default=0.8, help="rougeL similarity score threshold")
    args, _ = parser.parse_known_args()

    seed_demos = read_jsonl(args.seed_demo_file)

    files = os.listdir(args.demo_dir)
    examples = load_demos(args.demo_dir)
    
    rule_funcs = [tool_using_order_rule, RepetitiveRequestRule(seed_demos, args.rougeL_thres)]

    print("Before filtering: {} examples. Filtering...".format(len(examples)))
    qualified_examples = [] + seed_demos
    for example in examples:
        if all([rule_f(example) for rule_f in rule_funcs]):
            qualified_examples.append(example)
        else:
            pass
    print("After filtering: {} examples.".format(len(qualified_examples)))
    
    now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
    fname =  os.path.join(args.demo_dir, "../filtered", f"filtered_{now}.jsonl")

    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))

    if len(examples) > 0:
        write_jsonl(qualified_examples, fname)
    
    print("Filtering completed. Demonstrations in those files are filtered: ")
    for f in files:
        print("    - {}".format(os.path.join(args.demo_dir, f)))

    print("Filtered demonstrations saved in {}.".format(fname))




if __name__ == "__main__":
    main()
    print("Over.")