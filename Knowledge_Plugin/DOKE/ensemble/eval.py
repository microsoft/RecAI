import re
import json
import argparse
import numpy as np

from metrics4rec import mapping_evaluate_all
from TFIDF_model import TFIDF_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str)
    parser.add_argument('--dataset', type=str)

    return parser.parse_args()

args = parse_args()

metadata = f"../../data/{args.dataset}/metadata.json"
item_list = []
for line in open(metadata):
    line = json.loads(line)
    title = "No title"
    if "app_name" in line:
        title = line['app_name']
    elif "title" in line:
        title = line['title']
    else:
        if "description" in line:
            title = line['description'][:50]
        elif "categories" in line:
            title = line['categories'][0][-1]
    item_list.append(title)
tfidf_model = TFIDF_model(item_list)

predicted_items = []
ground_truth = []
for line in open(f"outputs/{args.dataset}/{args.method}/answer.jsonl"):
        line = json.loads(line)
        target =  line["target"]
        result =  line["result"]
        predicted_items.append(result)
        ground_truth.append([target])

all_metrics=dict()
k_list = [1,5,10,20]
result_str = ""
for topk in k_list:
    result = mapping_evaluate_all(tfidf_model, predicted_items, ground_truth, topk)
    metrics = result[1]
    all_metrics={**all_metrics, **metrics}
    result_str += result[0]
print(result_str)