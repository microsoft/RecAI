# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import re
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from .metrics4rec import evaluate_all, evaluate_all_id
from .TFIDF_model import TFIDF_model, string_match_score

# compute ranking metrics for retrieval/ranking/searching
def compute_metrics_on_title_recommend(result_file_path, metadata_file_path, sim_threshold=0.6):
    results = []
    rankings = []
    ground_truths = []
    for line in open(result_file_path, "r", encoding="utf-8"):
        line = json.loads(line)
        if "result" not in line:
            line["result"] = parse_recommendations(line["answer"])
        rankings.append(line["result"])
        ground_truths.append([line["target"]])
        results.append(line)
    
    fd = open(result_file_path, "w", encoding="utf-8")
    for line in results:
        fd.write(json.dumps(line, ensure_ascii=False)+'\n')
    fd.close()

    item_list = []
    if not os.path.exists(metadata_file_path):
        if os.path.exists(metadata_file_path+ "l"):
            metadata_file_path = metadata_file_path + "l"
    for line in open(metadata_file_path):
        line = json.loads(line)
        if 'TitleName' in line:
            item_list.append(line['TitleName'])
        elif 'app_name' in line:
            item_list.append(line['app_name'])
        elif 'GameName' in line:
            item_list.append(line['GameName'])
        elif 'title' in line:
            item_list.append(line['title'])

    tfidf_model = TFIDF_model(item_list)

    all_metrics=dict()
    for topk in [1, 5, 10, 20]:
        metrics = evaluate_all(tfidf_model, rankings, ground_truths, topk, sim_threshold)[1]
        all_metrics={**all_metrics, **metrics}

    return all_metrics

def compute_metrics_on_multi_choices(result_file_path):
    score = 0
    length = 0

    for line in open(result_file_path, "r", encoding="utf-8"):
        line = json.loads(line)  
        length += 1 

        stripped_result = line["result"].strip()
        match = re.search(r'([a-zA-Z])', stripped_result)
        
        if match:
            answer_letter = match.group(1).lower() 
            target_letter = line["target"][0].lower() 
            if answer_letter == target_letter: 
                score += 1 

    if length != 0:
        score /= length 
    return score  

# compute ranking metrics for retrieval/ranking/searching
def compute_metrics_on_id_recommend(result_file_path):
    results = []
    rankings = []
    ground_truths = []
    for line in open(result_file_path, "r", encoding="utf-8"):
        line = json.loads(line)
        rankings.append(line["result"])
        ground_truths.append([line["target"]])
        results.append(line)

    all_metrics=dict()
    for topk in [1, 5, 10, 20]:
        metrics = evaluate_all_id(rankings, ground_truths, topk)[1]
        all_metrics={**all_metrics, **metrics}

    return all_metrics

def LCS(list_a, list_b, model, sim_threshold):  
    len_a = len(list_a)  
    len_b = len(list_b)  
  
    # Create a table to store lengths of longest common suffixes of substrings  
    lcs_table = [[0] * (len_b + 1) for _ in range(len_a + 1)]  
  
    # Build the table in bottom-up fashion  
    for i in range(len_a + 1):  
        for j in range(len_b + 1):  
            if i == 0 or j == 0:  
                lcs_table[i][j] = 0  
            elif string_match_score(model, [list_a[i - 1]], [list_b[j - 1]])[0]>=sim_threshold:
                lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1  
            else:  
                lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])  
  
    # The length of the LCS is stored in the bottom-right corner of the table  
    return lcs_table[len_a][len_b]  


def compute_errors_on_title_ranking(result_file_path, metadata_file_path, sim_threshold):
    item_list = ['padding']
    if not os.path.exists(metadata_file_path):
        if os.path.exists(metadata_file_path+ "l"):
            metadata_file_path = metadata_file_path + "l"
    for line in open(metadata_file_path):
        line = json.loads(line)
        if 'TitleName' in line:
            item_list.append(line['TitleName'])
        elif 'app_name' in line:
            item_list.append(line['app_name'])
        elif 'title' in line:
            item_list.append(line['title'])

    tfidf_model = TFIDF_model(item_list)

    history_error, duplicate_error, candidate_error = 0, 0, 0
    all_data = []
    for line in open(result_file_path, encoding="utf-8"):
        all_data.append(json.loads(line))

    copy_error = 0

    for idx, data in enumerate(tqdm(all_data)):
        ranking = data["result"]
        if "history" in data:
            history = data["history"]
        else:
            history = []
        if "candidate" in data:
            candidates = data["candidate"]
        else:
            candidates = []
        history_flag, duplicate_flag, candidate_flag = 0, 0, 0
        already_have = []
        for item in ranking:
            ## recommend items that are already in the history
            ## this implementation is not optimal, because items in a series may cause mistakes, such as Halo 1 and Halo 2. That's why we need to set a higher sim_threshold.
            if history and string_match_score(tfidf_model, [item], history)[0]>=sim_threshold:  
                history_flag = 1
            if already_have and string_match_score(tfidf_model, [item], already_have)[0]>=sim_threshold:  # recommend duplicate items
                duplicate_flag = 1
            if candidates and string_match_score(tfidf_model, [item], candidates)[0] < sim_threshold:  # recommend items that are not in the candidate set
                candidate_flag = 1
            already_have.append(item)
        history_error += history_flag
        duplicate_error += duplicate_flag
        candidate_error += candidate_flag
    
        x = LCS(ranking, history + candidates, tfidf_model, sim_threshold)
        copy_error += x

    print(f"total count: {len(all_data)}, history error count: {history_error}, duplicate error count: {duplicate_error}, candidate error count: {candidate_error}.")
    print(f"history error rate: {history_error/len(all_data)}, duplicate error rate: {duplicate_error/len(all_data)}, candidate error rate: {candidate_error/len(all_data)}.")
    print(f"copy error: {copy_error/len(all_data)}")

## parse the recommendation list returned by the llm, the format maybe invalid
def parse_recommendations(ranking_str):
    if type(ranking_str) == list:
        ranking_str = ranking_str[0]
    if type(ranking_str) != str:
        return ["invalid ranking result"]
    
    ranking = []
    lines = ranking_str.split("\n")
    for idx, line in enumerate(lines):
        if ranking == [] and len(line.split(", ")) >= 2:
            if "Based on your" in line:
                ranking.extend(": ".join(line.split(": ")[1:]).split(", "))
            else:
                ranking.extend(line.split(", "))
        elif ranking == [] and len(line.split(". ")) >= 3 and line.split(". ")[0].isdigit():
            ranking.extend([" ".join(x.split(" ")[:-1]) for x in line.split(". ")[1:]])
        elif len(line.split(". ")) >= 2 and line.split(". ")[0].isdigit():
            ranking.append(line.split(". ")[1])
    
    if ranking == []:
        if len(lines) >= 5:
            i1 = 0
            while i1 < len(lines) and lines[i1] != "": i1 += 1
            i2 = i1 + 1
            while i2 < len(lines) and lines[i2] != "": i2 += 1
            for i in range(i1 + 1, i2):
                ranking.append(lines[i].strip())

    if ranking == []:
        ranking = [lines[-1]]

    return ranking

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="steam")
    parser.add_argument("--output_dir", type=str, help="path to the llm output")
    parser.add_argument("--tasks", type=str, default="retrieval", help="tasks to evaluate")
    parser.add_argument("--sim_threshold", type=float, default=0.6, help="similarity threshold to tell whether two item names are the same")

    args = parser.parse_args()
    args.sequential_file_path = f"./data/{args.dataset}/sequential_data.txt"
    args.metadata_file_path = f"./data/{args.dataset}/metadata.json"
    args.negative_file_path = f"./data/{args.dataset}/negative_samples.txt"

    args.retrieval_file_path = f"{args.output_dir}/retrieval_results.pkl"
    args.ranking_file_path = f"{args.output_dir}/ranking_results.pkl"
    args.tagging_file_path = f"{args.output_dir}/tagging_results.pkl"
    args.searching_file_path = f"{args.output_dir}/searching_results.pkl"

    return args

if __name__ == "__main__":
    args = get_args()

    if "retrieval" in args.tasks:
        print("Evaluating retrieval results...")
        metrics = compute_metrics_on_title_recommend(args.retrieval_file_path, args.metadata_file_path, args.sim_threshold)
    
    if "ranking" in args.tasks:
        print("Evaluating ranking results...")
        metrics = compute_metrics_on_title_recommend(args.ranking_file_path, args.metadata_file_path, args.sim_threshold)
