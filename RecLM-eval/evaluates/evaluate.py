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
from .TFIDF_model import TFIDF_model, string_match_score, _clean_string

# compute ranking metrics for retrieval/ranking/searching
def compute_metrics_on_title_recommend(result_file_path, metadata_file_path, sim_threshold=0.6):
    results = []
    rankings = []
    ground_truths = []
    for line in open(result_file_path, "r", encoding="utf-8"):
        line = json.loads(line)
        # -----------------------------------------------------------------
        # Build the final ranking list.
        # If the original candidate list is present, use a robust parser that
        # relies on longest-match search instead of naive ", " splitting.
        # This prevents titles that contain ", " inside them from being
        # accidentally cut and counted as OOV → candidate_error.
        # -----------------------------------------------------------------
        if "result" not in line:
            if "candidate" in line and isinstance(line["candidate"], list):
                line["result"] = parse_recommendations_with_candidates(
                    line["answer"], line["candidate"]
                )
            else:
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
    """Compute accuracy for A–J multiple-choice questions with a TF-IDF fallback.

    If the model output does not contain a standalone letter between A and J, we compute the TF-IDF cosine similarity between the full output text and each of the ten candidate titles. The letter corresponding to the candidate with the highest similarity is taken as the prediction. If that similarity is still below the threshold, the answer is considered incorrect."""

    score = 0
    total = 0
    none_cnt = 0
    # --- debug: collect up to 20 wrong cases for manual inspection ---
    wrong_shown = 0  # counter for examples already printed
    none_cnt = 0     # count predictions that cannot be recognized

    for idx, raw in enumerate(open(result_file_path, "r", encoding="utf-8")):
        data = json.loads(raw)
        total += 1

        # --- Unify the result field ---
        if "result" not in data:
            data["result"] = data.get("answer", "")

        output_text = str(data["result"]).strip()

        # --  Pre-processing: strip leading numbering such as "1. " or "1) " --
        output_text_norm = re.sub(r"^\s*\d+\s*[\.\)]\s*", "", output_text, count=1)

        # -- Search for the first *stand-alone* uppercase letter A–J (no dot/parenthesis required)
        m = re.search(r"\b([A-J])\b", output_text_norm)
        if m:
            pred_letter = m.group(1).lower()
        else:
            # -- Title→letter mapping (substring, then difflib, finally TF-IDF)
            letters_items = []  # [(letter, clean_title)]
            raw_items = []      # Raw titles for TF-IDF
            if "candidates" in data and isinstance(data["candidates"], list):
                for cand in data["candidates"]:
                    if not isinstance(cand, str):
                        continue
                    m_c = re.match(r"\s*([A-Ja-j])[\.\)]\s*(.+)", cand)
                    if m_c:
                        letters_items.append((m_c.group(1).lower(), _clean_string(m_c.group(2))))
                        raw_items.append(m_c.group(2))

            if not letters_items:
                pred_letter = None
                none_cnt += 1
            else:
                clean_output = _clean_string(output_text_norm)

                # -- Direct substring containment
                found = None
                for lt, ct in letters_items:
                    if ct and ct in clean_output:
                        found = lt
                        break

                # -- difflib similarity
                if found is None:
                    import difflib
                    best_sim, best_lt = 0.0, None
                    for lt, ct in letters_items:
                        sim = difflib.SequenceMatcher(None, clean_output, ct).ratio()
                        if sim > best_sim:
                            best_sim, best_lt = sim, lt
                    if best_sim >= 0.55:
                        found = best_lt

                # -- (removed) — previously TF-IDF fallback. We now stop here.

                pred_letter = found

        target_letter = str(data["target"])[0].lower()
        if pred_letter is None:
            none_cnt += 1

        if pred_letter is not None and pred_letter == target_letter:
            score += 1
        # No debug printing when prediction is incorrect
        else:
            pass

    acc1 = score / total if total else 0.0
    none_ratio = none_cnt / total if total else 0.0
    return acc1, none_ratio

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
            ## Recommend items that are already in the history
            ## Note: this implementation is not optimal – items in a series (e.g. Halo 1 vs Halo 2) may trigger false positives, hence the higher sim_threshold.
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

    # --- return a dictionary so that upstream can record it ---
    return {
        "history_error_rate": history_error/len(all_data),
        "duplicate_error_rate": duplicate_error/len(all_data),
        "candidate_error_rate": candidate_error/len(all_data),
        "copy_error": copy_error/len(all_data),
    }

END_TOKEN = "<end>"
# parse the recommendation list returned by the llm, the format maybe invalid
def parse_recommendations(ranking_str):
    """Split the generated ranking string by the special END_TOKEN marker.

    The model is instructed to output each title followed *immediately* by
    the literal token "<end>".  We therefore recover the original list by
    splitting on that token and stripping whitespace.
    """
    # Accept list response
    if isinstance(ranking_str, list):
        ranking_str = ranking_str[0]
    if not isinstance(ranking_str, str):
        return []

    parts = [p.strip() for p in ranking_str.split(END_TOKEN) if p.strip()]
    return parts

# ------------------------------------------------------------------
# With the new <end> delimiter we no longer need candidate-based
# longest-substring matching.  The helper now simply falls back to the
# universal parser above to avoid duplicated code.
# ------------------------------------------------------------------

def parse_recommendations_with_candidates(ranking_str: str, candidates: list):
    return parse_recommendations(ranking_str)

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
