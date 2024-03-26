# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn.functional as F
import math
from tqdm import tqdm

def evaluate_ILD(item_embed, predicted_items, topk):
    ILD = 0
    all_cnt = 0
    for preds in tqdm(predicted_items, desc=f"ILD@{topk}"):
        total_sum = 0
        total_cnt = 0
        topk_preds = []
        for x in preds:
            try:
                if int(x) < len(item_embed):
                    topk_preds.append(x)
            except:
                pass
        topk_preds = topk_preds[:topk]
        if len(topk_preds) > 1:
            for i in topk_preds:
                for j in topk_preds:
                    if i != j:
                        embi = torch.FloatTensor(item_embed[int(i)])
                        embj = torch.FloatTensor(item_embed[int(j)])
                        cos_sim = F.cosine_similarity(embi, embj, dim=0)
                        total_sum += cos_sim
                        total_cnt += 1
        if total_cnt:
            ILD += total_sum.item() / total_cnt
        all_cnt += total_cnt
    return {f"ILD@{topk}": 1 - ILD / len(predicted_items)}

def evaluate_IC(predicted_items, total_item_num, topk):
    covered_cates = set()
    for preds in tqdm(predicted_items, desc=f"IC@{topk}"):
        topk_preds = []
        for x in preds:
            try:
                if int(x) < total_item_num:
                    topk_preds.append(x)
            except:
                pass
        topk_preds = topk_preds[:topk]
        for i in topk_preds:
            covered_cates.add(i)
    return {f"IC@{topk}": len(covered_cates) / total_item_num}

def match_score(topk_preds, targets):
    topk_preds = [int(x) for x in topk_preds]
    targets = [int(x) for x in targets]
    return [1 if x in targets else 0 for x in topk_preds]


def string_match_score(model, predictions, truths):
    sims = model.match(predictions, truths)
    scores = []
    for pred in sims:
        max_score = 0
        for sim in pred:
            if sim > max_score:
                max_score = sim
        scores.append(max_score)
    return scores
