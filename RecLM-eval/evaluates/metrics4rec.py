# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import math
import numpy as np
import heapq
from tqdm import tqdm
import string
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer=WordNetLemmatizer()

## This file is adapted from https://github.com/orcax/LOGER/blob/main/metrics4rec.py

def recall_at_k(r, k, all_pos_num):
    r = np.asarray(r)[:k]
    return np.sum(r) / all_pos_num

def hit_at_k(r, k):
    r = np.asarray(r)[:k]
    if np.sum(r) > 0:
        return 1.0
    else:
        return 0.0

def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1.0 / (r[0] + 1) if r.size else 0.0 for r in rs])

def r_precision(r):
    """Score is precision after all relevant documents have been retrieved
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> r_precision(r)
    0.33333333333333331
    >>> r = [0, 1, 0]
    >>> r_precision(r)
    0.5
    >>> r = [1, 0, 0]
    >>> r_precision(r)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        R Precision
    """
    r = np.asarray(r) != 0
    z = r.nonzero()[0]
    if not z.size:
        return 0.0
    return np.mean(r[: z[-1] + 1])

def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError("Relevance score length < k")
    return np.mean(r)

def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.0
    return np.mean(out)

def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])

def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asarray(r, dtype=float)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError("method must be 0 or 1.")
    return 0.0

def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k, method) / dcg_max

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

def evaluate_once(model, topk_preds, groundtruth, sim_threshold=0.6):
    """Evaluate one user performance.
    Args:
        topk_preds: list of <item_title>. length of the list is topK.
        groundtruth: list of <item_title>.
    Returns:
        dict of metrics.
    """
    gt_set = set(groundtruth)
    topk = len(topk_preds)
    rel = []
    hit = 0
    str_sim_scores = string_match_score(model, topk_preds, groundtruth)
    for idx, str_sim_score in enumerate(str_sim_scores):
        if str_sim_score >= sim_threshold and hit < len(groundtruth):
            rel.append(1)
            hit += 1
        else:
            rel.append(0)
    return {
        "precision@k": precision_at_k(rel, topk),
        "recall@k": recall_at_k(rel, topk, len(gt_set)),
        "ndcg@k": ndcg_at_k(rel, topk, 1),
        "hit@k": hit_at_k(rel, topk),
        "ap": average_precision(rel),
        "rel": rel,
    }

def evaluate_all(model, predicted_items, groudtruths, topk=10, sim_threshold=0.6):
    avg_prec, avg_recall, avg_ndcg, avg_hit = 0.0, 0.0, 0.0, 0.0
    rs = []
    cnt = 0
    pbar = tqdm(total = len(predicted_items),desc=f"metric {topk}", leave=False)
    for topk_preds, ground_truth in zip(predicted_items, groudtruths):
        result = evaluate_once(model, topk_preds[:topk], ground_truth, sim_threshold)
        avg_prec += result["precision@k"]
        avg_recall += result["recall@k"]
        avg_ndcg += result["ndcg@k"]
        avg_hit += result["hit@k"]
        rs.append(result["rel"])
        cnt += 1
        pbar.update(1)
    pbar.close()

    avg_prec = avg_prec / cnt
    avg_recall = avg_recall / cnt
    avg_ndcg = avg_ndcg / cnt
    avg_hit = avg_hit / cnt
    map_ = mean_average_precision(rs)
    mrr = mean_reciprocal_rank(rs)
    msg = "\nNDCG@{}\tRec@{}\tHits@{}\tPrec@{}\tMAP@{}\tMRR@{}".format(topk, topk, topk, topk, topk, topk)
    msg += "\n{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(avg_ndcg, avg_recall, avg_hit, avg_prec, map_, mrr)
    print(msg)
    res = {
        f'ndcg@{topk}': avg_ndcg,
        f'map@{topk}': map_,
        f'recall@{topk}': avg_recall,
        f'precision@{topk}': avg_prec,
        f'mrr@{topk}': mrr,
        f'hit@{topk}': avg_hit,
    }
    return msg, res

def evaluate_once_id(topk_preds, groundtruth):
    """Evaluate one user performance.
    Args:
        topk_preds: list of <item_title>. length of the list is topK.
        groundtruth: list of <item_title>.
    Returns:
        dict of metrics.
    """
    gt_set = set(groundtruth)
    topk = len(topk_preds)
    rel = [int(x in groundtruth) for x in topk_preds]
    return {
        "precision@k": precision_at_k(rel, topk),
        "recall@k": recall_at_k(rel, topk, len(gt_set)),
        "ndcg@k": ndcg_at_k(rel, topk, 1),
        "hit@k": hit_at_k(rel, topk),
        "ap": average_precision(rel),
        "rel": rel,
    }

def evaluate_all_id(predicted_items, groudtruths, topk=10):
    avg_prec, avg_recall, avg_ndcg, avg_hit = 0.0, 0.0, 0.0, 0.0
    rs = []
    cnt = 0
    pbar = tqdm(total = len(predicted_items),desc=f"metric {topk}", leave=False)
    for topk_preds, ground_truth in zip(predicted_items, groudtruths):
        result = evaluate_once_id(topk_preds[:topk], ground_truth)
        avg_prec += result["precision@k"]
        avg_recall += result["recall@k"]
        avg_ndcg += result["ndcg@k"]
        avg_hit += result["hit@k"]
        rs.append(result["rel"])
        cnt += 1
        pbar.update(1)
    pbar.close()

    avg_prec = avg_prec / cnt
    avg_recall = avg_recall / cnt
    avg_ndcg = avg_ndcg / cnt
    avg_hit = avg_hit / cnt
    map_ = mean_average_precision(rs)
    mrr = mean_reciprocal_rank(rs)
    msg = "\nNDCG@{}\tRec@{}\tHits@{}\tPrec@{}\tMAP@{}\tMRR@{}".format(topk, topk, topk, topk, topk, topk)
    msg += "\n{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(avg_ndcg, avg_recall, avg_hit, avg_prec, map_, mrr)
    print(msg)
    res = {
        f'ndcg@{topk}': avg_ndcg,
        f'map@{topk}': map_,
        f'recall@{topk}': avg_recall,
        f'precision@{topk}': avg_prec,
        f'mrr@{topk}': mrr,
        f'hit@{topk}': avg_hit,
    }
    return msg, res

