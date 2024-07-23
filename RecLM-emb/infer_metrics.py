"""
The following code is modified from
https://github.com/orcax/LOGER/blob/main/metrics4rec.py
"""

import os
import sys
import json
import pickle
import argparse
from tqdm import tqdm
import random
import pandas as pd
import datetime

import torch
import numpy as np

from accelerate import Accelerator
from accelerate.utils import set_seed

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__) )))
from preprocess.utils import get_item_text
from src.huggingface_model_infer import run_model_embedding

def coverage_at_k(r, k):
    r = np.asarray(r)[:k]
    return np.sum(r) / k

def recall_at_k(r, k, all_pos_num):
    r = np.asarray(r)[:k]
    return np.sum(r) / all_pos_num

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

def hit_at_k(r, k):
    r = np.asarray(r)[:k]
    if np.sum(r) > 0:
        return 1.0
    else:
        return 0.0

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
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError("method must be 0 or 1.")
    return 0.0

def ndcg_at_k(r, k, gt_set, method=1):
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
    gt_rel = [1]*len(gt_set) + [0]*(k-len(gt_set)) if len(gt_set)<k else [1]*k
    dcg_max = dcg_at_k(gt_rel, k, method)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k, method) / dcg_max

def evaluate_once_id(topk_preds, groundtruth):
    """Evaluate one user performance.
    Args:
        topk_preds: list of <item_title>. length of the list is topK.
        groundtruth: list of <item_title>.
    Returns:
        dict of metrics.
    """
    gt_set = set(groundtruth) if isinstance(groundtruth, list) else set([groundtruth])
    topk = len(topk_preds)
    rel = [int(x in gt_set) for x in topk_preds]
    return {
        "precision@k": precision_at_k(rel, topk),
        "recall@k": recall_at_k(rel, topk, len(gt_set)),
        "coverage@k": coverage_at_k(rel, topk),
        "ndcg@k": ndcg_at_k(rel, topk, gt_set),
        "hit@k": hit_at_k(rel, topk),
        "rel": rel,
    }

def evaluate_all_id(predicted_items, groudtruths, topk=10):
    avg_prec, avg_recall, avg_coverage, avg_ndcg, avg_hit = 0.0, 0.0, 0.0, 0.0, 0.0
    rs = []
    cnt = 0
    for topk_preds, ground_truth in zip(predicted_items, groudtruths):
        result = evaluate_once_id(topk_preds[:topk], ground_truth)
        avg_prec += result["precision@k"]
        avg_recall += result["recall@k"]
        avg_coverage += result["coverage@k"]
        avg_ndcg += result["ndcg@k"]
        avg_hit += result["hit@k"]
        rs.append(result["rel"])
        cnt += 1

    avg_prec = avg_prec / cnt
    avg_recall = avg_recall / cnt
    avg_coverage = avg_coverage / cnt
    avg_ndcg = avg_ndcg / cnt
    avg_hit = avg_hit / cnt
    # msg = "\nNDCG@{}\tHits@{}".format(topk, topk)
    # msg += "\n{:.4f}\t{:.4f}".format(avg_ndcg, avg_hit)
    # print(msg)
    res = {
        'precision@'+str(topk): avg_prec,
        'recall@'+str(topk): avg_recall,
        'coverage@'+str(topk): avg_coverage,
        'ndcg@'+str(topk): avg_ndcg,
        'hit@'+str(topk): avg_hit,
    }
    return res

def compute_metrics_on_id_recommend(args):
    results = []
    rankings = []
    ground_truths = []
    for line in open(args.answer_file, "r"):
        line = json.loads(line)
        rankings.append(line["result"])
        ground_truths.append(line["target"])
        results.append(line)

    all_metrics=dict()
    for topk in eval(args.topk):
        metrics = evaluate_all_id(rankings, ground_truths, topk)
        all_metrics={**all_metrics, **metrics}

    return all_metrics

def cal_feature_coverage(ground_truth, meta):
    flag=1
    for key, value in ground_truth:
        if key not in meta:
            flag=0
            break
        if isinstance(value, list):
            for v in value:
                if v not in meta[key]:
                    flag=0
                    break
        else:
            if value != meta[key]:
                flag=0
                break
        if flag==0:
            break
    return flag

def cal_vague_feature_coverage(ground_truth, meta):
    flag=1
    if ground_truth[0] in [0, 2, 3]:#check price
        if meta['price']==None:
            flag=0
        elif ground_truth[1][0]=='less than' and meta['price']>=ground_truth[1][1]:
            flag=0
        elif ground_truth[1][0]=='more than' and meta['price']<=ground_truth[1][1]:
            flag=0
    
    if flag==1 and ground_truth[0] in [1, 2, 3]:#check release date
        if ground_truth[2]:
            ground_date = datetime.datetime.strptime(ground_truth[3][1], "%B %d, %Y")
        else:
            ground_date = ground_truth[3][1]
        if meta['release date']==None:
            flag=0
        elif ground_truth[3][0]=='before':
            if not ground_truth[2]:
                if meta['release date'].year>=ground_date:
                    flag=0
            elif (meta['release date'].year, meta['release date'].month)>=(ground_date.year, ground_date.month):
                flag=0
        elif ground_truth[3][0]=='after':
            if not ground_truth[2]:
                if meta['release date'].year<=ground_date:
                    flag=0
            elif (meta['release date'].year, meta['release date'].month)<=(ground_date.year, ground_date.month):
                flag=0
        elif ground_truth[3][0]=='in':
            if not ground_truth[2]:
                if meta['release date'].year!=ground_date:
                    flag=0
            elif meta['release date'].year!=ground_date.year or meta['release date'].month!=ground_date.month:
                flag=0
    return flag

def compute_metrics_on_query(args):
    item2meta = {}
    for line in open(item_embedding_prompt_path, "r"):
        line = json.loads(line)
        if line['id']==0:
            continue
        if args.task_type=='vaguequery2item':
            item2meta[line['id']] = {'price': line['price'], 'release date': datetime.datetime.strptime(line['release date'], "%B %d, %Y") if line['release date']!=None else None, 
                                 'next month': datetime.datetime.strptime(line['next month'], "%B %d, %Y") if line['next month']!=None else None, 
                                 'last month': datetime.datetime.strptime(line['last month'], "%B %d, %Y") if line['last month']!=None else None}
        else:
            item2meta[line['id']] = {line['title'][0]: line['title'][1]}
            for v in line['features']:
                item2meta[line['id']][v[0]] = v[1]

    ground_truths = []
    results = []
    for line in open(args.answer_file, "r"):
        line = json.loads(line)
        results.append(line['result'])
        ground_truths.append(line["target"])

    all_metrics=dict()
    for topk in eval(args.topk):
        conv = 0.0
        count = 0.0
        for ground_truth, topk_preds in zip(ground_truths, results):
            for item_id in topk_preds[:topk]:
                count += 1
                meta = item2meta[item_id]
                if args.task_type=='vaguequery2item':
                    flag = cal_vague_feature_coverage(ground_truth, meta)
                else:
                    flag = cal_feature_coverage(ground_truth, meta)
                
                if flag==1:
                    conv += 1
        topk_coverage = {"hit_count@"+str(topk): conv, "total_count@"+str(topk): count, "coverage@"+str(topk): conv/count if count!=0.0 else 0.0}
        all_metrics={**all_metrics, **topk_coverage}
        
    return all_metrics

def compute_metrics(args):
    if args.task_type in ['user2item', 'title2item', 'item2item', 'queryuser2item', 'misspell2item']:
        return compute_metrics_on_id_recommend(args)
    elif args.task_type in ['query2item', 'vaguequery2item']:
        return compute_metrics_on_query(args)

def get_topk_index(scores, topk):
    r"""Get topk index given scores with numpy. The returned index is sorted by scores descendingly."""
    scores = - scores
    topk_ind = np.argpartition(scores, topk, axis=1)[:, :topk] 
    topk_scores = np.take_along_axis(scores, topk_ind, axis=-1)
    sorted_ind_index = np.argsort(topk_scores, axis=1)
    sorted_index = np.take_along_axis(topk_ind, sorted_ind_index, axis=-1)
    return sorted_index

def gen_retrieval_result(args, item_embedding_path, user_embedding_path):
    os.makedirs(os.path.dirname(args.answer_file), exist_ok=True)
    fd = open(args.answer_file, "w", encoding='utf-8')

    item_embeddings = torch.tensor(pickle.load(open(item_embedding_path, "rb")))
    user_embeddings = torch.tensor(pickle.load(open(user_embedding_path, "rb")))
    print("shape of item embeddings: ", item_embeddings.shape)
    print("shape of user embeddings: ", user_embeddings.shape)

    scores = torch.softmax(torch.matmul(user_embeddings, item_embeddings.T), -1).squeeze().numpy()
    targets = []
    for idx, line in enumerate(open(args.user_embedding_prompt_path)):
        # if idx==10000:
        #     break
        data = json.loads(line)
        targets.append(data['ground_truth'])
        scores[idx][0] = -2
        if args.task_type=='user2item' or args.task_type=='queryuser2item':
            scores[idx][data["history"]] = -2
        elif args.task_type=='item2item':
            scores[idx][data["item_id"]] = -2 # remove the item itself
        
    topk_index = get_topk_index(scores, max(eval(args.topk)))
    for target, index in zip(targets, topk_index):
        line = {'target': target, 'result': index.tolist()}
        fd.write(json.dumps(line, ensure_ascii=False)+'\n')
    fd.close()

def parse_args():
    parser = argparse.ArgumentParser(description="infer case")
    parser.add_argument(
        "--in_seq_data", type=str, help=""
    )
    parser.add_argument(
        "--in_meta_data", type=str, help=""
    )
    parser.add_argument(
        "--model_path_or_name", type=str, help="Path to pretrained model or model identifier from huggingface.co/models"
    )
    parser.add_argument(
        "--user_embedding_prompt_path", type=str, help="Path to query file"
    )
    parser.add_argument(
        "--answer_file", type=str, help=""
    )
    parser.add_argument(
        "--all_metrics_file", type=str, help="the file to save all metrics"
    )
    parser.add_argument(
        "--topk", type=str, default="[5, 10]", help=""
    )
    parser.add_argument(
        "--seed", type=int, default=2023, help=""
    )
    parser.add_argument(
        "--query_max_len", type=int, default=512, help=""
    )
    parser.add_argument(
        "--passage_max_len", type=int, default=128, help=""
    )
    parser.add_argument(
        "--per_device_eval_batch_size", type=int, default=128, help=""
    )
    parser.add_argument(
        "--task_type", type=str, default='user2item', help="", choices=['user2item', 'query2item', 'title2item', 'item2item', 'queryuser2item', 'misspell2item', 'vaguequery2item']
    )
    parser.add_argument(
        "--sentence_pooling_method", type=str, default='cls', help="the pooling method, should be cls, mean or last", choices=['cls', 'mean', 'last']
    )
    parser.add_argument(
        "--normlized", action='store_true', help=""
    )
    parser.add_argument(
        "--has_template", action='store_true', help=""
    )
    parser.add_argument(
        "--peft_model_name", type=str, default=None, help=""
    )
    parser.add_argument(
        "--torch_dtype", type=str, default=None, help="", choices=["auto", "bfloat16", "float16", "float32"]
    )
    args = parser.parse_args()
    return args

def write_metrics_to_file(metrics_dict, task_name, file_name):    
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  
    output = {'time': datetime_str, 'task_name': task_name, 'metrics': metrics_dict}
    with open(file_name, 'a') as f:
        f.write(json.dumps(output, ensure_ascii=False) + '\n') 

if __name__ == '__main__':
    openai_model_names = ['ada_embeddings', 'text-embedding-ada-002', 'text-embedding-3-large']
    args = parse_args()
    set_seed(args.seed)
    if args.model_path_or_name in openai_model_names:
        from src.openai_model_infer import run_api_embedding

    accelerator = Accelerator()

    ## get the dir of args.user_embedding_prompt_path
    cache_dir =  os.path.dirname(args.all_metrics_file)
    item_embedding_prompt_path = os.path.join(cache_dir, 'item_embedding_prompt.jsonl')
    item_embedding_path = os.path.join(cache_dir, 'item_embedding.pkl')
    user_embedding_path = os.path.join(cache_dir, 'user_embedding.pkl')

    if not os.path.exists(item_embedding_path):
        if accelerator.is_main_process:
            os.makedirs(cache_dir, exist_ok=True)
            get_item_text(args.in_meta_data, save_item_prompt_path=item_embedding_prompt_path)
        accelerator.wait_for_everyone()
        
        accelerator.print("infer item embedding")
        if args.model_path_or_name in openai_model_names:
            run_api_embedding(args.model_path_or_name, item_embedding_prompt_path, item_embedding_path)
        else:
            run_model_embedding(args.model_path_or_name, max_seq_len=args.passage_max_len, batch_size=args.per_device_eval_batch_size, prompt_path=item_embedding_prompt_path, emb_path=item_embedding_path, accelerator=accelerator, args=args, qorp='passage')

    accelerator.print("infer user embedding")
    if args.model_path_or_name in openai_model_names:
        run_api_embedding(args.model_path_or_name, args.user_embedding_prompt_path, user_embedding_path)
    else:
        run_model_embedding(args.model_path_or_name, max_seq_len=args.query_max_len, batch_size=args.per_device_eval_batch_size, prompt_path=args.user_embedding_prompt_path, emb_path=user_embedding_path, accelerator=accelerator, args=args, qorp='query')

    if accelerator.is_main_process:
        gen_retrieval_result(args, item_embedding_path, user_embedding_path)
        all_metric = compute_metrics(args)
        print(all_metric)
        task_name = args.user_embedding_prompt_path.split('/')[-1].split('.jsonl')[0]
        write_metrics_to_file(all_metric, task_name, args.all_metrics_file)