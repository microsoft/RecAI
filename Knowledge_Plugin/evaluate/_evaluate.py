# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm

from metrics4rec import evaluate_all, mapping_evaluate_all
from metrics4recdiv import evaluate_IC, evaluate_ILD, string_match_score
from utils import bleu_score, rouge_score
from TFIDF_model import TFIDF_model

def compute_metrics_on_id_recommend(result_file_path, sequential_data_path, metadata_file_path, valid_or_test, k_list):
    results=pickle.load(open(result_file_path,"rb"))[:1000] # list of rankings
    if isinstance(results[0], list) or isinstance(results[0], np.ndarray):
        predicted_items = results
        if valid_or_test == "valid":
            ind = -2
        else:
            ind = -1
        if isinstance(predicted_items[0][0], int) or isinstance(predicted_items[0][0], np.int64):
            ground_truth=[[int(line.strip().split(" ")[ind])] for line in open(sequential_data_path)][:1000]
        else:
            ground_truth=[[str(line.strip().split(" ")[ind])] for line in open(sequential_data_path)][:1000]
    else:
        predicted_items = [result[0] for result in results]
        ground_truth = [[result[1]] for result in results]


    all_metrics=dict()
    result_str = ""
    for topk in k_list:
        metrics = evaluate_all(predicted_items, ground_truth, topk)[1]
        all_metrics={**all_metrics, **metrics}
        for metric in metrics:
            result_str += f'{metric}@{topk}: {round(np.mean(metrics[metric]), 4)}, '

    print("result: ", result_str)

    return all_metrics


def compute_metrics_on_title_recommend(result_file_path, sequential_data_path, metadata_file_path, valid_or_test, k_list, string_match_method):
    rankings = pickle.load(open(result_file_path, "rb"))[:100] # list of (rankings, ground_truth)
    predicted_items = [ranking[0] for ranking in rankings]
    ground_truth = [[ranking[1]] for ranking in rankings]

    item_list = []
    for line in open(metadata_file_path):
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

    all_metrics=dict()
    result_str = ""
    for topk in k_list:
        metrics = mapping_evaluate_all(tfidf_model, predicted_items, ground_truth, topk)[1]
        all_metrics={**all_metrics, **metrics}
        for metric in metrics:
            result_str += f'{metric}@{topk}: {round(np.mean(metrics[metric]), 4)},'

    print("result: ", result_str)

    return all_metrics


def compute_errors_on_title_retrieval(result_file_path, available_item_path, sequential_data_path, metadata_file_path, valid_or_test, sim_threshold=0.6):
    rankings = pickle.load(open(result_file_path, "rb"))[:1000] # list of (rankings, ground_truth)
    predicted_items = [ranking[0] for ranking in rankings]
    available_items = pickle.load(open(available_item_path, "rb")) # all available items in that domain
    try:
        metadata = [eval(line)['app_name'] for line in open(metadata_file_path)]
    except:
        metadata = [eval(line)['title'] for line in open(metadata_file_path)]

    tfidf_model = TFIDF_model(metadata)

    user_history = []
    with open(sequential_data_path) as fr:
        for lidx, line in enumerate(fr):
            line = line.strip().split(" ")[1:-1]
            line = [metadata[int(item)-1] for item in line]
            user_history.append(line)
            if len(user_history) == len(predicted_items):
                break

    history_error, duplicate_error, available_error = 0, 0, 0

    for ranking, history in zip(predicted_items, user_history):
        history_flag, duplicate_flag, available_flag = 0, 0, 0
        already_have = []
        for item in ranking:
            if history and string_match_score(tfidf_model, [item], history)[0]>=sim_threshold:  # recommend items that are already in the history
                history_flag = 1
            if already_have and string_match_score(tfidf_model, [item], already_have)[0]>=sim_threshold:  # recommend duplicate items
                duplicate_flag = 1
            if available_items and string_match_score(tfidf_model, [item], available_items)[0]<sim_threshold:  # recommend items that are not available
                available_flag = 1
            already_have.append(item)
        history_error += history_flag
        duplicate_error += duplicate_flag
        available_error += available_flag

    print(f"total count: {len(predicted_items)}, history error: {history_error}, duplicate error: {duplicate_error}, available error: {available_error}.")
    print(f"history error rate: {history_error/len(predicted_items)}, duplicate error rate: {duplicate_error/len(predicted_items)}, available error rate: {available_error/len(predicted_items)}.")


def compute_errors_on_title_ranking(result_file_path, sequential_data_path, metadata_file_path, negative_file_path, valid_or_test, sim_threshold=0.6):
    rankings = pickle.load(open(result_file_path, "rb"))[:1000] # list of (rankings, ground_truth)
    predicted_items = [ranking[0] for ranking in rankings]
    try:
        metadata = [eval(line)['app_name'] for line in open(metadata_file_path)]
    except:
        metadata = [eval(line)['title'] for line in open(metadata_file_path)]

    tfidf_model = TFIDF_model(metadata)

    user_history = []
    with open(sequential_data_path) as fr:
        for line in fr:
            line = line.strip().split(" ")[1:-1]
            line = [metadata[int(item)-1] for item in line]
            user_history.append(line)
            if len(user_history) == len(predicted_items):
                break
    candidate_samples = []
    with open(negative_file_path) as fr:
        for line in fr:
            line = line.strip().split(" ")[1:]
            line = [metadata[int(item)-1] for item in line]
            candidate_samples.append(line)
            if len(candidate_samples) == len(predicted_items):
                break
    
    history_error, duplicate_error, candidate_error = 0, 0, 0
    
    for idx, (ranking, history, candidates) in enumerate(zip(predicted_items, user_history, candidate_samples)):
        history_flag, duplicate_flag, candidate_flag = 0, 0, 0
        already_have = []
        for item in ranking:
            if history and string_match_score(tfidf_model, [item], history)[0]>=sim_threshold:  # recommend items that are already in the history
                history_flag = 1
                # print("idx: ", idx, "history: ", history, "item: ", item)
            if already_have and string_match_score(tfidf_model, [item], already_have)[0]>=sim_threshold:  # recommend duplicate items
                duplicate_flag = 1
            if candidates and string_match_score(tfidf_model, [item], candidates)[0] < sim_threshold:  # recommend items that are not in the candidate set
                candidate_flag = 1
            already_have.append(item)
        history_error += history_flag
        duplicate_error += duplicate_flag
        candidate_error += candidate_flag
    
    print(f"total count: {len(predicted_items)}, history error: {history_error}, duplicate error: {duplicate_error}, candidate error: {candidate_error}.")
    print(f"history error rate: {history_error/len(predicted_items)}, duplicate error rate: {duplicate_error/len(predicted_items)}, candidate error rate: {candidate_error/len(predicted_items)}.")


def compute_metrics_on_explanations(explanation_file_path):
    explanations=pickle.load(open(explanation_file_path,"rb"))
    exp_list, truth_list = [], []
    for exp, truth in explanations:
        exp_list.append(exp)
        truth_list.append(truth)
    bleu=bleu_score(truth_list,exp_list)
    rouge=rouge_score(truth_list,exp_list)
    print("BLEU score: {}".format(bleu))
    print("ROUGE score: {}".format(rouge))

    return bleu, rouge


def compute_metrics_on_tags(tag_file_path):
    tags=pickle.load(open(tag_file_path,"rb"))
    overlap_list = []
    target = set()
    pred_top5 = set()
    pred_top10 = set()
    target_top5 = set()
    target_top10 = set()
    total = 0
    cnt = 0
    for tag, truth in tags:
        tag = tag.split(", ")
        total += len(tag)
        cnt += 1
        truth = truth.split(", ")
        overlap = set(tag) & set(truth)
        overlap_list.append(len(overlap))
        pred_top5 = pred_top5 | set(tag[:5])
        pred_top10 = pred_top10 | set(tag[:10])
        target = target | set(truth)
        target_top5 = target_top5 | set(truth[:5])
        target_top10 = target_top10 | set(truth[:10])

    print(f"everage tag num: {total/cnt}")
    print("#Overlap tags: {}".format(sum(overlap_list)/len(overlap_list)))
    print("#Coverage tags@5 : {}".format(len(pred_top5&target_top5)))
    print("#Coverage tags@10: {}".format(len(pred_top10&target_top10)))
    print("#Coverage tags@5 total truth  : {}".format(len(pred_top5&target)))
    print("#Coverage tags@10 total truth : {}".format(len(pred_top10&target)))

    return sum(overlap_list)/len(overlap_list)

def compute_diversity_on_retrieval(metadata_file_path, result_file_path, appr_index_file_path, k_list):
    all_metrics=dict()
    
    item_embeddings = [eval(line)['bert_rep'] for line in open(metadata_file_path)]
    item_embeddings = [item_embeddings[0]] + item_embeddings

    results=pickle.load(open(result_file_path,"rb"))[:1000] # list of rankings
    if isinstance(results[0], list) or isinstance(results[0], np.ndarray):
        predicted_items = results
    else:
        predicted_items = [result[0] for result in results]

    
    item_list = ['padding_item']
    for line in open(metadata_file_path):
        line = json.loads(line)
        if 'title' in line:
            item_list.append(line['title'])
        elif 'app_name' in line:
            item_list.append(line['app_name'])

    tfidf_model = TFIDF_model(item_list)

    # ILD and IC
    if not str(predicted_items[0][0]).isdigit(): # title_based need approximate index
        if not os.path.exists(appr_index_file_path):
            appr_predicted_results = [tfidf_model.map_index(x) for x in tqdm(predicted_items, desc="building appr. index results")]
            pickle.dump(appr_predicted_results, open(appr_index_file_path, "wb"))
        appr_predicted_items=pickle.load(open(appr_index_file_path, "rb"))[:1000]
        for topk in k_list:
            metrics = evaluate_ILD(item_embeddings, appr_predicted_items, topk)
            all_metrics={**all_metrics, **metrics}
            total_item_num = len(item_embeddings)
            metrics = evaluate_IC(appr_predicted_items, total_item_num, topk)
            all_metrics={**all_metrics, **metrics}
        print('calcing ILD and IC using approximate index')
    else: # id_based
        for topk in k_list:
            metrics = evaluate_ILD(item_embeddings, predicted_items, topk)
            all_metrics={**all_metrics, **metrics}
            total_item_num = len(item_embeddings)
            metrics = evaluate_IC(predicted_items, total_item_num, topk)
            all_metrics={**all_metrics, **metrics}
        print('calcing ILD and IC using id')

    msg = ""
    for topk in k_list:
        msg += f"ILD@{topk}\tIC@{topk}\n"
        msg += f"{all_metrics[f'ILD@{topk}']}\t{all_metrics[f'IC@{topk}']}\n"
    print(msg)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default="/home/jingyao/weixu/P5_title/output/backbone_t5-small_losses_item_import,retrieval,sequential,explanation,tagging_lr_0.001_bs_16_seed_2023_maxlen_512/")
    parser.add_argument("--data_dir", type=str, default="/home/jingyao/weixu/P5_title/data")

    parser.add_argument("--MF_embedding_file_path", type=str, default="/home/jingyao/projects/pretrain_recommendation/Baselines/model/RecModel/steam/embeddings.pkl")
    parser.add_argument("--MF_metadata_file_path", type=str, default="/home/jingyao/projects/pretrain_recommendation/RecoGPT/data/steam/metadata.json")
    
    ## ground truth

    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--use_gpu", action='store_true', default=False)
    parser.add_argument("--mode", type=str, default="title_retrieval,sequential,explanation,tagging")
    parser.add_argument("--valid_or_test", type=str, default="test")
    parser.add_argument("--string_match", type=str, default="TF-IDF")

    args = parser.parse_args()
    args.sequential_data_file_path = os.path.join(args.data_dir, "sequential_data.txt")
    args.metadata_file_path = os.path.join(args.data_dir, "metadata.json")
    args.datamaps_file_path = os.path.join(args.data_dir, "datamaps.json")
    args.negative_file_path = os.path.join(args.data_dir, "negative_samples.txt")
    args.available_item_path = os.path.join(args.data_dir, "available_items.pkl")

    args.emb_file_path = os.path.join(args.output_dir, "embeddings.pkl")
    args.retrieval_file_path = os.path.join(args.output_dir, "retrieval_results.pkl")
    if not os.path.exists(args.retrieval_file_path):
        args.retrieval_file_path = os.path.join(args.output_dir, "retrieval_id_results.pkl")
    args.appr_index_retrieval_file_path = os.path.join(args.output_dir, "appr_index_retrieval_results.pkl")
    args.ranking_file_path = os.path.join(args.output_dir, "ranking_results.pkl")
    if not os.path.exists(args.ranking_file_path):
        args.ranking_file_path = os.path.join(args.output_dir, "ranking_id_results.pkl")
    args.explanation_file_path = os.path.join(args.output_dir, "explanations.pkl")
    args.tag_file_path = os.path.join(args.output_dir, "tags.pkl")

    return args


if __name__=="__main__":
    args = get_args()
    args.mode=args.mode.split(",")
    print("args: ", args)

    if "id_retrieval" in args.mode:
        print("evaluate id retrieval...")
        if not os.path.exists(args.retrieval_file_path):
            from retriever import search_by_faiss
            item_embed, query_embed = pickle.load(open(args.emb_file_path, "rb"))
            query_embed = query_embed.astype('float32')
            item_embed = item_embed.astype('float32')
            search_by_faiss(query_embed, item_embed, args.retrieval_file_path, depth=args.depth, use_gpu=args.use_gpu)
        
        metrics = compute_metrics_on_id_recommend(args.retrieval_file_path, args.sequential_data_file_path, \
                                                     args.metadata_file_path, args.valid_or_test, [5, 10, 20])


    if "title_retrieval" in args.mode:
        print("evaluate title retrieval...")
        metrics = compute_metrics_on_title_recommend(args.retrieval_file_path, args.sequential_data_file_path, \
                                                     args.metadata_file_path, args.valid_or_test, [5, 10, 20], args.string_match)
        
    if 'id_ranking' in args.mode:
        print("evaluate id ranking recommendation...")
        metrics = compute_metrics_on_id_recommend(args.ranking_file_path, args.sequential_data_file_path, \
                                                     args.metadata_file_path, args.valid_or_test, [1, 5, 10, 20])
    
    if "title_ranking" in args.mode:
        print("evaluate title ranking recommendation...")
        metrics = compute_metrics_on_title_recommend(args.ranking_file_path, args.sequential_data_file_path, \
                                                     args.metadata_file_path, args.valid_or_test, [1, 5, 10, 20], args.string_match)
    
    if "explanation" in args.mode:
        print("evaluate explanation...")
        metrics = compute_metrics_on_explanations(args.explanation_file_path)

    if "tagging" in args.mode:
        print("evaluate tagging...")
        metrics = compute_metrics_on_tags(args.tag_file_path)

    if "retrieval_error" in args.mode:
        print("evaluate retrieval error...")
        compute_errors_on_title_retrieval(args.retrieval_file_path, args.available_item_path, \
                                          args.sequential_data_file_path, args.metadata_file_path, args.valid_or_test)
    
    if "ranking_error" in args.mode:
        print("evaluate ranking error...")
        compute_errors_on_title_ranking(args.ranking_file_path, args.sequential_data_file_path, \
                                        args.metadata_file_path, args.negative_file_path, args.valid_or_test)

    if "diversity" in args.mode:
        print("evaluate diversity...")

        compute_diversity_on_retrieval(
            args.metadata_file_path,
            args.retrieval_file_path, 
            args.appr_index_retrieval_file_path,
            [5, 10, 20])
