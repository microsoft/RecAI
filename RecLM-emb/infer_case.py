# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
import json
import pickle
import argparse
from tqdm import tqdm
import random

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from preprocess.utils import get_item_text
from src.huggingface_model_infer import run_model_embedding

def gen_retrieval_result(item_embedding_prompt_path, answer_file, topk, item_embedding_path, user_embedding_path):
    itemid2title = []
    with open(item_embedding_prompt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line.strip())
            itemid2title.append(line['title'])

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    fd = open(answer_file, "w", encoding='utf-8')

    item_embeddings = torch.tensor(pickle.load(open(item_embedding_path, "rb")))
    user_embeddings = torch.tensor(pickle.load(open(user_embedding_path, "rb")))
    print("shape of item embeddings: ", item_embeddings.shape)
    print("shape of user embeddings: ", user_embeddings.shape)

    for user_emb in user_embeddings:
        scores = torch.softmax(torch.matmul(user_emb, item_embeddings.T), -1).squeeze().tolist()
        scores = [(index, score) for index, score in enumerate(scores) if index!=0]
        top_itemids = sorted(scores, key=lambda x:-x[1])[:topk]
        data = {
            "result": [(x[0], itemid2title[x[0]][1]) for x in top_itemids]
        }
        fd.write(json.dumps(data, ensure_ascii=False)+'\n')
    fd.close()

def parse_args():
    parser = argparse.ArgumentParser(description="infer case")
    parser.add_argument(
        "--in_meta_data", type=str, help=""
    )
    parser.add_argument(
        "--user_embedding_prompt_path", type=str, help="Path to query file"
    )
    parser.add_argument(
        "--model_path_or_name", type=str, help="Path to pretrained model or model identifier from huggingface.co/models"
    )
    parser.add_argument(
        "--answer_file", type=str, help=""
    )
    parser.add_argument(
        "--topk", type=int, default=10, help=""
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


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator()

    ## get the dir of args.user_embedding_prompt_path
    cache_dir =  os.path.dirname(args.answer_file)
    item_embedding_prompt_path = os.path.join(cache_dir, 'item_embedding_prompt.jsonl')
    item_embedding_path = os.path.join(cache_dir, 'item_embedding.pkl')
    user_embedding_path = os.path.join(cache_dir, 'user_embedding.pkl')

    if accelerator.is_main_process:
        os.makedirs(cache_dir, exist_ok=True)
        get_item_text(args.in_meta_data, save_item_prompt_path=item_embedding_prompt_path)
    accelerator.wait_for_everyone()

    print("infer item embedding")
    run_model_embedding(args.model_path_or_name, max_seq_len=args.passage_max_len, batch_size=args.per_device_eval_batch_size, prompt_path=item_embedding_prompt_path, emb_path=item_embedding_path, accelerator=accelerator, args=args, qorp='passage')

    # if accelerator.is_main_process:
    #     gen_user_embedding_prompt(args.item_embedding_prompt_path, args.user_embedding_prompt_path)
    # accelerator.wait_for_everyone()
    
    print("infer user embedding")
    run_model_embedding(args.model_path_or_name, max_seq_len=args.query_max_len, batch_size=args.per_device_eval_batch_size, prompt_path=args.user_embedding_prompt_path, emb_path=user_embedding_path, accelerator=accelerator, args=args, qorp='query')

    if accelerator.is_main_process:
        gen_retrieval_result(item_embedding_prompt_path, args.answer_file, args.topk, item_embedding_path, user_embedding_path)