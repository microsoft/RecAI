# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import faiss
import pickle
import logging
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

class BaseFaissIPRetriever:
    def __init__(self, reps_dim: int):
        index = faiss.IndexFlatIP(reps_dim)
        self.index = index

    def add(self, p_reps: np.ndarray):
        self.index.add(p_reps)

    def search(self, q_reps: np.ndarray, k: int):
        return self.index.search(q_reps, k)

    def batch_search(self, q_reps: np.ndarray, k: int, batch_size: int):
        num_query = q_reps.shape[0]
        all_scores = []
        all_indices = []
        for start_idx in tqdm(range(0, num_query, batch_size), total=num_query // batch_size):
            nn_scores, nn_indices = self.search(q_reps[start_idx: start_idx + batch_size], k)
            all_scores.append(nn_scores)
            all_indices.append(nn_indices)
        all_scores = np.concatenate(all_scores, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)

        return all_scores, all_indices

def search_queries(retriever, q_reps, depth, batch_size):
    if batch_size > 0:
        all_scores, all_indices = retriever.batch_search(q_reps, depth, batch_size)
    else:
        all_scores, all_indices = retriever.search(q_reps, depth)

    return all_scores, all_indices

def write_ranking(corpus_indices, corpus_scores, q_lookup, ranking_save_file):
    with open(ranking_save_file, 'w') as f:
        for qid, q_doc_scores, q_doc_indices in zip(q_lookup, corpus_scores, corpus_indices):
            score_list = [(s, idx) for s, idx in zip(q_doc_scores, q_doc_indices)]
            score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
            rank = 1
            for s, idx in score_list:
                f.write(f'{qid}\t{idx}\t{rank}\t{s}\n')
                rank += 1

def search_by_faiss(query_emb, item_emb, save_file_path, batch_size=512, depth=1000, use_gpu=False):
    print("shape of query embeddings: ", np.shape(query_emb))
    print("shape of item embeddings: ", np.shape(item_emb))

    retriever = BaseFaissIPRetriever(np.shape(item_emb)[-1])

    faiss.omp_set_num_threads(64)
    if use_gpu:
        print('use GPU for Faiss')
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        retriever.index = faiss.index_cpu_to_all_gpus(
            retriever.index,
            co=co
        )
    retriever.add(item_emb)

    logger.info('Index Search Start')
    all_scores, all_indices = search_queries(retriever, query_emb, depth, batch_size)
    logger.info('Index Search Finished')
    # print("all_scores: ", all_scores, "all_indices: ", all_indices, all_indices.shape)

    pickle.dump(all_indices, open(save_file_path, "wb"))