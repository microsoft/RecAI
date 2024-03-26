# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import pickle
import gzip
from tqdm import tqdm
import numpy as np
import random
import argparse

import re
from typing import List, Tuple
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _clean_string(string: str) -> str:
    """ Only keep alphanumerical characters """
    string = re.sub(r'[^A-Za-z0-9 ]+', '', string.lower())
    string = re.sub('\s+', ' ', string).strip()
    return string


class TFIDF_model:
    def _create_ngrams(self, string: str) -> List[str]:
        """ Create n_grams from a string

        Steps:
            * Extract character-level ngrams with `self.n_gram_range` (both ends inclusive)
            * Remove n-grams that have a whitespace in them
        """
        string = _clean_string(string)

        result = []
        for n in range(2, 4):
            ngrams = zip(*[string[i:] for i in range(n)])
            ngrams = [''.join(ngram) for ngram in ngrams]
            result.extend(ngrams)

        return result
    
    def __init__(self, corpus):
        self.vectorizer = TfidfVectorizer(min_df=1, analyzer=self._create_ngrams).fit(corpus)

    def match(self, from_docs, to_docs):
        froms = self.vectorizer.transform(from_docs)
        tos = self.vectorizer.transform(to_docs)
        cosine_sim = cosine_similarity(froms, tos)
        return cosine_sim
    
    def map_index(self, from_docs, to_docs):
        score = self.match(from_docs, to_docs)
        index_list = []
        for x in score:
            max_score = -999
            max_index = 0
            for index, y in enumerate(x):
                if y > max_score:
                    max_score = y
                    max_index = index
            index_list.append(max_index)
        return index_list

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def ReadLineFromFile(path):
    lines = []
    with open(path,'r',encoding="utf8") as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def parse(path):
    if path.endswith('gz'):
        g = gzip.open(path, 'r',encoding="utf8")
    else:
        g = open(path, 'r',encoding="utf8")
    nan_default = {'NaN': "", 'false': "", 'true': ""}
    for l in g:
        yield eval(l, nan_default)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default='../out/result/steam/')
    parser.add_argument('--dataset', type=str, default='steam')
    args = parser.parse_args()

    title_list = ['padding_title']
    title_2_id = {}
    for line in parse(f"../data/{args.dataset}/metadata.json"):
        title = "No title"
        if "app_name" in line:
            title = line['app_name']
        elif "title" in line:
            title = line['title']
            tmp = title.split('(', 1)[0].split(',')
            if '(' in title:
                others = ' (' + title.split('(', 1)[1]
            else:
                others = ''
            if len(tmp) == 2 and tmp[1].strip() == "The":
                title = "The " + tmp[0].strip() + others
        else:
            if "description" in line:
                title = line['description'][:50]
            elif "categories" in line:
                title = line['categories'][0][-1]
        title_list.append(title)

    tfidf = TFIDF_model(title_list)

    candidate_lists = []
    history_lists = []
    for seq_line, neg_line in zip(ReadLineFromFile(f"../data/{args.dataset}/sequential_data.txt"), ReadLineFromFile(f"../data/{args.dataset}/negative_samples_pop.txt")):
        _, seq_ids = seq_line.strip().split(' ', 1)
        _, neg_ids = neg_line.strip().split(' ', 1)
        seq_ids = [int(x) for x in seq_ids.split(' ')]
        neg_ids = [int(x) for x in neg_ids.split(' ')]
        candidate_list = neg_ids[:19] + [seq_ids[-1]]
        random.shuffle(candidate_list)
        candidate_lists.append(candidate_list)
        history_lists.append(seq_ids[:-1])
        
    candidate_titles = [[title_list[x] for x in l] for l in candidate_lists]
    history_titles = [[title_list[x] for x in l] for l in history_lists]

    ranking_result = load_pickle(f"{args.result_path}/ranking_results.pkl")
    candidate_error_cnt_list = []
    history_error_cnt_list = []
    for i, result in enumerate(tqdm(ranking_result[:1000])):
        result_titles = result[0]
        candidate_error_cnt = 0
        history_error_cnt = 0
        for title in result_titles:
            match_scores = tfidf.match([title], candidate_titles[i])[0]
            if max(match_scores) < 0.6:
                # print(f"Candidate ERROR: \"{title}\" is not found in candidate_titles.")
                candidate_error_cnt += 1
            match_scores = tfidf.match([title], history_titles[i])[0]
            for idx, score in enumerate(match_scores):
                if score >= 0.8:
                    # print(f"History ERROR: \"{title}\" is found in history. {history_titles[i][idx]}")
                    history_error_cnt += 1
                    break
        
        candidate_error_cnt_list.append(candidate_error_cnt)
        history_error_cnt_list.append(history_error_cnt)

    print(f"Candidate Errors: ", Counter(candidate_error_cnt_list))
    print(f"History Errors: ", Counter(history_error_cnt_list))

    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_title("candidate_error_cnt")
    ax2.set_title("history_error_cnt")
    a = np.arange(0, 1, 20, dtype=None)
    sns.histplot(data=candidate_error_cnt_list, bins=20, binrange=(0,20), ax=ax1)
    sns.histplot(data=history_error_cnt_list, bins=20, binrange=(0,20), ax=ax2)
    fig.savefig(f"{args.result_path}/error.png")