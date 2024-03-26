# process the knowledge graph wikidata

import re
import os
import json
import gzip
import argparse
import pickle
from tqdm import tqdm
from typing import List, Tuple
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from multiprocessing import Process, JoinableQueue, Lock

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
        for n in range(3, 4):
            ngrams = zip(*[string[i:] for i in range(n)])
            # ngrams = [''.join(ngram) for ngram in ngrams if ' ' not in ngram]
            ngrams = [''.join(ngram) for ngram in ngrams]
            result.extend(ngrams)

        return result
    
    def __init__(self, corpus):
        self.corpus = corpus
        self.vectorizer = TfidfVectorizer(min_df=1, analyzer=self._create_ngrams).fit(corpus)

    def match(self, from_docs, to_docs):
        froms = self.vectorizer.transform(from_docs)
        tos = self.vectorizer.transform(to_docs)
        cosine_sim = cosine_similarity(froms, tos)
        return cosine_sim

    def map_index(self, docs):
        score = self.match(docs, self.corpus)
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
        
    def retrieve(self, doc, limit):
        score = self.match([doc], self.corpus)
        results = sorted([(x,y) for x,y in zip(score[0], self.corpus) if x>=0.6], key=lambda x:-x[0])
        # return [x[1] for x in results[:limit]]
        return results[:limit]

def producer(queue:JoinableQueue, recdata_path):
    mention_path = os.path.join(recdata_path, "entity2id.txt")
    with open(mention_path, "r") as fr:
        for index, line in enumerate(tqdm(fr, desc="mapping mention.")):
            line = line.rstrip().split("\t")
            if len(line) < 2:
                continue
            if line[0].startswith("User") or (len(line) == 3 and line[2] == "Term"):
                continue
            id = int(line[1])
            title = line[0]
            queue.put((id, title))

def process_title(i, tfidf, queue:JoinableQueue ,lock, args):
    linkentity_f = open(os.path.join(args.output_path, "linkentities.txt"), "a", encoding="utf-8")
    while True:
        try:
            itemid, title = queue.get()
            retrieved_labels = tfidf.retrieve(title, 50)
            result = []
            for score, label in retrieved_labels:
                for entity in label_2_entities[label]:
                    result.append((itemid, entity, f"{score:.4f}", f"{title}->{label}: {entity_2_description[entity]}"))
            lock.acquire()
            for x in result:
                linkentity_f.write(f"{x[0]}\t{x[1]}\t{x[2]}\t{x[3]}\n")
                linkentity_f.flush()
            lock.release()
        except:
            pass
        finally:
            queue.task_done()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--kgdata_path', type=str, default="../data/KG_data/")
    parser.add_argument('--recdata_path', type=str, default="../data/ml1m/")
    parser.add_argument('--output_path', type=str, default="../data/ml1m")
    parser.add_argument('--num_works', type=int, default=18)
    args = parser.parse_args()
    entity_2_labels = {}
    entity_2_description = {}
    entity_2_instance = defaultdict(list)
    candidate_entities = set()
    label_2_entities = defaultdict(list)
    
    for line in tqdm(open(os.path.join(args.kgdata_path, "label.txt")), desc="loading label"):
        line = line.strip().split("\t")
        entity = line[0].strip()
        labels = "\t".join(line[1:])
        entity_2_labels[entity] = labels
    
    for line in tqdm(open(os.path.join(args.kgdata_path, "description.txt")), desc="loading description"):
        line = line.split("\t")
        entity = line[0].strip()
        description = line[1].strip()
        entity_2_description[entity] = description

    not_found = 0
    for line in tqdm(open(os.path.join(args.kgdata_path, "triplets.txt")), desc="loading triple"):
        head, relation, tail = line.strip().split('\t')
        if relation == "P31":
            try:
                entity_2_instance[head].append(entity_2_labels[tail])
            except:
                not_found += 1
    print("#Not found: ", not_found)

    for entity in tqdm(entity_2_description, desc="search candidate entities."):
        if "ml" in args.recdata_path:
            if "film" in entity_2_description[entity] or "movie" in entity_2_description[entity]:
                candidate_entities.add(entity)
            elif "film" in ", ".join(entity_2_instance[entity]) or "movie" in ", ".join(entity_2_instance[entity]):
                candidate_entities.add(entity)
        elif "beauty" in args.recdata_path:
            if "cosmetics" in entity_2_description[entity] or "product" in entity_2_description[entity] or "company" in entity_2_description[entity]:
                candidate_entities.add(entity)
            elif "cosmetics" in ", ".join(entity_2_instance[entity]) or "product" in ", ".join(entity_2_instance[entity]) or "company" in ", ".join(entity_2_instance[entity]) or "business" in ", ".join(entity_2_instance[entity]):
                candidate_entities.add(entity)

    one_hop_candidates = set()
    for line in tqdm(open(os.path.join(args.kgdata_path, "triplets.txt")), desc="search candidate entities."):
        head, relation, tail = line.strip().split('\t')
        if "ml" in args.recdata_path:
            if head in candidate_entities and ((tail in entity_2_description and ("film" in entity_2_description[tail] or "movie" in entity_2_description[tail])) or (tail in entity_2_instance and ("film" in ", ".join(entity_2_instance[tail]) or "movie" in ", ".join(entity_2_instance[tail])))):
                one_hop_candidates.add(tail)
        elif "beauty" in args.recdata_path:
            if head in candidate_entities and ((tail in entity_2_description and ("cosmetics" in entity_2_description[tail] or "product" in entity_2_description[tail] or "company" in entity_2_description[tail])) or (tail in entity_2_instance and ("cosmetics" in ", ".join(entity_2_instance[tail]) or "product" in ", ".join(entity_2_instance[tail]) or "company" in ", ".join(entity_2_instance[tail]) or "business" in ", ".join(entity_2_instance[tail])))):
                one_hop_candidates.add(tail)
    candidate_entities = candidate_entities.union(one_hop_candidates)

    for entity in tqdm(candidate_entities, desc="get candidate entity labels."):
        if entity in entity_2_labels:
            labels = entity_2_labels[entity]
            for label in labels.split("\t"):
                label_2_entities[label].append(entity)

    print(len(label_2_entities.keys()))
    tfidf = TFIDF_model(label_2_entities.keys())

    linkentity_f = open(os.path.join(args.output_path, "linkentities.txt"), "w", encoding="utf-8")
    linkentity_f.close()

    write_lock = Lock()

    queue = JoinableQueue(100)
    pc = Process(target=producer, args=(queue,args.recdata_path,))
    pc.start()
    
    workercount = args.num_works
    for i in range(workercount):
        worker = Process(target=process_title, args=(i, tfidf, queue, write_lock, args,))
        worker.daemon = True
        worker.start()
    pc.join()
    queue.join()