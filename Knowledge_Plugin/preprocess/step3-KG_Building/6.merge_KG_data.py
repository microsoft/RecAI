'''
Merge the triplets of the raw dataset and the triplets of the external knowledge graph.
'''
import os
import json
import pickle
import argparse
from tqdm import tqdm
import networkx as nx
from collections import defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--kgdata_path', type=str, default="../data/KG_data/")
    parser.add_argument('--recdata_path', type=str, default="../data/ml1m/")
    args = parser.parse_args()

    entity_freq = defaultdict(int)
    relation_freq = defaultdict(int)
    entity2id = {}
    relation2id = {}
    entity2type = pickle.load(open(os.path.join(args.recdata_path, "entity2type.pkl"), "rb"))
    with open(os.path.join(args.recdata_path, "entity2id.txt"), "r") as fr:
        for line in tqdm(fr, desc="loading entity2id"):
            line = line.rstrip().split('\t')
            if len(line) < 3:
                continue
            entity2id[line[0]] = int(line[1])

    with open(os.path.join(args.recdata_path, "relation2id.txt"), "r") as fr:
        for line in tqdm(fr, desc="loading relation2id"):
            line = line.rstrip().split('\t')
            if len(line) < 2:
                continue
            relation2id[line[0]] = int(line[1])

    triplets = set()
    with open(os.path.join(args.recdata_path, "triplets.tsv"), "r") as fr:
        for line in tqdm(fr, desc="loading dataset triplets"):
            head, relation, tail = line.strip("\n").split('\t')
            # print(head + ";" + relation + ";" + tail)
            assert head in entity2id, "head error: " + head
            assert tail in entity2id, "tail error: " + tail
            assert relation in relation2id, "relation error: " + relation
            triplets.add((head, relation, tail))
            entity_freq[tail] += 1
            relation_freq[relation] += 1
    
    pruned_relations = set()
    with open(os.path.join(args.recdata_path, "prune_incorporate_relations.txt"), "r") as fr:
        for line in fr:
            pruned_relations.add(line.split('\t')[0])
    
    steam_relation_mapping = {"genre": "HasGenre", "developer": "HasDeveloper", "publisher": "HasPublisher"}
    ml1m_relation_mapping = {"genre": "HasGenre"}
    beauty_relation_mapping = {"brand": "ProducedBy"}
    
    incorporate_triplets = set()
    with open(os.path.join(args.recdata_path, "incorporate_triplets.tsv"), "r") as fr:
        for line in tqdm(fr, desc="loading incorporated triplets"):
            head, relation, tail = line.rstrip("\n").split('\t')
            if relation not in pruned_relations:
                continue

            if "steam" in args.recdata_path and relation in steam_relation_mapping:
                relation = steam_relation_mapping[relation]
            if "ml1m" in args.recdata_path and relation in ml1m_relation_mapping:
                relation = ml1m_relation_mapping[relation]
            if "beauty" in args.recdata_path and relation in beauty_relation_mapping:
                relation = beauty_relation_mapping[relation]
            if (head, relation, tail) in triplets:
                continue

            if head not in entity2id:
                entity2id[head] = len(entity2id)
                entity2type[head] = "Entity"
            if tail not in entity2id:
                entity2id[tail] = len(entity2id)
                entity2type[tail] = "Entity"
            if relation not in relation2id:
                relation2id[relation] = len(relation2id)
            incorporate_triplets.add((head, relation, tail))
            entity_freq[tail] += 1
            relation_freq[relation] += 1
    print("incorporate triplets: ", len(incorporate_triplets))

    sorted_entity_freq = sorted(entity_freq.items(), key=lambda x: x[1], reverse=True)
    print(sorted_entity_freq[:100])

    triplets = triplets.union(incorporate_triplets)
    with open(os.path.join(args.recdata_path, "merged_triplets.tsv"), "w") as fw:
        for head, relation, tail in tqdm(triplets, desc="writing  triplets"):
            if entity_freq[head] > 2000 and entity2type[head] not in ("Game", "Movie", "Product", "User"): 
                continue
            if entity_freq[tail] > 2000 and entity2type[tail] not in ("Game", "Movie", "Product", "User"):
                continue
            fw.write("{}\t{}\t{}\n".format(head, relation, tail))

    with open(os.path.join(args.recdata_path, "merged_entity2id.txt"), "w") as fw:
        fw.write(f"{len(entity2id)}\n")
        idx = 0
        for entity, id in entity2id.items():
            if entity_freq[entity] > 2000 and entity2type[entity] == "Entity": 
                continue
            fw.write(f"{entity}\t{idx}\t{entity2type[entity]}\n")
            idx += 1

    with open(os.path.join(args.recdata_path, "merged_relation2id.txt"), "w") as fw:
        fw.write(f"{len(relation2id)}\n")
        for relation, id in relation2id.items():
            fw.write(f"{relation}\t{id}\n")

    pickle.dump(entity2type, open(os.path.join(args.recdata_path, "merged_entity2type.pkl"), "wb"))