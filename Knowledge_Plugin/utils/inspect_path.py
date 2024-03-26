import pickle
import os
import json
import argparse

def build_entity_relation_labels():
    print("build entity labels...")
    entity_relation_label = {}
    with open("data/kg/prune_steam/label.txt") as fr:
        for line in fr:
            entity, label = line.strip().split("\t")
            entity_relation_label[entity] = label
    
    return entity_relation_label

def build_ent_rel_2id(args): # entity_id_dict {Pxxx: 1}, relation_id_dict {Qxxx: 1}, id start from 1
    entity_id_dict = {}
    for line in open(f"../data/{args.dataset}/onlyitem_entity2id.txt"):
        if len(line.strip().split()) == 1:
            continue
        linesplit = line.split('\n')[0].split('\t')
        entity_id_dict[linesplit[0]] = int(linesplit[1])+1 # 0 for padding

    relation_id_dict = {}
    for line in open(f"../data/{args.dataset}/onlyitem_relation2id.txt"):
        if len(line.strip().split()) == 1:
            continue
        linesplit = line.split('\n')[0].split('\t')
        relation_id_dict[linesplit[0]] = int(linesplit[1])+1 # 0 for padding
    
    return entity_id_dict, relation_id_dict #note that the id starts from 1

parser = argparse.ArgumentParser(description='data processing')
parser.add_argument('-p', '--path', default="data/kprn/train_data.json", type=str)
parser.add_argument('-d', '--dataset', default="steam", type=str)
args = parser.parse_args()

entity_id_dict, relation_id_dict = build_ent_rel_2id(args)
id_entity_dict = {v: k for k, v in entity_id_dict.items()}
id_relation_dict = {v: k for k, v in relation_id_dict.items()}
# item2entity = pickle.load(open("data/kg/prune_steam/item2entity.pkl", "rb")) # {itemid: entity}
# entity_relation_label = build_entity_relation_labels()

count = 0
for line in open(args.path):
    if count > 100:
        break
    data = json.loads(line.strip())
    item1, item2, label, paths, edges = data['item1'], data['item2'], data['label'], data['paths'], data['edges']
    print(item1, item2, label)
    if paths:
        if isinstance(paths[0], list):
            for path, edge in zip(paths, edges):
                for node, relation in zip(path[:-1], edge):
                    print(f"{id_entity_dict[node]}->({id_relation_dict[relation]})->", end="")
                print(id_entity_dict[path[-1]])
        else:
            for node, relation in zip(paths[:-1], edges):
                print(f"{id_entity_dict[node]}->({id_relation_dict[relation]})->", end="")
            print(id_entity_dict[paths[-1]])

        count += 1
    print("\n================================================\n")