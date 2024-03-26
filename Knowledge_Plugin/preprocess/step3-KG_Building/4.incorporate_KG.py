import os
import json
import pickle
import argparse
from tqdm import tqdm
import networkx as nx
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--kgdata_path', type=str, default="../data/KG_data/")
    parser.add_argument('--recdata_path', type=str, default="../data/ml1m/")
    args = parser.parse_args()

    # loading external graph
    if not os.path.exists(os.path.join(args.kgdata_path, "all_graph.pkl")):
        G = nx.DiGraph()
        for line in tqdm(open(os.path.join(args.kgdata_path, "triplets.txt")), desc="loading kg triple"):
            h, r, t = line.strip().split()
            G.add_edge(h, t)
            if "r" not in G[h][t]:
                G[h][t]["r"] = []
            G[h][t]["r"].append(r)
    
        pickle.dump(G, open(os.path.join(args.kgdata_path, "all_graph.pkl"), "wb"))
        exit()
    else:
        print("load kg graph...")
        G = pickle.load(open(os.path.join(args.kgdata_path, "all_graph.pkl"), "rb"))
    
    entity_2_id, id_2_entity = {}, {}
    entity_2_type = pickle.load(open(os.path.join(args.recdata_path, "entity2type.pkl"), "rb"))
    for line in tqdm(open(os.path.join(args.recdata_path, "entity2id.txt")), desc="loading entity2id"):
        if len(line.strip().split('\t')) < 3:
            continue
        entity, entity_id, entity_type = line.split('\t')
        entity_2_id[entity] = entity_id
        id_2_entity[entity_id] = entity

    total_linked_entities = set()
    linked_entities = {}
    linked_entities_2_id = {}
    for line in tqdm(open(os.path.join(args.recdata_path, "linkentities.txt")), desc="loading linked entities"):
        entityid, entity, score, desc = line.strip().split('\t')
        total_linked_entities.add(entityid)
        entity_type = entity_2_type[id_2_entity[entityid]]
        if ("Genre" in entity_type or "Tag" in entity_type) and "genre" not in desc.lower():
            continue
        if "Movie" in entity_type and id_2_entity[entityid][-5:-1] not in desc.split("->")[1]:
            continue
        if "Brand" in entity_type and "business" not in desc.lower() and "company" not in desc.lower():
            continue
        if "Category" in entity_type and "category" not in desc.lower() and "product" not in desc.lower():
            continue
        if (entityid not in linked_entities) and float(score) > 0.65:   
            linked_entities[entityid] = entity
            linked_entities_2_id[entity] = entityid
            # print(line)
    print("#Valid Linked Entities: ", len(linked_entities), " #Total Linked Entities: ", len(total_linked_entities))

    sources = list(set(linked_entities.values()))
    sources = [x for x in sources if x in G]

    def incorporate_2hop_nodes(sources):
        nodes_1hop = set()
        nodes_2hop = set()
        print("#Item Entity: ", len(sources))
        ##########################################################################################
        for source in tqdm(sources, desc="mining 1 hop nodes"):
            if source not in G:
                continue
            nodes_1hop |= set(G.successors(source))
        print("#1-hop Neighbors: ", len(nodes_1hop))
        ratios = []
        for node in tqdm(nodes_1hop, desc="filtering 1 hop nodes"):
            neighbors = list(G.predecessors(node))
            ratio = len(set(neighbors) & set(sources)) / len(neighbors)
            ratios.append((node, ratio))
        ratios = sorted(ratios, key=lambda x:-x[1])
        nodes_1hop = set([x[0] for x in ratios[:170000]])
        ##########################################################################################
        for node in tqdm(nodes_1hop, desc="mining 2 hop nodes"):
            nodes_2hop |= set(G.successors(node))
        ##########################################################################################
        print("#2-hop Neighbors: ", len(nodes_2hop))
        triples = set()
        entity2id = {}
        relation2id = {}
        entity_2_labels = {}
        
        for line in tqdm(open(os.path.join(args.kgdata_path, "label.txt")), desc="loading label"):
            line = line.strip().split("\t")
            entity = line[0].strip()
            entity_2_labels[entity] = line[1:]
        
        for node in tqdm(set(sources), desc="add triple 0-1"):
            for neighbor in set(list(G.successors(node))) & nodes_1hop:
                for relation in G[node][neighbor]['r']:
                    triples.add((node, relation, neighbor))
        for node in tqdm(nodes_1hop, desc="add triple 1-2"):
            for neighbor in set(list(G.successors(node))) & nodes_2hop:
                for relation in G[node][neighbor]['r']:
                    triples.add((node, relation, neighbor))

        with open(os.path.join(args.recdata_path, "incorporate_triplets.tsv"), "w") as f:
            for node, relation, neighbor in tqdm(triples, desc="writing incorporated triplets"):
                if relation not in relation2id:
                    relation2id[relation] = len(relation2id)
                if node in sources and neighbor in sources:
                    continue
                if node not in entity_2_labels or neighbor not in entity_2_labels:
                    continue
                if node in sources:
                    f.write(f"{id_2_entity[linked_entities_2_id[node]]}\t{entity_2_labels[relation][0]}\t{entity_2_labels[neighbor][0]}\n")
                elif neighbor in sources:
                    f.write(f"{entity_2_labels[node][0]}\t{entity_2_labels[relation][0]}\t{id_2_entity[linked_entities_2_id[neighbor]]}\n")
                else:
                    f.write(f"{entity_2_labels[node][0]}\t{entity_2_labels[relation][0]}\t{entity_2_labels[neighbor][0]}\n")

        with open(os.path.join(args.recdata_path, "incorporate_relations.txt"), "w") as f:
            for relation in tqdm(relation2id, desc="writing incorporated relations"):
                f.write(f"{relation}\t{entity_2_labels[relation][0]}\n")

    incorporate_2hop_nodes(sources)