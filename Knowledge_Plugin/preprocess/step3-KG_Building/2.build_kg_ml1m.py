'''
Build a knowledge graph based on the MovieLen-1M dataset.
Entity Types: Movie, Genre
Relation Types: HasGenre, GenreOf
'''
import os
import json
import pickle
from tqdm import tqdm
import networkx as nx

def entity_linking(mentions, kgdata_path):
    entity_2_description = {}
    for line in tqdm(open(os.path.join(kgdata_path, "description.txt")), desc="loading description"):
        line = line.strip().split("\t")
        entity = line[0].strip()
        description = line[1].strip()
        entity_2_description[entity] = description

if __name__ == "__main__":
    meta_data_file = "../data/ml1m/metadata.json"
    interaction_file = "../data/ml1m/sequential_data.txt"

    G = nx.DiGraph()
    metadata = ["padding"]
    triples = set()
    entity2id = {}
    relation2id = {}
    entity2type = {}
    item2entity = {}
    with open(meta_data_file, "r") as fr:
        for line in tqdm(fr, desc="loading metadata"):
            line = json.loads(line)
            metadata.append(line)
            if "title" not in line:
                line["title"] = "No Title"
            line["title"] = '{} ({})'.format(line["title"], line["year"])
            if line["title"] not in entity2id:
                entity2id[line["title"]] = len(entity2id)
                entity2type[line["title"]] = "Movie"

            if "genre" in line:
                for genre in line["genre"].split(", "):
                    if genre not in entity2id:
                        entity2id[genre] = len(entity2id)
                        entity2type[genre] = "Genre"
                    if entity2type[genre] == "Genre":
                        G.add_edge(line["title"], genre, relation="HasGenre")
                        G.add_edge(genre, line["title"], relation="GenreOf")

    with open(interaction_file, "r") as fr:
        for line in tqdm(fr, desc="loading interaction"):
            line = line.strip().split(" ")
            for item in line[1:]:
                if item not in item2entity:
                    item2entity[item] = metadata[int(item)]["title"]

    print("Number of nodes: ", G.number_of_nodes())
    print("Number of edges: ", G.number_of_edges())

    os.makedirs("../data/ml1m", exist_ok=True)
    with open("../data/ml1m/triplets.tsv", "w") as fw:
        for edge in G.edges:
            relation = G.edges[edge]["relation"]
            if relation not in relation2id:
                relation2id[relation] = len(relation2id)
            fw.write(f"{edge[0]}\t{relation}\t{edge[1]}\n")
            # fw.write(f"{entity2id[edge[0]]}\t{relation2id[relation]}\t{entity2id[edge[1]]}\n")
    
    with open("../data/ml1m/entity2id.txt", "w") as fw:
        fw.write(f"{len(entity2id)}\n")
        for entity, id in entity2id.items():
            fw.write(f"{entity}\t{id}\t{entity2type[entity]}\n")
    
    with open("../data/ml1m/relation2id.txt", "w") as fw:
        fw.write(f"{len(relation2id)}\n")
        for relation, id in relation2id.items():
            fw.write(f"{relation}\t{id}\n")

    pickle.dump(entity2type, open("../data/ml1m/entity2type.pkl", "wb"))
    pickle.dump(item2entity, open("../data/ml1m/item2entity.pkl", "wb"))