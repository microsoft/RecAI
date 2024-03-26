'''
Build a knowledge graph based on the 5-core Amazon Beauty (or others) dataset.
Entity Types: Item, Feature, Brand, Category
Relation Types: Mention, Described_by, Belong_to, Producted_by, Also_bought, Also_viewed, Bought_together
'''
import os
import json
import pickle
import numpy as np
from tqdm import tqdm
import networkx as nx
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfTransformer

def entity_linking(mentions, kgdata_path):
    entity_2_description = {}
    for line in tqdm(open(os.path.join(kgdata_path, "description.txt")), desc="loading description"):
        line = line.strip().split("\t")
        entity = line[0].strip()
        description = line[1].strip()
        entity_2_description[entity] = description

def compute_tfidf_fast(texts, vocab):
    tf_matrix = np.zeros((len(texts), len(vocab)))
    for idx, (item, text) in tqdm(enumerate(texts), desc="computing tfidf"):
        for word in text.split():
            if word in vocab:
                tf_matrix[idx, vocab[word]] += 1
    
    print("tf_matrix: ", tf_matrix.shape)
    transformer = TfidfTransformer(smooth_idf=True)
    tfidf = transformer.fit_transform(tf_matrix).toarray()
    print("tfidf: ", tfidf.shape)
    return tfidf


if __name__ == "__main__":
    meta_data_file = "../data/beauty/metadata.json"
    interaction_file = "../data/beauty/sequential_data.txt"
    # review_file = "../data/beauty/reviews_Beauty.pickle"
    review_file = "../data/beauty/review_splits.pkl"
    raw_meta_file = "../data/beauty/meta_Beauty.json"
    datamap_file = "../data/beauty/datamaps.json"

    G = nx.DiGraph()
    metadata = ["padding"]
    asin2id = {}
    triples = set()
    entity2id = {}
    relation2id = {}
    entity2type = {}
    item2entity = {}

    # loading metadata, add entity item, brand, category
    with open(meta_data_file, "r") as fr:
        for line in tqdm(fr, desc="loading metadata"):
            line = json.loads(line)
            metadata.append(line)
            asin2id[line["asin"]] = len(metadata) - 1

            if "title" not in line:
                if "description" in line:
                    line["title"] = line["description"][:50]
                else:
                    line["title"] = line["asin"]
            line["title"] = line["title"].replace("\n", " ").replace("\t", " ").replace("\r", " ")
            if line["title"] not in entity2id:
                entity2id[line["title"]] = len(entity2id)
                entity2type[line["title"]] = "Product"

            if "categories" in line:
                for category in line["categories"]:
                    category = category[-1].replace("\t", " ")
                    if category not in entity2id:
                        entity2id[category] = len(entity2id)
                        entity2type[category] = "Category"
                    if entity2type[category] == "Category":
                        G.add_edge(line["title"], category, relation="BelongTo")
                        G.add_edge(category, line["title"], relation="CategoryOf")
            
            if "brand" in line:
                line["brand"] = line["brand"].replace("\t", " ")
                if line["brand"] not in entity2id:
                    entity2id[line["brand"]] = len(entity2id)
                    entity2type[line["brand"]] = "Brand"
                if entity2type[line["brand"]] == "Brand":
                    G.add_edge(line["title"], line["brand"], relation="ProducedBy")
                    G.add_edge(line["brand"], line["title"], relation="BrandOf")
                    
    with open(interaction_file, "r") as fr:
        for line in tqdm(fr, desc="loading interaction"):
            line = line.strip().split(" ")
            for item in line[1:]:
                if item not in item2entity:
                    item2entity[item] = metadata[int(item)]["title"]

    # add relations also_bought, also_viewed, bought_together
    with open(raw_meta_file, "r") as fr:
        for line in tqdm(fr, desc="loading raw metadata"):
            line = eval(line)
            if line["asin"] not in asin2id:
                continue
            if "related" in line:
                if "also_bought" in line["related"]:
                    for also_bought in line["related"]["also_bought"]:
                        if also_bought not in asin2id:
                            continue
                        G.add_edge(metadata[asin2id[line["asin"]]]["title"], metadata[asin2id[also_bought]]["title"], relation="AlsoBought")
                        G.add_edge(metadata[asin2id[also_bought]]["title"], metadata[asin2id[line["asin"]]]["title"], relation="AlsoBought")
                
                if "also_viewed" in line["related"]:
                    for also_viewed in line["related"]["also_viewed"]:
                        if also_viewed not in asin2id:
                            continue
                        G.add_edge(metadata[asin2id[line["asin"]]]["title"], metadata[asin2id[also_viewed]]["title"], relation="AlsoViewed")
                        G.add_edge(metadata[asin2id[also_viewed]]["title"], metadata[asin2id[line["asin"]]]["title"], relation="AlsoViewed")
                
                if "bought_together" in line["related"]:
                    for bought_together in line["related"]["bought_together"]:
                        if bought_together not in asin2id:
                            continue
                        G.add_edge(metadata[asin2id[line["asin"]]]["title"], metadata[asin2id[bought_together]]["title"], relation="BoughtTogether")
                        G.add_edge(metadata[asin2id[bought_together]]["title"], metadata[asin2id[line["asin"]]]["title"], relation="BoughtTogether")

    print("Number of nodes: ", G.number_of_nodes())
    print("Number of edges: ", G.number_of_edges())  

    # add entity description features, relation describe
    english_stopwords = list(stopwords.words('english'))
    vocab_count = defaultdict(int)
    vocab2id = {}
    reviews = []

    for line in tqdm(metadata[1:], desc="loading metadata"):
        if "description" in line:
            item = line["title"]
            review = line["description"].lower().strip().replace("\t", " ")
            for word in review.split(" "):
                if word in english_stopwords:
                    continue
                vocab_count[word] += 1
                if word not in vocab2id:
                    vocab2id[word] = len(vocab2id)
            reviews.append((item, review))
    
    review_tfidf = compute_tfidf_fast(reviews, vocab2id)
    for idx, (item, review) in tqdm(enumerate(reviews), desc="adding review features"):
        tfidf = review_tfidf[idx]
        for word in review.split(" "):
            if word in english_stopwords:
                continue
            score = tfidf[vocab2id[word]]
            if score > 0.1 and vocab_count[word] >= 10 and vocab_count[word] < 3000:
                if word not in entity2id:
                    entity2id[word] = len(entity2id)
                    entity2type[word] = "Term"

                G.add_edge(item, word, relation="Describedby")
                G.add_edge(word, item, relation="Describe")
    
    print("Number of nodes: ", G.number_of_nodes())
    print("Number of edges: ", G.number_of_edges())

    os.makedirs("../data/beauty", exist_ok=True)
    with open("../data/beauty/triplets.tsv", "w") as fw:
        for edge in G.edges:
            relation = G.edges[edge]["relation"]
            if relation not in relation2id:
                relation2id[relation] = len(relation2id)
            fw.write(f"{edge[0]}\t{relation}\t{edge[1]}\n")
    
    with open("../data/beauty/entity2id.txt", "w") as fw:
        fw.write(f"{len(entity2id)}\n")
        for entity, id in entity2id.items():
            fw.write(f"{entity}\t{id}\t{entity2type[entity]}\n")
    
    with open("../data/beauty/relation2id.txt", "w") as fw:
        fw.write(f"{len(relation2id)}\n")
        for relation, id in relation2id.items():
            fw.write(f"{relation}\t{id}\n")

    pickle.dump(entity2type, open("../data/beauty/entity2type.pkl", "wb"))
    pickle.dump(item2entity, open("../data/beauty/item2entity.pkl", "wb"))