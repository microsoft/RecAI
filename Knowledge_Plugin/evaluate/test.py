# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import string
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from polyfuzz import PolyFuzz

wordnet_lemmatizer=WordNetLemmatizer()

def compute_string_match_score(result,candidate):
    result=result.lower()
    candidate=candidate.lower()

    result_bag_of_words=[wordnet_lemmatizer.lemmatize(w) for w in word_tokenize(result) if w not in string.punctuation]
    candidate_bag_of_words=[wordnet_lemmatizer.lemmatize(w) for w in word_tokenize(candidate) if w not in string.punctuation]

    set_result=set(result_bag_of_words)
    set_candidate=set(candidate_bag_of_words)

    Jaccard=len(set_result.intersection(set_candidate))/len(set_result.union(set_candidate))

    return Jaccard

model_polyfuzz = PolyFuzz("TF-IDF")
def get_match_list_to_list(from_list, to_list):
    model_polyfuzz.match(from_list, to_list)
    results=model_polyfuzz.get_matches()
    matches=list(results.To)
    similarity=list(results.Similarity)
    return (matches, similarity)

def get_match_retrieval(model,query_list):
    results = model.transform(query_list)['TF-IDF']
    matches = list(results.To)
    similarity = list(results.Similarity)
    return (matches, similarity)

if __name__=="__main__":
    # print(compute_string_match_score("Come on , you are the best !","He comes to my country. Do you know?"))
    # print(compute_string_match_score("Come on , you are the best !", "Come on , you are the best !"))

    # example of reranking
    query_list = ["apple", "apples", "mouse"]
    key_list = ["apple", "apples", "appl", "recal", "similarity"]
    print(get_match_list_to_list(query_list,key_list))

    #example of retrieval
    model = PolyFuzz("TF-IDF")
    model.fit(key_list)
    print(get_match_retrieval(model,query_list))
