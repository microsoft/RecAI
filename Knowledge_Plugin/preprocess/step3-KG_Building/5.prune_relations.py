import os
import time
import pickle
import json
import argparse
import random
import pandas as pd
from tqdm import tqdm
import openai

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--recdata_path', type=str, default="../../data/steam/")
    args = parser.parse_args()

    key = ["your openai keys"]
    openai.api_key = key
    openai.api_base = "https://api.openai.com/v1"

    relation2triple = {}
    for line in open(os.path.join(args.recdata_path, "incorporate_triplets.tsv")):
        head, relation, tail = line.strip().split('\t')
        if relation not in relation2triple:
            relation2triple[relation] = []
        relation2triple[relation].append(f"{head} -> ({relation}) -> {tail}")

    relation2triple = {relation: random.sample(triples, min(len(triples),5)) for relation, triples in relation2triple.items()}
            
    cases = (len(relation2triple) + 29) // 30
    relation2triple = list(relation2triple.items())
    relation_2_result = {}
    for index in tqdm(range(cases), desc="pruning relations"):
        if "steam" in args.recdata_path:
            prompt = "To help make game recommendation, I have mined a lot of triplets from the knowledge graph that provide external information about the games. But there are too many noisy relations that contribute little to the game recommendation. So I want to prune out some useless relations and maintain the useful ones." \
                "I will first give you some examples of useful relations along with several triplets corresponding to this relation in the knowledge graph. These useful relations usually indicate features that affect user prefernces about games." \
                "1. genre: \n" \
                "Jazzpunk -> (genre) -> adventure game\n" \
                "Knights of Light -> (genre) -> role-playing video game\n" \
                "Maurice Leblanc -> (genre) -> science fiction\n" \
                "Lifestream - A Haunting Text Adventure -> (genre) -> adventure game\n" \
                "Town of Salem -> (genre) -> strategy video game\n" \
                "2. developer: \n" \
                "Magical Diary -> (developer) -> Hanako Games\n" \
                "Rayman 2: The Great Escape -> (developer) -> Ubisoft Montpellier\n" \
                "Pizza Frenzy -> (developer) -> PopCap Games\n" \
                "Putrefaction -> (developer) -> Oleg Kazakov\n" \
                "Half-Life: Blue Shift -> (developer) -> Valve Corporation\n" \
                "Then I will give you some examples of useless relations along with several corresponding triplets in the knowledge graph. These useless relations usually indicate features that do not affect user prefernces about games and have little relatedness with recommendation." \
                "1. second family name in Spanish name: \n" \
                "Manuel L. Quezon -> (second family name in Spanish name) -> Molina\n" \
                "Steven Bauer -> (second family name in Spanish name) -> Samson\n" \
                "Roberto Encinas -> (second family name in Spanish name) -> Duval\n" \
                "2. diplomatic relation: \n" \
                "Mexico -> (diplomatic relation) -> Sweden\n" \
                "Iraq -> (diplomatic relation) -> France\n" \
                "Iran -> (diplomatic relation) -> Turkey\n" \
                "3. shares border with: \n" \
                "Arab League -> (shares border with) -> Niger\n" \
                "Grayson County -> (shares border with) -> Cooke County\n" \
                "Hubei -> (shares border with) -> Jiangxi\n" \
                "Now I will give you some unlabeled relations from the wikidata knowledge graph. I want you to judge whether each relation indicate useful features that affect user prefernces about games. \n"

        elif "ml1m" in args.recdata_path:
            prompt = "To help make movie recommendation, I have mined a lot of triplets from the knowledge graph that provide external information about the movies. But there are too many noisy relations that contribute little to the movie recommendation. So I want to prune out some useless relations and maintain the useful ones." \
                "I will first give you some examples of useful relations along with several triplets corresponding to this relation in the knowledge graph. These useful relations usually indicate features that affect user prefernces about movies." \
                "1. genre: \n" \
                "Police Academy: Mission to Moscow -> (genre) -> comedy film\n" \
                "Scream 3 (2000) -> (genre) -> horror film\n" \
                "Jackie Chan's First Strike (1996) -> (genre) -> comedy film\n" \
                "Flipper (1996) -> (genre) -> family genre\n" \
                "2. director: \n" \
                "Saturday Night Live -> (director) -> Chel White\n" \
                "Lake Placid 2 -> (director) -> David Flores\n" \
                "Teenage Mutant Ninja Turtles II: The Secret of the Ooze (1991) -> (director) -> Michael Pressman\n" \
                "For a Few Dollars More (1965) -> (director) -> Sergio Leone\n" \
                "Then I will give you some examples of useless relations along with several corresponding triplets in the knowledge graph. These useless relations usually indicate features that do not affect user prefernces about movies and have little relatedness with recommendation." \
                "1. country of citizenship: \n" \
                "Christian Sievert -> (country of citizenship) -> Denmark\n" \
                "Kylie Belling -> (country of citizenship) -> Australia\n" \
                "Peter King -> (country of citizenship) -> United Kingdom\n" \
                "2. office held by head of the organization: \n" \
                "Yale University -> (office held by head of the organization) -> President of Yale University\n" \
                "University of Oxford -> (office held by head of the organization) -> Chancellor of the University of Oxford\n" \
                "British Film Institute -> (office held by head of the organization) -> Chair of the British Film Institute\n" \
                "3. languages spoken, written or signed: \n" \
                "Ferdy Mayne -> (languages spoken, written or signed) -> German\n" \
                "Roger Michell -> (languages spoken, written or signed) -> English\n" \
                "Calvin Lockhart -> (languages spoken, written or signed) -> English\n" \
                "Now I will give you some unlabeled relations from the wikidata knowledge graph. I want you to judge whether each relation indicate useful features that affect user prefernces about games. \n"

        elif "beauty" in args.recdata_path:
            prompt = "To help make game recommendation, I have mined a lot of triplets from the knowledge graph that provide external information about the games. But there are too many noisy relations that contribute little to the game recommendation. So I want to prune out some useless relations and maintain the useful ones." \
                "I will first give you some examples of useful relations along with several triplets corresponding to this relation in the knowledge graph. These useful relations usually indicate features that affect user prefernces about games." \
                "1. genre: \n" \
                "Jazzpunk -> (genre) -> adventure game\n" \
                "Knights of Light -> (genre) -> role-playing video game\n" \
                "Maurice Leblanc -> (genre) -> science fiction\n" \
                "Lifestream - A Haunting Text Adventure -> (genre) -> adventure game\n" \
                "Town of Salem -> (genre) -> strategy video game\n" \
                "2. developer: \n" \
                "Magical Diary -> (developer) -> Hanako Games\n" \
                "Rayman 2: The Great Escape -> (developer) -> Ubisoft Montpellier\n" \
                "Pizza Frenzy -> (developer) -> PopCap Games\n" \
                "Putrefaction -> (developer) -> Oleg Kazakov\n" \
                "Half-Life: Blue Shift -> (developer) -> Valve Corporation\n" \
                "Then I will give you some examples of useless relations along with several corresponding triplets in the knowledge graph. These useless relations usually indicate features that do not affect user prefernces about games and have little relatedness with recommendation." \
                "1. second family name in Spanish name: \n" \
                "Manuel L. Quezon -> (second family name in Spanish name) -> Molina\n" \
                "Steven Bauer -> (second family name in Spanish name) -> Samson\n" \
                "Roberto Encinas -> (second family name in Spanish name) -> Duval\n" \
                "2. diplomatic relation: \n" \
                "Mexico -> (diplomatic relation) -> Sweden\n" \
                "Iraq -> (diplomatic relation) -> France\n" \
                "Iran -> (diplomatic relation) -> Turkey\n" \
                "3. shares border with: \n" \
                "Arab League -> (shares border with) -> Niger\n" \
                "Grayson County -> (shares border with) -> Cooke County\n" \
                "Hubei -> (shares border with) -> Jiangxi\n" \
                "Now I will give you some unlabeled relations from the wikidata knowledge graph. I want you to judge whether each relation indicate useful features that affect user prefernces about games. \n"
        
        for idx, (relation, triples) in enumerate(relation2triple[index*30:(index+1)*30]):
            prompt += f"{idx+1}. {relation}: \n"
            for triple in triples:
                prompt += triple + '\n'
        prompt += "For each relation, only output whether this relation is useful or not for game recommendation, i.e. \"Useful\" or \"Not Useful\"."

        response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
        response = response['choices'][0]['message']['content']
        # print(response)
        print("Sleep 20 seconds before sending next request")
        time.sleep(20)
        prompt = prompt.strip().split("I want you to judge whether each relation indicate useful features that affect user prefernces about games.")[-1].split("For each relation, only output whether this relation is useful or not for game recommendation.")[0].strip()
        
        ## parse the response
        relations, results = [], []
        for line in prompt.split("\n"):
            line = line.strip()
            if not line:
                continue
            if "->" not in line:
                relation = line.split('.', 1)[1][:-1].strip()
                relations.append(relation)
        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.split('.')[0].isdigit():
                if "Not Useful" in line:
                    results.append(0)
                elif "Useful" in line:
                    results.append(1)
        for relation, result in zip(relations, results):
            relation_2_result[relation] = result
        print(relation_2_result)
    with open(os.path.join(args.recdata_path, "prune_incorporate_relations.txt"), "w") as fw:
        for relation, result in relation_2_result.items():
            if result == 1:
                fw.write(f"{relation}\t{result}\n")