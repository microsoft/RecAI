# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pandas
import json
from tqdm import tqdm
import random
import argparse
import os
from utils import get_item_text

random.seed(2023)

def parse_args():
    parser = argparse.ArgumentParser(description="genera_query_file")
    parser.add_argument(
        "--in_seq_data", type=str, help=""
    )
    parser.add_argument(
        "--in_meta_data", type=str, help=""
    )
    parser.add_argument(
        "--out_u2i_file", type=str, help=""
    )
    parser.add_argument(
        "--out_q2i_file", type=str, help=""
    )
    parser.add_argument(
        "--out_q2i_misspell_file", type=str, help=""
    )
    parser.add_argument(
        "--out_query_file", type=str, help=""
    )
    parser.add_argument(
        "--task_type", type=str, default='train', choices=['train', 'test']
    )
    args = parser.parse_args()
    return args


def gen_user2item(itemid2title, args):
    count=0
    threshold = 0.008 if args.task_type=='train' else 0.007
    with open(args.out_u2i_file, 'w') as f:
        for line in tqdm(open(args.in_seq_data)):
            if random.random() > threshold:
                continue
            userid, itemids = line.strip().split(' ', 1)
            itemids = itemids.split(' ')
            
            if args.task_type=='train':
                query_items = itemids[:-2][::-1]
                target_item = int(itemids[-2])
                history = [int(itemid) for itemid in itemids[:-2]]
            elif args.task_type=='test':
                query_items = itemids[:-1][::-1]
                target_item = int(itemids[-1])
                history = [int(itemid) for itemid in itemids[:-1]]
            else:
                raise NotImplementedError
            query_items = query_items[:20] # truncate to 20

            query = ''
            for item in query_items:
                query += itemid2title[int(item)][1] + ', '

            query = query.strip().strip(',')          

            output = {
                'userid': userid,
                'target_id': target_item,
                'history': history,
                'query': query,
            }
            f.write(json.dumps(output) + '\n')
            count += 1
            # if count > 5:
            #     break
    print('gen_user2item total samples: ', count)

def gen_query2item(itemid2text, args):
    target_set = set()
    with open(args.out_u2i_file, 'r') as f:
        for line in f:
            line = json.loads(line)
            target_set.add(int(line['target_id']))
    print('target set size: ', len(target_set))
    count = 0
    threshold = 0.2 if args.task_type=='train' else 0.15
    with open(args.out_q2i_file, 'w') as f:
        for i, text in enumerate(itemid2text):
            if i == 0:
                continue
            if i in target_set or random.random() <= threshold:
                output = {
                    'item_id': i,
                    'query': text,  
                }
                f.write(json.dumps(output) + '\n')
                count += 1
                # if count > 5:
                #     break
    print('gen_query2item total samples: ', count)

def gen_query2item_misspell(itemid2title, args):
    count = 0
    threshold = 0.15 if args.task_type=='train' else 0.05
    with open(args.out_q2i_misspell_file, 'w') as f:
        for i, text in enumerate(itemid2title):
            if i == 0:
                continue
            if random.random() > threshold:
                continue
            output = {
                'item_id': i,
                'query': text[1],  
            }
            f.write(json.dumps(output) + '\n')
            count += 1
            # if count > 5:
            #     break
    print('gen_query2item_misspell total samples: ', count)

def gen_query_file(args):
    u2i_template = "Based on a user's gaming history, create a concise user summary under 100 words, highlighting patterns and traits from past games so that it is helpful to identify preferences and predict future game choices. Randomly select a writing style from first-person, third-person, or non-narrative. Do not mention the chosen style in the summary. Do not mention the future predictions in the summary. Here is his game play history: {}"

    q2i_template = "I will give you some properties of a video game. Supppse you are interested in this game and once you find it, you will download and play it.  But currently you don't know the exact title of this game, so you are search for it with some queries. You only have a coarse idea about a very few of the properties of the game. What search queries you may possible use to look for this game?  Please print about 10 possible queries, concatenate them with '#SEP#' without order numbers and output only one string line. Queries should not include the the exact item title. Here are some examples:\n \
    Game properties:\ntitle: Tropico 4, tags: City Builder,Simulation,Strategy,Management,Comedy,Sandbox,Singleplayer,Great Soundtrack,Economy,Real-Time with Pause,Politics,Building,Resource Management,Funny,Political,Atmospheric,RTS,Cold War,Capitalism,God Game, game details: Single-player,Steam Achievements,Steam Trading Cards, publisher: Kalypso Media Digital, developer: Haemimont Games, price: 14.99, release date: 2011-09-01, description: The world is changing and Tropico is moving with the times - geographical powers rise and fall and the world market is dominated by new players with new demands and offers - and you, as El Presidente, face a whole new set of challenges.\n \
    Possible queries:\nA city building game released before 2015, cold war#SEP#city simulation, strategy, and like comedy#SEP#city simulator by  Kalypso Media Digital #SEP#simulation and strategy, under $15 #SEP#city simulation with wars#SEP#city simulation game after 2010#SEP#City-building and political simulation game#SEP#Game where you play as an island ruler#SEP#Manage a tropical paradise game\n \
    Game properties:\ntitle: Pertinence, tags: Indie,Action,Adventure,Exploration,Puzzle,Difficult,Metroidvania,Minimalist, game details: Single-player,Steam Achievements,Steam Trading Cards,Partial Controller Support, publisher: Not Magic, developer: Not Magic, price: 3.99, release date: 2016-03-25, description: Pertinence is a top-down adventure with minimalist graphics that explores various puzzle and action mechanics. The goal of the game is to freely explore rooms in a grid layout while collecting as many \"alloy\" as possible. These will unlock new areas and abilities.\n \
    Possible queries:\n indie action and exploration game with puzzle#SEP#game: metroidvania, puzzle, cheap#SEP#Board game with relevance theme#SEP#Card game about pertinence#SEP#Strategic game related to relevance#SEP#Game focusing on significance and importance#SEP#Unique board game with pertinence concept#SEP#indie puzzle and exploration game in 2016#SEP#game that explore rooms in a grid layout\n \
    Now here are the properties of the target video game: {} Please give your answer:"

    q2i_misspell_template = "I will give you a game title. Please rewrite the game title to imitate the spelling errors that humans may make when searching for this game. Please print 10 possible misspellings, concatenate them with '#SEP#' without order numbers and output only one string line. These 10 misspellings are required to be as diverse as possible, and each misspelling can contain one or multiple spelling errors. Here is the game title: {}. Please give your answer:"
    ## generate an empty pandas dataframe, with columns: 'question'
    df = pandas.DataFrame(columns=['question'])
    ## insert a row to the dataframe with values: 'question': 'What is the meaning of life?'
    
    with open(args.out_u2i_file, 'r') as f:
        for line in f:
            line = json.loads(line)
            query = line['query']
            df.loc[len(df)] = [u2i_template.format(query)]
    
    with open(args.out_q2i_file, 'r') as f:
        for line in f:
            line = json.loads(line)
            query = line['query']
            df.loc[len(df)] = [q2i_template.format(query)]

    with open(args.out_q2i_misspell_file, 'r') as f:
        for line in f:
            line = json.loads(line)
            query = line['query']
            df.loc[len(df)] = [q2i_misspell_template.format(query)]
    ## save the dataframe to a csv file
    
    # split_index = len(df) // 2   
    # first_half = df.iloc[:split_index, :]  
    # second_half = df.iloc[split_index:, :] 

    # first_half.to_csv(args.out_query_file+"_1.csv", index=False)
    # second_half.to_csv(args.out_query_file+"_2.csv", index=False)
    df.to_csv(args.out_query_file+".csv", index=False)

if __name__ == '__main__':
    args = parse_args() 
    os.makedirs(os.path.dirname(args.out_u2i_file), exist_ok=True)
    
    itemid2text, itemid2title, itemid2features, _ = get_item_text(args.in_meta_data)
    gen_user2item(itemid2title, args)
    gen_query2item(itemid2text, args)
    gen_query2item_misspell(itemid2title, args)
    gen_query_file(args)


    