# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

'''
Recommendation Tasks -- 5 Prompt Families (1, 2, 3, 4, 5)
'''

all_tasks = {}

# retrieval without candidates
# =====================================================
# Task Subgroup 1 -- Retrieval
# =====================================================

template = {}
'''
Input template:
You are a game recommender. This is the playing history of a user:
{{history item list of {{item_title}}}}.
You should recommend 10 Steam games that he/she is most likely to play next. You should order them by probability and compact them in one line split by commas. Do not output other words.

Target template:
{{item_titles}}

Metrics:
HR, NDCG, MRR
'''

template['source'] = "You are a game recommender. This is the playing history of a user:\n{}.\nYou should recommend 10 Steam games that he/she is most likely to play next. You should order them by probability and compact them in one line split by commas. Do not output other words."
template['target'] = "{}"
template['task'] = "retrieval"
template['source_argc'] = 1
template['source_argv'] = ['click_history']
template['target_argc'] = 1
template['target_argv'] = ['item_title']
template['id'] = "1-1"

all_tasks["retrieval"] = template


# =====================================================
# Task Subgroup 2 -- Ranking
# =====================================================

template = {}
'''
Input template:
You are a game recommender. This is the playing history of a user:\n
{{history item list of {{item_title}}}}.\n
There are 20 candidate Steam games that the user can play next:\n
{{candidate {{item_title}}}}.\n
You should rank the 20 candidate Steam games by measuring the probabilities that the user will play them next, according to the playing history. You should compact them in one line split by commas. Do not output other words.

Target template:
{{item_title}}

Metrics:
HR, NDCG, MRR
'''

template['source'] = "You are a game recommender. This is the playing history of a user:\n{}.\nThere are 20 candidate Steam games that the user can play next:\n{}.\nYou should rank the 20 candidate Steam games by measuring the probabilities that the user will play them next, according to the playing history. You should compact them in one line split by commas. Do not output other words."
template['target'] = "{}"
template['task'] = "ranking"
template['source_argc'] = 2
template['source_argv'] = ['click_history', 'candidates']
template['target_argc'] = 1
template['target_argv'] = ['item_title']
template['id'] = "2-1"

all_tasks['ranking'] =  template


# ====================================================
# Task Subgroup 3 -- Explanation
# ====================================================

template = {}
'''
Input template:
You are an explainable game recommender that will provide an explanation about why a given game is recommended to a user according to his or her playing history. You should summarize the explanation within 100 words.
The playing history of the user is\n{{history item list of {{item_title}}}}.\nThe given game to be recommended is {item_title}.
Please provide an explanation:

Target template:
{{explanation}}
'''

template['source'] = "You are an explainable game recommender that will provide an explanation about why a given game is recommended to a user according to his or her playing history. You should summarize the explanation within 100 words.\nThe playing history of the user is\n{}.\nThe given game to be recommended is {}.\nPlease provide an explanation:"
template['target'] = "{}"
template['task'] = "explanation"
template['source_argc'] = 2
template['source_argv'] = ['click_history', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['explanation']
template['id'] = "3-1"

all_tasks['explanation'] =  template


# ========================================================
# Task Subgroup 4 -- Tagging
# ========================================================

template = {}
'''
Input template:
You are a game tag labeler now. There is a Steam game {{item_title}}. Please summarize as many tags as possible for this game, which can accurately reflect its characteristics. You should compact them in one line split by commas. Do not output other words.
 
Target template:
{{tags}}
'''

template['source'] = "You are a game tag labeler now. There is a Steam game {}. Please summarize as many tags as possible for this game, which can accurately reflect its characteristics. You should compact them in one line split by commas. Do not output other words."
template['target'] = "{}"
template['task'] = "tagging"
template['source_argc'] = 1
template['source_argv'] = ['item_title']
template['target_argc'] = 1
template['target_argv'] = ['tags']
template['id'] = "4-1"

all_tasks['tagging'] =  template


# ========================================================
# Task Subgroup 5 -- Search
# ========================================================

template = {}
'''
Input template:
You are a Steam game search engine, which could retrieve relevant games based on the input query. Users may not know exactly the name of the game but some side information, or they may even only know the existence of the game but no other information at all. The input query is {query}. Please output the name of 10 most relevant games. You should compact them in one line split by commas. Do not output other words.

Target template:
{{queries}}
'''

template['source'] = "You are a Steam game search engine, which could retrieve relevant games based on the input query. Users may not know exactly the name of the game but some side information, or they may even only know the existence of the game but no other information at all. The input query is {}. Please output the name of 10 most relevant games. You should compact them in one line split by commas. Do not output other words."
template['target'] = "{}"
template['task'] = "search"
template['source_argc'] = 1
template['source_argv'] = ['queries']
template['target_argc'] = 1
template['target_argv'] = ['item_title']
template['id'] = "5-1"

all_tasks['search'] =  template