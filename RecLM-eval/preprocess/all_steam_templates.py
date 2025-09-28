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
You are a recommender system. This is the interation history of a user:
{{history item list of {{item_title}}}}.
You should recommend 10 items that he/she is most likely to choose next. You should order them by probability and compact them in one line split by commas. Do not output other words.

Target template:
{{item_titles}}

Metrics:
HR, NDCG, MRR
'''

template['source'] = "You are a recommender system. This is the interation history of a user:\n{}.\nYou should recommend 10 items that he/she is most likely to play next. You should order them by probability and compact them in one line split by commas. Do not output other words."
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
You are a recommender system. This is the interation history of a user:\n
{{history item list of {{item_title}}}}.\n
There are 20 candidate items that the user can choose next:\n
{{candidate {{item_title}}}}.\n
You should rank the 20 candidate items by measuring the probabilities that the user will choose them next, according to the interation history. You should compact them in one line split by commas. Do not output other words.

Target template:
{{item_title}}

Metrics:
HR, NDCG, MRR
'''

template['source'] = (
    "You are a recommender system. Your task is to rank 20 candidate item titles for a user based on their interaction history.\n\n"
    "Here is the interaction history of the user:\n{}\n"
    "The candidates are listed below. EACH candidate title is wrapped by the delimiters <SOI> and <EOI> on its own line:\n{}\n\n"
    "NOTE: The order of candidates is RANDOM and carries NO meaning about relevance. Generating an output list identical to this random order is almost certainly sub-optimal and should be treated as an error unless it is verifiably the optimal ranking under your own scoring.\n\n"
    "### INTERNAL ANALYSIS PROTOCOL (DO NOT OUTPUT) ###\n"
    "Step 1 — SCORING: First estimate the probability that the user will choose each candidate next, **then** assign a UNIQUE score from 1 (worst) to 20 (best) that strictly preserves this probability ranking (higher probability → higher score).\n"
    "Step 2 — SORTING: Sort the 20 candidates by score DESC to obtain LIST_RANKED.\n"
    "Step 3 — SELF-CHECK: Ensure LIST_RANKED length==20, all titles are unique (case-insensitive) and none appear in the history.\n"
    "Step 4 — OUTPUT: Convert LIST_RANKED into the final single-line format described below. If ANY check fails, REPEAT from Step 1 silently. Never reveal this protocol.\n\n"
    "REQUIREMENTS\n"
    "1. Use ONLY titles that appear between <SOI> and <EOI> in the candidate list. Titles that appear in the history are STRICTLY forbidden.\n"
    "2. Copy every chosen title VERBATIM (byte-for-byte) without the surrounding <SOI> <EOI> delimiters; do NOT add, delete, truncate, or modify any character.\n"
    "3. Output MUST contain EXACTLY 20 unique titles (no duplicates; uniqueness is case-insensitive).\n"
    "4. Concatenate the 20 titles into ONE single line with NO newline characters. Immediately append <end> after each title with NO spaces. Do NOT output any other text.\n"
    "5. After generating the line, silently verify rules above. If ANY check fails or the count ≠ 20, regenerate the entire line until ALL checks pass. NEVER reveal this process.\n"
    "6. Your final ranking MUST differ from the original candidate order unless that order is demonstrably optimal under the probability-based scores from Step 1.\n"
    "\n"
    "FINAL OUTPUT FORMAT (single line):\n"
    "Title1<end>Title2<end>...Title20<end>\n"
    )
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
You are an explainable recommender system that will provide an explanation about why a given item is recommended to a user according to his or her interation history. You should summarize the explanation within 100 words.
The interation history of the user is\n{{history item list of {{item_title}}}}.\nThe given item to be recommended is {item_title}.
Please provide an explanation:

Target template:
{{explanation}}
'''

template['source'] = "You are an explainable recommender system that will provide an explanation about why a given item is recommended to a user according to his or her interation history. You should summarize the explanation within 100 words.\nThe interation history of the user is\n{}.\nThe given item to be recommended is {}.\nPlease provide an explanation:"
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
You are a item tag labeler now. There is a item {{item_title}}. Please summarize as many tags as possible for this item, which can accurately reflect its characteristics. You should compact them in one line split by commas. Do not output other words.
 
Target template:
{{tags}}
'''

template['source'] = "You are a item tag labeler now. There is a item {}. Please summarize as many tags as possible for this item, which can accurately reflect its characteristics. You should compact them in one line split by commas. Do not output other words."
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
You are a item search engine, which could retrieve relevant items based on the input query. Users may not know exactly the name of the item but some side information, or they may even only know the existence of the item but no other information at all. The input query is {query}. Please output the name of 10 most relevant items. You should compact them in one line split by commas. Do not output other words.

Target template:
{{queries}}
'''

template['source'] = "You are a item search engine, which could retrieve relevant items based on the input query. Users may not know exactly the name of the item but some side information, or they may even only know the existence of the item but no other information at all. The input query is {}. Please output the name of 10 most relevant games. You should compact them in one line split by commas. Do not output other words."
template['target'] = "{}"
template['task'] = "search"
template['source_argc'] = 1
template['source_argv'] = ['queries']
template['target_argc'] = 1
template['target_argv'] = ['item_title']
template['id'] = "5-1"

all_tasks['search'] =  template


# ========================================================
# Task Subgroup 6 -- cf_ranking_mc
# ========================================================

template = {}
'''
Input template:
You are a recommender system. This is the interation history of a user (items are not in chronological order):
{{history item list of {{item_title}}}}.
There are 10 candidate items:
{{candidate {{item_title}}}}.
Based on the user's interation history, which item is most likely to be in the user's history? Please choose the correct answer from the 10 candidates. Only output the letter (A-J) corresponding to your item.

Target template:
{{correct_answer}}

Metrics:
Accuracy
'''

template['source'] = '''You are a recommendation assistant. The user's interaction history (order not guaranteed) is:
{}.

Here are 10 candidate items labeled from A to J:
{}.

### INTERNAL ANALYSIS PROTOCOL (DO NOT OUTPUT) ###
You must follow these steps silently. The user will not see this process.

1. **SYNTHESIZE USER PROFILE:** Analyze the provided history to form a clear, concise understanding of the user's core preferences, themes, and styles.

2. **HOLISTIC CANDIDATE SCAN:** Review all 10 candidates from A to J before making any judgments. This is mandatory to prevent positional bias.

3. **FORCED RANKING:** Evaluate each candidate against the user profile. Assign a unique score from 1 (worst) to 10 (best) to every candidate. Ties are forbidden. This forces a clear decision.

4. **FINAL SELECTION:** Identify the single candidate that received the score of 10. This is your final answer.

### FINAL OUTPUT COMMAND (ABSOLUTE RULE) ###
Your response in the output MUST conform to the following rules without exception:
- Your entire output MUST BE a single uppercase letter from the set {{A, B, C, D, E, F, G, H, I, J}}.
- There must be ABSOLUTELY NO other text, symbols, newlines, spaces, or explanations before or after this single letter.'''
template['target'] = "{}"
template['task'] = "cf_ranking_mc"
template['source_argc'] = 2
template['source_argv'] = ['click_history', 'candidates']
template['target_argc'] = 1
template['target_argv'] = ['correct_answer']
template['id'] = "6-1"

all_tasks['cf_ranking_mc'] =  template


# ========================================================
# Task Subgroup 7 -- seq_ranking_mc
# ========================================================

template = {}
'''
Input template:
You are a recommender system. This is the interation history of a user in chronological order:
{{history item list of {{item_title}}}}.
There are 10 candidate items:
{{candidate {{item_title}}}}.
Based on the user's sequential interation history, which item is most likely to be the next item the user will choose? Please choose the correct answer from the 10 candidates. Only output the number (1-10) corresponding to your item.

Target template:
{{correct_answer}}

Metrics:
Accuracy
'''

template['source'] = '''You are a recommendation assistant. The user's interaction history (chronological order) is:
{}.

Here are 10 candidate items labeled from A to J:
{}.

### INTERNAL ANALYSIS PROTOCOL (DO NOT OUTPUT) ###
You must follow these steps silently. The user will not see this process.

1. **SYNTHESIZE USER PROFILE:**  Analyze the provided history to form a clear, concise understanding of the user's core preferences, themes, and styles.

2. **HOLISTIC CANDIDATE SCAN:** Review all 10 candidates from A to J before making any judgments. This is mandatory to prevent positional bias.

3. **FORCED RANKING:** Evaluate each candidate against the user profile. Assign a unique score from 1 (worst) to 10 (best). Ties are forbidden.

4. **FINAL SELECTION:** Identify the single candidate that received the score of 10. This is your final answer.

### FINAL OUTPUT COMMAND (ABSOLUTE RULE) ###
Your response in the output MUST conform to the following rules without exception:
- Your entire output MUST BE a single uppercase letter from the set {{A, B, C, D, E, F, G, H, I, J}}.
- There must be ABSOLUTELY NO other text, symbols, newlines, spaces, or explanations before or after this single letter.'''
template['target'] = "{}"
template['task'] = "seq_ranking_mc"
template['source_argc'] = 2
template['source_argv'] = ['click_history', 'candidates']
template['target_argc'] = 1
template['target_argv'] = ['correct_answer']
template['id'] = "7-1"

all_tasks['seq_ranking_mc'] =  template
