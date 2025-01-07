# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# need to be formatted: item, Item, ITEM


TOOL_NAMES = {
    "BufferStoreTool": "{Item} Candidates Storing Tool",
    "LookUpTool": "{Item} Information Look Up Tool",
    "HardFilterTool": "{Item} Properties Filtering Tool",
    "SoftFilterTool": "{Item} Similarity Filtering Tool",
    "RankingTool": "{Item} Candidates Ranking Tool",
    "MapTool": "Mapping Tool",
}


CANDIDATE_STORE_TOOL_DESC = """
The tool is useful to save candidate {item}s into buffer as the initial candidates, following tools would filter or ranking {item}s from those canidates. \
For example, "Please select the most suitable {item} from those {item}s". \
Don't use this tool when the user hasn't specified that they want to select from a specific set of {item}s. \
The input of the tool should be a string of {item} names split by two ';', such as "{ITEM}1;; {ITEM}2;; {ITEM}3". 
"""


LOOK_UP_TOOL_DESC = """
The tool is used to look up {item}'s detailed information in a {item} information table (including statistical information), like number of {item}s, description of {item}s, price and so on. \

The input of the tools should be a SQL command (in one line) converted from the search query, which would be used to search information in {item} information table. \
You should try to select as less columns as you can to get the necessary information. \
Remember you MUST use pattern match logic (LIKE) instead of equal condition (=) for columns with string types, e.g. "title LIKE '%xxx%'". \

For example, if asking for "how many xxx {item}s?", you should use "COUNT()" to get the correct number. If asking for "description of xxx", you should use "SELECT {item}_description FROM xxx WHERE xxx". \

The tool can NOT give recommendations. DO NOT SELECT id information!
"""


HARD_FILTER_TOOL_DESC = """
The tool is a hard-condition {item} filtering tool. The tool is useful when human want {item}s with some hard conditions on {item} properties. \
The input of the tool should be a one-line SQL SELECT command converted from hard conditions. Here are some rules: \
1. {item} titles can not be used as conditions in SQL;
2. always use pattern match logic for columns with string type;
3. only one {item} information table is allowed to appear in SQL command;
4. select all {item}s that meet the conditions, do not use the LIMIT keyword;
5. try to use OR instead of AND;
6. use given related values for categorical columns instead of human's description.
"""


SOFT_FILTER_TOOL_DESC = """
The tool is a soft condition {item} filtering tool to find similar {item}s for specific seed {item}s. \
Use this tool ONLY WHEN human explicitly want to find similar {item}s with seed {item}s. \
The tool can not recommend {item}s based on human's history. \
There is a similarity score threshold in the tool, only {item}s with similarity above the threshold would be kept. \
Besides, the tool could be used to calculate the similarity scores with seed {item}s for {item}s in candidate buffer for ranking tool to refine. \
The input of the tool should be a list of seed {item} titles/names, which should be a Python list of strings. \
Do not fake any {item} names.
"""


RANKING_TOOL_DESC = """
The tool is useful to refine {item}s order (for better experiences) or remove unwanted {item}s (when human tells the {item}s he does't want) in conversation. \
The input of the tool should be a json string, which may consist of three keys: "schema", "prefer" and "unwanted". \
"schema" represents ranking schema, optional choices: "popularity", "similarity" and "preference", indicating rank by {item} popularity, rank by similarity, rank by human preference ("prefer" {item}s). \
The "schema" depends on previous tool using and human preference. If "prefer" info here not empty, "preference" schema should be used. If similarity filtering tool is used before, prioritize using "similarity" except human want popular {item}s.
"prefer" represents {item} names that human has enjoyed or human has interacted with before (human's history), which should be an array of {item} titles. Keywords: "used to do", "I like", "prefer".
"unwanted" represents {item} names that human doesn't like or doesn't want to see in next conversations, which should be an array of {item} titles. Keywords: "don't like", "boring", "interested in". 
"prefer" and "unwanted" {item}s should be extracted from human request and previous conversations. Only {item} names are allowed to appear in the input. \
The human's feedback for you recommendation in conversation history could be regard as "prefer" or "unwanted", like "I have tried those items you recommend" or "I don't like those".
Only when at least one of "prefer" and "unwanted" is not empty, the tool could be used. If no "prefer" info, {item}s would be ranked based on the popularity.\
Do not fake {item}s.
"""

MAP_TOOL_DESC = """
The tool is useful when you want to convert {item} id to {item} title before showing {item}s to human. \
The tool is able to get stored {item}s in the buffer and randomly select a specific number of {item}s from the buffer. \
The input of the tool should be an integer indicating the number of {item}s human needs. \
The default value is 5 if human doesn't give.
"""


_TOOL_DESC = {
    "CANDIDATE_STORE_TOOL_DESC": CANDIDATE_STORE_TOOL_DESC,
    "LOOK_UP_TOOL_DESC": LOOK_UP_TOOL_DESC,
    "HARD_FILTER_TOOL_DESC": HARD_FILTER_TOOL_DESC,
    "SOFT_FILTER_TOOL_DESC": SOFT_FILTER_TOOL_DESC,
    "RANKING_TOOL_DESC": RANKING_TOOL_DESC,
    "MAP_TOOL_DESC": MAP_TOOL_DESC,
}

OVERALL_TOOL_DESC = """
There are several tools to use:
- {BufferStoreTool}: {CANDIDATE_STORE_TOOL_DESC}
- {LookUpTool}: {LOOK_UP_TOOL_DESC}
- {HardFilterTool}: {HARD_FILTER_TOOL_DESC}
- {SoftFilterTool}: {SOFT_FILTER_TOOL_DESC}
- {RankingTool}: {RANKING_TOOL_DESC}
- {MapTool}: {MAP_TOOL_DESC}
""".format(
    **TOOL_NAMES, **_TOOL_DESC
)

# """
# There are several tools to use:
# - {BufferStoreTool}: useful to store candidate {item} given by user into the buffer for further filtering and ranking. For example, user may say "I want the cheapest in the list" or "A and B, which is more suitable for me".
# - {LookUpTool}: used to look up some {item} information in a {item} information table (including statistical information), like number of {item}, description of {item} and so on.
# - {HardFilterTool}: useful when user expresses intentions about {item}s with some hard conditions on {item} properties. The input should be a SQL command. The tool would return {item} ids.
# - {SoftFilterTool}: useful when user gives specific {item} names (seed {item}s) and is looking for similar {item}s. The input could only be {item} titles. The tool would return {item} ids.
# - {RankingTool}: useful to refine {item}s order based on user profile (including user prefer {item}s, like play history or liked {item}s and unwanted {item}s, like disliked {item}s) expressed in conversation. The input is a json str with 3 keys: "schema", "prefer" and "unwanted". If the two informations are both empty, the tool should not be used.The tool would return {item} ids.
# - {MapTool}: useful to convert {item} id to {item} titles.
# """


TOOLBOX_DESC = """
The tool is a big tool box consisting of all tools metioned above. The tool box is used to execute tools when plan is maked. 
The input is a List of Dict, indicating the tool using plan and input to each tool. There are two keys "tool_name" and "input" in Dict. 
The format should be a JSON string, like: \"[{\\\"tool_name\\\": TOOL-1, \\\"input\\\": INPUT-1}, ..., {\\\"tool_name\\\": TOOL-N, \\\"input\\\": INPUT-N}]\".
The input string will be parsed by json.loads() function. 
"""
