# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

user2item_template = [
    "I have the following histories: {}.",
    "Recommend an item for me based on my history interactions: {}.",
    "I have played several games before: {}. Can you recommend other games for me?",
    "{} are games I have just played, are there any other games suitable for me?",
    "Given the following item history {}, predict next possible item to be played?",
    "Here are the history games a user has played {}. What to recommend next for the user?",
    "Find a game for me based on these previous games: {}.",
    "I've played these games before: {}. What other games should I try?",
    "Played games list: {}. Suggest a new game for me to play.",
    "My gaming history includes: {}. What game should I play next?",
    "These are the games I've experienced: {}. Offer some new recommendations.",
    "My game history: {}. Find a suitable game for me.",
    "Played these games: {}. What's a good game to try next?",
    "Considering my previous games {}, what would you recommend?",
    "My game background includes: {}. Any recommendations?",
    "Games I've played before: {}. What should I play now?",
    "Played and enjoyed these games: {}. What's next?",
    "Here's my gaming history: {}. Can you suggest a new game?",
    "Check out my game history {}. What game do you recommend?",
    "Histories: {}",
    "I have played {}.",
    "Enjoyed games: {}.",
    "Played games: {}. Recommend a new game for me.",
    "I've played these games: {}.",
    "{} are games I've played. Any recommendations?",
    "History interactions include: {}.",
]

query2item_template = [
    "{}",
    "I want to find some games with several features: {}.", 
    "Recommend games for a user who likes games that have these features {}.", 
    "I prefer games with the following type: {}, can you recommend some for me?",
    "Recommend an item that meets the following requirements: {}.",
    "I'm looking for games that include these elements: {}.",
    "Suggest some games for someone who enjoys features like {}.",
    "I have a taste for this kind of games {}, what do you recommend?",
    "Find me a game that matches these criteria: {}.",
    "I'm in search of games with these characteristics: {}.",
    "What games can you suggest that offer these features? {}",
    "I'd like to try games that incorporate {}. Any recommendations?",
    "Help me find games with the following attributes: {}.",
    "Can you recommend games that focus on elements like {}",
    "I enjoy games with {}, can you suggest any?",
    "I'm interested in games that provide elements like {}. What are my options?",
    "What are some good games that emphasize {}?",
    "I'd love to discover games that have {}. Any suggestions?",
    "I'm a fan of this style games: {}. Can you offer some recommendations?",
    "Looking for game suggestions with qualities such as {}",
    "I'm curious about games that highlight {}. What can you recommend?",
    "Games containing {}.",
    "I want to find games featuring {}.",
    "Recommend games with {}.",
    "I'm looking for games that have {}.",
    "I'm interested in games that include {}.",
    "{} are features I'm looking for.",
    "query: {}",
    "Search for games with {}.",
    "Search for {}",
]

title2item_template = [
    "{}",
    "I'm looking for this game: {}.",
    "I want to find this game: {}.",
    "I'm looking for a game called {}.",
    "Search for this game: {}.",
    "I'm trying to find this game: {}.",
    "Please help me locate this game: {}.",
    "Can you find this game for me: {}?",
    "I'm in search of a game named {}.",
    "Could you search for a game titled {}?",
    "I need assistance finding this game: {}.",
    "I'm interested in locating a game called {}.",
    "Please look up this game for me: {}.",
    "I'd like to locate the game {}.",
    "Help me find this game, please: {}.",
]

item2item_template = [
    "Can you recommend me some items similar to {}?",
    "I have played this game: {} and find it interesting. Are there other games similar to this one?",
    "Looking for games like this: {}. Any suggestions?",
    "I enjoyed this item: {}, can you recommend anything similar?",
    "Any recommendations for games related to {}?",
    "Find me more games like {} please.",
    "Which games share similarities with this game? {}",
    "Just played this game: {}, and I'm craving more with the same feel. Any ideas?",
    "{} was so much fun! What other similar games would you recommend?",
    "Are there any other games with the same feel as this game? {}",
    "Games similar to this one {}",
    "I'm looking for games similar to {}",
    "Recommend games similar to this game: {}.",
    "Similar games to {}.",
    "Games like {}.",
    "Items similar to {}.",
    "Items like {}.",
    "Recommend items similar to this one: {}.",
    "Related games to {}.",
]

queryuser2item_template = [
    "target: {1},  histories: {0}",
    "histories: {0}, target: {1}",
    "I want to find some games with several features: {1}. These are games I have played before: {0}.",
    "Given the following item history {0}, predict the next possible item meeting the following requirements: {1}",
    "Based on my gaming history {0}, suggest some games that have these attributes: {1}.",
    "Considering my previous games {0}, please recommend new games possessing these qualities: {1}.",
    "Using my game history {0} as reference, provide game suggestions with the following characteristics: {1}.",
    "I'm looking for games with these features: {1}. Here are games I've played before: {0}.",
    "Show me games with these traits: {1}, taking into account my past games {0}.",
    "In light of the games I've enjoyed {0}, suggest new games that include these aspects: {1}.",
    "Discover games that have the following attributes: {1}, based on my previous gaming experience {0}.",
    "Find games that contain these properties: {1}, keeping my game history in mind: {0}.",
    "Search for games with these elements: {1}, considering my gaming background {0}.",
    "With my past games {0} as a guide, present new games featuring these characteristics: {1}.",
    "Given my previous gaming choices {0}, I'd like to explore games that offer these features: {1}.",
    "My past gaming experience {0} leads me to seek games with these characteristics: {1}.",
    "Following my game history {0}, kindly suggest games with these specific features: {1}.",
    "Please provide game recommendations with these qualities: {1}, in relation to my gaming past {0}.",
    "features: {1}, history interactions: {0}",
    "history interactions: {0}, features: {1}",
    "histories: {0}, features: {1}",
    "features: {1}, histories: {0}",
    "I have played {0}, recommend games with features: {1}.",
    "Characteristics: {1}, games I have played: {0}.",
    "Look for games contain {1}, based on my gaming history {0}.",
    "Enjoy features: {1}. Played games: {0}.",
    "Played games: {0}. Enjoy features: {1}.",
]

querysummary2item_template = [
    "user summary: {0}, query: {1} ",
    "Here is summary of the user history: {0}, following this query: {1}",
    "user query: {1}, summary of user preference: {0}",
    "search request: {1}, User's background: {0}",
    "Overview of user's interests: {0}, related inquiry: {1}",
    "User profile highlights: {0}, associated question: {1}",
]

vaguequery2item_template = [
    "{}",
    "I want to find some games with several features: {}.", 
    "Recommend games for a user who likes games that have these features: {}.", 
    "I prefer games with the following type: {}, can you recommend some for me?",
    "Recommend an item that meets the following requirements: {}.",
    "I'm looking for games that include these elements: {}.",
    "Suggest some games for someone who enjoys features like {}.",
    "Find me a game that matches these criteria: {}.",
    "I'm in search of games with these characteristics: {}.",
    "What games can you suggest that offer these features? {}",
    "Help me find games with the following attributes: {}.",
    "I'm interested in games that provide these elements: {}. What are my options?",
    "Games: {}.",
    "I want to find games with {}.",
    "Recommend games with {}.",
    "I'm looking for games that have {}.",
    "I'm interested in games that include {}.",
    "query: {}",
    "Search for games: {}.",
]

relativequery2item_template = {
    "recent": [
        "Recommend me some recently released games.",
        "Could you suggest some new games that have just been launched?",
        "Can you tell me about some of the latest games on the market?",
        "I'm looking for some newly released games, can you help?",
        "What are some of the newest games that have just come out?",
        "Can you list some of the latest video games that have been released recently?",
        "Do you have any recommendations for games that have been released lately?",
        "I'm interested in the latest games, could you provide some suggestions?",
    ],
    "cheap": [
        "Recommend me some cheap games.",
        "I'm looking for some cheap games, can you help?",
        "What are some of the cheapest games you can recommend?",
        "Could you suggest some inexpensive games?",
        "Can you tell me about some of the most affordable games?",
        "I'm interested in the cheapest games, could you provide some suggestions?",
        "Could you suggest some affordable games for me?", 
        "Can you help me find some budget-friendly games?", 
        "I'm looking for some low-cost games, can you assist?", 
        "What are some cost-effective games that I could buy?", 
        "Can you list some of the inexpensive games available?", 
        "Do you have any recommendations for games that are cheap?", 
        "I'm interested in games that won't break the bank, could you provide some suggestions?",
    ],
    "expensive": [
        "Recommend me some expensive games.",
        "I'm looking for some expensive games, can you help?",
        "What are some of the most expensive games you can recommend?",
        "Could you suggest some pricey games?",
        "Can you tell me about some of the most costly games?",
        "I'm interested in the most expensive games, could you provide some suggestions?",
        "Can you help me find some high-priced games?", 
        "I'm looking for some high-cost games, can you assist?", 
        "Can you list some of the expensive games available?", 
        "Do you have any recommendations for games that are costly?", 
        "Can you help me find some premium-priced games?",
        "I'm looking for some costly games, can you assist?",
        "What are some of the high-priced games that I could buy?",
        "Can you list some of the pricy games available?",
        "Do you have any recommendations for games that are expensive?",
        "I'm interested in luxury games, could you provide some suggestions?",
    ],
    "popular": [
        "Recommend me some popular games.",
        "I'm looking for some popular games, can you help?",
        "What are some of the most popular games you can recommend?",
        "Could you suggest some popular games?",
        "Can you tell me about some of the most popular games?",
        "I'm interested in the most popular games, could you provide some suggestions?",
        "Can you help me find some popular games?", 
        "I'm looking for some popular games, can you assist?", 
        "Can you list some of the popular games available?", 
        "Do you have any recommendations for games that are popular?", 
        "I'm interested in games that are trending, could you provide some suggestions?",
        "Could you suggest some trending games for me?",
        "Can you help me find some of the most played games right now?",
        "I'm looking for some hit games, can you assist?",
        "What are some of the hot games that are popular right now?",
        "Can you list some of the most downloaded games at the moment?",
        "Do you have any recommendations for games that are currently popular?",
        "I'm interested in mainstream games, could you provide some suggestions?",
    ],
}

negquery2item_template = [
    "not including {}",
    "Games excluding these features: {}",
    "I want to find some games excluding several features: {}.", 
    "Recommend games for a user who likes games that don't have these features: {}.", 
    "Recommend an item that meets the following requirements: not including {}.",
    "I'm looking for games that don't include these elements: {}.",
    "Find me a game that matches these criteria: excluding {}.",
    "I'd love to discover games that do not have {}. Any suggestions?",
    "Games not containing {}.",
    "I want to find games not featuring {}.",
    "Recommend games excluding these elements: {}.",
    "I'm looking for games that do not have {}.",
    "I'm interested in games that do not include {}.",
    "Search for games excluding these features: {}.",
    "Search for games not including {}",
]

dialog_template = '''Assume that a user is looking for some games recommendation, and the user would chat with a conversational recommendation assistant for help. 
    And user's historical games are:

    {{history}}

    Information about target game that the user is looking for: 
                        
    {{target_info}}
                        
    Please generate a conversation between the user and the recommendation assistant. Here are some rules:
    1. Do not mention games not in history and do not mention the target item's name.
    2. The assistant doesn't know the user's history, so the user should tell the history in conversation.
    3. In the final turn of the conversation, the assistant should recommend the target the user is looking for. Use '<item>' as placeholder to represent the target.
    4. Do not give too much information in one message and keep each message short.
    5. The conversation should consist of {{num_round}} rounds.
    6. Only the user has the information about target item in his mind. The assistant could only guess from user's messages.
    7. The conversation require {{difficulty}} level education to understand.{{neg_requirement}}
                        
    Use the following format:
    
    [{"role": "User", "text": "xxxxx"}, {"role": "Assistant", "text": "xxxxx"}, ...]
    '''

usersummary_template = '''You are a human game enthusiast. Please make your own user game summary based on your historical play records.
    Please adhere to the following guidelines: 
    1.The user summary should be under {num_words} words.
    2.The user summary should highlight patterns and traits from past games so that it is helpful to identify preferences and predict future game choices.
    3.The writing style should be {writing_style}. 
    4.Do not mention the future predictions in the summary. 
    5.The summary is {clarity} and requires {difficulty} level education to comprehend.
    
    Here is the game play history: {history}
'''

query_template = '''
Based on the given game properties, please generate a query for conveying your likings or search intentions for certain game properties.          

Information about the game is given in json format as follows, with several key value pairs.
{target_info}
 
Please adhere to the following guidelines: 
1. Do not mention the name of the game in the query.
2. Select several key value pairs from the game properties and generate a query based on them.
3. You can rewrite some values while keeping the meaning unchanged, such as replacing it with synonyms or replacing it with some relative values. For example, "price: 22$" can be replaced with "Cost less than 30$", "release date: September 30, 2022" can be replaced with "Released in 2022" and so on. But do not change the value of "developer" and "publisher".
4. The writing style should be {writing_style}.
5. The query should be under {num_words} words.
6. The query is {clarity} and requires {difficulty} level education to comprehend.
'''
neg_query_template = '''
Based on the given game properties, please generate a query for conveying your dislikings or hatings for certain game properties and wanting to exclude those properties when searching for games.          

Information about the game is given in json format as follows, with several key value pairs.
{target_info}

Please adhere to the following guidelines: 
1. Do not mention the name of the game in the query.
2. Select several key value pairs from the game properties and generate a query based on them. Be as diverse as possible.
3. The writing style should be {writing_style}.
4. The query should be under {num_words} words.
5. The query is {clarity} and requires {difficulty} level education to comprehend.
'''