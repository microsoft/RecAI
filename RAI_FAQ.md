# RecAI: Responsible AI FAQ

## What is RecAI?

RecAI includes some techniques bridging LLMs with traditional recommender models, with the goal of building next-generation intelligence recommender systems.  Large Language Models (LLMs) offer significant potential for the development of cutting-edge recommender systems, particularly in terms of enhancing interactivity, explainability, and controllability. These are aspects that have traditionally posed challenges. However, the direct application of a general-purpose LLM for recommendation purposes is not viable due to the absence of specific domain knowledge. The RecAI project aims to bridge this gap by investigating effective strategies to integrate LLMs with recommender systems, a concept we term as LLM4Rec. The goal is to reflect the real-world needs of LLM4Rec through a comprehensive review and experimentation of various methodologies. 

## What can RecAI do?

RecAI utilizes pre-trained domain-specific recommendation-related models (such as SQL tools, id-based recommendation models) as tools, and a large language model (LLM) as the brain, to implement an interactive, conversational recommendation agent. 

RecAI's input is user's input text; in the middle, the LLM will understand user's intention, call recommender tools, get the necessary item information, pass the information to the LLM to summary a result, then finally deliver the result back to the user.
 
In RecAI, the LLM primarily engages in user interaction and parses user interests as input for the recommendation tools, which are responsible for finding suitable items. RecAI will not modify the LLM or the provided tools. RecAI only serves as a connector to bridge the LLM and tools.

## What is/are RecAI’s intended use(s)?

1.	Convert traditional recommender systems into an interactive, explainable, and controllable recommender system.

2.	Empower a generic LLM with the domain-specific recommendation ability.

## How was RecAI evaluated? What metrics are used to measure performance?

To enable the quantitative assessment of RecAI, we have designed two evaluation strategies:

1.	User Simulator. We have designed a role-playing prompt to guide GPT-4 in simulating users interacting with conversational recommendation agents. A user’s historical behavior is integrated into the prompt as their profile, with the last item in their history serving as the target item they wish to find. In this manner, GPT-4 behaves from the user’s perspective and promptly responds to the recommended results, creating a more realistic dialogue scenario. This strategy is employed to evaluate the performance of InteRecAgent in multi-turn dialogue settings.

2.	One-Turn Recommendation. Given a user’s history, we design a prompt that enables GPT-4 to generate a dialogue, simulating the communication between a user and a rec-agent. The goal is to test whether a rec-agent can accurately recommend the ground truth item in the next response. We evaluate both entire space retrieval and candidate-provided ranking tasks. Specifically, the dialogue context is supplied to the recommendation agent, along with the instruction. Please give me k recommendations based on the chat history for retrieval task, and the instruction Please rank these candidate items base on the chat history for ranking task. 

Detailed experimental results please refer to our paper [*Recommender AI Agent: Integrating Large Language Models for Interactive Recommendations*](https://arxiv.org/abs/2308.16505).

## What are the limitations of RecAI? How can users minimize the impact of RecAI’s limitations when using the system?

The response speed is 2 to 3 times slower than a direct response from an LLM such as GPT-4. This is because in the back end, multiple rounds of LLM inference happen before the result is returned to the users. 

[uses for which the system was not designed] real-time recommendations such as homepage recommendations.

[steps to minimize errors] use a stronger LLM in RecAI (such as use GPT-4 instead of GPT-3.5)

## What operational factors and settings allow for effective and responsible use of RecAI?

Consider that (1) RecAI will not modify the provided LLM and recommender tools; and (2) RecAI is focused on connecting LLM and recommender tools, itself will not produce text content to users, all the generated content are from the given LLM and recommender tools, thus, when users want to use RecAI, they should use trustworthy LLMs (such as GPT-4) and recommender tools (such as trained on their own dataset).
Below is a detailed list of choices that end users can customize:

1.	LLM: A large language model, which serves as a brain. Such as GPT-4 and Llama 2.

2.	Item profile table: A table containing item informations, whose columns consists of id, title, tag, description, price, release date, popularity, et al.

3.	Query module: A SQL module to query item information in the item profile table.
 
4.	Retrieval module: The module aims to retrieve item candidates from the all item corups according to user's intention (requirements). Note that the module does not function in deal with user's personal profile, like user history, user age, et al. Instead, it focuses on what user wants, like "give me some sports games", "I want some popular games". The module should consist of at least two kinds of retrieval tools:

5.	SQL tool: The tool is used to deal with complex search condition, which is related to item information. For example, "I want some popular sports games". Then the tool would use SQL command to search in the item profile table.
 
6.	Item similarity tool: The tools aims to retrieve items according to item similarity. Sometimes, user's intention is not clear enough to organized as SQL command, for example, "I want some games similar with Call of Duty", where the requirements is expressed implicitly through item similarity instead of explicit item features.

7.	Ranking module: Refine the rank of item candidates according to schema (popularity, similarity, preference). User preference comprises prefer and unwanted. The module could be a traditional recommender model, which inputs user and item features and outputs relevant score.

## What system prompt framework should be considered for reponsible AI?

We have followed the instructions in https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/system-message to design the system prompts. For users who want to add or change prompts to fit specific scenarios, please also consider these instructions.