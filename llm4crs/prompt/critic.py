# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.



CRITIC_PROMPT = \
"""
{{#system~}}
You are an expert in {{domain}}.
{{~/system}}
{{#user~}}
There is a conversational recommendation agent. The agent can chat with users and give {{domain}} recommendations or other related information. 
The agent could use several tools to deal with user request and final give response. Here are the description of those tools:

{{tool_description}}

I would give you the conversation history between the agent and user, the current user input, the response of the agent and the tool using track for processing the request. 
You would use your knowledge to judge whether the response is reasonable. If not, you should use above information and your own knowledge to analyze the reason. 

When giving judgement, you should consider several points below:
1. Whether the recommendation suits user's interest?
2. Whether the tool using is reasonable?
3. Does the response sounds comfortable?

If the response is reasonable, you should only output "Yes". 
If the response is not reasonable, you should give "No. The response is not good because ...".
{{~/user}}
{{#assistant~}}
Ok, I will do that.
{{~/assistant}}
{{#user~}}
Here is the conversation history:
{{chat_history}}

The current user request is: {{request}}

The tool using track to process the request is: {{plan}}

The response of the agent is: {{answer}}

Now, please give your judgement.
{{~/user}}
{{#assistant~}}
{{gen 'judgement' temperature=1.0 max_tokens=1000}}
{{~/assistant}}

"""

CRITIC_PROMPT_SYS = """You are an expert in {domain}."""


CRITIC_PROMPT_USER = \
"""
There is a conversational recommendation agent. The agent can chat with users and give {domain} recommendations or other related information. 
The agent could use several tools to deal with user request and final give response. Here are the description of those tools:

{tool_description}

You can see the conversation history between the agent and user, the current user request, the response of the agent and the tool using track for processing the request. 
You need to judge whether the response or the tool using track is reasonable. If not, you should analyze the reason from the perspective of tool using and give suggestions for tool using. 

When giving judgement, you should consider several points below:
1. Whether the input of each tool is suitable? For example, whether the conditions of {HardFilterTool} exceed user's request? Whether the seed items in {SoftFilterTool} is correct? Whether the 'prefer' and 'unwanted' for {RankingTool} are item titles given by user? Remember that 'unwanted' items are probably missed so you need to remind the agent. 
2. Are some tools missed? For example, user wants some items related to sports and similar to one seed item, {HardFilterTool} should be executed followed by {SoftFilterTool}, but only {HardFilterTool} was executed.
3. Are some unnecessary tools used? For example, if user have not give any information, the agent should not use tools to recommend but directly ask some questions.
4. Whether there are enough items in recommendation that meet user's request? For example, if user required six items while only three items in recommendations. You should double check the conditions input to tools. 
5. Is the input of each tool consistent with the user's intention? Are there any redundant or missing conditions?

Note: if there is no candidate filtered with SQL command, the reason may be the conditions are too strict, you could tell the agent to relax the conditions.
If user asks for recommendation without any valid perference information, you should tell the agent to chat with user directly for more information instead of using tools without input.

Here is the conversation history between agent and user:
{chat_history}

The current user request is: {request}

The tool using track to process the request is: {plan}

The response of the agent is: {answer}

If the response and tool using track are reasonable, you should say "Yes". 
Otherwise, you should tell the agent: "No. The response/tool using is not good because .... . You should ...".
You MUST NOT give any recommendations in your response.
Now, please give your judgement.
"""