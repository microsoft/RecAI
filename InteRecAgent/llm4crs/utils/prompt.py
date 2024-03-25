# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
from typing import *
from typing import Any, List
from loguru import logger

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate, StringPromptTemplate
from langchain.schema import BaseMessage
from langchain.schema import AgentAction, AgentFinish, HumanMessage



class CRSChatPrompt(StringPromptTemplate):
    intermediate_steps: str
    template: str 
    tools: List[Tool]
    buffer: Any
    memory: str
    examples: str
    reflection: str
    table_info: str


    def format(self, **kwargs: Any) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        self.intermediate_steps = thoughts[:-9]
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        kwargs["buffer_state"] = self.buffer.state()
        kwargs["history"] = self.memory
        kwargs["examples"] = self.examples
        if self.reflection:
            kwargs["reflection"] = self.reflection
        else:
            kwargs["reflection"] = ''
        kwargs["table_info"] = self.table_info
        return self.template.format(**kwargs)



class CRSOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            logger.debug(f"LLM's output not matched: {llm_output.strip()}")
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.strip()},
                log=llm_output,
            )
        else:
            action = match.group(1).strip()
            action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)



if __name__ == "__main__":
    print()