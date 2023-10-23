# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import random
import time
from typing import *

import openai
from langchain.llms import OpenAI
from langchain.agents import (AgentExecutor, LLMSingleActionAgent, Tool)
from langchain.callbacks import get_openai_callback
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from llm4crs.critic import Critic
from llm4crs.demo import DemoSelector
from llm4crs.prompt import *
from llm4crs.utils.prompt import CRSChatPrompt, CRSOutputParser
from loguru import logger


class CRSAgent:

    def __init__(self, domain: str, tools: Dict[str, Callable], candidate_buffer, item_corups, engine: str='text-davinci-003', bot_type: str="completion", 
                 enable_shorten: bool=False, demo_mode: str='zero', demo_dir_or_file: str=None, num_demos: int=3, critic: Critic=None, reflection_limits: int=3, 
                 verbose: bool = True, **kwargs):
        self.domain = domain
        self._domain_map = {'item': self.domain, 'Item': self.domain.capitalize(), 'ITEM': self.domain.upper()}
        self._tool_names = {k: v.name for k,v in tools.items()}
        self._tools = list(tools.values())
        self.candidate_buffer = candidate_buffer
        self.tools = self.setup_tools(self._tools)  # TODO: rename `tools` to `lc_tools`
        self.item_corups = item_corups
        self.engine = engine
        assert bot_type in {'chat', 'completion'}, f"`bot_type` should be `chat` or `completion`, while got {bot_type}"
        self.bot_type = bot_type
        self.mode = 'accuracy'
        self.enable_shorten = enable_shorten
        self.kwargs = kwargs
        if demo_mode == "zero":
            self.selector = None
        else:
            if demo_dir_or_file:
                self.selector = DemoSelector(demo_mode, demo_dir_or_file, k=num_demos, domain=domain)
            else:
                self.selector = None

        self.critic = critic

        # reflection
        self.reflection_limits = reflection_limits
        self._reflection_cnt = 0

        self.verbose = verbose

    @property
    def failed_times(self):
        return 0


    def init_agent(self, temperature: float=0.0):
        llm = self.setup_llm(self.engine, temperature, bot_type=self.bot_type, **self.kwargs)
        self.memory = ConversationBufferMemory(memory_key='chat_history')
        self._memory = ''
        self.prompt = self.setup_prompts(self.tools)
        chain = LLMChain(
            llm=llm,
            prompt=self.prompt,
        )
        self.agent = LLMSingleActionAgent(
            llm_chain=chain,
            output_parser=CRSOutputParser(),
            stop=["\nObservation:"],
            allowed_tools=[tool.name for tool in self.tools]
        )
        self.agent_exe = AgentExecutor.from_agent_and_tools(self.agent, self.tools, verbose=self.verbose, memory=self.memory, max_iterations=5)
        # self.agent = initialize_agent(self.tools, llm, AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION)
        pass

    
    def set_mode(self, mode: str):
        assert mode in ['diversity', 'accuracy'], "`mode` should be `diversity` or `accuracy`"
        self.mode = mode
        for tool in self._tools:
            if hasattr(tool, 'mode'):
                tool.mode = mode
        logger.debug(f"Mode changed to {mode}.")

    def set_style(self, style: str):
        logger.debug(f"Reply style change not supported for {self.__class__.__name__}.")


    def setup_llm(self, engine: str, temperature: float, max_retries: int=10, max_tokens: int=1024, bot_type: str='chat', **kwargs):
        self.set_oai_env()

        if bot_type == 'chat':
            llm = ChatOpenAI(model=engine, temperature=temperature, max_retries=max_retries, max_tokens=max_tokens, engine=engine, request_timeout=60)
        else:
            llm = OpenAI(temperature=temperature, model=engine, max_retries=max_retries, max_tokens=max_tokens, engine=engine, request_timeout=60)
        return llm


    def set_oai_env(self):
        api_type = os.environ.get("OPENAI_API_TYPE", "")
        if len(api_type) > 0: 
            openai.api_base = os.environ.get("OPENAI_API_BASE")
            openai.api_version = os.environ.get("OPENAI_API_VERSION", "2022-12-01" if '4' not in self.engine else "2023-03-15-preview")
            openai.api_type = api_type
        else:
            pass
        openai.api_key = os.environ.get("OPENAI_API_KEY")
    

    def setup_tools(self, tools: List[Callable]) -> List[Tool]:
        """ Wrap tool function into Tool """
        res = []
        for t in tools:
            res.append(
                Tool(
                    name = t.name,
                    func = getattr(t, 'run'),
                    description = t.desc
                )
            )
        return res


    def setup_prompts(self, tools: List[Tool]):
        prompt = CRSChatPrompt(
            table_info=self.item_corups.info(),
            intermediate_steps="",
            template=SYSTEM_PROMPT.format(table_info=self.item_corups.info(), **self._tool_names, **self._domain_map),
            tools=tools,
            input_variables=["input", "intermediate_steps"],
            buffer=self.candidate_buffer,
            memory="",
            examples="",
            reflection=""
        )
        return prompt

    
    def run(self, input: Dict[str, str], chat_history: str=None):
        """Given input text, return response"""
        logger.debug("Input: {}".format(input['input']))
        self.set_oai_env()
        self.candidate_buffer.clear_tracks()
        self.candidate_buffer.clear()
        if chat_history is not None:
            # for one-turn evaluation
            self.prompt.memory = chat_history
        else:   # interactive chat
            if self.enable_shorten:
                self.prompt.memory = self.shorten_history()
            else:
                self.prompt.memory = self.memory.buffer
        if self.selector:
            self.prompt.examples = self.selector(input['input'])
        self.prompt.table_info = self.item_corups.info(query=input['input'])
        
        try:
            response = self.agent_exe.run(input)
        except Exception as e:
            response = "An error raised: {}".format(e)

        # reflection 
        if self.critic is not None:
            rechain, reflection = self.critic(input['input'], response, self.prompt.memory, self.candidate_buffer.track_info)
            if rechain:
                logger.debug("Need Reflection! Reflection information: {}".format(reflection))
                if self._reflection_cnt < self.reflection_limits:
                    self._reflection_cnt += 1
                    # do rechain
                    self.prompt.reflection = reflection
                    # remove the last response from memory
                    self.memory.chat_memory.messages = self.memory.chat_memory.messages[: -2]
                    response = self.run(input)
                    self._reflection_cnt = 0
                else:
                    logger.debug("Reflection exceeds max times, adopt current response.")
            else:
                logger.debug("No rechain needed, good response.")
                self.prompt.reflection = ""
        return response


    def run_gr(self, state):
        text = state[-1][0]
        logger.debug(f"\nProcessing run_text, Input text: {text}\nCurrent state: {state}\n"
            f"Current Memory: {self.memory.buffer}")
        try:
            with get_openai_callback() as cb:
                tic = time.time()
                response = self.run({"input": text.strip()})
                toc = time.time()
                logger.debug(cb)
                logger.debug("Time collapsed: {} s".format(toc-tic))
        except ValueError as e:
            res = str(e)
            if not res.startswith("Could not parse LLM output: `"):
                response = "Something went wrong, please try again."
                logger.debug(f"ValueError: {e}")
            else:
                response = res.removeprefix("Could not parse LLM output: `").removesuffix("`")
                response = response.replace('Question: ', '')
        except Exception as e:
            logger.debug(f"Here is the exception: {e}")
            response = "Something went wrong, please try again."
        state[-1][1] = response
        return state, state


    def clear(self):
        self.memory.clear()
        logger.debug("History Cleared!")


    def shorten_history(self) -> str:
        if len(self.memory.buffer) < 100:
            return self.memory.buffer
        else:
            api_type = os.environ.get("OPENAI_API_TYPE", "")
            if len(api_type) > 0: 
                openai.api_base = os.environ.get("OPENAI_API_BASE")
                openai.api_version = os.environ.get("OPENAI_API_VERSION")
                openai.api_type = api_type
            else:
                pass
            openai.api_key = os.environ.get("OPENAI_API_KEY")
            system_prompt = 'You are a helpful assistant to summarize conversation history and make it shorter. The output should be like:\nHuman: xxxx\nAI: xxx. '
            user_prompt = 'Please help me to shorten the conversational history below. \n{}'.format(self.memory.buffer)
            retry_cnt = 6
            for retry in range(retry_cnt):
                try:
                    if self.bot_type == 'chat':
                        msg = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
                        chat = openai.ChatCompletion.create(
                                model=self.engine, 
                                engine=self.engine,
                                temperature=0.0,
                                messages=msg,
                                request_timeout=10,
                                max_tokens=100
                            )
                        reply = chat.choices[0].message.content
                    else:
                        complete = openai.Completion.create(
                            model=self.engine, 
                            engine=self.engine,
                            prompt=system_prompt + '\n' + user_prompt,
                            temperature=0.0,
                            request_timeout=10,
                            max_tokens=100
                        )
                        reply = complete.choices[0].text
                    break
                except Exception as e:
                    logger.debug(f"Shorten History: An error occurred while making the API call: {e}")
                    reply = self.memory.buffer
                    time.sleep(random.randint(1, 10))
            return reply



__all__ = [
    "CRSAgent"
]
