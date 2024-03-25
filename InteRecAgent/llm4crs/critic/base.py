# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import re
from typing import *

import openai
from llm4crs.prompt import *
from llm4crs.utils.open_ai import OpenAICall



def parse_output(s: str) -> Tuple[bool, str]:
    yes_pattern = r"Yes.*"
    no_pattern = r"No\. (.*)"
    yes_matches = re.findall(yes_pattern, s)
    if yes_matches:
        need_reflection = False
        info = ""
    else:
        no_matches = re.findall(no_pattern, s)
        if no_matches:
            need_reflection = True
            info = no_matches[0]
        else:
            need_reflection = False
            info = ""
    return need_reflection, info


class Critic:
    def __init__(
        self,
        model,
        engine,
        buffer,
        domain: str = "video games",
        bot_type="chat",
        temperature: float = 0.0,
        timeout: int = 60,
    ):
        self.model = model
        self.engine = engine
        self.buffer = buffer
        self.domain = domain
        self.timeout = timeout

        assert bot_type in {
            "chat",
            "completion",
        }, f"`bot_type` should be `chat` or `completion`, while got {bot_type}"
        self.bot_type = bot_type
        self.bot = OpenAICall(
            model=self.engine,
            api_key=os.environ["OPENAI_API_KEY"],
            api_type=os.environ.get("OPENAI_API_TYPE", None),
            api_base=os.environ.get("OPENAI_API_BASE", None),
            api_version=os.environ.get("OPENAI_API_VERSION", None),
            temperature=temperature,
            model_type=self.bot_type,
            timeout=self.timeout,
        )

    def __call__(self, request: str, answer: str, history: str, tracks: str):
        output = self._call(request, answer, history, tracks)
        need_reflection, info = parse_output(output)
        if need_reflection:
            info = f"Your previous response: {answer}. \nThe response is not reasonable. Here is the advice: {info} "
        return need_reflection, info

    def _call(self, request: str, answer: str, history: str, tracks: str):
        sys_msg = CRITIC_PROMPT_SYS.format(domain=self.domain)
        item_map = {
            "item": self.domain,
            "Item": self.domain.capitalize(),
            "ITEM": self.domain.upper(),
        }
        tool_names = {k: v.format(**item_map) for k, v in TOOL_NAMES.items()}
        usr_msg = CRITIC_PROMPT_USER.format(
            domain=self.domain,
            tool_description=OVERALL_TOOL_DESC.format(**tool_names, **item_map),
            chat_history=history,
            request=request,
            plan=tracks,
            answer=answer,
            **TOOL_NAMES,
        )

        reply = self.bot.call(
            sys_prompt=sys_msg,
            user_prompt=usr_msg,
            max_tokens=128
        )
        return reply
