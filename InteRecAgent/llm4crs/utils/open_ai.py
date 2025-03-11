# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
import json
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, List, Tuple, Union
import traceback

import openai
from openai import OpenAI, AzureOpenAI

TOKEN_USAGE_VAR = ContextVar(
    "openai_token_usage",
    default={
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "total_tokens": 0,
        "OAI": 0,
    },
)


@contextmanager
def get_openai_tokens():
    TOKEN_USAGE_VAR.set({"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0, "OAI": 0})
    yield TOKEN_USAGE_VAR
    TOKEN_USAGE_VAR.set({"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0, "OAI": 0})


class OpenAICall:
    def __init__(
        self,
        model: str,
        api_key: str,
        api_type: str = "open_ai",
        api_base: str = "https://api.openai.com/v1",
        api_version: str = None,
        temperature: float = 0.0,
        model_type: str = "chat_completion",
        timeout: int = 60,
        # engine: str = None,
        retry_limits: int = 5,
        stop_words: Union[str, List[str]] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.api_type = api_type if api_type else "open_ai"
        self.api_base = api_base if api_base else "https://api.openai.com/v1"
        self.api_version = api_version if api_type!="open_ai" else None
        self.temperature = temperature
        self.model_type = model_type
        self.retry_limits = retry_limits
        self.timeout = timeout
        self.stop_words = stop_words

        if (self.api_type) and (self.api_type not in {"open_ai", "azure"}):
            raise ValueError(
                f"Only open_ai/azure API are supported, while got {api_type}."
            )

        model_type = "chat_completion" if "chat" in model_type else model_type
        if model_type not in {"chat_completion", "completion"}:
            raise ValueError(
                f"Only chat_completion and completion types are supported, while got {model_type}"
            )

        if self.api_type == "azure":
            self.openai_client = AzureOpenAI(
                azure_endpoint=self.api_base,
                api_key=self.api_key,
                api_version=self.api_version,
            )
        else:
            self.openai_client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
            )

    def call(
            self,
            user_prompt: str,
            sys_prompt: str="You are a helpful assistent.",
            max_tokens: int = 512, 
            temperature: float = None
            ) -> str:
        errors = [
            openai.Timeout,
            openai.APIError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.InternalServerError,
        ]
        temperature = temperature if (temperature is not None) else self.temperature
        retry = False
        success = False
        sleep_time = 2
        for _ in range(self.retry_limits):
            try:
                if self.model_type.startswith("chat"):
                    prompt = [
                        {
                            "role": "system",
                            "content": sys_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ]
                    result = self._chat_completion(prompt, max_tokens, temperature)
                else:
                    prompt = f"{sys_prompt} {user_prompt}"
                    result = self._completion(prompt, max_tokens, temperature)
                if result[0]:   # content is not None
                    success = True
                    break
            except Exception as e:
                print(traceback.format_exc())
                for err in errors:
                    if isinstance(e, err):
                        retry = True
                        break
                if retry:
                    result = "Something went wrong, please retry.", {}
                    time.sleep(sleep_time)
                    sleep_time = min(1.5 * sleep_time, 10)
                else:
                    raise e

        _prev_usuage = TOKEN_USAGE_VAR.get()
        _total_usuage = {
            k: _prev_usuage.get(k, 0) + result[1].get(k, 0)
            for k in _prev_usuage.keys()
            if "token" in k
        }
        _total_usuage["OAI"] = _prev_usuage.get("OAI", 0) + 1
        TOKEN_USAGE_VAR.set(_total_usuage)
        if not success:
            reply = "Something went wrong, please retry."
        else:
            reply = result[0]
        return reply

    def _chat_completion(self, msgs: List, max_tokens: int, temperature: float) -> Tuple[str, Dict]:
        kwargs = {
            "model": self.model,
            "messages": msgs,
            "temperature": temperature,
            "timeout": self.timeout,
            "max_tokens": max_tokens,
        }
        if self.stop_words:
            kwargs["stop"] = self.stop_words
        model_resp = self.openai_client.chat.completions.create(**kwargs)
        resp = json.loads(model_resp.json())
        if "choices" in resp:
            message = resp["choices"][0].get("message", None)
            if message:
                content: str = message.get("content", None)
                if content:
                    content = content.strip()
            else:
                content = None
        else:
            content = None

        usage = resp["usage"]

        return content, usage

    def _completion(self, prompt: str, max_tokens: int, temperature: float) -> Tuple[str, Dict]:
        kwargs = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "timeout": self.timeout,
            "max_tokens": max_tokens,
        }
        if self.stop_words:
            kwargs["stop"] = self.stop_words
        model_resp = self.openai_client.chat.completions.create(**kwargs)
        resp = json.loads(model_resp.json())
        if "choices" in resp:
            content: str = resp["choices"][0].get("text", None)
            if content:
                content = content.strip()
        else:
            content = None

        usage = resp["usage"]

        return content, usage


__all__ = ["OpenAICall", "get_openai_tokens"]


if __name__ == "__main__":
    import os
    prompt_msgs = "Which city is the capital of the US?"

    # fastchat API - Vicuna
    # see https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md
    llm0 = OpenAICall(
        model="vicuna-7b-v1.5-16k",
        api_key="EMPTY",
        api_base="http://localhost:8000/v1",
        model_type="chat_completion",
    )

    print(f"Completion from vicuna: {llm0.call(prompt_msgs)}")

    # azure OpenAI key
    azure_api_key = os.getenv("AZURE_API_KEY")
    azure_api_engine = "gpt-4o"
    azure_api_base = os.getenv("AZURE_API_BASE")
    azure_api_version = os.getenv("AZURE_API_VERSION")
    llm1 = OpenAICall(
        model=azure_api_engine,
        api_key=azure_api_key,
        api_type="azure",
        api_base=azure_api_base,
        api_version=azure_api_version,
        model_type="chat_completion",
        stop_words=["\n"]
    )

    print("Azure OpenAI: ", llm1.call(prompt_msgs))

    # personal OpenAI key
    personal_api_key = os.getenv("OPENAI_API_KEY")
    personal_api_base = os.getenv("OPENAI_API_BASE")
    llm2 = OpenAICall(
        model="gpt-3.5-turbo",
        api_base=personal_api_base,
        api_key=personal_api_key,
        model_type="chat_completion",
        api_type="open_ai",
    )
    print("OpenAI: ", llm2.call(prompt_msgs))
