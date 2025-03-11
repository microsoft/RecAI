"""
The following code is modified from
https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
"""

from dataclasses import dataclass
from typing import List

TEXT = ["sharegpt"]
ID2TEXT = ["iid2title", "iid2feature", "iid2description", "iid2brand", "iid2tags", "iid2sim", "feature2iid", "description2iid", "uid2hist", "uid2summary", "uid2next"]
IDS2TEXT = ["uidiid2review", "uidiid2rank", "uidiid2binary", "uidiid2explan", "demo"]


@dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = "{system_message}"
    # The system message
    system_message: str = ""
    # The names of two roles
    roles: List[str] = (("USER", "ASSISTANT"),)
    # All messages. Each item is (role, message).
    messages: List[List[str]] = ()
    # The number of few shot examples
    offset: int = 0
    # The separator style and configurations
    sep_style: str = 'ADD_COLON_TWO'
    sep: str = "\n"
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: str = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None
    roles_mapping = {"[INST]": "USER", "[/INST]": "ASSISTANT", "user": "USER", "assistant": "ASSISTANT", "<|user|>\n": "USER", "<|assistant|>\n": "ASSISTANT",}

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def get_prompt(self, name, template=None) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.sep_style == 'ADD_COLON_TWO':
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if i%2==0:
                        if name in TEXT:
                            m = message
                        elif name in ID2TEXT:
                            m = template[role].format(message)
                        elif name in IDS2TEXT:
                            m = template[role].format(message[0], message[1])
                        ret += role + ": " + m + seps[0]
                    else:
                        if name in TEXT:
                            m = message
                        elif name in ID2TEXT + IDS2TEXT:
                            m = template[role].format(message)
                        ret += role + ": " + m + seps[1]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == "LLAMA2":
            seps = [self.sep, self.sep2]
            if self.system_message:
                ret = system_prompt
            else:
                ret = "[INST] "
            for i, (role, message) in enumerate(self.messages):
                tag = self.roles[i % 2]
                if message:
                    if i%2==0:
                        if name in TEXT:
                            m = message
                        elif name in ID2TEXT:
                            m = template[self.roles_mapping[role]].format(message)
                        elif name in IDS2TEXT:
                            m = template[self.roles_mapping[role]].format(message[0], message[1])

                        if i == 0:
                            ret += m + " "
                        else:
                            ret += tag + " " + m + seps[0]
                    else:
                        if name in TEXT:
                            m = message
                        elif name in ID2TEXT + IDS2TEXT:
                            m = template[self.roles_mapping[role]].format(message)
                        ret += tag + " " + m + seps[1] #tag + m + seps[1]
                else:
                    ret += tag
            return ret
        elif self.sep_style == "LLAMA3":
            ret = "<|begin_of_text|>"
            if self.system_message:
                ret += system_prompt
            else:
                ret += ""
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if i%2==0:
                        if name in TEXT:
                            m = message
                        elif name in ID2TEXT:
                            m = template[self.roles_mapping[role]].format(message)
                        elif name in IDS2TEXT:
                            m = template[self.roles_mapping[role]].format(message[0], message[1])
                        ret += f"<|start_header_id|>{role}<|end_header_id|>\n\n{m.strip()}<|eot_id|>"
                    else:
                        if name in TEXT:
                            m = message
                        elif name in ID2TEXT + IDS2TEXT:
                            m = template[self.roles_mapping[role]].format(message)
                        ret += f"<|start_header_id|>{role}<|end_header_id|>\n\n{m.strip()}<|eot_id|>"
                else:
                    ret += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
            return ret
        elif self.sep_style == "PHI3":
            if self.system_message:
                ret = system_prompt
            else:
                ret = ""
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if i%2==0:
                        if name in TEXT:
                            m = message
                        elif name in ID2TEXT:
                            m = template[self.roles_mapping[role]].format(message)
                        elif name in IDS2TEXT:
                            m = template[self.roles_mapping[role]].format(message[0], message[1])
                        ret += role + m + self.stop_str
                    else:
                        if name in TEXT:
                            m = message
                        elif name in ID2TEXT + IDS2TEXT:
                            m = template[self.roles_mapping[role]].format(message)
                        ret += role + m + self.stop_str
                else:
                    ret += role
            return ret

template_dict = {
    'vicuna': Conversation(
        name="vicuna",
        system_message="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("USER", "ASSISTANT"),
        messages=(),
        offset=0,
        sep_style='ADD_COLON_TWO',
        sep=" ",
        sep2="</s>",
    ),
    'mistral': Conversation(
        name="mistral",
        system_template="[INST] {system_message}\n",
        roles=("[INST]", "[/INST]"),
        sep_style="LLAMA2",
        sep=" ",
        sep2="</s>",
    ),
    'llama-2': Conversation(
        name="llama-2",
        system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
        roles=("[INST]", "[/INST]"),
        sep_style="LLAMA2",
        sep=" ",
        sep2=" </s><s>",
    ),
    'llama-3': Conversation(
        name="llama-3",
        system_template="<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>",
        roles=("user", "assistant"),
        sep_style="LLAMA3",
        sep="",
        stop_str="<|eot_id|>",
        stop_token_ids=[128001, 128009],
    ),
    'phi3': Conversation(
        name="phi3",
        system_message="You are a helpful AI assistant.",
        system_template="<|system|>\n{system_message}<|end|>\n",
        roles=("<|user|>\n", "<|assistant|>\n"),
        sep_style="PHI3",
        sep="",
        stop_str="<|end|>\n",
    )
}