"""
The following code is modified from
https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
"""

from dataclasses import dataclass
from typing import List

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

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def get_prompt(self, template=None) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.sep_style == 'ADD_COLON_TWO':
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == 'ID2TITLE':
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if i%2==0:
                        ret += role + ": " + template[role].format(message) + seps[0]
                    else:
                        ret += role + ": " + template[role].format(message) + seps[1]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == 'IDs2REVIEW':
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if i%2==0:
                        ret += role + ": " + template[role].format(message[0], message[1]) + seps[0]
                    else:
                        ret += role + ": " + template[role].format(message) + seps[1]
                else:
                    ret += role + ":"
            return ret

template_dict = {
    'sharegpt': Conversation(
        name="sharegpt",
        system_message="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("USER", "ASSISTANT"),
        messages=(),
        offset=0,
        sep_style='ADD_COLON_TWO',
        sep=" ",
        sep2="</s>",
    ),
    'iid2title': Conversation(
        name="iid2title",
        system_message="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("USER", "ASSISTANT"),
        messages=(),
        offset=0,
        sep_style='ID2TITLE',
        sep=" ",
        sep2="</s>",
    ),
    'uid2hist': Conversation(
        name="uid2hist",
        system_message="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("USER", "ASSISTANT"),
        messages=(),
        offset=0,
        sep_style='ID2TITLE',
        sep=" ",
        sep2="</s>",
    ),
    'uidiid2review': Conversation(
        name="uidiid2review",
        system_message="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("USER", "ASSISTANT"),
        messages=(),
        offset=0,
        sep_style='IDs2REVIEW',
        sep=" ",
        sep2="</s>",
    ),
}