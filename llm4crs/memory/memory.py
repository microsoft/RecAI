
import json

from llm4crs.utils.open_ai import OpenAICall

_FEW_SHOT_EXAMPLES = \
"""
> Conversations
User: My history is ITEM-1, ITEM-2, ITEM-3. Now I want something new.
Assistent: Based on your preference, I recommend you ITEM-17, ITEM-19, ITEM-30.
User: I don't like those items, give me more options.
Assistent: Based on your feedbacks, I recommend you ITEM-5, ITEM-100.
User: I think ITEM-100 may be very interesting. I may like it. 
> Profiles
{"history": ["ITEM-1", "ITEM-2", "ITEM-3"], "like": ["ITEM-100"], "unwanted": ["ITEM-17", "ITEM-19", "ITEM-30"]}

> Conversations
User: I used to enjoy ITEM-89, ITEM-11, ITEM-78, ITEM-67. Now I want something new.
Assistent: Based on your preference, I recommend you ITEM-53, ITEM-10.
User: I think ITEM-10 may be very interesting, but I don't like it.
Assistent: Based on your feedbacks, I recommend you ITEM-88, ITEM-70.
User: I don't like those items, give me more options.
> Profiles
{"history": ["ITEM-89", "ITEM-11", "ITEM-78", "ITEM-67"], "like": [], "unwanted": ["ITEM-10", "ITEM-88", "ITEM-70"]}

"""

class UserProfileMemory:
    """
    The memory is used to store long-term user profile. It can be updated by the conversation and used as the input for recommendation tool.

    The memory consists of three parts: history, like and unwanted. Each part is a set. The history is a set of items that the user has interacted with. The like is a set of items that the user likes. The unwanted is a set of items that the user dislikes.
    """
    def __init__(self, llm_engine=None, **kwargs) -> None:
        if llm_engine:
            self.llm_engine = llm_engine
        else:
            self.llm_engine = OpenAICall(**kwargs)
        self.profile = {
            "history": set([]),
            "like": set([]),
            "unwanted": set([]),
        }

    def conclude_user_profile(self, conversation: str) -> str:
        prompt = "Your task is to extract user profile from the conversation."
        prompt += f"The profile consists of three parts: history, like and unwanted.Each part is a list. You should return a json-format string.\nHere are some examples.\n{_FEW_SHOT_EXAMPLES}\nNow extract user profiles from below conversation: \n> Conversation\n{conversation}\n> Profiles\n"
        return self.llm_engine.call(
            user_prompt=prompt,
            temperature=0.0
        )


    def correct_format(self, err_resp: str) -> str:
        prompt = "Your task is to correct the string to json format. Here are two examples of the format:\n{\"history\": [\"ITEM-1\", \"ITEM-2\", \"ITEM-3\"], \"like\": [\"ITEM-100\"], \"unwanted\": [\"ITEM-17\", \"ITEM-19\", \"ITEM-30\"]}\nThe string to be corrected is {err_resp}. It can not be parsed by Python json.loads(). Now give the corrected json format string.".replace("{err_resp}", err_resp)
        return self.llm_engine.call(
            user_prompt=prompt,
            sys_prompt="You are an assistent and good at writing json string.",
            temperature=0.0
        )


    def update(self, conversation: str):
        cur_profile: str = self.conclude_user_profile(conversation)
        parse_success = False
        limit = 3
        tries = 0
        while not parse_success and tries < limit:
            try:
                cur_profile_dict = json.loads(cur_profile)
                parse_success = True
            except json.decoder.JSONDecodeError as e:
                cur_profile = self.correct_format(cur_profile)
            tries += 1
        if parse_success:
            # update profile
            self.profile['like'] -= set(cur_profile_dict.get('unwanted', []))
            self.profile['like'].update(cur_profile_dict.get('like', []))
            self.profile['unwanted'] -= set(cur_profile_dict.get('like', []))
            self.profile['unwanted'].update(cur_profile_dict.get('unwanted', []))
            self.profile['history'].update(cur_profile_dict.get('history', []))

    def get(self) -> dict:
        return {k: list(v) for k, v in self.profile.items()}
    

    def clear(self):
        self.profile = {
            "history": set([]),
            "like": set([]),
            "unwanted": set([]),
        }
        