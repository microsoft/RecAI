# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import random
import re
import sys
import math
import json
import time
from loguru import logger
import openai
import argparse
import threading  
from typing import *
import numpy as np
import pandas as pd
from tqdm import tqdm
from rapidfuzz import fuzz
from datetime import datetime
from faiss import IndexFlatIP
from time import strftime, localtime
from langchain.callbacks import get_openai_callback
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings


from llm4crs.prompt import *
from llm4crs.utils import FuncToolWrapper, OpenAICall, get_openai_tokens
from llm4crs.corups import BaseGallery
from llm4crs.agent import CRSAgent
from llm4crs.agent_plan_first import CRSAgentPlanFirst
from llm4crs.agent_plan_first_openai import CRSAgentPlanFirstOpenAI
from llm4crs.environ_variables import *
from llm4crs.critic import Critic
from llm4crs.mapper import MapTool
from llm4crs.query import QueryTool
from llm4crs.ranking import RecModelTool
from llm4crs.buffer import CandidateBuffer
from llm4crs.retrieval import SQLSearchTool, SimilarItemTool


user_simulator_sys_prompt = \
"""
You are a user chatting with a recommender for {domain} recommendation in turn. 
"""

user_simulator_template = \
"""
Your history is {history}. Your target items: {target}. 
You must follow the instructions below during chat. 
If the recommender recommends {target}, you should accept. 
If the recommender recommends other items, you should refuse them and provide the information about {target}. 
You should never directly tell the target item title. 
If the recommender asks for your preference, you should provide the information about {target}. 
You could provide your history. 
You should never directly tell the target item title. 
Now lets start, you first, act as a user. 
Your output is only allowed to be the words from the user you act.
If you think the conversation comes to an ending, output a <END>.
You should never directly tell the target item. 
Here is the information about target you could use: {target_item_info}.
Only use the provided information about the target.  
Never give many details about the target items at one time. Less than 3 conditions is better.
Never recommend items to the assistant. Never tell the target item title.
"""

def read_jsonl(fpath: str) -> List[Dict]:
    res = []
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            res.append(data)
    return res


class Conversation:

    def __init__(self, user_prefix='User', agent_prefix='Assistant', save_path: str=None):
        self.user_prefix = user_prefix
        self.agent_prefix = agent_prefix
        self.all_history = []
        self.history = []
        self.save_path = save_path

    def add_user_msg(self, msg) -> None:
        self.history.append({'role': self.user_prefix, 'msg': msg})
        return

    def add_agent_msg(self, msg) -> None:
        self.history.append({'role': self.agent_prefix, 'msg': msg})
        return

    @property
    def total_history(self) -> str:
        res = ""
        for h in self.history:
            res += f"{h['role']}: {h['msg']}\n"
        res = res[:-1]
        return res

    @property
    def turns(self) -> int:
        return math.ceil(len(self.history) / 2)

    def __len__(self) -> int:
        return len(self.history)

    def clear(self, data_index: int, label: int) -> None:
        if len(self.history) > 0:
            data = {'id': data_index, 'conversation': self.history, 'label': label}
            self.all_history.append(data)
            if self.save_path:
                with open(self.save_path, 'a', encoding='utf-8') as f:
                    line = json.dumps(data, ensure_ascii=False) + "\n"
                    f.write(line)
        self.history = []

    def dump(self, fpath: str):
        with open(fpath, 'w', encoding='utf-8') as f:
            for entry in self.all_history:
                json.dump(entry, f)
                f.write('\n')


class OpenAIBot:
    def __init__(
        self,
        domain: str,
        engine: str,
        api_key: str,
        api_type: str,
        api_base: str,
        api_version: str,
        conversation: Conversation,
        timeout: int,
        model_type: str = "chat_completion"
    ):
        self.domain = domain
        self.engine = engine
        self.api_key = api_key
        self.api_type = api_type
        self.api_base = api_base
        self.api_version = api_version
        self.conversation = conversation
        self.timeout = timeout
        self.engine = OpenAICall(
            model=engine,
            api_key=api_key,
            api_type=api_type,
            api_base=api_base,
            api_version=api_version,
            temperature=0.8,
            model_type=model_type,
            timeout=timeout
        )

    def run(self, inputs: Dict) -> str:
        sys_prompt = f"You are a helpful conversational agent who is good at {self.domain} recommendation. "

        usr_prompt = (
            f"Here is the conversation history: \n{self.conversation.total_history}\n"
            f"User: {inputs['input']} \nAssistant: "
        )

        reply = self.engine.call(
            user_prompt=usr_prompt,
            sys_prompt=sys_prompt,
            max_tokens=256,
            temperature=0.8
        )
        return reply


class ChatRec:
    """ ChatRec method, referred in the paper "Chat-rec: Towards interactive and explainable llms-augmented recommender system". Paper link: https://arxiv.org/abs/2303.14524

    The main idea of ChatRec is to combine a text-embedding model with ChatGPT. 
    """
    def __init__(
        self,
        domain: str,
        engine: str,
        api_key: str,
        api_type: str,
        api_base: str,
        api_version: str,
        conversation: Conversation,
        timeout: int,
        item_corups: BaseGallery,
        model_type: str = "chat_completion",
        embed_vec_dir_path: str = None,
        embedding_model_deployment_name="text-embedding-ada-002",
    ):
        self.domain = domain
        self.engine = engine
        self.api_key = api_key
        self.api_type = api_type
        self.api_base = api_base
        self.api_version = api_version
        self.conversation = conversation
        self.timeout = timeout
        self.engine = OpenAICall(
            model=engine,
            api_key=api_key,
            api_type=api_type,
            api_base=api_base,
            api_version=api_version,
            temperature=0.8,
            model_type=model_type,
            timeout=timeout
        )
        if self.api_type == 'openai':
            kwargs = {
                'model': "text-embedding-ada-002",
                'openai_api_key': self.api_key
            }
        else:
            kwargs = {
                'model': "text-embedding-ada-002",
                'openai_api_key': self.api_key,
                'deployment': embedding_model_deployment_name,
                'openai_api_base': self.api_base,
                'openai_api_type': self.api_type
            }
        self.embedding_model = OpenAIEmbeddings(**kwargs)
        success_load = self.load_emb(embed_vec_dir_path)
        if not success_load:
            self.item_title, self.item_vecs = self.encode_items(item_corups)
            self.save_emb(embed_vec_dir_path)
        self.search_engine = IndexFlatIP(self.item_vecs.shape[1])
        self.search_engine.add(self.item_vecs)
        self.num_candidates = 20

    
    def encode_items(self, item_corups):
        item_texts = self._get_all_item_text(item_corups.corups)
        embed_vecs = self.embedding_model.embed_documents(item_texts)
        item_title = item_corups.corups['title'].to_numpy()
        embed_vecs = np.array(embed_vecs)
        return item_title, embed_vecs
    
    def save_emb(self, dir_name: str):
        if dir_name is None:
            timestamp = strftime('%Y%m%d-%H%M%S',localtime())
            dir_name = f"./chatrec_embed_vec_cache/{self.domain}/{timestamp}"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(f"{dir_name}/item_names.txt", "w") as f:
            for t in self.item_title:
                f.write(f"{t}\n")
        np.save(f"{dir_name}/item_vectors.npy", self.item_vecs)
        logger.info(f"ChatRec embedding file saved in {dir_name}.")
    
    
    def load_emb(self, dir_name: str) -> bool:
        if dir_name is None:
            return False
        if not os.path.exists(dir_name):
            return False
        if not os.path.exists(f"{dir_name}/item_names.txt"):
            logger.info("Not found ChatRec item names file.")
            return False
        if not os.path.exists(f"{dir_name}/item_vectors.npy"):
            logger.info("Not found ChatRec item vectors file.")
            return False
        with open(f"{dir_name}/item_names.txt", "r") as f:
            item_titles = f.readlines()
        self.item_title = np.array(item_titles)
        self.item_vecs = np.load(f"{dir_name}/item_vectors.npy")
        logger.info(f"Load vectors from {dir_name}.")
        return True
        

    def find_candidates(self, query_emb):
        query_emb = np.array(query_emb)[None, :]
        score, index = self.search_engine.search(query_emb, self.num_candidates)
        titles = self.item_title[index]
        return titles
    
    
    def _get_all_item_text(self, corups: pd.DataFrame):
        item_texts = []
        dict_records = corups.to_dict(orient='records')
        for r in dict_records:
            text = ''.join([f"The {key} of the item is {value}." for key, value in r.items()])
            item_texts.append(text)
        return item_texts
        

    def run(self, inputs: Dict) -> str:
        sys_prompt = "You need to recommend items to a user based on conversation. "
        user_input = f"{self.conversation.total_history}\nUser: {inputs['input']}"
        query_emb = self.embedding_model.embed_query(user_input)
        candidate_list = self.find_candidates(query_emb)

        usr_prompt = (
            f"Here is the conversation history: \n{self.conversation.total_history}\n"  
            f"Here is a list of items that he is likely to like: {candidate_list}\n"
            "Please select less than five items from the list to recommend. Only output the item name."
            f"User: {inputs['input']} \nAssistant: "
        )

        reply = self.engine.call(
            user_prompt=usr_prompt,
            sys_prompt=sys_prompt,
            max_tokens=256,
            temperature=0.8
        )
        return reply



class Simulator:

    def __init__(
            self,
            domain: str,
            engine: str,
            api_key: str,
            api_type: str,
            api_base: str,
            api_version: str,
            model_type: str,
            conversation: Conversation,
            timeout: int
        ):
        self.conversation = conversation
        self.engine = engine
        self.domain = domain
        self.history = None
        self.target = None
        self.target_info = None
        self.timeout = timeout
        self.engine = OpenAICall(
            model=engine,
            api_key=api_key,
            api_type=api_type,
            api_base=api_base,
            api_version=api_version,
            temperature=0.8,
            model_type=model_type,
            timeout=timeout
        )

    def set(self, history: str, target: str, target_info: str) -> None:
        self.history = history
        self.target = target
        self.target_info = target_info

    def __call__(self) -> str:
        sys_prompt = user_simulator_sys_prompt.format(domain=self.domain)
        user_prompt = user_simulator_template.format(
            domain=self.domain,
            history=self.history,
            target=self.target,
            target_item_info=self.target_info
        )
        if len(self.conversation) == 0:
            pass
        else:
            user_prompt += f"\nHere are the conversation history: \n{self.conversation.total_history}"

        user_prompt += "User: "
        target_related_score = 100

        n_simulator_tries = 0
        while (target_related_score >= 60) and (n_simulator_tries < 5):
            # detect whether simulator gives the target directly
            if n_simulator_tries > 0:
                user_prompt += "\nDo not tell the target item directly."
            reply = self.engine.call(
                sys_prompt=sys_prompt,
                user_prompt=user_prompt,
                max_tokens=256,
                temperature=0.8
            )
            target_related_score = fuzz.partial_ratio(self.target, reply)
            n_simulator_tries += 1

        self.conversation.add_user_msg(reply)

        reply = reply.replace("User:", "").strip()
        return reply


def hit_judge(msg: str, target: str, thres: float=80):
    msg = re.sub(r"[^a-zA-Z0-9\s]", "", msg.lower())
    target = re.sub(r"[^a-zA-Z0-9\s]", "", target.lower())
    if fuzz.partial_ratio(msg, target) > thres:
        return True
    else:
        return False

def conversation_eval(data: List[Dict], agent: CRSAgent, simulator: Simulator, conversation: Conversation, item_corup: BaseGallery, max_turns: int=10, 
                      recbot: bool=False, start: int=0, end: int=None):
    hit_turn = []
    hit_num = 0
    n_oai = []
    n_fail = []
    end = len(data) if end is None else end
    i = start
    N = 0
    pbar = tqdm(total=end-start)
    hit_flag = False
    while i < end:
        if recbot:
            if getattr(agent, "planning_recording_file", None):
                agent.save_plan(reward=int(hit_flag))
            agent.clear()
            if (i > start) and ((i-start) % 10 == 0):
                agent.clear()
                time.sleep(random.randint(30, 60))  # wait for a while at the begining of each conv
        d = data[i]
        try:
            target = item_corup.fuzzy_match(d['target'], 'title')
            target_info = item_corup.convert_title_2_info(target)
            target_info = {k: str(v) for k,v in target_info.items() if k not in {'id', 'visited_num'}}
            simulator.set(**d, target_info=json.dumps(target_info))  # {'history': xxxx, 'target': xxx}
            n = 0
            oai = 0
            fail = 0
            hit_flag = False
            while n < max_turns:
                try:
                    usr_msg = ""
                    agent_msg = ""
                    usr_msg = simulator()   # user simulartor
                    tqdm.write(f"User: {usr_msg}")
                    if "<END>" in usr_msg:
                        break
                    if recbot:
                        time.sleep(random.randint(5, 15))  # wait for a while
                        with get_openai_tokens() as cb:
                            agent_msg = agent.run({'input': usr_msg})   # conversational agent
                            oai += cb.get()['OAI']
                            fail += agent.failed_times
                    else:
                        agent_msg = agent.run({'input': usr_msg})   # conversational agent
                    tqdm.write(f"Assistant: {agent_msg}")
                    conversation.add_agent_msg(agent_msg)
                    if hit_judge(agent_msg, d['target']):
                        hit_num += 1
                        hit_turn.append(n+1)
                        hit_flag = True
                        break
                    else:
                        pass
                    n += 1
                except openai.error.InvalidRequestError as e:
                    if "content management policy" in str(e):
                        print("Prompt trigger content management policy.")
                        break
            if not hit_flag:  # not hit
                hit_turn.append(max_turns+1)
            n_oai.append(oai)
            n_fail.append(fail)
            conversation.clear(data_index=i, label=int(hit_flag))
            AT = sum(hit_turn) / (N+1)
            oai_report = sum(n_oai) / (N+1)
            fail_report = sum(n_fail) / (N+1)
            tqdm.write(f"Sample {i}: Hit={hit_num}/{N+1}={(hit_num/(N+1)):.4f}, AT={AT:.4f}, OAI={oai_report:.4f}, Fail={fail_report:.4f}")
            pbar.update(1)
            N += 1
            i += 1
        except KeyboardInterrupt:
            print("Catch Keyboard Interrupt.")
            break

    res = {}
    if recbot:
        OAI = sum(n_oai) / (len(n_oai) + 1e-10)
        failed_times = sum(n_fail) / (len(n_fail) + 1e-10)
        res.update({'OAI': OAI, 'failed_plan': failed_times})
    hit_ratio = hit_num / (N)
    AT = sum(hit_turn) / (N)
    res.update({"hit": hit_ratio, "AT": AT})
    return res


def main():
    parser = argparse.ArgumentParser("Evaluator")
    parser.add_argument("--data", type=str, default="./data/steam/simulator_test_data.jsonl")
    parser.add_argument("--start_test_num", type=int, default=0, help="the start point in test data, for continual evaluation")
    parser.add_argument("--end_test_num", type=int, help="the end point in test data, for continual evaluation")
    parser.add_argument("--max_turns", type=int, default=5, help="max turns limit for evaluation")
    parser.add_argument("--save", type=str, help='path to save conversation text')
    parser.add_argument("--timeout", type=int, default=5, help="Timeout threshold when calling OAI. (seconds)")

    parser.add_argument('--agent', type=str, help='agent type, "recbot" is our method and others are baselines')

    parser.add_argument('--max_candidate_num', type=int, default=1000, help="Number of max candidate number of buffer")
    parser.add_argument('--similar_ratio', type=float, default=0.1, help="Ratio of returned similar items / total games")
    parser.add_argument('--rank_num', type=int, default=100, help="Number of games given by ranking tool")
    parser.add_argument('--max_output_tokens', type=int, default=512, help="Max number of tokens in LLM output")

    # chat history shortening
    parser.add_argument('--enable_shorten', type=int, choices=[0,1], default=0, help="Whether to enable shorten chat history with LLM")

    # dynamic demonstrations
    parser.add_argument('--demo_mode', type=str, choices=["zero", "fixed", "dynamic"], default="zero", help="Directory path of demonstrations")
    parser.add_argument('--demo_dir_or_file', type=str, help="Directory or file path of demonstrations")
    parser.add_argument('--chatrec_vec_dir', type=str, help="Directory to save or load embedding vector")
    parser.add_argument('--num_demos', type=int, default=3, help="number of demos for in-context learning")

    # reflection mechanism
    parser.add_argument('--enable_reflection', type=int, choices=[0,1], default=0, help="Whether to enable reflection")
    parser.add_argument('--reflection_limits', type=int, default=3, help="limits of reflection times")

    # plan first agent
    parser.add_argument('--plan_first', type=int, choices=[0,1], default=0, help="Whether to use plan first agent")

    parser.add_argument("--langchain", type=int, choices=[0, 1], default=0, help="Whether to use langchain in plan-first agent")
    
    parser.add_argument("--plan_record_file", type=str, help="The file path to save records of plans")
    
    args, _ = parser.parse_known_args()

    logger.remove()
    logger.add(sys.stdout, colorize=True, level="DEBUG", format="<lc>[{time:YYYY-MM-DD HH:mm:ss} {level}] <b>{message}</b></lc>")

    eval_data = read_jsonl(args.data)
    # save conversation
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    if args.save is not None:
        save_path = args.save
    else:
        start = args.start_test_num
        end = len(eval_data) if args.end_test_num is None else args.end_test_num
        save_path = os.path.join(os.path.dirname(args.data), f"saved_conversations_{args.agent}_{os.environ.get('OPENAI_ENGINE', 'None')}_from_{start}_to_{end}_{current_time}_{os.path.basename(args.data)}")

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    domain = os.environ["DOMAIN"]
    print(f"Domain = {domain}.")

    domain_map = {'item': domain, 'Item': domain.capitalize(), 'ITEM': domain.upper()}


    conversation = Conversation(save_path = save_path)
    simulator = Simulator(
        domain=domain,
        engine=os.environ['SIMULATOR_ENGINE'],
        api_key=os.environ['SIMULATOR_API_KEY'],
        api_type=os.environ.get('SIMULATOR_API_TYPE', 'open_ai'),
        api_version=os.environ.get('SIMULATOR_API_VERSION', None),
        api_base=os.environ.get('SIMULATOR_API_BASE', 'https://api.openai.com/v1'),
        model_type=os.environ.get('SIMULATOR_ENGINE_TYPE', 'chat_completion'),
        conversation=conversation,
        timeout=args.timeout
    )

    item_corups = BaseGallery(
        fpath=GAME_INFO_FILE,
        column_meaning_file=TABLE_COL_DESC_FILE,
        name=f'{domain}_information',
        columns=USE_COLS,
        fuzzy_cols=['title'] + CATEGORICAL_COLS,
        categorical_cols=CATEGORICAL_COLS
    )

    if args.agent == 'recbot':

        tool_names = {k: v.format(**domain_map) for k,v in TOOL_NAMES.items()}

        candidate_buffer = CandidateBuffer(item_corups, num_limit=args.max_candidate_num)

        # The key of dict here is used to map to the prompt
        tools = {
                "BufferStoreTool": FuncToolWrapper(func=candidate_buffer.init_candidates, name=tool_names['BufferStoreTool'], 
                                                desc=CANDIDATE_STORE_TOOL_DESC.format(**domain_map)),
                "LookUpTool": QueryTool(name=tool_names['LookUpTool'], desc=LOOK_UP_TOOL_DESC.format(**domain_map), item_corups=item_corups, buffer=candidate_buffer),
                "HardFilterTool": SQLSearchTool(name=tool_names['HardFilterTool'], desc=HARD_FILTER_TOOL_DESC.format(**domain_map), item_corups=item_corups, 
                                                buffer=candidate_buffer, max_candidates_num=args.max_candidate_num),
                "SoftFilterTool": SimilarItemTool(name=tool_names['SoftFilterTool'], desc=SOFT_FILTER_TOOL_DESC.format(**domain_map), item_sim_path=ITEM_SIM_FILE, 
                                                item_corups=item_corups, buffer=candidate_buffer, top_ratio=args.similar_ratio),
                "RankingTool": RecModelTool(name=tool_names['RankingTool'], desc=RANKING_TOOL_DESC.format(**domain_map), model_fpath=MODEL_CKPT_FILE, 
                                            item_corups=item_corups, buffer=candidate_buffer, rec_num=args.rank_num),
                "MapTool": MapTool(name=tool_names['MapTool'], desc=MAP_TOOL_DESC.format(**domain_map), item_corups=item_corups, buffer=candidate_buffer),
        }


        if args.enable_reflection:
            critic = Critic(
                model = 'gpt-4' if "4" in os.environ.get("OPENAI_ENGINE", "") else 'gpt-3.5-turbo',
                engine = os.environ.get("OPENAI_ENGINE", ""),
                buffer = candidate_buffer,
                domain = domain
            )
        else:
            critic = None


        if args.plan_first:
            if args.langchain:
                AgentType = CRSAgentPlanFirst
            else:
                AgentType = CRSAgentPlanFirstOpenAI
        else:
            AgentType = CRSAgent


        bot = AgentType(domain, tools, candidate_buffer, item_corups, os.environ.get("OPENAI_ENGINE", ""),
                    os.environ.get("OPENAI_ENGINE_TYPE", "chat"), max_tokens=args.max_output_tokens,
                    enable_shorten=args.enable_shorten,  # history shortening
                    demo_mode=args.demo_mode, demo_dir_or_file=args.demo_dir_or_file, num_demos=args.num_demos,    # demonstration
                    critic=critic, reflection_limits=args.reflection_limits, reply_style='concise',   # reflexion
                    planning_recording_file=args.plan_record_file,   # save plan, default None
                    enable_summarize=0,
                )   
        
        bot.init_agent()

    
    elif args.agent == 'gpt4' or args.agent == 'chatgpt':
        bot = OpenAIBot(
            domain = domain,
            engine = os.environ.get("OPENAI_ENGINE"),
            api_base = os.environ.get("OPENAI_API_BASE"),
            api_key = os.environ.get("OPENAI_API_KEY"),
            api_version = os.environ.get("OPENAI_API_VERSION"),
            api_type = os.environ.get("OPENAI_API_TYPE"),
            conversation = conversation,
            timeout=args.timeout
        )


    elif args.agent.startswith('llama') or args.agent.startswith('vicuna'):  # refer to fastchat to build API
        bot = OpenAIBot(
            domain = domain,
            engine = 'gpt-3.5-turbo',
            api_base = os.environ.get("OPENAI_API_BASE"),
            api_key = "EMPTY",
            api_version = "",
            api_type = "open_ai",
            conversation = conversation,
            timeout=args.timeout
        )

    elif args.agent.lower() == "chatrec":
        bot = ChatRec(
            domain = domain,
            engine = os.environ.get("OPENAI_ENGINE"),
            api_base = os.environ.get("OPENAI_API_BASE"),
            api_key = os.environ.get("OPENAI_API_KEY"),
            api_version = os.environ.get("OPENAI_API_VERSION"),
            api_type = os.environ.get("OPENAI_API_TYPE"),
            conversation = conversation,
            timeout=args.timeout,
            item_corups=item_corups,
            embed_vec_dir_path=args.chatrec_vec_dir,
            embedding_model_deployment_name=os.environ.get("OPENAI_EMB_MODEL", "text-embedding-ada-002")
        )


    else:
        raise ValueError("Not support for such agent.")


    metrics = conversation_eval(eval_data, bot, simulator, conversation, item_corups, max_turns=args.max_turns, recbot=(args.agent=='recbot'), start=args.start_test_num, end=args.end_test_num)
    
    conversation.dump(save_path)
    print("Conversation history saved in {}.".format(save_path))

    print(metrics)
    result_file = os.path.join(os.path.dirname(args.data), 'result', args.agent, 
                               f"metric-{args.agent}_from_{start}_to_{end}_{current_time}_{os.path.basename(args.data)}.txt")
    if not os.path.exists(os.path.dirname(result_file)):
        os.makedirs(os.path.dirname(result_file))
        
    with open(result_file, 'w') as f:
        f.write(json.dumps(metrics))

if __name__ == "__main__":
    main()