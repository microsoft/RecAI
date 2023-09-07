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
from tqdm import tqdm
from rapidfuzz import fuzz
from datetime import datetime
from langchain.callbacks import get_openai_callback


from llm4crs.prompt import *
from llm4crs.utils import FuncToolWrapper
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
"""

def read_jsonl(fpath: str) -> List[Dict]:
    res = []
    with open(fpath, 'r') as f:
        for line in f:
            data = json.loads(line)
            res.append(data)
    return res




class Conversation:

    def __init__(self, user_prefix='User', agent_prefix='Assistent', save_path: str=None):
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
            res += "{}: {}\n".format(h['role'], h['msg'])
        res = res[:-1]
        return res

    @property
    def turns(self) -> int:
        return math.ceil(len(self.history) / 2)

    def __len__(self) -> int:
        return len(self.history)

    def clear(self, data_index: int) -> None:
        if len(self.history) > 0:
            data = {'id': data_index, 'conversation': self.history}
            self.all_history.append(data)
            if self.save_path:
                with open(self.save_path, 'a') as f:
                    line = json.dumps(data, ensure_ascii=False) + "\n"
                    f.write(line)
        self.history = []

    def dump(self, fpath: str):
        with open(fpath, 'w') as f:
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
        fschat: bool=False
    ):
        self.domain = domain
        self.engine = engine
        self.api_key = api_key
        self.api_type = api_type
        self.api_base = api_base
        self.api_version = api_version
        self.conversation = conversation
        self.timeout = timeout
        self.fschat = fschat
        

    def run(self, inputs: Dict) -> str:
        if 'azure' in self.api_type: 
            openai.api_base = self.api_base
            openai.api_version = self.api_version
            openai.api_type = self.api_type
        else:
            openai.api_base = self.api_base
            openai.api_version = None
            openai.api_type = self.api_type
        if self.fschat:
            openai.api_base = self.api_base
        openai.api_key = self.api_key
        prompt = "You are a helpful conversational agent who is good at {domain} recommendation. "
        sys_msg = {'role': 'system', 'content': prompt.format(domain=self.domain)}

        usr_prompt = "Here is the conversation history: \n{chat_history} \nUser: {u_msg} \nAssistent: "
        usr_msg = {'role': 'user', 'content': usr_prompt.format(chat_history=self.conversation.total_history, u_msg=inputs['input'])}
        
        msg = [sys_msg, usr_msg]

        retry_cnt = 6
        sleep_interval = 4
        for retry in range(retry_cnt):
            try:    
                kwargs = {
                    "model": self.engine, 
                    "temperature": 0.8,
                    "messages": msg,
                    "max_tokens": 256,
                    "request_timeout": self.timeout
                }
                if (not self.fschat) and (openai.api_type != 'open_ai'):
                    kwargs["engine"] = self.engine

                chat = openai.ChatCompletion.create(**kwargs)
                reply = chat.choices[0].message.content
                break
            except Exception as e:
                print(f"An error occurred while making the API call: {e}")
                reply = "Something went wrong, please retry."
                time.sleep(sleep_interval)
                sleep_interval = min(sleep_interval*1.5, 15)
        return reply



class Simulator:

    def __init__(self, conversation: Conversation, engine: str, domain: str, timeout: int):
        self.conversation = conversation
        self.engine = engine
        self.domain = domain
        self.history = None
        self.target = None
        self.target_info = None
        self.timeout = timeout

    def set(self, history: str, target: str, target_info: str) -> None:
        self.history = history
        self.target = target
        self.target_info = target_info

    def __call__(self) -> str:
        api_type = os.environ.get("SIMULATOR_API_TYPE", "")
        if 'azure' in api_type: 
            openai.api_base = os.environ.get("SIMULATOR_API_BASE")
            openai.api_version = os.environ.get("SIMULATOR_API_VERSION")
            openai.api_type = api_type
        else:
            openai.api_base = os.environ.get("SIMULATOR_API_BASE")
            openai.api_version = None
            openai.api_type = api_type
        openai.api_key = os.environ.get("SIMULATOR_API_KEY")
        msg = [{'role': 'system', 'content': user_simulator_sys_prompt.format(domain=self.domain)}]
        usr_msg = user_simulator_template.format(
            domain=self.domain, 
            history=self.history,
            target=self.target,
            target_item_info=self.target_info
        )
        if len(self.conversation) == 0:
            pass
        else:
            usr_msg += "\nHere are the conversation history: \n{}".format(self.conversation.total_history)

        msg.append({'role': 'user', 'content': usr_msg})
        target_related_score = 100

        n_simulator_tries = 0
        while (target_related_score >= 60) and (n_simulator_tries < 5):
            # detect whether simulator gives the target directly
            retry_cnt = 6
            sleep_interval = 4
            for retry in range(retry_cnt):
                if n_simulator_tries > 0:
                    msg[-1]['content'] = msg[-1]['content'] + "\nDo not tell the target directly."
                try:
                    kwargs = {
                        "model": self.engine, 
                        "temperature": 0.8,
                        "messages": msg,
                        "max_tokens": 256,
                        "request_timeout": self.timeout
                    }
                    if (openai.api_type != 'open_ai'):
                        kwargs["engine"] = self.engine
                    chat = openai.ChatCompletion.create(**kwargs)
                    reply = chat.choices[0].message.content
                    break
                except Exception as e:
                    print(f"An error occurred while making the API call: {e}")
                    reply = "I'm offline, wait me for a while."
                    # time.sleep(random.randint(1, 5))
                    time.sleep(sleep_interval)
                sleep_interval = min(sleep_interval*1.5, 15)
            target_related_score = fuzz.partial_ratio(self.target, reply)
            n_simulator_tries += 1

        
        self.conversation.add_user_msg(reply)
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
    pbar = tqdm(total=end-i)
    while i < end:
        if recbot:
            agent.clear()
            if (i > start) and ((i-start) % 10 == 0):
                agent.clear()
                time.sleep(random.randint(30, 60))  # wait for a while at the begining of each conv
        d = data[i]
        try:
            target = item_corup.fuzzy_match(d['target'], 'title')
            target_info = item_corup.convert_title_2_info(target)
            target_info = {k: str(v) for k,v in target_info.items() if k not in {'id'}}
            simulator.set(**d, target_info=json.dumps(target_info))  # {'history': xxxx, 'target': xxx}
            n = 0
            oai = 0
            fail = 0
            hit_flag = False
            while n < max_turns:
                # if recbot:
                #     time.sleep(random.randint(10,20))   # wait for a while
                usr_msg = simulator()   # user simulartor
                tqdm.write(f"User: {usr_msg}")
                if "<END>" in usr_msg:
                    break
                if recbot:
                    time.sleep(random.randint(10, 30))  # wait for a while
                    with get_openai_callback() as cb:
                        agent_msg = agent.run({'input': usr_msg})   # conversational agent
                        oai += cb.successful_requests
                        fail += agent.failed_times
                else:
                    agent_msg = agent.run({'input': usr_msg})   # conversational agent
                tqdm.write(f"Assistent: {agent_msg}")
                conversation.add_agent_msg(agent_msg)
                if hit_judge(agent_msg, d['target']):
                    hit_num += 1
                    hit_turn.append(n+1)
                    hit_flag = True
                    break
                else:
                    pass
                n += 1
            if not hit_flag:  # not hit
                hit_turn.append(max_turns+1)
            n_oai.append(oai)
            n_fail.append(fail)
            conversation.clear(data_index=i)
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
    parser.add_argument('--engine', type=str, default='text-davinci-003',
                    help='Engine of OpenAI API to use as user simulator. The default is text-davinci-003') 
    parser.add_argument("--timeout", type=int, default=5, help="Timeout threshold when calling OAI. (seconds)")

    # parser.add_argument('--domain', type=str, default='game')
    parser.add_argument('--agent', type=str, help='agent type, "recbot" is our method and others are baselines')

    parser.add_argument('--max_candidate_num', type=int, default=1000, help="Number of max candidate number of buffer")
    parser.add_argument('--similar_ratio', type=float, default=0.1, help="Ratio of returned similar items / total games")
    parser.add_argument('--rank_num', type=int, default=100, help="Number of games given by ranking tool")
    parser.add_argument('--max_output_tokens', type=int, default=512, help="Max number of tokens in LLM output")
    parser.add_argument('--bot_type', type=str, default='completion', choices=['chat', 'completion'],
                    help='Type OpenAI models. The default is completion. Options [completion, chat]') 

    # chat history shortening
    parser.add_argument('--enable_shorten', type=int, choices=[0,1], default=0, help="Whether to enable shorten chat history with LLM")

    # dynamic demonstrations
    parser.add_argument('--demo_mode', type=str, choices=["zero", "fixed", "dynamic"], default="zero", help="Directory path of demonstrations")
    parser.add_argument('--demo_dir_or_file', type=str, help="Directory or file path of demonstrations")
    parser.add_argument('--num_demos', type=int, default=3, help="number of demos for in-context learning")

    # reflection mechanism
    parser.add_argument('--enable_reflection', type=int, choices=[0,1], default=0, help="Whether to enable reflection")
    parser.add_argument('--reflection_limits', type=int, default=3, help="limits of reflection times")

    # plan first agent
    parser.add_argument('--plan_first', type=int, choices=[0,1], default=0, help="Whether to use plan first agent")

    parser.add_argument("--langchain", type=int, choices=[0, 1], default=0, help="Whether to use langchain in plan-first agent")
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
        save_path = os.path.join(os.path.dirname(args.data), f"saved_conversations_{args.agent}_from_{start}_to_{end}_{current_time}_{os.path.basename(args.data)}")

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    domain = os.environ["DOMAIN"]
    print(f"Domain = {domain}.")

    domain_map = {'item': domain, 'Item': domain.capitalize(), 'ITEM': domain.upper()}


    conversation = Conversation(save_path = save_path)
    simulator = Simulator(
        conversation=conversation,
        engine=os.environ.get("SIMULATOR_ENGINE", "gpt-4"),
        domain=domain,
        timeout=args.timeout
    )

    item_corups = BaseGallery(GAME_INFO_FILE, TABLE_COL_DESC_FILE, f'{domain}_information',
                          columns=USE_COLS, 
                          fuzzy_cols=['title'] + CATEGORICAL_COLS, 
                          categorical_cols=CATEGORICAL_COLS)

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
                # "BufferClearTool": buffer_replan_tool
        }



        if args.enable_reflection:
            critic = Critic(
                model = 'gpt-4' if "4" in os.environ.get("AGENT_ENGINE", "") else 'gpt-3.5-turbo',
                engine = os.environ.get("AGENT_ENGINE", ""),
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


        bot = AgentType(domain, tools, candidate_buffer, item_corups, os.environ.get("AGENT_ENGINE", ""), 
                    args.bot_type, max_tokens=args.max_output_tokens, 
                    enable_shorten=args.enable_shorten,  # history shortening
                    demo_mode=args.demo_mode, demo_dir_or_file=args.demo_dir_or_file, num_demos=args.num_demos,    # demonstration
                    critic=critic, reflection_limits=args.reflection_limits, reply_style='concise')   # reflexion
        
        bot.init_agent()

    
    elif args.agent == 'gpt4' or args.agent == 'chatgpt':
        bot = OpenAIBot(
            domain = domain,
            engine = os.environ.get("AGENT_ENGINE"),
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
            timeout=args.timeout,
            fschat=True
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