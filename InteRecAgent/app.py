# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os
import sys
import gradio as gr
from loguru import logger

from llm4crs.agent import CRSAgent
from llm4crs.agent_plan_first import CRSAgentPlanFirst
from llm4crs.agent_plan_first_openai import CRSAgentPlanFirstOpenAI
from llm4crs.buffer import CandidateBuffer
from llm4crs.corups import BaseGallery
from llm4crs.critic import Critic
from llm4crs.environ_variables import *
from llm4crs.mapper import MapTool
from llm4crs.prompt import *
from llm4crs.ranking import RecModelTool
from llm4crs.retrieval import SimilarItemTool, SQLSearchTool
from llm4crs.query import QueryTool
from llm4crs.utils import FuncToolWrapper

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default="7891")
# parser.add_argument('--domain', type=str, default='game')
parser.add_argument(
    "--max_candidate_num",
    type=int,
    default=1000,
    help="Number of max candidate number of buffer",
)
parser.add_argument(
    "--similar_ratio",
    type=float,
    default=0.05,
    help="Ratio of returned similar items / total games",
)
parser.add_argument(
    "--rank_num", type=int, default=100, help="Number of games given by ranking tool"
)
parser.add_argument(
    "--max_output_tokens",
    type=int,
    default=1024,
    help="Max number of tokens in LLM output",
)
parser.add_argument(
    "--engine",
    type=str,
    default="text-davinci-003",
    help="Engine of OpenAI API to use. The default is text-davinci-003",
)
parser.add_argument(
    "--bot_type",
    type=str,
    default="chat",
    choices=["chat", "completion"],
    help="Type OpenAI models. The default is completion. Options [completion, chat]",
)

# chat history shortening
parser.add_argument(
    "--enable_shorten",
    type=int,
    choices=[0, 1],
    default=0,
    help="Whether to enable shorten chat history with LLM",
)

# dynamic demonstrations
parser.add_argument(
    "--demo_mode",
    type=str,
    choices=["zero", "fixed", "dynamic"],
    default="dynamic",
    help="Directory path of demonstrations",
)
parser.add_argument(
    "--demo_dir_or_file",
    type=str,
    default="./demonstration/seed_demos_placeholder.jsonl",
    help="Directory or file path of demonstrations"
)
parser.add_argument(
    "--num_demos", type=int, default=5, help="number of demos for in-context learning"
)

# reflection mechanism
parser.add_argument(
    "--enable_reflection",
    type=int,
    choices=[0, 1],
    default=0,
    help="Whether to enable reflection",
)
parser.add_argument(
    "--reflection_limits", type=int, default=3, help="limits of reflection times"
)

# plan first agent
parser.add_argument(
    "--plan_first",
    type=int,
    choices=[0, 1],
    default=1,
    help="Whether to use plan first agent",
)

# whether to use langchain in plan-first agent
parser.add_argument(
    "--langchain",
    type=int,
    choices=[0, 1],
    default=0,
    help="Whether to use langchain in plan-first agent",
)

# the reply style of the assistent
parser.add_argument(
    "--reply_style",
    type=str,
    choices=["concise", "detailed"],
    default="detailed",
    help="Reply style of the assistent. If detailed, details about the recommendation would be give. Otherwise, only item names would be given.",
)

args = parser.parse_args()

domain = os.environ.get("DOMAIN", "game")
domain_map = {"item": domain, "Item": domain.capitalize(), "ITEM": domain.upper()}

default_chat_value = [
    None,
    "Hello, I'm a conversational {item} recommendation assistant. I'm here to help you discover your interested {item}s.".format(
        **domain_map
    ),
]


tool_names = {k: v.format(**domain_map) for k, v in TOOL_NAMES.items()}

item_corups = BaseGallery(
    GAME_INFO_FILE,
    TABLE_COL_DESC_FILE,
    f"{domain}_information",
    columns=USE_COLS,
    fuzzy_cols=["title"] + CATEGORICAL_COLS,
    categorical_cols=CATEGORICAL_COLS,
)

candidate_buffer = CandidateBuffer(item_corups, num_limit=args.max_candidate_num)


# The key of dict here is used to map to the prompt
tools = {
    "BufferStoreTool": FuncToolWrapper(
        func=candidate_buffer.init_candidates,
        name=tool_names["BufferStoreTool"],
        desc=CANDIDATE_STORE_TOOL_DESC.format(**domain_map),
    ),
    "LookUpTool": QueryTool(
        name=tool_names["LookUpTool"],
        desc=LOOK_UP_TOOL_DESC.format(**domain_map),
        item_corups=item_corups,
        buffer=candidate_buffer,
    ),
    "HardFilterTool": SQLSearchTool(
        name=tool_names["HardFilterTool"],
        desc=HARD_FILTER_TOOL_DESC.format(**domain_map),
        item_corups=item_corups,
        buffer=candidate_buffer,
        max_candidates_num=args.max_candidate_num,
    ),
    "SoftFilterTool": SimilarItemTool(
        name=tool_names["SoftFilterTool"],
        desc=SOFT_FILTER_TOOL_DESC.format(**domain_map),
        item_sim_path=ITEM_SIM_FILE,
        item_corups=item_corups,
        buffer=candidate_buffer,
        top_ratio=args.similar_ratio,
    ),
    "RankingTool": RecModelTool(
        name=tool_names["RankingTool"],
        desc=RANKING_TOOL_DESC.format(**domain_map),
        model_fpath=MODEL_CKPT_FILE,
        item_corups=item_corups,
        buffer=candidate_buffer,
        rec_num=args.rank_num,
    ),
    "MapTool": MapTool(
        name=tool_names["MapTool"],
        desc=MAP_TOOL_DESC.format(**domain_map),
        item_corups=item_corups,
        buffer=candidate_buffer,
    ),
}


if args.enable_reflection:
    critic = Critic(
        model="gpt-4" if "4" in args.engine else "gpt-3.5-turbo",
        engine=args.engine,
        buffer=candidate_buffer,
        domain=domain,
        bot_type=args.bot_type,
    )
else:
    critic = None

if args.plan_first and args.langchain:
    AgentType = CRSAgentPlanFirst
elif args.plan_first and not args.langchain:
    AgentType = CRSAgentPlanFirstOpenAI
else:
    AgentType = CRSAgent

bot = AgentType(
    domain,
    tools,
    candidate_buffer,
    item_corups,
    args.engine,
    args.bot_type,
    max_tokens=args.max_output_tokens,
    enable_shorten=args.enable_shorten,  # history shortening
    demo_mode=args.demo_mode,
    demo_dir_or_file=args.demo_dir_or_file,
    num_demos=args.num_demos,  # demonstration
    critic=critic,
    reflection_limits=args.reflection_limits,  # reflexion
    verbose=True,
    reply_style=args.reply_style,  # only supported for CRSAgentPlanFirstOpenAI
)

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    level="DEBUG",
    format="<lc>[{time:YYYY-MM-DD HH:mm:ss} {level}] <b>{message}</b></lc>",
)


def user(user_message, history):
    return "", history + [[user_message, None]], history + [[user_message, None]]


bot.init_agent()
bot.set_mode("accuracy")

css = """
#chatbot .overflow-y-auto{height:600px}
#send {background-color: #FFE7CF}
"""
with gr.Blocks(css=css, elem_id="chatbot") as demo:
    with gr.Row(visible=True) as btn_raws:
        with gr.Column(scale=5):
            mode = gr.Radio(
                ["diversity", "accuracy"], value="accuracy", label="Recommendation Mode"
            )
        with gr.Column(scale=5):
            style = gr.Radio(
                ["concise", "detailed"], value=getattr(bot, 'reply_style', 'concise'), label="Reply Style"
            )
    chatbot = gr.Chatbot(
        elem_id="chatbot", label=f"{domain_map['Item']} RecAgent"
    )
    state = gr.State([])
    with gr.Row(visible=True) as input_raws:
        with gr.Column(scale=4):
            txt = gr.Textbox(
                show_label=False, placeholder="Enter text and press enter", container=False
            )
        with gr.Column(scale=1, min_width=0):
            send = gr.Button(value="Send", elem_id="send", variant="primary")
        with gr.Column(scale=1, min_width=0):
            clear = gr.ClearButton(value="Clear")

    state.value = [default_chat_value]
    chatbot.value = [default_chat_value]

    txt.submit(user, [txt, state], [txt, state, chatbot]).then(
        bot.run_gr, [state], [chatbot, state]
    )
    txt.submit(candidate_buffer.clear, [], [])
    txt.submit(lambda: "", None, txt)

    send.click(user, [txt, state], [txt, state, chatbot]).then(
        bot.run_gr, [state], [chatbot, state]
    )
    send.click(candidate_buffer.clear, [], [])
    send.click(lambda: "", None, txt)
    mode.change(bot.set_mode, [mode], None)
    style.change(bot.set_style, [style], None)

    clear.click(bot.clear)
    clear.click(lambda: [default_chat_value], None, chatbot)
    clear.click(lambda: [default_chat_value], None, state)

demo.launch(share=False)
