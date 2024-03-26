from call_models.openai_models import gen_openai_answer

import argparse
import importlib

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str)
    parser.add_argument('--dataset', type=str)

    return parser.parse_args()

def get_gen_answer_func(method_name):
    if importlib.util.find_spec(f'method.{method_name}', __name__):
        model_module = importlib.import_module(f'method.{method_name}', __name__)
        method_func = getattr(model_module, "gen_answer")
        return method_func
    else:
        raise NotImplementedError(f"'{method_name}' not exist!")
    
args = parse_args()

gen_answer_func = get_gen_answer_func(args.method)
gen_openai_answer(
    "gpt-3.5-turbo-0301", 
    f"outputs/{args.dataset}/{args.method}/question.jsonl",
    f"outputs/{args.dataset}/{args.method}/answer.jsonl",
    gen_answer_func
)