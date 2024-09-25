# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import time
from datetime import datetime
import json
import pickle
import argparse
import pandas as pd
from collections import defaultdict
import openai
import yaml
import threading
import pytz
from datetime import datetime
lock = threading.Lock()
cost = {}   # model_name: [input_tokens, output_tokens]
api_config = yaml.safe_load(open("openai_api_config.yaml"))

if api_config["API_TYPE"] == "azure":
    from openai import AzureOpenAI
    from azure.identity import get_bearer_token_provider, AzureCliCredential  
    credential = AzureCliCredential() 
    token_provider = get_bearer_token_provider( credential, 
    api_config['TOKEN_URL']) 
    client = AzureOpenAI(
        api_version= api_config['API_VERSION'],
        azure_ad_token_provider=token_provider, 
        azure_endpoint = api_config['API_BASE']
    )
else:
    from openai import OpenAI
    if api_config.get('API_BASE'):
        client = OpenAI(  
            api_base = api_config['API_BASE'],
            api_key=api_config["API_KEY"]
        )       
    else: 
        client = OpenAI(  
            api_key=api_config["API_KEY"]
        )

MAX_THREADS = api_config['MAX_THREADS']
MAX_RETRIES = api_config['MAX_RETRIES']
INTERVAL = api_config['SLEEP_INP_INTERVAL']
SLEEP_SECONDS = api_config['SLEEP_SECONDS']

def cost_save():
    with lock:
        global cost

        try:
            with open(api_config['API_COST_FILE'], 'r') as file:
                previous_data = json.load(file)
        except FileNotFoundError:
            previous_data = []
        total_input_cost = 0
        total_output_cost = 0
        for record in previous_data:
            if "models" in record:
                for model, model_costs in record['models'].items():
                    total_input_cost += model_costs['input_cost']
                    total_output_cost += model_costs['output_cost']
                total_all_cost = total_input_cost + total_output_cost


        us_timezone = pytz.timezone('America/New_York')
        us_time = datetime.now(us_timezone).strftime("%Y-%m-%d %H:%M")
        saved_cost_data = {'time': us_time, 'models': {}}

        filepath = "api_cost.jsonl"  
        cost_per_token = []
        with open(filepath, 'r') as f:
            for line in f:
                cost_per_token.append(json.loads(line.strip())) 


        cost_dict = {}
        for item in cost_per_token:
            for model, tokens in item.items():
                cost_dict[model.lower()] = tokens

        for model, tokens in cost.items():
            input_tokens = tokens[0]
            output_tokens = tokens[1]

            if model.lower() not in cost_dict:
                print(f"Warning: {model} not found in cost_dict, will count as gpt-4o")
                input_cost = input_tokens / 1_000_000 * cost_dict["gpt-4o"]['input']
                output_cost = output_tokens / 1_000_000 * cost_dict["gpt-4o"]['output']
            else:
                input_cost = input_tokens / 1_000_000 * cost_dict[model.lower()]['input']
                output_cost = output_tokens / 1_000_000 * cost_dict[model.lower()]['output']
            total_cost = input_cost + output_cost
            total_input_cost += input_cost
            total_output_cost += output_cost
            saved_cost_data['models'][model] = {
                'input_cost': round(input_cost, 4),
                'output_cost': round(output_cost, 4),
                'all_cost': round(total_cost, 4)
            }
            

        total_all_cost = total_input_cost + total_output_cost
        summarize = {
            'summarize_input_cost': round(total_input_cost, 4),
            'summarize_output_cost': round(total_output_cost, 4),
            'summarize_all_cost': round(total_all_cost, 4)
        }

        if previous_data == [] or 'summarize_input_cost' not in previous_data[0]:
            previous_data.insert(0, summarize)  
        else:
            previous_data[0] = summarize  

        previous_data.append(saved_cost_data)
        cost = {}

    if not api_config['API_COST_FILE'] or api_config['API_COST_FILE'] == "":
        with open("cost.json", 'w') as file:
            json.dump(previous_data, file, indent=4)
    else:
        with open(api_config['API_COST_FILE'], 'w') as file:
            json.dump(previous_data, file, indent=4)
            
def get_eval(model_name, user_prompt, sys_prompt, max_tokens=1024, temp=0.0):
    for i in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                    model=model_name, 
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {
                            "role": "user",
                            "content": user_prompt,
                        },
                    ],
                    temperature=temp,
                    max_tokens=max_tokens
                )
            content = response.choices[0].message.content
            tokens = response.usage
            with lock:
                global cost
                if model_name not in cost:
                    cost[model_name] =[0, 0]
                cost[model_name][0]+=int(tokens.prompt_tokens)
                cost[model_name][1]+=int(tokens.completion_tokens)
            return content
        except Exception as e:
            print(e)
            time.sleep(INTERVAL*(i+1))
    print(f"Failed after {MAX_RETRIES} retries.")
    return "error"



def generate_prompt(question, answer1, answer2, task, reviewer_jsons):
    if task in reviewer_jsons:
        reviewer = reviewer_jsons[task]
    else:
        reviewer = reviewer_jsons["recommend"]
    sys_prompt = reviewer["system_prompt"]
    prompt_template = reviewer["prompt_template"]
    defaults = reviewer["defaults"]
    prompt = prompt_template.format(
        question=question, answer_1=answer1, answer_2=answer2, **defaults
    )
    return sys_prompt, prompt

def parse_score(review):
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            raise Exception("Invalid score pair.")
    except Exception as e:
        print(f"{e}\nContent: {review}\n" "You must manually fix the score pair.")
        return [-1, -1]
        

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="steam")
    parser.add_argument("--answer1_path", type=str, help="path to the answer1 file.")
    parser.add_argument("--answer2_path", type=str, help="path to the answer2 file.")
    parser.add_argument("--tasks", type=str, default="retrieval", help="tasks to evaluate")
    parser.add_argument("--compare_num", type=int, default=10, help="number of samples to compare for each task")
    parser.add_argument("--model_name", type=str, default="gpt-35-turbo", help="openai model name")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    reviewer_jsons = {}
    with open("./evaluate/reviewer.jsonl", "r") as fr:
        for line in fr:
            reviewer = json.loads(line)
            reviewer_jsons[reviewer["category"]] = reviewer

    test_dataset = pd.read_csv(f'./data/{args.dataset}/test.csv')
    task_questions = defaultdict(list)
    for idx in range(len(test_dataset)):
        prompt = test_dataset.iloc[idx]['prompt']
        task = test_dataset.iloc[idx]['task']
        task_questions[task].append(prompt)

    total_reviews = []
    
    if "retrieval" in args.tasks:
        retrieval_answers1 = pickle.load(open(os.path.join(args.answer1_path, "retrieval_results.pkl"), "rb"))
        retrieval_answers2 = pickle.load(open(os.path.join(args.answer2_path, "retrieval_results.pkl"), "rb"))
        for idx in range(args.compare_num):
            question = task_questions["retrieval"][idx]
            answer1 = retrieval_answers1[idx][0]
            answer2 = retrieval_answers2[idx][0]
            sys_prompt, prompt = generate_prompt(question, answer1, answer2, "retrieval", reviewer_jsons)
            review = get_eval(args.model_name, sys_prompt, prompt, 1024)
            total_reviews.append({
                "answer1": answer1,
                "answer2": answer2,
                "review": review,
                "task": "retrieval",
            })
    
    if "ranking" in args.tasks:
        ranking_answers1 = pickle.load(open(os.path.join(args.answer1_path, "ranking_results.pkl"), "rb"))
        ranking_answers2 = pickle.load(open(os.path.join(args.answer2_path, "ranking_results.pkl"), "rb"))
        for idx in range(args.compare_num):
            question = task_questions["ranking"][idx]
            answer1 = ranking_answers1[idx][0]
            answer2 = ranking_answers2[idx][0]
            sys_prompt, prompt = generate_prompt(question, answer1, answer2, "ranking", reviewer_jsons)
            review = get_eval(args.model_name, sys_prompt, prompt, 1024)
            total_reviews.append({
                "answer1": answer1,
                "answer2": answer2,
                "review": review,
                "task": "ranking",
            })
    
    if "tagging" in args.tasks:
        tagging_answers1 = pickle.load(open(os.path.join(args.answer1_path, "tagging_results.pkl"), "rb"))
        tagging_answers2 = pickle.load(open(os.path.join(args.answer2_path, "tagging_results.pkl"), "rb"))
        for idx in range(args.compare_num):
            question = task_questions["tagging"][idx]
            answer1 = tagging_answers1[idx][0]
            answer2 = tagging_answers2[idx][0]
            sys_prompt, prompt = generate_prompt(question, answer1, answer2, "tagging", reviewer_jsons)
            review = get_eval(args.model_name, sys_prompt, prompt, 1024)
            total_reviews.append({
                "answer1": answer1,
                "answer2": answer2,
                "review": review,
                "task": "tagging",
            })

    if "explanation" in args.tasks:
        explanation_answers1 = pickle.load(open(os.path.join(args.answer1_path, "explanation_results.pkl"), "rb"))
        explanation_answers2 = pickle.load(open(os.path.join(args.answer2_path, "explanation_results.pkl"), "rb"))
        for idx in range(args.compare_num):
            question = task_questions["explanation"][idx]
            answer1 = explanation_answers1[idx][0]
            answer2 = explanation_answers2[idx][0]
            sys_prompt, prompt = generate_prompt(question, answer1, answer2, "explanation", reviewer_jsons)
            review = get_eval(args.model_name, sys_prompt, prompt, 1024)
            total_reviews.append({
                "answer1": answer1,
                "answer2": answer2,
                "review": review,
                "task": "explanation",
            })
    
    if "searching" in args.tasks:
        searching_answers1 = pickle.load(open(os.path.join(args.answer1_path, "searching_results.pkl"), "rb"))
        searching_answers2 = pickle.load(open(os.path.join(args.answer2_path, "searching_results.pkl"), "rb"))
        for idx in range(args.compare_num):
            question = task_questions["searching"][idx]
            answer1 = searching_answers1[idx][0]
            answer2 = searching_answers2[idx][0]
            sys_prompt, prompt = generate_prompt(question, answer1, answer2, "searching", reviewer_jsons)
            review = get_eval(args.model_name, sys_prompt, prompt, 1024)
            total_reviews.append({
                "answer1": answer1,
                "answer2": answer2,
                "review": review,
                "task": "searching",
            })

    answer1_name = args.answer1_path.split("/")[-1].strip()
    answer2_name = args.answer2_path.split("/")[-1].strip()
    with open(f"./output/{args.dataset}/review_{answer1_name}_{answer2_name}.json", "w") as fw:
        for review in total_reviews:
            scores = parse_score(review['review'])
            review['score'] = scores
            fw.write(json.dumps(review) + "\n")