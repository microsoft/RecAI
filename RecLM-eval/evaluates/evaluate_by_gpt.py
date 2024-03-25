# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import time
import json
import pickle
import argparse
import pandas as pd
from collections import defaultdict
import openai
import yaml

api_config = yaml.safe_load(open("openai_api_config.yaml"))

if api_config["API_TYPE"] == "azure":
    client = AzureOpenAI(
        api_key=api_config('API_KEY'),
        api_version= api_config('API_VERSION'),
        azure_endpoint = api_config('API_BASE')
    )
else:
    client = OpenAI(  
        api_key=api_config["API_KEY"]
    )


MAX_THREADS = api_config['MAX_THREADS']
MAX_RETRIES = api_config['MAX_RETRIES']
INTERVAL = api_config['SLEEP_INP_INTERVAL']
SLEEP_SECONDS = api_config['SLEEP_SECONDS']


def get_eval(model_name, user_prompt, sys_prompt, max_tokens=1024):
    for i in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                    model=model_name, 
                    messages=create_prompt(model_name, [
                        {"role": "system", "content": sys_prompt},
                        {
                            "role": "user",
                            "content": user_prompt,
                        },
                    ]),
                    temperature=temp,
                    max_tokens=max_tokens
                )
            content = response.choices[0].message.content
            return content
        except Exception as e:
            print(e)
            time.sleep(INTERVAL*(i+1))
    print(f"Failed after {MAX_RETRIES} retries.")
    return "error"

def create_prompt(model_name, messages):
    if model_name == "gpt-35-turbo":
        prompt = ""
        for message in messages:
            prompt += f"\n<|im_start|>{message['role']}\n{message['content']}\n<|im_end|>"
        prompt += "\n<|im_start|>assistant\n"
    else:
        prompt = ""
        for message in messages:
            prompt += f"\n{message['role']}: {message['content']}\n"
        prompt += "\nassistant:"
    
    print(prompt)
    return prompt

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