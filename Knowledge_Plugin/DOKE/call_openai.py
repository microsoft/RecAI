# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.



import os
import time
import pickle
from tqdm import tqdm
import json
import openai
import argparse
import random
import requests

import queue
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import openai
from openai import OpenAI

api_keys = [
    "your openai keys"
]

def generate_Davinci(api_key, text):
    openai.api_key = api_key
    openai.api_base =  "https://api.openai.com/v1" # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
    for i in range(MAX_RETRIES):
        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=text,
                temperature=0,
                max_tokens=800,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )['choices'][0]['text']
            return response
        except Exception as e:
            print(f"{api_key}\nError occurred: {e}. Retrying...")
    print(f"Failed to get response for prompt: {prompt} after {MAX_RETRIES} retries.")
    return "None"

def generate_chatgpt(api_key, prompt, version):
    # 使用你的 API 密钥初始化 OpenAI GPT-3
    client = OpenAI(api_key=api_key)
    text = [{'role': 'user', 'content': prompt}]
    if version == "0301":
        model = "gpt-3.5-turbo-0301"
    else:
        model = "gpt-3.5-turbo"

    for i in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=text,
                temperature=0.0,
                max_tokens=2048,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            content = response.choices[0].message.content.strip()
            return content
        except Exception as e:
            print(f"{api_key}\nError occurred: {e}. Retrying...")
            time.sleep(INTERVAL)
    print(f"Failed to get response for prompt: {prompt} after {MAX_RETRIES} retries.")
    return "None"

def generate_gpt4(api_key, prompt):
    client = OpenAI(api_key=api_key)
    text = [{'role': 'user', 'content': prompt}]
    for _ in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=text,
                temperature=0.0,
                max_tokens=2048,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            content = response.choices[0].message.content.strip()
            return content
        except Exception as e:
            print(f"{api_key}\nError occurred: {e}. Retrying...")
            time.sleep(INTERVAL)
    
    print("out of max_retry_times")
    return "Error"

def parse_gpt_response(candidate_list, response):
    try:
        response = ' '.join(response.strip().split('\n'))
        response = response.split('[', 1)[1]
        response = ''.join(response.split(']')[:-1])
        result = json.loads("["+response+"]")
        if len(result) == 0:
            result = ["1"]
    except:
        result = ["1"]
    return result

def parse_davinci_response(candidate_list, response):
    # response = response.strip().split('\n')
    # results = []
    # for line in response:
    #     result = line.strip()
    #     if result.split('.')[0].isdigit() and len(result.split('.')) > 1:
    #         result = result.split('.', 1)[1].strip()
    #     if len(result) == 0:
    #         continue
    #     if result[0] == "'" and result[-1] == "'":
    #         result = result[1:-1]
    #     results.append(result)
    # if len(results) == 0:
    #     results = ['1']
    # return results
    response = response.strip('[]\n.')
    if len(response.split('\n')) > 10:
        response = response.split('\n')
    else:
        response = response.split(',')
    results = []
    for result in response:
        result = result.strip()
        if result.split('.')[0].isdigit() and len(result.split('.')) > 1:
            result = result.split('.', 1)[1].strip()
        if len(result) == 0:
            continue
        if result[0] == "'" and result[-1] == "'":
            result = result[1:-1]
        if result[0] == "\"" and result[-1] == "\"":
            result = result[1:-1]
        if result.isdigit():
            if int(result) < len(candidate_list):
                results.append(candidate_list[int(result)])
        else:
            results.append(result)
    if len(results) == 0:
        results = ['1']
    return results

def worker(i, model, version):
    while not prompts_queue.empty():
        index, prompt = prompts_queue.get()
        api_key = api_keys[i % len(api_keys)]
        if model == "GPT4":
            result = generate_gpt4(api_key, prompt)
        if model == "ChatGPT":
            result = generate_chatgpt(api_key, prompt, version)
        elif model == "Davinci":
            result = generate_Davinci(api_key, prompt)
        results.put((index, result))
        # if model == "ChatGPT":
        #     time.sleep(INTERVAL)  # 控制调用频率
        with num_completed.get_lock():
            num_completed.value += 1  # 更新完成任务数量

# 创建进度条更新进程
def progress_monitor(total):
    with tqdm(total=total) as pbar:
        while True:
            completed = num_completed.value
            pbar.n = completed
            pbar.refresh()
            if completed >= total:
                break
            time.sleep(0.1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prompt construction')
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--version', type=str, default="latest")
    parser.add_argument('--dataset', type=str, default='ml1m')
    args = parser.parse_args()
    
    ranking_results = []
    task_name = f"{args.model}_{args.prompt.split('/')[-1].split('.')[0]}"
    if args.version == "0301":
        task_name += "-0301"
    os.makedirs(f"out/result/{args.dataset}/{task_name}", exist_ok=True)
    fd = open(f"out/result/{args.dataset}/{task_name}/ranking_results.txt", "w",encoding="utf8")

    prompts = []
    targets = []
    candidate_lists = []
    for line in tqdm(open(args.prompt, encoding="utf8").readlines()):
        data = json.loads(line)
        prompts.append(data['prompt'])
        targets.append(data['ground_truth'])
        candidate_lists.append(data['candidate'])

    results = queue.PriorityQueue()
    MAX_THREADS = len(api_keys)
    INTERVAL = 20
    MAX_RETRIES = 3
    num_completed = multiprocessing.Value('i', 0)

    # 将所有的提示放入一个 Queue 中
    prompts_queue = queue.Queue()
    for i, prompt in enumerate(prompts):
        prompts_queue.put((i, prompt))

    # 使用 ThreadPoolExecutor 创建线程池
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        # 启动进度条更新进程
        progress_process = multiprocessing.Process(target=progress_monitor, args=(len(prompts),))
        progress_process.start()

        for i in range(MAX_THREADS):
            executor.submit(worker, i, args.model, args.version)

    # 现在我们的结果队列中应该有所有的 API 调用结果，可以像这样处理它们：
    final_results = []
    while not results.empty():
        final_results.append(results.get())

    # 对结果按照索引进行排序
    final_results.sort()
    
    # 等待进度条进程完成
    progress_process.join()

    # 打印排序后的结果
    for index, result in final_results:
        # print(f"Prompt: {prompts[index]}\nResult: {result}\n")
        if args.model in ["GPT4", "ChatGPT"]:
            rank_list = parse_gpt_response(candidate_lists[index], result)
        if args.model == "Davinci":
            rank_list = parse_davinci_response(candidate_lists[index], result)
        ranking_results.append((rank_list, targets[index], candidate_lists[index]))
        result = result.replace('\n', '\\n')
        prompts[index] = prompts[index].replace('\n', '\\n')
        fd.write(f"{len(rank_list)} {rank_list}\t {targets[index]} \t {result} \t {prompts[index]}\n")
        fd.flush()
    pickle.dump(ranking_results, open(f"out/result/{args.dataset}/{task_name}/ranking_results.pkl", "wb"))