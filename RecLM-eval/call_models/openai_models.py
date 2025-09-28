# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import time
import yaml
import pandas as pd
from tqdm import tqdm 
import pytz
import queue
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import threading
from datetime import datetime
lock = threading.Lock()
api_config = yaml.safe_load(open("openai_api_config.yaml"))

from openai import AzureOpenAI

# If the user provides TOKEN_URL, we treat it as Azure AD token flow authentication;
# Otherwise, use api_key authentication directly to avoid relying on local Azure CLI.   
if api_config.get('TOKEN_URL'):
    from azure.identity import get_bearer_token_provider, AzureCliCredential

    try:
        credential = AzureCliCredential()  # depends on locally installed azure-cli
    except Exception:
        # Azure CLI unavailable, automatically fallback to api_key authentication
        credential = None

    if credential is not None:
        token_provider = get_bearer_token_provider(credential, api_config['TOKEN_URL'])
        client = AzureOpenAI(
            api_version=api_config['API_VERSION'],
            azure_ad_token_provider=token_provider,
            azure_endpoint=api_config['API_BASE']
        )
    else:
        client = AzureOpenAI(
            api_key=api_config['API_KEY'],
            api_version=api_config['API_VERSION'],
            azure_endpoint=api_config['API_BASE']
        )
else:
    client = AzureOpenAI(
        api_key=api_config['API_KEY'],
        api_version=api_config['API_VERSION'],
        azure_endpoint=api_config['API_BASE']
    )


cost = {}   # model_name: [input_tokens, output_tokens]
MAX_THREADS = api_config['MAX_THREADS']
MAX_RETRIES = api_config['MAX_RETRIES']
INTERVAL = api_config['SLEEP_INP_INTERVAL']
SLEEP_SECONDS = api_config['SLEEP_SECONDS']

def get_openai_client():
    global client
    if client is None:
        if api_config.get('API_BASE'):
            client = OpenAI(
                base_url=api_config['API_BASE'],
                api_key=api_config["API_KEY"]
            )
        else: 
            client = OpenAI(  
                api_key=api_config["API_KEY"]
            )
    return client



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



        
    
def call_openai_chat(model, prompt, system_prompt, temp=0.0):
    if system_prompt:
        text = [{"role":"system", "content": system_prompt}]
    else:
        text = []
    if isinstance(prompt, str):
        text += [{'role': 'user', 'content': prompt}]
    else:
        text += prompt
    for i in range(MAX_RETRIES):
        try:
            client = get_openai_client()
            response = client.chat.completions.create(
                model=model, 
                messages=text,
                temperature=temp,
                # top_p=0.95,
            )
            content = response.choices[0].message.content
            tokens = response.usage
            with lock:
                global cost
                if model not in cost:
                    cost[model] =[0, 0]
                cost[model][0]+=int(tokens.prompt_tokens)
                cost[model][1]+=int(tokens.completion_tokens)
                                
            return content
        except Exception as e:
            sleep_time = INTERVAL * (i + 1)
            print(f"Error occurred: {e}. \nWill retry after {sleep_time} seconds...")
            time.sleep(sleep_time)  
    print(f"Failed to get response for prompt: {prompt} after {MAX_RETRIES} retries.")
    return "None"

def call_openai_embedding(model, text):
    for i in range(MAX_RETRIES):
        try:
            client = get_openai_client()
            response = client.embeddings.create(
                model=model,
                input = [text]
            )
            tokens = response.usage
            with lock:
                global cost
                if model not in cost:
                    cost[model] =[0, 0]
                cost[model][0]+=int(tokens.prompt_tokens)
                cost[model][1]+=0
            return response.data[0].embedding
        except Exception as e:
            print(f"Error occurred: {e}. Retrying...")
            time.sleep(INTERVAL) 
        print(f"Failed to get response for text: {text} after {MAX_RETRIES} retries.")
        return "None"

def worker_chat(i, model, system_prompt, verify, verify_prompt_config):
    while not prompts_queue.empty():
        index, prompt, data = prompts_queue.get()
        # multi prompts in one case
        if isinstance(prompt, list) and isinstance(prompt[0], str) or isinstance(prompt[0], list):
            result = []
            for single_prompt in prompt:
                if not verify:
                    single_result = call_openai_chat(model, single_prompt, system_prompt)
                    result.append(single_result)
                else:
                    while True:
                        single_result = call_openai_chat(model, single_prompt, system_prompt, 0.7)
                        if single_prompt[-1]["role"] == "user":
                            rec_response = single_prompt[-1]["content"]
                        else:
                            rec_response = "empty"
                        verify_prompt = verify_prompt_config["prompt_template"].format(
                            target=data["target"],
                            user_response=single_result,
                            rec_response=rec_response
                        )
                        verify_result = call_openai_chat(model, verify_prompt, system_prompt=None)
                        if "YES" in verify_result:
                            break
        else:
            if not verify:
                result = call_openai_chat(model, prompt, system_prompt)
            else:
                while True:
                    result = call_openai_chat(model, prompt, system_prompt, 0.7)
                    if prompt[-1]["role"] == "user":
                        rec_response = prompt[-1]["content"]
                    else:
                        rec_response = "empty"
                    verify_prompt = verify_prompt_config["prompt_template"].format(
                        target=data["target"],
                        user_response=result,
                        rec_response=rec_response
                    )
                    verify_result = call_openai_chat(model, verify_prompt, system_prompt=None)
                    if "YES" in verify_result:
                        break
        results.put((index, result))
        if SLEEP_SECONDS > 0:
            time.sleep(SLEEP_SECONDS)
        with num_completed.get_lock():
            num_completed.value += 1   

def worker_embedding(i, model):
    while not prompts_queue.empty():
        index, prompt = prompts_queue.get()
        result = call_openai_embedding(model, prompt)
        results.put((index, result))
        with num_completed.get_lock():
            num_completed.value += 1


def progress_monitor(total):
    with tqdm(total=total) as pbar:
        while True:
            completed = num_completed.value
            pbar.n = completed
            pbar.refresh()
            if completed >= total:
                break
            time.sleep(0.1)

def gen_api_chat_answer(model, question_file, answer_file, args, system_prompt, verify=False, verify_prompt_config=None):
    prompts = []
    test_data = []
    for line in open(question_file):
        data = json.loads(line)
        test_data.append(data)
        prompts.append(data['prompt'])

    global results, num_completed, prompts_queue
    results = queue.PriorityQueue()
    num_completed = multiprocessing.Value('i', 0)

 
    prompts_queue = queue.Queue()
    for i, (prompt, data) in enumerate(zip(prompts, test_data)):
        prompts_queue.put((i, prompt, data))
 
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        progress_process = multiprocessing.Process(target=progress_monitor, args=(len(prompts),))
        progress_process.start()

        for i in range(MAX_THREADS):
            executor.submit(worker_chat, i, model, system_prompt, verify, verify_prompt_config)


    final_results = []
    while not results.empty():
        final_results.append(results.get())

    final_results.sort()
    

    progress_process.join()


    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    fd = open(answer_file, "w", encoding='utf-8')
    for data, (index, result) in zip(test_data, final_results):
        data["answer"] = result
        fd.write(json.dumps(data, ensure_ascii=False) + '\n')
    fd.close()
    try:
        cost_save()
    except Exception as e:
        print(e)

def gen_api_embedding_answer(model, question_file, answer_file):
    prompts = []
    test_data = []
    for line in open(question_file):
        data = json.loads(line)
        test_data.append(data)
        prompts.append(data['prompt'])

    global results, num_completed, prompts_queue
    results = queue.PriorityQueue()
    num_completed = multiprocessing.Value('i', 0)


    prompts_queue = queue.Queue()
    for i, prompt in enumerate(prompts):    
        prompts_queue.put((i, prompt))


    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        progress_process = multiprocessing.Process(target=progress_monitor, args=(len(prompts),))
        progress_process.start()

        for i in range(MAX_THREADS):
            executor.submit(worker_embedding, i, model, )


    final_results = []
    while not results.empty():
        final_results.append(results.get())


    final_results.sort()
    

    progress_process.join()


    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    fd = open(answer_file, "w", encoding='utf-8')
    for data, (index, result) in zip(test_data, final_results):
        data["answer"] = result
        fd.write(json.dumps(data, ensure_ascii=False) + '\n')
    fd.close()
    try:
        cost_save()
    except Exception as e:
        print(e)
        