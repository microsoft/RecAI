# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import time
import yaml
import pandas as pd
from tqdm import tqdm
import openai
from openai import OpenAI, AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider, AzureCliCredential
import pickle

import queue
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

MAX_THREADS = 1
MAX_RETRIES = 5
INTERVAL = 5

api_key = os.environ.get('OPENAI_API_KEY') if os.environ.get('OPENAI_API_KEY') else None
api_base =  os.environ.get('OPENAI_API_BASE') if os.environ.get('OPENAI_API_BASE') else None
api_type = os.environ.get('OPENAI_API_TYPE') if os.environ.get('OPENAI_API_TYPE') else None
api_version =  os.environ.get('OPENAI_API_VERSION') if os.environ.get('OPENAI_API_VERSION') else None

if api_key:
    if api_type == "azure":
        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=api_base
        )
    else:
        client = OpenAI(  
            api_key=api_key
        )
else:
    credential = AzureCliCredential()    

    token_provider = get_bearer_token_provider(
        credential,
        "https://cognitiveservices.azure.com/.default"
    )

    client = AzureOpenAI(
        azure_endpoint=api_base,
        azure_ad_token_provider=token_provider,
        api_version=api_version,
        max_retries=MAX_RETRIES,
    )

def call_openai_embedding(model, text):
    for i in range(MAX_RETRIES):
        try:
            return client.embeddings.create(
                model=model,
                input = [text]
            ).data[0].embedding
        except Exception as e:
            print(f"Error occurred: {e}. Retrying...")
            time.sleep(INTERVAL*(i+1))  # Wait for INTERVAL seconds before retrying
    print(f"Failed to get response for text: {text} after {MAX_RETRIES} retries.")
    return "None"

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

def run_api_embedding(model, question_file, answer_file):
    prompts = pd.read_json(question_file, lines=True)
    prompts = prompts['text'].tolist()

    global results, num_completed, prompts_queue
    results = queue.PriorityQueue()
    num_completed = multiprocessing.Value('i', 0)

    # add all prompts to the queue
    prompts_queue = queue.Queue()
    for i, prompt in enumerate(prompts):
        prompts_queue.put((i, prompt))

    # use ThreadPoolExecutor to create a pool of threads to process the prompts
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        # start a progress monitor process
        progress_process = multiprocessing.Process(target=progress_monitor, args=(len(prompts),))
        progress_process.start()

        for i in range(MAX_THREADS):
            executor.submit(worker_embedding, i, model, )

    # collect results from the queue
    final_results = []
    while not results.empty():
        final_results.append(results.get())

    # sort the results by index
    final_results.sort()
    
    # wait for the progress monitor to finish
    progress_process.join()

    valid_index = 0
    for i, result in enumerate(final_results):
        if result[1] != "None":
            valid_index = i
            break
    embed_size = len(final_results[valid_index][1])
    embeddings = [result[1] if result[1] != "None" else [0.0]*embed_size for result in final_results]
    print("shape of result lists: ", len(embeddings))
    print(f'embedding size: {len(final_results[valid_index][1])}')
    pickle.dump(embeddings, open(answer_file, "wb"))