# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
import os
import time
import argparse
import pandas as pd
import tiktoken
import os.path as osp
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI, AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider, AzureCliCredential

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
        max_retries=5,
    )

MODEL = os.environ.get('MODEL')

if MODEL.startswith("gpt-3"):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # gpt-3.5-turbo gpt-4 gpt-4-0314 
else:
    encoding = tiktoken.encoding_for_model("gpt-4")

def call_chatgpt(prompt):
    max_retry_cnt = 5
    result = "NULL"
    for i in range(max_retry_cnt):
        try:
            response = client.chat.completions.create(
                model=MODEL,  
                messages=[
                    {"role": "system",
                    "content": "You are a helpful assistant. \n"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=1.0,
                top_p=1.0,
            )
            result = response.choices[0].message.content
            break
        except Exception as e:
            error_msg = str(e)
            print(f"OpenAI API Error: {error_msg}")            
            if "content filtering" in error_msg:
                break            
            if "time" in error_msg or "exceeded token rate limit" in error_msg:
                print("Rate limit reached. Waiting for 20 seconds...")
                time.sleep(20) 
    if not result:
        result = "NULL"
    return result


def process_row(writer, sample, columns):
    question = sample['question'] 
    input_token_num = len(encoding.encode(question))
    output = call_chatgpt(question)
    all_writes = [sample[col] for col in columns]
    all_writes.append(output)
    writer.writerow(all_writes)
    # writer.writerow([question, output])
    # writer.writerow([output, sample['model'], sample['label'], sample['history'], sample['target item'], question])
    # writer.writerow([sample['uid'], sample['iid'], sample['target'], sample['type'], sample['ground_truth'], question, output])
    output_token_num = len(encoding.encode(output))
    return input_token_num, output_token_num


def process_hf_data(dataset, output_file, args):
    filename = os.path.basename(output_file)
    with open(output_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(args.input_columns.split(',')+['answer'])
        # writer.writerow(['question', 'answer'])
        # writer.writerow(['score', 'model', 'label', 'history', 'target item', 'question'])
        # writer.writerow(['uid', 'iid', 'target', 'type', 'ground_truth', 'question', 'response'])
        total_input_token_num, total_output_token_num = 0, 0

        try:
            with ThreadPoolExecutor(max_workers=args.num_process) as executor:
                futures = []
                for i, sample in enumerate(dataset):
                    futures.append(executor.submit(process_row, writer, sample, args.input_columns.split(',')))

                for future in tqdm(futures, desc=filename):
                    input_token_num, output_token_num = future.result()
                    total_input_token_num += input_token_num
                    total_output_token_num += output_token_num
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, exiting...")
            for future in futures:  
                future.cancel()  
        
            executor.shutdown(wait=False)  
        
            print("Results of completed tasks:")  
            for future in futures:
                if future.done() and not future.cancelled():
                    try:  
                        input_token_num, output_token_num = future.result()
                        total_input_token_num += input_token_num
                        total_output_token_num += output_token_num
                    except Exception as e:  
                        print(f"Task generated an exception: {e}")  
            
        
    return total_input_token_num, total_output_token_num

## return a list of dictionaries
def load_jsonl_from_disk(file_path):
    ## read a csv file with pd, the first line is header 
    df = pd.read_csv(file_path)    
    return df.to_dict(orient='records')


def main(args):
    total_token_num, total_cost = 0, 0

    infile = args.input_file
    outfile = args.output_file
    data_as_list = load_jsonl_from_disk(infile)

    total_input_token_num, total_output_token_num = process_hf_data(data_as_list, outfile, args)
    # cost = 0.015 * total_input_token_num / 1000 + 0.0020 * total_output_token_num / 1000
    # cost = 0.01 * total_input_token_num / 1000 + 0.03 * total_output_token_num / 1000
    cost = 10 * total_input_token_num / 1000000 + 30 * total_output_token_num / 1000000
    total_token_num = total_input_token_num + total_output_token_num
    print(">> Task done. Use {:d} tokens in total, and cost $ {:.4f}.".format(total_token_num, cost)) 
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_process", type=int, default=1)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--input_columns", type=str, default="question")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    main(args)
