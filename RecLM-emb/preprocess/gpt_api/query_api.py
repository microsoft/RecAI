# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pandas as pd
import openai
from openai import OpenAI, AzureOpenAI
import argparse
from tqdm import tqdm
from collections import defaultdict
import time
import os

QUERY_COL = 'question'
LABEL_COL = 'target'
RESPONSE_COL = 'response'
DATAFRAME_SEP = ','


engine = os.environ.get('ENGINE')
api_key = os.environ.get('OPENAI_API_KEY')
api_base =  os.environ.get('OPENAI_API_BASE') if os.environ.get('OPENAI_API_BASE') else None
api_type = os.environ.get('OPENAI_API_TYPE') if os.environ.get('OPENAI_API_TYPE') else None
api_version =  os.environ.get('OPENAI_API_VERSION') if os.environ.get('OPENAI_API_VERSION') else None
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

def query_gpt_gpt4(batch_queries, system_message='', sleep=60, engine=None):    
    responses = []
 
    success = True
    exception_content_filter = False
    try:
        batch_response = client.chat.completions.create(
            model=engine, # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
            temperature=0.8,
            messages=[ 
                {"role": "system", "content": system_message},
                {"role": "user", "content": batch_queries[0]} 
            ]
        )
        # print('batch_response = {0}'.format(batch_response))
    except Exception as e:
        success = False
        print(str(e))
        if 'The response was filtered due to the prompt triggering Azure OpenAI\'s content management policy.' in str(e):
            exception_content_filter = True
            print('The query is :\n{0}'.format(batch_queries[0]))
    # print( batch_response['choices'][0]['message']['content'])
    # return
 
    if success: 
        for i, _ in enumerate(batch_queries): 
            try:
                text = batch_response.choices[i].message.content
                if text:
                    text = text.replace('\t', ' ').replace('\n', ' ')
            except Exception as e:
                print('Error in extract content from response:')
                print(f'batch_response = {batch_response}')
                print(str(e))
                text = "Fail"
            responses.append(text)
    else:
        for _ in batch_queries:
            responses.append('Fail')
    time.sleep(sleep) 
    return success, responses, exception_content_filter


 
def retry_gpt_api_func(batch_names, gpt_api_func, system_message, engine):
    print('Retrying GPT API Func..')
    retry_cnt = 5
    sleep_seconds = 2
    responses = []
    for name in batch_names:
        T = retry_cnt
        status = True
        while True:
            T -= 1
            status, cur_response, _ = gpt_api_func([name], sleep=sleep_seconds*(retry_cnt-T), system_message=system_message, engine=engine)
            if status == True or T <= 0:
                break
        if status:
            responses.append(cur_response[0])
        else:
            responses.append('Fail')
    return responses
 

r'''
    arguments:
        query_file: a csv file which will be loaded as a pandas dataframe.
                    Its QUERY_COL column contains the query content and the result will be filled in the RESPONSE_COL column (may be created automatically).
        
        system_message: a string which will be used as the system message before the prompt. It is optional. You can put everything in the prompt, without using system_message.
        
        prompt: a string which will be used to wrap the user input. 
                Should be something like 'text-before-user-input {0} text-after-user-input.', which contains one '{0}' in the prompt.
                The default is '{0}' which make the query content exactly the same as user input.

'''
def batch_queries_gpt(query_file, outfile, engine=None, system_message=None, prompt='{0}'):    
    gpt_api_func = query_gpt_gpt4
    sleep_seconds = 3 #2
    batch_size = 1 ## currently gpt-4 only support batch_size = 1    

    ## load pandas dataframe from query_file as query_profile
    query_profile = pd.read_csv(query_file, header=0, sep=DATAFRAME_SEP)
    n_rows = len(query_profile)

    ## create a new column name 'response' to query_profile, leave it empty for now
    if RESPONSE_COL not in query_profile:
        query_profile[RESPONSE_COL] = ['TODO'] * n_rows
    
    out_dir = os.path.dirname(outfile)
    os.makedirs(out_dir, exist_ok=True)
    
    def do_gpt_query(batch_queries, batch_idx):
        cur_status, cur_responses, excep_content_filter = gpt_api_func(batch_queries, sleep=sleep_seconds, system_message=system_message, engine=engine)
        if not cur_status and not excep_content_filter:
            cur_responses = retry_gpt_api_func(batch_queries, gpt_api_func, system_message, engine=engine)
            time.sleep(sleep_seconds) 
        for _i, _r in zip(batch_idx, cur_responses):
            query_profile.at[_i, RESPONSE_COL] = _r
        return excep_content_filter

        
    batch_queries = []
    batch_idx = []
    content_filter_cnt = 0
    
    for idx, row in tqdm(query_profile.iterrows(), desc=f'Querying {engine}', total=n_rows): 
        batch_queries.append(prompt.format(row[QUERY_COL]))
        batch_idx.append(idx)
        if len(batch_queries) >= batch_size:            
            excep_content_filter = do_gpt_query(batch_queries, batch_idx)
            if excep_content_filter:
                content_filter_cnt += 1
            batch_queries = []
            batch_idx = []
        
        if idx % 10 == 0:
            print(f'Saving checkpoint.. Content filter trigger cnt: {content_filter_cnt}')
            query_profile.to_csv(outfile, sep=DATAFRAME_SEP, index=False)


    if len(batch_queries) > 0:
        excep_content_filter = do_gpt_query(batch_queries, batch_idx)
        if excep_content_filter:
            content_filter_cnt += 1
        batch_queries = []
        batch_idx = []
     
    query_profile.to_csv(outfile, sep=DATAFRAME_SEP, index=False)

 



def recsys_job(): 
    global QUERY_COL, DATAFRAME_SEP, RESPONSE_COL, engine 
    QUERY_COL = 'question'
    RESPONSE_COL = f'response-{engine}'
    DATAFRAME_SEP = ',' 
    args = parse_args()
    if args.response_col is not None:
        RESPONSE_COL = args.response_col
    batch_queries_gpt(
        args.query_file,
        args.response_file,
        engine,
        system_message='',
        prompt='{0}'
    )

def parse_args():
    parser = argparse.ArgumentParser(description="data process")
    parser.add_argument(
        "--query_file", type=str, help=""
    )
    parser.add_argument(
        "--response_file", type=str, help=""
    )
    parser.add_argument(
        "--response_col", type=str, help="", default=None
    )
    args = parser.parse_args()
    return args  
 
def main():
    
    recsys_job()



if __name__ == '__main__':
    main()
