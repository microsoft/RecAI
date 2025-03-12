import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from tqdm import tqdm
from transformers import AutoTokenizer

choices = ["A", "B", "C", "D", "E"]


def load_jsonl_data(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data


def get_prompt(train_data, test_item, train_num, tokenizer):
    messages = [
        {'role': 'system', 'content': "You are a helpful, respectful and honest assistant. The following are multiple choice questions (with answers). You need to select the correct answer."},
    ]
    for idx in range(min(train_num, len(train_data))):
        question = train_data[idx]['question']['stem']
        choices_text = "\n".join([f"{choice['label']}. {choice['text']}" for choice in train_data[idx]['question']['choices']])
        answer = train_data[idx]['answerKey']
        messages.append({'role': 'user', 'content': f"Question: {question}\n{choices_text}\nAnswer:"})
        messages.append({'role': 'assistant', 'content':  f'{answer}'})
    question = test_item['question']['stem']
    choices_text = "\n".join([f"{choice['label']}. {choice['text']}" for choice in test_item['question']['choices']])
    messages.append({'role': 'user', 'content': f"Question: {question}\n{choices_text}\nAnswer:"})
    return tokenizer.apply_chat_template(messages, tokenize=False)


def evaluate():
    def sample_process(i):
        prompt = get_prompt(dev_data, test_data[i], args.ntrain, tokenizer)
        pload = {
            "model": args.model_name,
            "prompt": prompt,
            "max_tokens": 1,
            "logprobs": 20,
            "temperature": 0.0
        }
        response = requests.post(f'http://127.0.0.1:{args.port}/v1/completions', json=pload, stream=False)
        output_data = json.loads(response.content)
        dist = output_data['choices'][0]['logprobs']['top_logprobs'][0]
        pred = ''
        max_logprobs = float("-inf")
        for a in ['▁A', '▁B', '▁C', '▁D', '▁E', '_A', '_B', '_C', '_D', '_E', 'A', 'B', 'C', 'D', 'E']:
            if a not in dist:
                continue
            if dist[a] > max_logprobs:
                pred = a
                max_logprobs = dist[a]
        if i == 0:
            print("prompt: ", prompt)
            print("pred: ", pred)
        return pred

    cors = []
    labels = [_['answerKey'] for _ in test_data]
    with ThreadPoolExecutor(max_workers=256) as executor:
        preds = list(tqdm(executor.map(sample_process, range(len(test_data))), total=len(test_data)))

    for p, l in zip(preds, labels):
        cor = l.lower() in p.lower()
        cors.append(cor)

    acc = np.mean(cors)
    cors = np.array(cors)
    print("acc: ", acc)
    return cors, acc


def main():
    start_time = time.time()
    evaluate()
    end_time = time.time()
    print(f'Time cost: {end_time - start_time:.2f}s')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=7)
    parser.add_argument("--data_dir", "-d", type=str, default="./")
    parser.add_argument("--port", type=int, default=13579)
    parser.add_argument("--model_name", type=str, default="snap/.../SFT_Epoch20/")
    args = parser.parse_args()
    print(args)

    dev_data_path = os.path.join(args.data_dir, "dev.jsonl")
    test_data_path = os.path.join(args.data_dir, "test.jsonl")
    dev_data = load_jsonl_data(dev_data_path)
    test_data = load_jsonl_data(test_data_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    main()

'''
[INST] <<SYS>>
You are a helpful, respectful and honest assistant.
<</SYS>>

The following are multiple choice questions (with answers). You need to select the correct answer.

 [/INST] Ok. </s><s>[INST] 
Question: A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?
A. bank
B. library
C. department store
D. mall
E. new york
Answer: [/INST] A </s><s>[INST] 
Question: What do people aim to do at work?
A. complete job
B. learn from each other
C. kill animals
D. wear hats
E. talk to each other
Answer: [/INST] A </s><s>[INST] 
Question: Where would you find magazines along side many other printed works?
A. doctor
B. bookstore
C. market
D. train station
E. mortuary
Answer: [/INST] B </s><s>[INST] 
Question: Where are  you likely to find a hamburger?
A. fast food restaurant
B. pizza
C. ground up dead cows
D. mouth
E. cow carcus
Answer: [/INST] A </s><s>[INST] 
Question: James was looking for a good place to buy farmland.  Where might he look?
A. midwest
B. countryside
C. estate
D. farming areas
E. illinois
Answer: [/INST] A </s><s>[INST] 
Question: What island country is ferret popular?
A. own home
B. north carolina
C. great britain
D. hutch
E. outdoors
Answer: [/INST] C </s><s>[INST] 
Question: In what Spanish speaking North American country can you get a great cup of coffee?
A. mildred's coffee shop
B. mexico
C. diner
D. kitchen
E. canteen
Answer: [/INST] B </s><s>[INST] 
Question: What do animals do when an enemy is approaching?
A. feel pleasure
B. procreate
C. pass water
D. listen to each other
E. sing
Answer: [/INST]
'''

'''
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful, respectful and honest assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

The following are multiple choice questions (with answers). You need to select the correct answer.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Ok.<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?
A. bank
B. library
C. department store
D. mall
E. new york
Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

A<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: What do people aim to do at work?
A. complete job
B. learn from each other
C. kill animals
D. wear hats
E. talk to each other
Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

A<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: Where would you find magazines along side many other printed works?
A. doctor
B. bookstore
C. market
D. train station
E. mortuary
Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

B<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: Where are  you likely to find a hamburger?
A. fast food restaurant
B. pizza
C. ground up dead cows
D. mouth
E. cow carcus
Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

A<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: James was looking for a good place to buy farmland.  Where might he look?
A. midwest
B. countryside
C. estate
D. farming areas
E. illinois
Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

A<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: What island country is ferret popular?
A. own home
B. north carolina
C. great britain
D. hutch
E. outdoors
Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

C<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: In what Spanish speaking North American country can you get a great cup of coffee?
A. mildred's coffee shop
B. mexico
C. diner
D. kitchen
E. canteen
Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

B<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: What do animals do when an enemy is approaching?
A. feel pleasure
B. procreate
C. pass water
D. listen to each other
E. sing
Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>


'''