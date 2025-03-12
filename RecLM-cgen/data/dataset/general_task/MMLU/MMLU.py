import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

choices = ["A", "B", "C", "D"]


def format_subject(subject):
    return ' '.join(subject.split("_"))


def get_prompt(train_df, subject, train_num, test_df, index, tokenizer):
    messages = [{'role': 'system', 'content': f"You are a helpful, respectful and honest assistant. The following are multiple choice questions (with answers) about {format_subject(subject)}. You need to select the correct answer."}]
    k = train_df.shape[1] - 2
    for idx in range(train_num):
        messages.append({'role': 'user', 'content': f"Question: {train_df.iloc[idx, 0]}\n" + "\n".join([f"{choices[j]}. {train_df.iloc[idx, j+1]}" for j in range(k)]) + '\nAnswer: '})
        messages.append({'role': 'assistant', 'content':  f"{train_df.iloc[idx, k+1]}\n\n"})
    messages.append({'role': 'user', 'content': f"Question: {test_df.iloc[index, 0]}\n" + "\n".join([f"{choices[j]}. {test_df.iloc[index, j+1]}" for j in range(k)]) + '\nAnswer: '})
    return tokenizer.apply_chat_template(messages, tokenize=False)


def evaluate(subject):
    def sample_process(i):
        prompt = get_prompt(dev_df, subject, args.ntrain, test_df, i, tokenizer)
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
        for a in ['▁A', '▁B', '▁C', '▁D', 'A', 'B', 'C', 'D']:
            if a not in dist:
                continue
            if dist[a] > max_logprobs:
                pred = a
                max_logprobs = dist[a]
        if i == 0:
            print("prompt: ", prompt)
            print("pred: ", pred)
        return pred

    dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
    test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)

    cors = []
    labels = test_df.iloc[:, test_df.shape[1]-1]
    with ThreadPoolExecutor(max_workers=256) as executor:
        preds = list(tqdm(executor.map(sample_process, range(test_df.shape[0])), total=test_df.shape[0]))

    for p, l in zip(preds, labels):
        cor = l.lower() in p.lower()
        cors.append(cor)

    acc = np.mean(cors)
    cors = np.array(cors)
    print(subject, "acc: ", acc)
    return cors, acc


def main(args):
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])
    subjects = subjects[args.subject_start:]

    print(subjects)
    print(args)
    print(args.model_name)
    all_cors = []
    results = []

    with torch.no_grad():
        start_time = time.time()
        for sub in subjects:
            result = evaluate(sub)
            results.append(result)
        end_time = time.time()
        print(f'time cost: {(end_time-start_time)}s')
        for cors, acc in results:
            print("{:.3f}".format(acc))
            all_cors.append(cors)

    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))

    with open(args.model_name.rstrip('/') + '_MMLU.txt', 'w') as f:
        f.write(str(np.mean(weighted_acc)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="./")
    parser.add_argument("--subject_start", type=int, default=0)
    parser.add_argument("--port", type=int, default=13579)
    parser.add_argument("--model_name", type=str, default="snap/.../SFT_Epoch20/")

    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    main(args)


# The following are multiple choice questions (with answers) about abstract algebra.
#
# Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.
# A. 0
# B. 1
# C. 2
# D. 3
# Answer: B
#
# Statement 1 | If aH is an element of a factor group, then |aH| divides |a|. Statement 2 | If H and K are subgroups of G then HK is a subgroup of G.
# A. True, True
# B. False, False
# C. True, False
# D. False, True
# Answer: B
#
# Statement 1 | Every element of a group generates a cyclic subgroup of the group. Statement 2 | The symmetric group S_10 has 10 elements.
# A. True, True
# B. False, False
# C. True, False
# D. False, True
# Answer: C
#
# Statement 1| Every function from a finite set onto itself must be one to one. Statement 2 | Every subgroup of an abelian group is abelian.
# A. True, True
# B. False, False
# C. True, False
# D. False, True
# Answer: A
#
# Find the characteristic of the ring 2Z.
# A. 0
# B. 3
# C. 12
# D. 30
# Answer: A
#
# Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.
# A. 0
# B. 4
# C. 2
# D. 6
# Answer:
