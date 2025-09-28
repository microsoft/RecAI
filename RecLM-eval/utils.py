# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import torch
import pickle
import argparse
from tqdm import tqdm
from evaluates.TFIDF_model import string_match_score

def parse_args():
    parser = argparse.ArgumentParser()

    # Allow evaluating MULTIPLE datasets in one run.  Both --bench-name and
    # --bench-names are supported for backward-compatibility, but they now map
    # to the same destination variable ``bench_names`` which is *a list*.
    # Example:  --bench-names steam Books Movies_and_TV
    parser.add_argument(
        "--bench-names", "--bench-name",
        dest="bench_names",
        type=str,
        nargs="+",
        default=["steam"],
        help="One or more benchmark dataset names (folder names under ./data).",
    )
    parser.add_argument(
        "--task-names",
        type=str,
        nargs="+",
        default=None,
        help="A list of tasks to be evaluated.",
    )
    parser.add_argument("--model_path_or_name", type=str, help="Directory containing your language model for recommendation.")


    ## huggingface inferencing config
    parser.add_argument('--nodes', type=int, default=1, help='num nodes')
    parser.add_argument('--gpus', type=int, default=-1, help='num gpus per node')
    parser.add_argument('--nr', type=int, default=0, help='ranking within the nodes')
    parser.add_argument("--master_port", type=str, default='12343')

    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum number of tokens to generate, prompt+max_new_tokens should be less than 2048.")
    parser.add_argument("--batch_size", type=int, default=4)

    # length of recommendation list (e.g. Recall@K). Used by vllm_models.py and
    # post-processing logic to pad/truncate outputs.
    parser.add_argument("--top_k", type=int, default=20,
                        help="Recommendation list length K for @K metrics.")

    # pair-wise evaluation config
    parser.add_argument("--judge-model", type=str, default="gpt-3.5-turbo", help="The model path or name used to perform judge during pairwise evaluation.")
    parser.add_argument("--baseline-model", type=str, default="gpt-3.5-turbo", help="The model path or name acts as a baseline during pairwise evaluation.")
    
    # conversation task config
    parser.add_argument("--simulator-model", type=str, default="gpt-3.5-turbo", help="The model path or name acts as a user simulator during conversation task.")
    parser.add_argument('--max_turn', type=int, default=5, help='Number of conversation turns.')
    
    # embedding task config
    parser.add_argument("--item_emb_type", type=str, choices=["title", "description"], default="title")
    parser.add_argument("--user_emb_type", type=str, choices=["title", "summary"], default="title")
    parser.add_argument("--summary-model", type=str, default="gpt-3.5-turbo", help="The name of the model used to summary user preference.")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Backward compatibility layer
    # ------------------------------------------------------------------
    # Most of the existing code expects ``args.bench_name`` to be a *single*
    # string.  We keep that attribute pointing to the **first** dataset so
    # that unchanged code continues to work when only one benchmark is given.
    # When more than one benchmark is supplied, the caller should iterate over
    # ``args.bench_names``;   the legacy ``args.bench_name`` is still defined
    # (pointing to the first element) to avoid AttributeError in old paths.
    if len(args.bench_names) > 0:
        args.bench_name = args.bench_names[0]

    return args

def ReadLineFromFile(path):
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def gen_judge_prompts(eval_model_answer_file, base_model_answer_file, question_file, prompt_template):
    os.makedirs(os.path.dirname(question_file), exist_ok=True)
    fd = open(question_file, "w", encoding='utf-8')
    for line_eval, line_base in zip(open(eval_model_answer_file), open(base_model_answer_file)):
        data_eval = json.loads(line_eval)
        data_base = json.loads(line_base)
        data = {
            "prompt": [
                prompt_template.format(
                    question=data_eval["prompt"],
                    answer_a=data_eval["answer"],
                    answer_b=data_base["answer"],
                ),
                prompt_template.format(
                    question=data_eval["prompt"],
                    answer_a=data_base["answer"],
                    answer_b=data_eval["answer"],
                ),
            ]
        }
        fd.write(json.dumps(data, ensure_ascii=False)+'\n')
    fd.close()

def parse_model_name_to_dirname(model_path_or_name):
    if os.path.exists(model_path_or_name):
        return os.path.basename(os.path.normpath(model_path_or_name)).replace('/', '_').strip('_')
    return model_path_or_name.replace('/', '_').strip('_')

# -------------------------------
# Metric recording helper
# -------------------------------

def record_metrics(bench_name: str, model_path_or_name: str, task_name: str, metrics):
    """Append evaluation metrics of a task to a *human-readable* TXT file.

    The file will be saved to::

        output/{bench_name}/{model_dir}/metrics.txt

    Example content::

        --------------------------------------------------
        [2025-08-07 13:22:14] Task: ranking
        NDCG@1  Rec@1  Hits@1 Prec@1 MAP@1  MRR@1
        0.4021  0.3500  0.3500 0.3500 0.4021 0.4021
        ...
        --------------------------------------------------
    """
    import datetime, os

    model_dir = parse_model_name_to_dirname(model_path_or_name)
    out_dir = os.path.join("output", bench_name, model_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "metrics.txt")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = ["-"*60,
             f"[{timestamp}] Task: {task_name}"]

    # ---- pretty table for @1/5/10/20 ----
    header_tpl = "{:<7}{:<7}{:<7}{:<7}{:<7}{:<7}"
    row_tpl    = "{:<7.4f}{:<7.4f}{:<7.4f}{:<7.4f}{:<7.4f}{:<7.4f}"
    lines.append("")
    for k in [1, 5, 10, 20]:
        if f"ndcg@{k}" not in metrics:
            continue
        # header row
        lines.append(header_tpl.format(f"NDCG@{k}", f"Rec@{k}", f"Hits@{k}", f"Prec@{k}", f"MAP@{k}", f"MRR@{k}"))
        # metric values
        lines.append(row_tpl.format(
            metrics.get(f"ndcg@{k}", 0.0),
            metrics.get(f"recall@{k}", 0.0),
            metrics.get(f"hit@{k}", 0.0),
            metrics.get(f"precision@{k}", 0.0),
            metrics.get(f"map@{k}", 0.0),
            metrics.get(f"mrr@{k}", 0.0),
        ))
        lines.append("")

    # ---- additional (non-@K) metrics, e.g. accuracy ----
    other_metrics = {k: v for k, v in metrics.items() if "@" not in k or k == "acc@1"}
    if other_metrics:
        for k in sorted(other_metrics.keys()):
            lines.append(f"{k} = {other_metrics[k]}")

    lines.append("-"*60)

    with open(out_file, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def gen_user_simulator_prompts(eval_model_answer_file, simulator_question_file):
    os.makedirs(os.path.dirname(simulator_question_file), exist_ok=True)
    fd = open(simulator_question_file, "w", encoding='utf-8')
    for line in open(eval_model_answer_file):
        data_eval = json.loads(line)
        data = {
            "history": data_eval["history"],
            "target": data_eval["target"],
            "user_simulator_system_prompt": data_eval["user_simulator_system_prompt"],
            "prompt": [
                {
                    'role': 'system', 
                    'content': data_eval["user_simulator_system_prompt"]
                }
            ]
        }
        if "prompt" in data_eval:
            for text in data_eval["prompt"]:
                if text["role"] == "user":
                    text["role"] = "assistant"
                else:
                    text["role"] = "user"
                data["prompt"].append(text)
        if "answer" in data_eval:
            data["prompt"].append(
                {
                    "role": "user",
                    "content": data_eval["answer"]
                }
            )
        fd.write(json.dumps(data, ensure_ascii=False)+'\n')
    fd.close()

def gen_eval_model_conversation_prompts(simulator_answer_file, eval_model_question_file):
    os.makedirs(os.path.dirname(eval_model_question_file), exist_ok=True)
    fd = open(eval_model_question_file, "w", encoding='utf-8')
    for line in open(simulator_answer_file):
        data_sim = json.loads(line)
        data = {
            "history": data_sim["history"],
            "target": data_sim["target"],
            "user_simulator_system_prompt": data_sim["user_simulator_system_prompt"],
            "prompt": []
        }
        for text in data_sim["prompt"][1:]:
            if text["role"] == "assistant":
                text["role"] = "user"
            else:
                text["role"] = "assistant"
            data["prompt"].append(text)
        data["prompt"].append(
            {
                "role": "user",
                "content": data_sim["answer"]
            }
        )
        fd.write(json.dumps(data, ensure_ascii=False)+'\n')
    fd.close()

def load_prompt_config():
    prompt_config = {}
    for line in open("prompt_config.jsonl"):
        data = json.loads(line)
        prompt_config[data["task_name"]] = data
    return prompt_config

def fuzzy_substring_matching(target, source, model, threshold):
    candidates = [source]
    for line in source.split('\n'):
        for i in range(len(line)-len(target)+1):
            candidates.append(line[i:i+len(target)])
    if string_match_score(model, [target], candidates)[0]>=threshold:
        return True
    return False

def gen_item_embedding_prompt(bench_name, item_emb_type, prompt_file_path):
    meta_data_file = f"data/{bench_name}/metadata.json"
    fd = open(prompt_file_path, "w", encoding='utf-8')
    for idx, line in enumerate(open(meta_data_file)):
        line = json.loads(line)
        title = "No title"
        if "app_name" in line:
            title = line['app_name']
        elif "title" in line:
            title = line['title']
        else:
            if "description" in line:
                title = line['description'][:50]
            elif "categories" in line:
                title = line['categories'][0][-1]
        description = ""
        if "game_description" in line and len(line["game_description"].strip())>0:
            description = line["game_description"].strip()
        elif "desc_snippet" in line and len(line["desc_snippet"].strip())>0:
            description = line["desc_snippet"].strip()
        elif "description" in line and len(line["description"].strip())>0:
            description = line["description"].strip()
        data = {}
        data["id"] = idx + 1
        if item_emb_type == "title":
            data["prompt"] = title
        else:
            data["prompt"] = title + " " + description
        fd.write(json.dumps(data, ensure_ascii=False) + '\n')
    fd.close()

def extract_embedding(answer_path, embedding_path):
    max_id = 0
    embedding_size = 0
    for line in open(answer_path):
        data = json.loads(line)
        max_id = max(max_id, data["id"])
        embedding_size = len(data["answer"])
    embeddings = torch.zeros(max_id+1, embedding_size).tolist()
    for line in open(answer_path):
        data = json.loads(line)
        embeddings[int(data["id"])] = data["answer"]
    pickle.dump(embeddings, open(embedding_path, "wb"))

def gen_user_embedding_prompt(bench_name, user_emb_type, prompt_file_path, prompt_config):
    metadata_path = f"data/{bench_name}/metadata.json"
    itemid2title = ['padding']
    for line in open(metadata_path):
        line = json.loads(line)
        title = "No title"
        if "app_name" in line:
            title = line['app_name']
        elif "title" in line:
            title = line['title']
        else:
            if "description" in line:
                title = line['description'][:50]
            elif "categories" in line:
                title = line['categories'][0][-1]
        itemid2title.append(title)

    sequential_data_file = f"data/{bench_name}/sequential_data.txt"
    fd = open(prompt_file_path, "w", encoding='utf-8')
    for idx, line in enumerate(open(sequential_data_file)):
        if idx == 1000:
            break
        userid, itemids = line.strip().split(' ', 1)
        itemids = itemids.split(' ')
        data = {}
        data["id"] = int(userid)
        item_titles = ', '.join([itemid2title[int(x)] for x in itemids[:-1]])
        data["prompt"] = prompt_config[user_emb_type].format(item_titles)
        fd.write(json.dumps(data, ensure_ascii=False) + '\n')
    fd.close()

def extract_summary(answer_path, prompt_path):
    fd = open(prompt_path, "w", encoding='utf-8')
    for line in open(answer_path):
        data = json.loads(line)
        data["summary_prompt"] = data["prompt"]
        data["prompt"] = data["answer"]
        data.pop("answer")
        fd.write(json.dumps(data, ensure_ascii=False)+'\n')
    fd.close()

def load_embedding_array(filename):
    data = pickle.load(open(filename, "rb"))
    n = len(data[0])
    flag_rows = []
    for i, row in enumerate(data):
        if row is None or row == 'None':
            flag_rows.append(i)
    for i in flag_rows:
        data[i] = [0.0] * n
    return data

def gen_ranking_result(user_embedding_path, item_embedding_path, sequential_path, negative_path, answer_file):
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    fd = open(answer_file, "w", encoding='utf-8')

    item_embeddings = torch.tensor(load_embedding_array(item_embedding_path))
    user_embeddings = torch.tensor(load_embedding_array(user_embedding_path))
    print("shape of item embeddings: ", item_embeddings.shape)
    print("shape of user embeddings: ", user_embeddings.shape)
    
    for line1, line2 in tqdm(zip(ReadLineFromFile(sequential_path)[:1000], ReadLineFromFile(negative_path)[:1000]), ncols=80):
        user, items = line1.strip().split(' ', 1)
        sequence = [int(x) for x in items.split(' ')]
        user, items = line2.strip().split(' ', 1)
        candidate = [int(x) for x in items.split(' ') if int(x) != sequence[-1]]
        candidates = [sequence[-1]] + candidate[:19]

        scores = torch.softmax(torch.matmul(user_embeddings[int(user):int(user)+1], item_embeddings[candidates].T), -1).squeeze().tolist()
        scores = [(candidates[index], score) for index, score in enumerate(scores)]
        top_itemids = sorted(scores, key=lambda x:-x[1])[:20]
        data = {
            "target": sequence[-1],
            "result": [x[0] for x in top_itemids]
        }
        fd.write(json.dumps(data, ensure_ascii=False)+'\n')
    fd.close()

def gen_retrieval_result(user_embedding_path, item_embedding_path, sequential_path, answer_file):
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    fd = open(answer_file, "w", encoding='utf-8')
    
    item_embeddings = torch.tensor(load_embedding_array(item_embedding_path))
    user_embeddings = torch.tensor(load_embedding_array(user_embedding_path))
    print("shape of item embeddings: ", item_embeddings.shape)
    print("shape of user embeddings: ", user_embeddings.shape)
    
    for line in tqdm(ReadLineFromFile(sequential_path)[:1000], ncols=80):
        user, items = line.strip().split(' ', 1)
        sequence = [int(x) for x in items.split(' ')]
        history = sequence[:-1]

        scores = torch.softmax(torch.matmul(user_embeddings[int(user):int(user)+1], item_embeddings.T), -1).squeeze().tolist()
        scores = [(index, score) for index, score in enumerate(scores) if index not in history]
        top_itemids = sorted(scores, key=lambda x:-x[1])[:20]
        data = {
            "target": sequence[-1],
            "result": [x[0] for x in top_itemids]
        }
        fd.write(json.dumps(data, ensure_ascii=False)+'\n')
    fd.close()