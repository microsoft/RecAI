import copy
import json
import os.path
import pickle
import re

import requests
import torch


def pad_sequence(seq: list[list], pad_token_id, device, pad_side='right'):
    max_len = max([len(s) for s in seq])
    for i, s in enumerate(seq):
        pad_seq = [pad_token_id] * (max_len - len(s))
        if pad_side == 'right':
            seq[i] = s + pad_seq
        elif pad_side == 'left':
            seq[i] = pad_seq + s
    return torch.tensor(seq, dtype=torch.long, device=device)


def rm_idx(s):
    return re.sub(r'^(\d+)\. *', '', s, count=1)


def match_idx(s):
    return re.match(r'^(\d+)\. *', s)


def load_pickle(filename):
    if filename is None or not os.path.exists(filename):
        return None
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_json(filename):
    if filename is None or not os.path.exists(filename):
        return None
    with open(filename, "r") as f:
        return json.load(f)


def save_pickle(data, filename):
    if filename is None:
        return
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_json(data, filename):
    if filename is None:
        return
    with open(filename, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def side_tokenizer(text: list[str] | list[list[str]], padding_side, tokenizer, **kwargs):
    tokenizer.padding_side = padding_side
    tokenizer.truncation_side = padding_side
    tokenizer_res = tokenizer.batch_encode_plus(text, **kwargs)
    if padding_side == 'right':
        input_ids = tokenizer_res['input_ids']
        for i, input_id in enumerate(input_ids):
            if input_id[-1] != -100:
                input_ids[i][-1] = tokenizer.eos_token_id
        tokenizer_res['input_ids'] = input_ids
    return tokenizer_res


def get_item_list(ip, users, sub_sequential, k, candidate_item_list=None, target_category=None, port=2024, immediately=True):
    """
    :param ip:
    :param users: user id
    :param sub_sequential: user history, [[item_1, ..., item_n]]
    :param k: top_k recommendation
    :param candidate_item_list: candidate items, [[item_1, ..., item_m]]
    :param target_category: '+C': only keep items in C. '-C': exclude items in C.
    :param port:
    :param immediately: don't wait for batching
    :return: return the recommendation list with k items that complying the params.
    """
    url = f"http://{ip}:{port}/inference"
    data = {
        "users": users,
        "item_lengths": [len(_) for _ in sub_sequential],
        "k": k,
        "item_lists": sub_sequential,
        "immediately": 1 if immediately else 0
    }
    if candidate_item_list is not None:
        data['candidate_item_lists'] = candidate_item_list
    if target_category is not None:
        data['target_category'] = target_category
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36 Edg/83.0.478.45",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive"
    }

    response = requests.post(url, json=data, headers=headers)
    assert response.status_code == 200
    return response.json()['inference'][0]


def masked_mean(seq, mask, dim=None):
    return (seq * mask).sum(dim=dim) / mask.sum(dim=dim)


def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out

    return inner


def get_history_text(output_titles: list[str]):
    history_text = ' → '.join(output_titles)
    return history_text


def get_output_text(output_titles: list[str], eos_token='', idx=False, user_control_symbol=False):
    if user_control_symbol:
        output_titles = [f'<SOI>{t}<EOI>' for t in output_titles]
    if not idx:
        output_text = '\n '.join(output_titles)
    else:
        output_text = '\n '.join([f'{i+1}. {t}' for i, t in enumerate(output_titles)])
    return output_text + eos_token


def get_ctrl_item(out_titles):
    pattern = r"<SOI>\s*(.*?)\s*<EOI>"
    item_list = re.findall(pattern, out_titles, re.MULTILINE)
    return item_list


def process_train_sample(input_texts, output_texts, tokenizer):
    system_input = 'You are an expert recommender engine as well as a helpful, respectful and honest assistant.'
    max_len = 500
    ignore_index = -100
    input_ids, complete_ids, labels = [], [], []
    system_text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_input}<|eot_id|>"
    infer_text = system_text
    temp_ids = tokenizer.encode(system_text)
    complete_ids.extend(temp_ids)
    labels.extend([ignore_index]*len(temp_ids))
    for idx in range(len(input_texts)):
        infer_text += f"<|start_header_id|>user<|end_header_id|>\n\n{input_texts[idx]}<|eot_id|>"
        temp_ids = tokenizer.batch_encode_plus(["<|start_header_id|>user<|end_header_id|>\n\n", f"{input_texts[idx]}<|eot_id|>"])['input_ids']
        complete_ids.extend(temp_ids[0])
        labels.extend([ignore_index] * len(temp_ids[0]))
        complete_ids.extend(temp_ids[1][-max_len:])
        labels.extend([ignore_index] * len(temp_ids[1][-max_len:]))

        temp_ids = tokenizer.batch_encode_plus(["<|start_header_id|>assistant<|end_header_id|>\n\n", f"{output_texts[idx]}<|eot_id|>"])['input_ids']
        complete_ids.extend(temp_ids[0])
        labels.extend([ignore_index] * len(temp_ids[0]))
        if idx == len(output_texts)-1:
            input_ids = copy.deepcopy(complete_ids)
            infer_text += f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            infer_text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{output_texts[idx]}<|eot_id|>"

        complete_ids.extend(temp_ids[1][:max_len])
        labels.extend(temp_ids[1][:max_len])
        if len(temp_ids[1]) > max_len:
            complete_ids[-1] = tokenizer.eos_token_id
            labels[-1] = tokenizer.eos_token_id

    return input_ids, complete_ids, labels, infer_text


def process_train_sample_llama2(input_texts, output_texts, tokenizer):
    system_input = 'You are an expert recommender engine as well as a helpful, respectful and honest assistant.'
    max_len = 500
    ignore_index = -100
    input_ids, complete_ids, labels = [], [], []
    system_text = f"<<SYS>>\n{system_input}\n<</SYS>>"
    infer_text = ""
    for idx in range(len(input_texts)):
        if idx == 0:
            input_text = f"<s>[INST] {system_text}\n\n{input_texts[idx]} [/INST]"
        else:
            input_text = f"<s>[INST] {input_texts[idx]} [/INST]"
        infer_text += input_text

        temp_ids = tokenizer.batch_encode_plus([input_text])['input_ids']
        complete_ids.extend(temp_ids[0][-max_len:])
        labels.extend([ignore_index] * len(temp_ids[0][-max_len:]))

        output_text = f"{output_texts[idx]} </s>"
        temp_ids = tokenizer.batch_encode_plus([output_text])['input_ids']
        if idx == len(output_texts)-1:
            input_ids = copy.deepcopy(complete_ids)
        else:
            infer_text += output_text

        complete_ids.extend(temp_ids[0][:max_len])
        labels.extend(temp_ids[0][:max_len])
        if len(temp_ids[0]) > max_len:
            complete_ids[-1] = tokenizer.eos_token_id
            labels[-1] = tokenizer.eos_token_id

    return input_ids, complete_ids, labels, infer_text


def gsm8K_extract_answer(completion):
    try:
        last_number = re.findall(r'-?[0-9]+[0-9.,]?', completion)[-1].rstrip('.').replace(',', '')
        return eval(last_number)
    except:
        return "[invalid]"


def gsm8K_is_correct(completion, answer):
    gold = gsm8K_extract_answer(answer)
    assert gold != "[invalid]", "No ground truth answer found in the document."
    return gsm8K_extract_answer(completion) == gold


def gsm8K_clean_answer(text):
    text = text.split("Question:")[0]
    return text
