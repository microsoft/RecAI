# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import requests
from tqdm import tqdm
from sft.dataset import Test_task_group_mapping, SFTDataset
from param import Config
from utils.metrics import Metrics
from utils.tools import match_idx, rm_idx, vague_map, GPT, load_pickle, save_pickle

headers = {"User-Agent": "Test Client"}


def quary_vllm_openai(input_text, args):
    for ii in range(args.try_num):
        pload = {
            "model": args.model_name,
            "prompt": input_text,
            "n": 1,
            "temperature": 0.0,
            "max_tokens": args.gen_max_length,
        }
        response = requests.post(f'http://127.0.0.1:{args.vllm_port}/v1/completions', headers=headers, json=pload, stream=False)
        output_data = json.loads(response.content)
        output_text = output_data["choices"][0]['text']
        output = output_text
        if output is None:
            continue
        return output


def quary_openai(input_text, args):
    for ii in range(args.try_num):
        output = gpt.call(input_text)
        if output is None:
            continue
        return output


def quary_api(d, args):
    global wrongtime
    try:
        if f'{args.model_name}_output' not in d:
            input_text = d['input_text']
            if args.general_llm:
                input_text = d['input_text'].split('\n')
                sub_text1 = input_text[1].strip()
                sub_text2 = input_text[4].split('[/INST]')[0].strip()
                sub_text3 = input_text[4].split('[/INST]')[1].split('[INST]')[1].strip()
                if args.SFT_test_task in ['SFTTestSeqRec', 'SFTTestItemCount']:
                    sub_text4 = 'Notice! Do not explain the reason or include any other words.'
                else:
                    sub_text4 = 'Notice! You need to output category information in the brackets after item titles in this template: 1. title (category). Do not explain the reason or include any other words in the front of titles.'
                input_text = f'{sub_text1} {sub_text2} {sub_text3} \n{sub_text4}'
                d['raw_input_text'] = input_text

            if args.vllm_port > 0:
                d[f'{args.model_name}_output'] = quary_vllm_openai(input_text, args)
            else:
                d[f'{args.model_name}_output'] = quary_openai(input_text, args)

        assert f'{args.model_name}_output' in d, f'no {args.model_name}_output'
        wrongtime = 0

    except Exception as e:
        print(str(e), flush=True)
        wrongtime += 1
        if wrongtime > 10:
            assert 1 == 0, 'wrong'

    return 1


if __name__ == "__main__":
    def process_api_output(d):
        if f'{args.model_name}_output' not in d:
            return d
        if d[f'{args.model_name}_output'] == "":
            d[f'{args.SFT_test_task}_output_title_list'] = []
            return d
        if f'{args.SFT_test_task}_output_title_list' in d:
            return d
        raw_output = d[f'{args.model_name}_output']
        if raw_output[0] == raw_output[-1] == '"' or raw_output[0] == raw_output[-1] == "'":
            raw_output = raw_output[1:-1]

        ts = raw_output.split('\n')
        ts = [rm_idx(_).strip().split('\n')[0].strip() for _ in ts if match_idx(_)]
        if args.SFT_test_task != 'SFTTestSeqRec':
            ts = [re.sub(r' *[(,\[](.*)[),\]]$', '', _) for _ in ts]

        ts = [t[1:-1] if t[0] == t[-1] == "'" or t[0] == t[-1] == "\"" else t for t in ts if t != '']
        ts = [t.strip() for t in ts]
        ts = ts[:d['input_field_data']['item_count']]

        ts = vague_map(ts, test_data.title2item)
        d[f'{args.SFT_test_task}_output_title_list'] = ts

        return d

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='data/dataset/sub_movie/', help="processed_data path")
    parser.add_argument('--SFT_test_task', type=str, default='', help='in {SFTTestSeqRec, SFTTestRanking, SFT+TestPersonalControlRec, SFT-TestPersonalControlRec, SFTTestPersonalCategoryRate_xx%, SFTTestItemCount}')
    parser.add_argument("--num_process", type=int, default=80)
    parser.add_argument("--model_name", type=str, default='Llama-2-7b-hf-chat', help="openai model")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--try_num", type=int, default=2, help="The number of attempts to call the API")
    parser.add_argument("--max_item_length", type=int, default=10)
    parser.add_argument("--max_token_length", type=int, default=512, help="The max length of input text to gpt")
    parser.add_argument("--gen_max_length", type=int, default=1024)
    parser.add_argument("--candidate_num", type=int, default=10)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--item_index", type=str, default='title64_t')
    parser.add_argument("--backup_ip", type=str, default='0.0.0.0')
    parser.add_argument("--llama2_chat_template", action='store_true')
    parser.add_argument("--idx", action='store_true')
    parser.add_argument("--sample", action='store_true')
    parser.add_argument("--reprocess", action='store_true')
    parser.add_argument("--teacher_port", type=int, default=12621)
    parser.add_argument("--vllm_port", type=int, default=13579)
    parser.add_argument("--general_llm", action='store_true')
    args = parser.parse_args()
    args.is_main_process = True
    print(Config(**vars(args)))
    assert args.output_path is not None
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    gpt = GPT()

    category2item = load_pickle(args.data_path + 'category.pickle')
    metas = load_pickle(args.data_path + 'meta.pickle')
    item2category = {}
    for c in category2item:
        for i in category2item[c]:
            if item2category.get(i) is None:
                item2category[i] = []
            item2category[i].append(c)
    title2item = {}
    for _ in metas:
        if title2item.get(metas[_][args.item_index]) is None:
            title2item[metas[_][args.item_index]] = []
        title2item[metas[_][args.item_index]].append(_)
    data = {
        'metas': metas,
        'category2item': category2item,
        'item2category': item2category,
        'title2item': title2item,
        'sequential': load_pickle(args.data_path + 'sequential.pickle'),
        'share_chat_gpt': None,
        'ranking_candidate': load_pickle(args.data_path + 'ranking_candidate.pickle'),
    }
    TestTaskTemplate = {args.SFT_test_task: Test_task_group_mapping[args.SFT_test_task.split('_')[0]]}
    TestTaskNum = {args.SFT_test_task: 1}
    if args.SFT_test_task in ['SFT+TestPersonalControlRec', 'SFT-TestPersonalControlRec'] or args.SFT_test_task.startswith('SFTTestPersonalCategoryRate'):
        TestSeqRec_Result_file = os.path.join(args.output_path, f'SFTTestSeqRec_Top{args.topk}_Result.pickle')
        data['SFTTestSeqRec_Result'] = load_pickle(TestSeqRec_Result_file)
    test_data = SFTDataset(args, TestTaskTemplate, TestTaskNum, data, None, 'test')
    metrics_dict = Metrics([args.SFT_test_task], args.topk, test_data.category2item, test_data.title2item)
    result_file = os.path.join(args.output_path, f'{args.SFT_test_task}_Top{args.topk}_Result{"_Sample" if args.sample else ""}.pickle')

    test_data_list = load_pickle(result_file)
    _test_data_list = [_ for _ in test_data]
    if test_data_list and len(test_data_list) == len(_test_data_list):
        for _, __ in zip(test_data_list, _test_data_list):
            _.update(__)
    else:
        test_data_list = _test_data_list

    if args.SFT_test_task in ['SFT+TestPersonalControlRec', 'SFT-TestPersonalControlRec'] or args.SFT_test_task.startswith('SFTTestPersonalCategoryRate'):
        remain_test_data_list = [_ for _, __ in zip(test_data_list, data['SFTTestSeqRec_Result'])
                                 if f'{args.model_name}_output' not in _ and 'SFTTestSeqRec_output_title_list' in __][:]
    else:
        remain_test_data_list = [_ for _ in test_data_list[:] if f'{args.model_name}_output' not in _]
    print(f"Loading Test Task: '{args.SFT_test_task}'. Remain Example Count: {len(remain_test_data_list)}")
    print(test_data_list[1]['input_text'])
    with ThreadPoolExecutor(max_workers=args.num_process) as executor:
        results = list(tqdm(executor.map(lambda d: quary_api(d, args), remain_test_data_list), total=len(remain_test_data_list)))

    if len(remain_test_data_list) > 0:
        save_pickle(test_data_list, result_file)
    if not args.general_llm:
        for step_i, example in tqdm(enumerate(test_data_list)):
            if f'{args.model_name}_output' not in example or (f'{args.SFT_test_task}_output_title_list' in example and not args.reprocess):
                continue
            output_title = example[f'{args.model_name}_output']
            output_title_list = [_.strip() for _ in output_title.strip().split('\n')]
            output_title_list = [rm_idx(_) if args.idx else _ for _ in output_title_list]
            output_title_list = vague_map(output_title_list, test_data.title2item)
            example[f'{args.SFT_test_task}_output_title_list'] = output_title_list
    else:
        with ProcessPoolExecutor(max_workers=args.num_process) as executor:
            results = list(tqdm(executor.map(process_api_output, test_data_list), total=len(test_data_list)))
        test_data_list = results

    for step_i, example in tqdm(enumerate(test_data_list)):
        if f'{args.SFT_test_task}_output_title_list' not in example or len(example[f'{args.SFT_test_task}_output_title_list']) == 0:
            continue
        output_label = [_.strip() for _ in example['output_text'].strip().split('\n')]
        output_label = [rm_idx(_) if args.idx else _ for _ in output_label]
        metrics_dict.add_sample(example['task'], example['input_field_data'],
                                example[f'{args.SFT_test_task}_output_title_list'], output_label, vague_mapping=False)

    metrics_dict.print()
    if len(remain_test_data_list) > 0:
        save_pickle(test_data_list, result_file)


