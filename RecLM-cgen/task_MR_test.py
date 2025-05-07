import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from Levenshtein import distance
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from train_utils.dataset import Test_task_group_mapping, SFTDataset
from train_utils.processor import FastPrefixConstrainedLogitsProcessor
from train_utils.metrics import Metrics
from train_utils.utils import save_json, get_ctrl_item, rm_idx, load_json, load_pickle, side_tokenizer, process_train_sample, gsm8K_clean_answer, gsm8K_is_correct

headers = {"User-Agent": "Test Client"}
GSM8K_Q1 = '''Question: In 2004, there were 60 kids at a cookout. In 2005, half the number of kids came to the cookout as compared to 2004. In 2006, 2/3 as many kids came to the cookout as in 2005. How many kids came to the cookout in 2006?'''
GSM8K_A1 = '''Let's think step by step.
In 2005, 60/2=30 kids came to the cookout.
In 2006, 30/3*2=20 kids came to the cookout.
The answer is 20'''


@torch.no_grad()
def process_dataset_hf(data_list):
    if len(data_list) == 0:
        return

    eot_token = "<|eot_id|>"
    eot_token_id = tokenizer.convert_tokens_to_ids(eot_token)
    num_beams = 1
    logits_processors = [
        FastPrefixConstrainedLogitsProcessor(test_data.item_prefix_tree.constrain_search_list, num_beams)
    ] if args.use_CBS else None
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map=args.gpu).eval()
    bs = args.batch_size
    for i in tqdm(range(0, len(data_list), bs)):
        input_texts = [
            d['input_text'] if 'input_text' in d else process_train_sample([GSM8K_Q1]+d['input_texts'], [GSM8K_A1]+d['output_texts'], tokenizer)[3]
            for d in data_list[i: i + bs]
        ]
        input_data = side_tokenizer(input_texts, 'left', tokenizer, padding=True, truncation=True,
                                    max_length=args.max_token_length, return_tensors='pt').to(device=args.gpu).data
        input_ids_length = input_data['input_ids'].shape[1]
        output_ids = model.generate(
            **input_data,
            logits_processor=logits_processors if args.use_CBS else None,
            max_length=args.max_token_length + args.gen_max_length,
            num_beams=num_beams,
            num_return_sequences=1,
            eos_token_id=eot_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        output_texts = tokenizer.batch_decode(output_ids[:, input_ids_length:],
                                              skip_special_tokens=False if args.use_control_symbol else True)

        for d, o in zip(data_list[i: i + bs], output_texts):
            d[f'{args.model_name}_output'] = o.split(tokenizer.eos_token)[0]

        if i == 0:
            print(output_texts[0])

        input_texts_gsm8k = [
            d['input_text'] if 'input_text' in d else
            process_train_sample(
                [GSM8K_Q1]+d['input_texts']+[d['gsm8k_question']],
                [GSM8K_A1]+d['output_texts'][:-1]+d[f'{args.model_name}_output'].split(eot_token)[:1]+[d['gsm8k_answer']],
                tokenizer
            )[3]
            for d in data_list[i: i + bs]
        ]
        if i == 0:
            print(input_texts_gsm8k[0])

        input_data = side_tokenizer(input_texts_gsm8k, 'left', tokenizer, padding=True, truncation=True,
                                    max_length=args.max_token_length, return_tensors='pt').to(device=args.gpu).data
        input_ids_length = input_data['input_ids'].shape[1]
        output_ids = model.generate(
            **input_data,
            logits_processor=logits_processors if args.use_CBS else None,
            max_length=args.max_token_length + args.gen_max_length,
            num_beams=num_beams,
            num_return_sequences=1,
            eos_token_id=eot_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        output_texts_gsm8k = tokenizer.batch_decode(output_ids[:, input_ids_length:],
                                                    skip_special_tokens=False if args.use_control_symbol else True)
        for d, o in zip(data_list[i: i + bs], output_texts_gsm8k):
            d[f'{args.model_name}_output_gsm8k'] = o.split(tokenizer.eos_token)[0]

        if i == 0:
            print(output_texts_gsm8k[0])


if __name__ == "__main__":
    def vague_mapping(ts):
        for idx, __ in enumerate(ts):
            if __ in test_data.title2item:
                continue
            for ___ in test_data.title2item:
                if distance(__, ___) <= 3:
                    ts[idx] = ___
                    break

    def process_api_output(d):
        if f'{args.model_name}_output' not in d:
            return d
        if d[f'{args.model_name}_output'] == "":
            d[f'{args.SFT_test_task}_output_title_list'] = []
            return d
        if f'{args.SFT_test_task}_output_title_list' in d:
            return d

        raw_output = d[f'{args.model_name}_output']
        if args.use_control_symbol:
            ts = get_ctrl_item(raw_output)
        else:
            ts = [_.strip() for _ in raw_output.strip().split('\n')]
            ts = [rm_idx(_) if args.idx else _ for _ in ts]

        vague_mapping(ts)
        d[f'{args.SFT_test_task}_output_title_list'] = ts

        return d

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='data/dataset/sub_movie/', help="processed_data path")
    parser.add_argument("--GSM8K_data_path", type=str, default='data/dataset/general_task/GSM8K/', help="GSM8K data path")
    parser.add_argument('--SFT_test_task', type=str, default='', help='in {SFTTestSeqRec, SFTTestRanking, SFT+TestPersonalControlRec, SFT-TestPersonalControlRec, SFTTestPersonalCategoryRate_xx%, SFTTestItemCount}')
    parser.add_argument("--num_process", type=int, default=128)
    parser.add_argument("--model_name", type=str, default='Llama-2-7b-hf-chat', help="openai model")
    parser.add_argument("--try_num", type=int, default=2, help="The number of attempts to call the API")
    parser.add_argument("--max_item_length", type=int, default=10)
    parser.add_argument("--max_token_length", type=int, default=1024, help="The max length of input text to gpt")
    parser.add_argument("--gen_max_length", type=int, default=1024)
    parser.add_argument("--candidate_num", type=int, default=10)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--item_index", type=str, default='title_t')
    parser.add_argument("--backup_ip", type=str, default='0.0.0.0')
    parser.add_argument("--idx", action='store_true')
    parser.add_argument("--use_control_symbol", action='store_true')
    parser.add_argument("--use_CBS", action='store_true')
    parser.add_argument("--CBS_type", type=int, default=2)
    parser.add_argument("--port", type=int, default=13579)
    parser.add_argument("--gpu", type=str, default='cuda:0')
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    args.is_main_process = True
    print(json.dumps(args.__dict__, ensure_ascii=False, indent=2))
    data = {
        'category': load_json(args.data_path + 'category.jsonl'),
        'metas': load_json(args.data_path + 'metas.jsonl'),
        'sequential': load_json(args.data_path + 'sequential.jsonl'),
        'share_chat_gpt': None,
    }
    TestTaskTemplate = {args.SFT_test_task: Test_task_group_mapping[args.SFT_test_task]}
    TestTaskNum = {args.SFT_test_task: 1}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = '<|reserved_special_token_250|>'
    tokenizer.pad_token_id = 128255
    tokenizer.soi_token = "<SOI>"
    tokenizer.eoi_token = "<EOI>"
    tokenizer.soi_token_id = tokenizer.convert_tokens_to_ids(tokenizer.soi_token)
    tokenizer.eoi_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eoi_token)
    tokenizer.eos_token = "<|eot_id|>"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    dataset = args.data_path.strip('/').split('/')[-1]

    test_data = SFTDataset(args, TestTaskTemplate, TestTaskNum, data, tokenizer, 'test')
    metrics_dict = Metrics([args.SFT_test_task], args.topk, test_data.category2item, test_data.title2item)

    GSM8K_dataset = load_from_disk(args.GSM8K_data_path)
    _test_data_list = [_ for _ in test_data][:len(GSM8K_dataset["test"])]
    for d, gsm8k_d in zip(_test_data_list, GSM8K_dataset["test"]):
        d['gsm8k_question'] = gsm8k_d['question']
        d['gsm8k_answer'] = gsm8k_d['answer']

    print(_test_data_list[1]['input_texts'] if 'input_texts' in _test_data_list[1] else _test_data_list[1]['input_text'])
    result_file = os.path.join(args.model_name, f'{dataset}_{args.SFT_test_task}_Top10{f"_CBS{args.CBS_type}" if args.use_CBS else ""}_test_MR_Result.jsonl')
    test_data_list = (load_json(result_file) or [])[:len(GSM8K_dataset["test"])]
    # test_data_list = []
    if test_data_list and len(test_data_list) == len(_test_data_list):
        for _, __ in zip(test_data_list, _test_data_list):
            _.update(__)
    else:
        test_data_list = _test_data_list

    remain_test_data_list = [_ for _ in test_data_list if f'{args.model_name}_output' not in _ or f'{args.model_name}_output_gsm8k' not in _][:]

    process_dataset_hf(remain_test_data_list)
    with ProcessPoolExecutor(max_workers=args.num_process) as executor:
        result = list(tqdm(executor.map(process_api_output, test_data_list), total=len(test_data_list)))
    test_data_list = result

    if len(remain_test_data_list) > 0:
        save_json(test_data_list, result_file)

    for step_i, example in tqdm(enumerate(test_data_list)):
        if f'{args.SFT_test_task}_output_title_list' not in example or len(example[f'{args.SFT_test_task}_output_title_list']) == 0:
            continue
        if args.use_control_symbol:
            output_label = [example['output_texts'][-1]]
        else:
            output_label = [_.strip() for _ in example['output_texts'][-1].strip().split('\n')]
            output_label = [rm_idx(_) if args.idx else _ for _ in output_label]
        metrics_dict.add_sample(example['task'], example['input_field_data'], example[f'{args.SFT_test_task}_output_title_list'], output_label, vague_mapping=False)

    metrics_dict.print()

    acc_res = []
    ctl_res = []
    for example in test_data_list:
        pred = gsm8K_clean_answer(example[f'{args.model_name}_output_gsm8k'])
        label = example["gsm8k_answer"]
        acc = gsm8K_is_correct(pred, label)
        acc_res.append(acc)
        ctl_res.append(example[f'{args.model_name}_output_gsm8k'].count('<SOI>') + example[f'{args.model_name}_output_gsm8k'].count('<EOI>'))

    print("GSM8K acc: ", np.mean(acc_res))
    print("CSN_{R2}^{n=0}: ", (len(ctl_res) - len(np.nonzero(ctl_res)[0])) / len(ctl_res))

    item_count1_res = [1 if len(example[f'{args.SFT_test_task}_output_title_list']) == args.topk else 0 for example in test_data_list]
    print(f"CSN_{{R1}}^{{n={args.topk}}}: ", sum(item_count1_res) / len(item_count1_res))

    if len(remain_test_data_list) > 0:
        save_json(test_data_list, result_file)

