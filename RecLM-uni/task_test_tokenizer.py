import argparse
import json
import os
import torch

from concurrent.futures import ProcessPoolExecutor
from Levenshtein import distance
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from train_utils.dataset import Test_task_group_mapping, SFTDataset
from train_utils.processor import FastPrefixConstrainedLogitsProcessor
from train_utils.metrics import Metrics
from train_utils.utils import (
    save_json,
    get_ctrl_item,
    rm_idx,
    load_json,
    side_tokenizer,
    process_train_sample,
    load_item_code_mapping,
    ITEM_CODE_FIELD,
)


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
            d['input_text'] if 'input_text' in d else process_train_sample(d['input_texts'], d['output_texts'], tokenizer)[3]
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
            d[f'{args.model_name}_output'] = o

        if i == 0:
            print(output_texts[0])


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
    parser.add_argument('--SFT_test_task', type=str, default='', help='in {SFTTestSeqRec, SFTTestRanking, SFT+TestPersonalControlRec, SFT-TestPersonalControlRec, SFTTestPersonalCategoryRate_xx%, SFTTestItemCount}')
    parser.add_argument("--num_process", type=int, default=128)
    parser.add_argument("--model_name", type=str, default='Llama-2-7b-hf-chat', help="openai model")
    parser.add_argument("--try_num", type=int, default=2, help="The number of attempts to call the API")
    parser.add_argument("--max_item_length", type=int, default=10)
    parser.add_argument("--max_token_length", type=int, default=512, help="The max length of input text to gpt")
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
    parser.add_argument("--ids", type=int, default=0)
    args = parser.parse_args()
    args.is_main_process = True
    print(json.dumps(args.__dict__, ensure_ascii=False, indent=2))
    data = {
        'category': load_json(args.data_path + 'category.jsonl'),
        'metas': load_json(args.data_path + 'metas.jsonl'),
        'sequential': load_json(args.data_path + 'sequential.jsonl'),
        'share_chat_gpt': None,
    }
    code_info = load_item_code_mapping(args.data_path)
    args.item_code_tokens = []
    if code_info:
        args.item_code_tokens = code_info.get('token_vocab', [])
        for item_id, token_seq in code_info.get('item_seq', {}).items():
            if item_id in data['metas']:
                data['metas'][item_id][ITEM_CODE_FIELD] = token_seq
        if args.item_code_tokens:
            print(f"Loaded item code mapping for {len(code_info.get('item_seq', {}))} items "
                  f"({len(args.item_code_tokens)} unique tokens, {code_info.get('missing', 0)} missing).")
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
    if getattr(args, 'item_code_tokens', None):
        added = tokenizer.add_special_tokens({'additional_special_tokens': args.item_code_tokens})
        if added > 0:
            print(f"Added {added} item code tokens to tokenizer for testing.")
    test_data = SFTDataset(args, TestTaskTemplate, TestTaskNum, data, tokenizer, 'test')
    dataset = args.data_path.strip('/').split('/')[-1]

    result_file = os.path.join(args.model_name, f'{dataset}_{args.SFT_test_task}_Top10{f"_CBS{args.CBS_type}" if args.use_CBS else ""}_test_Result_{args.ids}.jsonl')
    print(f"load file from {result_file}")
    test_data_list = load_json(result_file)
    _test_data_list = [_ for _ in test_data]
    if test_data_list and len(test_data_list) == len(_test_data_list):
        for _, __ in zip(test_data_list, _test_data_list):
            _.update(__)
    else:
        test_data_list = _test_data_list

    remain_test_data_list = [_ for _ in test_data_list if f'{args.model_name}_output' not in _][:]
    print(f"Loading Test Task: '{args.SFT_test_task}'. Remain Example Count: {len(remain_test_data_list)}")
    print(test_data_list[1]['input_texts'] if 'input_texts' in test_data_list[1] else test_data_list[1]['input_text'])

    process_dataset_hf(remain_test_data_list)

    if len(remain_test_data_list) > 0:
        save_json(test_data_list, result_file)

    with ProcessPoolExecutor(max_workers=args.num_process) as executor:
        result = list(tqdm(executor.map(process_api_output, test_data_list), total=len(test_data_list)))
    test_data_list = result

    metrics_dict = Metrics([args.SFT_test_task], args.topk, test_data.category2item, test_data.title2item)
    for step_i, example in tqdm(enumerate(test_data_list[:])):
        if f'{args.SFT_test_task}_output_title_list' not in example or len(example[f'{args.SFT_test_task}_output_title_list']) == 0:
            continue
        if args.use_control_symbol:
            output_label = [example['output_texts'][-1]]
        else:
            output_label = [_.strip() for _ in example['output_texts'][-1].strip().split('\n')]
            output_label = [rm_idx(_) if args.idx else _ for _ in output_label]
        metrics_dict.add_sample(example['task'], example['input_field_data'], example[f'{args.SFT_test_task}_output_title_list'], output_label, vague_mapping=False)

    metrics_dict.print()

    if len(remain_test_data_list) > 0:
        save_json(test_data_list, result_file)
