# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/usr/bin/env python
# coding=utf-8
import json
import logging
import math
import os
import random
from pathlib import Path
import pandas as pd

import datasets
import torch
import transformers
from datasets import load_dataset
from huggingface_hub import Repository
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoTokenizer,
    GenerationConfig
)
from transformers.utils import get_full_repo_name

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DummyOptim, DummyScheduler, set_seed, DistributedDataParallelKwargs

from arguments import parse_args
from dataset.infer_dataset import InferDataset4Exp
from dataset.data_collator import ExpCollator
from models.LLM4Exp import LLM4Exp
from peft import LoraConfig, TaskType, get_peft_model

logger = get_logger(__name__)

def calculate_rank_metrics(preds, labels, topk=5):
    rel = [1000, 100, 10, 1, 0] #[100, 50, 20, 5, 0] #
    # idcg = sum([(topk-i)/math.log2(i+2) for i in range(topk)])
    idcg = rel[0]/math.log2(2) + rel[1]/math.log2(3) + rel[2]/math.log2(4) + rel[3]/math.log2(5) + rel[4]/math.log2(6)
    ndcgs = []
    for pred, label in zip(preds, labels):
        pred = pred[:topk]
        dcg = 0.0
        visited = set()
        for i, y in enumerate(label):
            for j, p in enumerate(pred):
                if y==p and j not in visited:
                    dcg += rel[i] / math.log2(j+2)
                    visited.add(j)
                    break
        ndcgs.append(dcg/idcg)
    return ndcgs

def calculate_rec_metrics(preds, labels, topk=5):
    # calculate hit, ndcg, one pos
    hits, ndcgs = [], []
    idcg = 1.0 / math.log2(2)
    for pred, label in zip(preds, labels):
        pred = pred[:topk]
        hit = 0.0
        dcg = 0.0
        for i, item in enumerate(pred):
            if item==label:
                hit = 1.0
                dcg = 1.0 / math.log2(i+2)
                break
        ndcgs.append(dcg/idcg)
        hits.append(hit)
    return hits, ndcgs

def main():
    args = parse_args()
    accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the inference seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    accelerator.wait_for_everyone()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.llm_model_name_or_path,
        cache_dir=args.cache_dir,
        model_max_length=args.llm_max_length - args.max_new_tokens,
        use_fast=not args.use_slow_tokenizer,
        padding_side='left',
    )
    tokenizer.add_special_tokens({"additional_special_tokens": ["<user>", "<item>"]})
    tokenizer.pad_token = tokenizer.unk_token
    
    llm_config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.llm_model_name_or_path,
        cache_dir=args.cache_dir,
    )
    llm_config.pad_token_id = tokenizer.pad_token_id
    model = LLM4Exp(args, llm_config)
    model.resize_token_embeddings(len(tokenizer))

    if args.use_lora:
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, 
                                    lora_dropout=0.1, fan_in_fan_out=False, modules_to_save=["lm_head", "embed_tokens"]) #["lm_head", "wte"])
        model.llm_model = get_peft_model(model.llm_model, peft_config)
        logger.info(f"model.llm_model: {model.llm_model.print_trainable_parameters()}")

    if args.llm_model_ckpt_path!=None and args.task_type!="none":
        logger.info(f"Loading model from {args.llm_model_ckpt_path}")
        state_dict = torch.load(args.llm_model_ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)

    model = model.half()
    # logger.info(f"model mixed precision: {model.llm_model.base_model.model.base_model.layers[0].self_attn.k_proj.weight.dtype}")
    # logger.info(f"model mixed precision: {model.item_connector[0].weight.dtype}")
    # logger.info(f"model mixed precision: {model.rec_model.item_embedding.weight.dtype}")

    # Download configuration from huggingface.co and cache.
    # generation_config = GenerationConfig.from_pretrained(args.llm_model_ckpt_path)

    # If you'd like to try a minor variation to an existing configuration, you can also pass generation
    # arguments to `.from_pretrained()`. Be mindful that typos and unused arguments will be ignored
    generation_config, unused_kwargs = GenerationConfig.from_pretrained(
        args.llm_model_name_or_path, return_unused_kwargs=True, return_dict_in_generate=True, 
        output_scores=True, #max_length=args.llm_max_length,
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        do_sample=args.do_sample,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        bad_words_ids=[[tokenizer.pad_token_id], [tokenizer.bos_token_id], [tokenizer.unk_token_id], 
                        [tokenizer.additional_special_tokens_ids[0]], [tokenizer.additional_special_tokens_ids[1]]],
    )
    
    infer_dataset = InferDataset4Exp(args, tokenizer, split="valid")

    # DataLoaders creation:
    infer_dataloader = DataLoader(
        infer_dataset, collate_fn=ExpCollator, batch_size=args.per_device_eval_batch_size, num_workers=args.preprocessing_num_workers, pin_memory=True
    )
    logger.info(f"args: {args}")
    # logger.info(f"model: {model}")

    # Prepare everything with our `accelerator`.
    model,  infer_dataloader = accelerator.prepare(model, infer_dataloader)

    # Infer!
    total_batch_size = (
        args.per_device_eval_batch_size * accelerator.num_processes
    )

    logger.info("***** Running inference *****")
    logger.info(f"  Num examples = {len(infer_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")
    logger.info(f"  Total infer batch size (w. parallel, distributed) = {total_batch_size}")
    
    model.eval()
    model = accelerator.unwrap_model(model)

    if args.inference_mode=="case study":
        # all_sequences = []
        # all_answers = []
        # all_scores = []
        column_names = ['label', 'history', 'target item', 'answer']  
        output_csv = pd.DataFrame(columns=column_names)
        for step, batch in tqdm(enumerate(infer_dataloader), desc="Infering...", total=len(infer_dataloader), ncols=80, disable=not accelerator.is_local_main_process):
            answers = batch.pop("answers")
            with torch.no_grad():
                outputs = model.generate(**batch, generation_config=generation_config)
            generated_tokens = outputs.sequences #[:, tokenizer.model_max_length:]
            sentences = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
            for i, (sent, answer) in enumerate(zip(sentences, answers)):
                sent = sent.split(" ASSISTANT:")[1].split("</s>")[0].strip()
                output_csv = output_csv._append({'label': answer[0], 'history': answer[1], 'target item': answer[2], 'answer': sent}, ignore_index=True)
            
            if step % 5 == 0:
                output_csv.to_csv(args.output_dir, index=False)

        output_csv.to_csv(args.output_dir, index=False)

            # all_sequences.extend(sentences)
            # all_answers.extend(answers)

        # with open(os.path.join(args.output_dir, f"case_study_{args.task_type}.txt"), "w") as f:
        #     for i, (sent, answer) in enumerate(zip(all_sequences, all_answers)):
        #         sent = "USER:" + sent.split("USER:")[1]
        #         f.write(f"case {i}: generated: {sent} %%%%%% ground_truth: {answer}\n\n")
        #     pad_generated_tokens = accelerator.pad_across_processes(generated_tokens, dim=-1, pad_index=tokenizer.pad_token_id, pad_first=True)
        #     sequences = accelerator.gather(pad_generated_tokens) # (batch_size*num_return_sequences, max_new_tokens)
        #     if args.num_beams > 1:
        #         scores = accelerator.gather(outputs.sequences_scores) # (batch_size*num_return_sequences, )
        #     else:
        #         transition_scores = model.compute_transition_scores(
        #                 sequences=outputs.sequences, scores=outputs.scores, normalize_logits=True
        #             )
        #         pad_scores = accelerator.pad_across_processes(transition_scores, dim=-1, pad_index=-10000, pad_first=True)
        #         scores = accelerator.gather(pad_scores) # (batch_size*num_return_sequences, max_new_tokens)
        #     all_sequences.append(sequences.cpu())
        #     all_scores.append(scores.cpu())
            
        # seq_count=0
        # accu_count=0
        # for scores, sequences in zip(all_scores, all_sequences):
        #     sentences = tokenizer.batch_decode(sequences, skip_special_tokens=False)
        #     for sent, score in zip(sentences, scores):
        #         logger.info(f"seq_id: {seq_count}-{accu_count} %%%%%%% sentence: {sent} %%%%%%% score: {score}")
        #         accu_count += 1
        #         if accu_count % args.num_return_sequences == 0:
        #             seq_count += 1
        #             accu_count = 0
    elif args.inference_mode=="uid2hist": 
        assert args.num_beams==args.num_return_sequences==1
        all_results = []
        for step, batch in tqdm(enumerate(infer_dataloader), desc="Infering...", total=len(infer_dataloader), ncols=80, disable=not accelerator.is_local_main_process):
            batch_results = [] 
            answers = batch.pop("answers")
            with torch.no_grad():
                outputs = model.generate(**batch, generation_config=generation_config)
            generated_tokens = outputs.sequences
            sentences = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
            for answer, sentence in zip(answers, sentences):
                pred = sentence.split(" ASSISTANT:")[1]
                pred = pred.split("</s>")[0]
                coverage = 0.0
                for item in pred.split(", ")[:len(answer)]:
                    if item.strip().lower() in answer:
                        coverage += 1/len(answer)
                batch_results.append(coverage)
            
            all_batch_results = accelerator.gather_for_metrics(torch.tensor(batch_results, device=accelerator.device)).cpu()
            all_results.append(all_batch_results)
        all_results = torch.cat(all_results)
        eval_converage = torch.mean(all_results)
        logger.info(f"task: {args.inference_mode} ; eval_converage: {eval_converage}")
    elif args.inference_mode=="uid2next":
        topk=5
        assert args.num_beams==args.num_return_sequences==topk
        all_hits = []
        all_ndcgs = []
        for step, batch in tqdm(enumerate(infer_dataloader), desc="Infering...", total=len(infer_dataloader), ncols=80, disable=not accelerator.is_local_main_process):
            batch_topk_preds = []
            batch_ground_truths = []
            answers = batch.pop("answers")
            with torch.no_grad():
                outputs = model.generate(**batch, generation_config=generation_config)
            generated_tokens = outputs.sequences
            sentences = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
            for i, answer in enumerate(answers):
                topk_pred=[]
                sents = sentences[i*args.num_return_sequences:(i+1)*args.num_return_sequences]
                for sent in sents:
                    sent = sent.split(" ASSISTANT:")[1]
                    sent = sent.split("</s>")[0]
                    sent = sent.strip().lower()
                    topk_pred.append(sent)
                batch_topk_preds.append(topk_pred)
                batch_ground_truths.append(answer)
            hits, ndcgs = calculate_rec_metrics(batch_topk_preds, batch_ground_truths, topk)
            all_batch_hits = accelerator.gather_for_metrics(torch.tensor(hits, device=accelerator.device)).cpu()
            all_batch_ndcgs = accelerator.gather_for_metrics(torch.tensor(ndcgs, device=accelerator.device)).cpu()
            all_hits.append(all_batch_hits)
            all_ndcgs.append(all_batch_ndcgs)

        all_hits = torch.cat(all_hits)
        all_ndcgs = torch.cat(all_ndcgs)
        eval_hits = torch.mean(all_hits)
        eval_ndcgs = torch.mean(all_ndcgs)
        logger.info(f"task: {args.inference_mode}; hit@{topk}: {eval_hits} ; ndcg@{topk}: {eval_ndcgs}")
        
    elif args.inference_mode=="uidiid2rank":
        assert args.num_beams==args.num_return_sequences==1
        all_ndcgs = []
        for step, batch in tqdm(enumerate(infer_dataloader), desc="Infering...", total=len(infer_dataloader), ncols=80, disable=not accelerator.is_local_main_process):
            batch_topk_preds = []
            batch_ground_truths = []
            answers = batch.pop("answers")
            with torch.no_grad():
                outputs = model.generate(**batch, generation_config=generation_config)
            generated_tokens = outputs.sequences
            sentences = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
            for answer, sentence in zip(answers, sentences):
                pred = sentence.split(" ASSISTANT:")[1]
                pred = pred.split("</s>")[0]
                batch_topk_preds.append([item.strip().lower() for item in pred.split(", ")])
                batch_ground_truths.append(answer)
            ndcgs = calculate_rank_metrics(batch_topk_preds, batch_ground_truths, 5)
            all_batch_ndcgs = accelerator.gather_for_metrics(torch.tensor(ndcgs, device=accelerator.device)).cpu()
            all_ndcgs.append(all_batch_ndcgs)
        all_ndcgs = torch.cat(all_ndcgs)
        eval_ndcgs = torch.mean(all_ndcgs)
        logger.info(f"task: {args.inference_mode}; ndcg@5: {eval_ndcgs}")

    elif args.inference_mode=="uidiid2binary": #3869: _Yes,  #1939: _No
        all_results = []
        for step, batch in tqdm(enumerate(infer_dataloader), desc="Infering...", total=len(infer_dataloader), ncols=80, disable=not accelerator.is_local_main_process):
            batch_results = [] 
            answers = batch.pop("answers")
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            for logit, answer in zip(logits, answers):
                if answer=="yes":
                    batch_results.append(int(logit[-1][3869]>=logit[-1][1939]))
                elif answer=="no":
                    batch_results.append(int(logit[-1][3869]<logit[-1][1939]))
                else:
                    raise NotImplementedError
            all_batch_results = accelerator.gather_for_metrics(torch.tensor(batch_results, device=accelerator.device, dtype=torch.float32)).cpu()
            all_results.append(all_batch_results)
        
        all_results = torch.cat(all_results)
        eval_accu = torch.mean(all_results)
        logger.info(f"task: {args.inference_mode} ; eval accuracy: {eval_accu}")       
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()
