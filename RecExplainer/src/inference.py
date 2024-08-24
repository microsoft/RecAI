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
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Optional
from tqdm.auto import tqdm
from transformers import (
    HfArgumentParser,
    AutoConfig,
    AutoTokenizer,
    GenerationConfig
)

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from dataset.infer_dataset import InferDataset4Exp
from dataset.data_collator import ExpCollator
from models.LLM4Exp import MistralForExp, LlamaForExp, Phi3ForExp
# from peft import LoraConfig, TaskType, get_peft_model

logger = get_logger(__name__)


@dataclass
class ScriptArguments:
    preprocessing_num_workers: Optional[int] = field(default=0, metadata={"help": "The number of processes to use for the preprocessing."})
    seed: Optional[int] = field(default=2024, metadata={"help": "A seed for reproducible training."})
    output_dir: Optional[str] = field(default="", metadata={"help": "the output directory"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "the batch size for evaluation"})
    use_fast_tokenizer: bool = field(default=True, metadata={"help": "Whether to use a fast tokenizer (backed by HuggingFace tokenizers library) or a slow one (backed by tokenizers library)"})
    model_name_or_path: Optional[str] = field(default="", metadata={"help": "the model checkpoint for inference"})
    validation_file: Optional[str] = field(default="", metadata={"help": "the validation file"})
    sequential_file: Optional[str] = field(default="", metadata={"help": "the sequential file"})
    cache_dir: Optional[str] = field(default="", metadata={"help": "the cache directory"})
    max_hist_len: Optional[int] = field(default=10, metadata={"help": "the maximum length of the user history"})
    model_max_length: Optional[int] = field(default=512, metadata={"help": "the maximum token length of the model"})
    task_type: Optional[str] = field(default="both", metadata={"help": "the task type, options are intention, behaviour, both, none"})
    template_name: Optional[str] = field(default="mistral", metadata={"help": "The template name to use. Options are: mistral, vicuna, llama-2, llama-3, phi3"})
    max_example_num: Optional[int] = field(default=10000000, metadata={"help": "the maximum number of examples"})

    inference_mode: Optional[str] = field(default="case study", metadata={"help": "what task to infer. Options are: case study, uid2hist, uid2next, uidiid2rank, uidiid2binary, iid2title"})
    metadata_file: Optional[str] = field(default="", metadata={"help": "Path to the metainfo file."})
    test_top_file: Optional[str] = field(default="", metadata={"help": "Path to the test top file."})
    do_sample: Optional[bool] = field(default=False, metadata={"help": "Whether or not to use sampling ; use greedy decoding otherwise."})
    max_new_tokens: Optional[int] = field(default=20, metadata={"help": "The max number of new tokens to generate."})
    min_new_tokens: Optional[int] = field(default=1, metadata={"help": "The min number of new tokens to generate."})
    num_beams: Optional[int] = field(default=1, metadata={"help": "The number of beams to use for beam search."})
    num_return_sequences: Optional[int] = field(default=1, metadata={"help": "The number of beams to use for beam search."})
    repetition_penalty: Optional[float] = field(default=1.0, metadata={"help": "The repetition penalty for the generation."})
    temperature: Optional[float] = field(default=1.0, metadata={"help": "the temperature for the generation"})

    torch_dtype: Optional[str] = field(default=None, metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    rec_model_type: Optional[str] = field(
        default="SASRec",
        metadata={
            "help": ("The type of the target recommender model."),
            "choices": ["SASRec", "MF"],
        },
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={"help": ("The attention implementation to use. Options are: 'eager', 'flash_attention_2'")},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

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
    parser = HfArgumentParser(ScriptArguments)
    args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
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
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        model_max_length=args.model_max_length - args.max_new_tokens,
        use_fast=args.use_fast_tokenizer,
        padding_side='left',
    )

    if tokenizer.pad_token is None:
        if args.template_name == "llama-3":
            tokenizer.pad_token_id = 128001
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("Tokenizer should have a pad token or an unk token or an eos token.")

    config_kwargs = {
        "cache_dir": args.cache_dir,
        "revision": args.model_revision,
        "token": args.token,
        "trust_remote_code": args.trust_remote_code,
    }
    config = AutoConfig.from_pretrained(args.model_name_or_path, **config_kwargs)
    config.pad_token_id = tokenizer.pad_token_id
    config.task_type = args.task_type
    config.rec_model_type = args.rec_model_type

    torch_dtype = (
        args.torch_dtype
        if args.torch_dtype in ["auto", None]
        else getattr(torch, args.torch_dtype)
    )

    if args.template_name == "mistral":
        classForExp = MistralForExp
    elif args.template_name in ["vicuna", "llama-2", "llama-3"]:
        classForExp = LlamaForExp
    elif args.template_name == "phi3":
        classForExp = Phi3ForExp
    else:
        raise ValueError("Model architecture not supported")
    
    model = classForExp.from_pretrained(
        args.model_name_or_path, 
        config=config,
        torch_dtype=torch_dtype,
        cache_dir=args.cache_dir,
        revision=args.model_revision,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        attn_implementation=args.attn_implementation,
    )
    logger.info(model)

    # Download configuration from huggingface.co and cache.
    # generation_config = GenerationConfig.from_pretrained(args.llm_model_ckpt_path)

    # If you'd like to try a minor variation to an existing configuration, you can also pass generation
    # arguments to `.from_pretrained()`. Be mindful that typos and unused arguments will be ignored
    if args.task_type=="none":
        bad_words_ids = [[tokenizer.pad_token_id], [tokenizer.bos_token_id]]
    else:
        bad_words_ids = [[tokenizer.pad_token_id], [tokenizer.bos_token_id], [tokenizer.additional_special_tokens_ids[0]], [tokenizer.additional_special_tokens_ids[1]]]
    if tokenizer.unk_token_id:
        bad_words_ids.append([tokenizer.unk_token_id])
    eos_token_id = [tokenizer.eos_token_id]
    if args.template_name == "llama-3":
        eos_token_id=[128009, 128001]
        if args.task_type=="none":
            bad_words_ids = [[i] for i in range(128000, 128256) if i not in [128009, 128001]]
        else:
            bad_words_ids = [[i] for i in range(128000, 128258) if i not in [128009, 128001]]
    elif args.template_name == "phi3":
        eos_token_id = [32007] # <|end|>
    generation_config, unused_kwargs = GenerationConfig.from_pretrained(
        args.model_name_or_path, return_unused_kwargs=True, return_dict_in_generate=True, 
        output_scores=True, #max_length=args.llm_max_length,
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        do_sample=args.do_sample,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        bad_words_ids=bad_words_ids,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_token_id,
    )
    
    infer_dataset = InferDataset4Exp(args, tokenizer)

    # DataLoaders creation:
    infer_dataloader = DataLoader(
        infer_dataset, collate_fn=ExpCollator, batch_size=args.per_device_eval_batch_size, num_workers=args.preprocessing_num_workers, pin_memory=True
    )
    logger.info(f"args: {args}")

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

    if args.template_name == "mistral":
        seps = [" [/INST]", ["</s>"]] # for vicuna-v1.3: [" ASSISTANT:", "</s>"]; for mistral-7b: [" [/INST]", "</s>"]
        Yes_token_index, No_token_index = 5592, 1770 #for vicuna-v1.3: [3869: _Yes, #1939: _No]; for mistral-7b: [5592: _Yes, 1770: _No]
        split_token = "; "
    elif args.template_name == "vicuna":
        seps = [" ASSISTANT:", ["</s>"]]
        Yes_token_index, No_token_index = 3869, 1939
        split_token = "; "
    elif args.template_name == "llama-2":
        seps = [" [/INST]", ["</s>"]] 
        Yes_token_index, No_token_index = 3869, 1939 #[3869: _Yes, #1939: _No]
        split_token = "; "
    elif args.template_name == "llama-3":
        seps = ["<|start_header_id|>assistant<|end_header_id|>\n\n", ["<|eot_id|>", "<|end_of_text|>"]]
        Yes_token_index, No_token_index = 9642, 2822 # for llama-3: [9642: Yes, 2822: No]
        split_token = "; " #split_token = ", "
    elif args.template_name == "phi3":
        seps = ["<|assistant|>\n", ["<|end|>"]]
        Yes_token_index, No_token_index = 3869, 1939
        split_token = "; "
        raise ValueError("Model architecture not supported")
    else:
        raise ValueError("Model architecture not supported")

    if args.inference_mode=="case study":
        column_names = ['label', 'history', 'target item', 'answer']  
        output_csv = pd.DataFrame(columns=column_names)
        for step, batch in tqdm(enumerate(infer_dataloader), desc="Infering...", total=len(infer_dataloader), ncols=80, disable=not accelerator.is_local_main_process):
            answers = batch.pop("answers")
            with torch.no_grad():
                outputs = model.generate(**batch, generation_config=generation_config)
            generated_tokens = outputs.sequences #[:, tokenizer.model_max_length:]
            sentences = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
            for i, (sent, answer) in enumerate(zip(sentences, answers)):
                sent = sent.split(seps[0])[1]
                for s in seps[1]:
                    sent = sent.split(s)[0].strip()
                output_csv = output_csv._append({'label': answer[0], 'history': answer[1], 'target item': answer[2], 'answer': sent}, ignore_index=True)
            
            if step % 5 == 0:
                output_csv.to_csv(args.output_dir, index=False)

        output_csv.to_csv(args.output_dir, index=False)

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
                pred = sentence.split(seps[0])[1]
                for s in seps[1]:
                    pred = pred.split(s)[0]
                coverage = 0.0
                for item in pred.split(split_token)[:len(answer)]:
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
                    sent = sent.split(seps[0])[1]
                    for s in seps[1]:
                        sent = sent.split(s)[0]
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
                pred = sentence.split(seps[0])[1]
                for s in seps[1]:
                    pred = pred.split(s)[0]
                batch_topk_preds.append([item.strip().lower() for item in pred.split(split_token)])
                batch_ground_truths.append(answer)
            ndcgs = calculate_rank_metrics(batch_topk_preds, batch_ground_truths, 5)
            all_batch_ndcgs = accelerator.gather_for_metrics(torch.tensor(ndcgs, device=accelerator.device)).cpu()
            all_ndcgs.append(all_batch_ndcgs)
        all_ndcgs = torch.cat(all_ndcgs)
        eval_ndcgs = torch.mean(all_ndcgs)
        logger.info(f"task: {args.inference_mode}; ndcg@5: {eval_ndcgs}")

    elif args.inference_mode=="uidiid2binary":
        all_results = []
        for step, batch in tqdm(enumerate(infer_dataloader), desc="Infering...", total=len(infer_dataloader), ncols=80, disable=not accelerator.is_local_main_process):
            batch_results = [] 
            answers = batch.pop("answers")
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            for logit, answer in zip(logits, answers):
                if answer=="yes":
                    batch_results.append(int(logit[-1][Yes_token_index]>=logit[-1][No_token_index]))
                elif answer=="no":
                    batch_results.append(int(logit[-1][Yes_token_index]<logit[-1][No_token_index]))
                else:
                    raise NotImplementedError
            all_batch_results = accelerator.gather_for_metrics(torch.tensor(batch_results, device=accelerator.device, dtype=torch.float32)).cpu()
            all_results.append(all_batch_results)
        
        all_results = torch.cat(all_results)
        eval_accu = torch.mean(all_results)
        logger.info(f"task: {args.inference_mode} ; eval accuracy: {eval_accu}")     
    elif args.inference_mode=="iid2title":
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
                pred = sentence.split(seps[0])[1]
                for s in seps[1]:
                    pred = pred.split(s)[0]
                acc=0.0
                if pred.strip().lower() == answer:
                    acc = 1.0
                batch_results.append(acc)
            
            all_batch_results = accelerator.gather_for_metrics(torch.tensor(batch_results, device=accelerator.device)).cpu()
            all_results.append(all_batch_results)
        all_results = torch.cat(all_results)
        eval_acc = torch.mean(all_results)
        logger.info(f"task: {args.inference_mode} ; eval_acc: {eval_acc}")
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()
