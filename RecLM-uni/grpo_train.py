import argparse
import math
import os
import random
import re
import torch

from typing import List, Optional

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from GRPO.rl_dataset import build_rl_samples

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None
    get_peft_model = None


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training on RecLM data")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset folder, e.g. data/steam/")
    parser.add_argument("--model_path", type=str, required=True, help="HF model directory produced by SFT merge.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store GRPO checkpoints.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=10, help="Recommendation list length.")
    parser.add_argument("--max_item_length", type=int, default=20, help="History window length.")
    parser.add_argument("--rl_max_samples", type=int, default=20000, help="Maximum RL samples to build.")
    parser.add_argument("--eval_ratio", type=float, default=0.05, help="Portion of samples for evaluation.")
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_steps", type=int, default=None, help="Run evaluation every N steps. Defaults to logging_steps.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.04, help="KL coefficient.")
    parser.add_argument("--max_completion_length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--reward_type", choices=["rule", "ranking"], default="ranking")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--use_control_symbol", action="store_true")
    parser.add_argument("--idx", action="store_true", help="Prefix recommendation list with indices.")
    parser.add_argument("--item_index", type=str, default="rq_token_seq", help="Meta field to use as item surface text.")
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma separated module names to apply LoRA on.",
    )
    return parser.parse_args()


def normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip().lower()


def extract_generated_items(text: str, use_control_symbol: bool) -> List[str]:
    if use_control_symbol:
        matches = re.findall(r"<SOI>\s*(.*?)\s*<EOI>", text, flags=re.MULTILINE)
        if matches:
            return [normalize_title(m) for m in matches]
    # fallback: split by newline or numbering
    lines = []
    for raw_line in text.splitlines():
        clean = raw_line.strip()
        if not clean:
            continue
        clean = re.sub(r"^\d+[\.\)]\s*", "", clean)
        lines.append(normalize_title(clean))
    return lines


def make_rule_reward(use_control_symbol: bool):
    def reward_fn(prompts, completions, target_title, **_):
        rewards = []
        for comp, target in zip(completions, target_title):
            pred_items = extract_generated_items(comp, use_control_symbol)
            target_norm = normalize_title(target)
            score = 1.0 if target_norm in pred_items else 0.0
            rewards.append(score)
        return rewards

    return reward_fn


def make_ndcg_reward(use_control_symbol: bool):
    def reward_fn(prompts, completions, target_title, **_):
        rewards = []
        for comp, target in zip(completions, target_title):
            pred_items = extract_generated_items(comp, use_control_symbol)
            target_norm = normalize_title(target)
            if target_norm in pred_items:
                rank = pred_items.index(target_norm)
                rewards.append(1.0 / math.log2(rank + 2))
            else:
                rewards.append(0.0)
        return rewards

    return reward_fn


def main():
    args = parse_args()
    random.seed(args.seed)

    samples = build_rl_samples(
        data_path=args.data_path,
        item_index_field=args.item_index,
        max_samples=args.rl_max_samples,
        seed=args.seed,
        topk=args.topk,
        max_item_length=args.max_item_length,
        use_control_symbol=args.use_control_symbol,
        use_idx=args.idx,
    )

    if len(samples) < 10:
        raise RuntimeError("RL sample size too small, please check data or increase rl_max_samples.")

    eval_size = max(1, int(len(samples) * args.eval_ratio))
    train_samples = samples[:-eval_size]
    eval_samples = samples[-eval_size:]

    def sample_to_dict(sample):
        return {
            "prompt": sample.prompt,
            "reference_response": sample.reference_response,
            "target_title": sample.target_title,
            "target_item": sample.target_item,
            "history": sample.history_text,
        }

    train_dataset = Dataset.from_list([sample_to_dict(s) for s in train_samples])
    eval_dataset = Dataset.from_list([sample_to_dict(s) for s in eval_samples])

    os.makedirs(args.output_dir, exist_ok=True)

    eval_steps = args.eval_steps if args.eval_steps and args.eval_steps > 0 else max(1, args.logging_steps)

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        beta=args.beta,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="epoch",
        temperature=args.temperature,
        bf16=True,
        save_total_limit=args.save_total_limit,
        report_to=[],
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)

    model_init: Optional[object]
    if args.use_lora:
        if LoraConfig is None or get_peft_model is None:
            raise ImportError("peft is required for LoRA training but was not found.")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16 if args.bf16 else None,
        )
        target_modules = [
            module.strip()
            for module in args.lora_target_modules.split(",")
            if module.strip()
        ]
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules or None,
        )
        model_init = get_peft_model(base_model, lora_config)
        try:
            model_init.print_trainable_parameters()
        except AttributeError:
            print("Note: print_trainable_parameters() method not available on this PEFT model type")
    else:
        model_init = args.model_path

    reward_funcs = [make_rule_reward(args.use_control_symbol)]
    if args.reward_type == "ranking":
        reward_funcs.append(make_ndcg_reward(args.use_control_symbol))

    trainer = GRPOTrainer(
        model=model_init,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
