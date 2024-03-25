#!/usr/bin/env python
# coding=utf-8
"""
The following code is modified from
https://github.com/huggingface/accelerate/blob/main/examples/by_feature/deepspeed_with_config_support.py
"""

import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path

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
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import get_full_repo_name

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DummyOptim, DummyScheduler, set_seed, DistributedDataParallelKwargs

from arguments import parse_args
from dataset.train_dataset import TrainingDataset4Exp
from dataset.data_collator import ExpCollator
from models.LLM4Exp import LLM4Exp
from peft import LoraConfig, TaskType, get_peft_model

logger = get_logger(__name__)

# New Code #
def evaluate(args, model, eval_dataloader, accelerator, eval_dataset):
    eval_4_loss = 0.0 # uid2next, uidiid2rank, uidiid2binary, uid2summary
    eval_4_count = 0
    eval_sharegpt_loss = 0.0
    eval_sharegpt_count = 0
    eval_uid2hist_loss = 0.0
    eval_uid2hist_count = 0
    model.eval()
    losses = []
    for step, batch in tqdm(enumerate(eval_dataloader), desc="Evaluating...", total=len(eval_dataloader), ncols=80):
        data_type = batch.pop('type')
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        if data_type[0] in ['uid2next', 'uidiid2rank', 'uidiid2binary', 'uid2summary']:
            eval_4_loss += loss.item()
            eval_4_count += 1
        elif data_type[0] == 'sharegpt':
            eval_sharegpt_loss += loss.item()
            eval_sharegpt_count += 1
        elif data_type[0] == 'uid2hist':
            eval_uid2hist_loss += loss.item()
            eval_uid2hist_count += 1

        losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))
    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")

    eval_4_loss = accelerator.reduce(torch.tensor(eval_4_loss, device=accelerator.device, dtype=torch.float32)).item()
    eval_4_count = accelerator.reduce(torch.tensor(eval_4_count, device=accelerator.device, dtype=torch.long)).item()
    eval_sharegpt_loss = accelerator.reduce(torch.tensor(eval_sharegpt_loss, device=accelerator.device, dtype=torch.float32)).item()
    eval_sharegpt_count = accelerator.reduce(torch.tensor(eval_sharegpt_count, device=accelerator.device, dtype=torch.long)).item()
    eval_uid2hist_loss = accelerator.reduce(torch.tensor(eval_uid2hist_loss, device=accelerator.device, dtype=torch.float32)).item()
    eval_uid2hist_count = accelerator.reduce(torch.tensor(eval_uid2hist_count, device=accelerator.device, dtype=torch.long)).item()

    try:
        eval_4_loss = eval_4_loss/eval_4_count
        eval_4_perplexity = math.exp(eval_4_loss)
    except:
        eval_4_loss = float("inf")
        eval_4_perplexity = float("inf")

    try:
        eval_sharegpt_loss = eval_sharegpt_loss/eval_sharegpt_count
        eval_sharegpt_perplexity = math.exp(eval_sharegpt_loss)
    except:
        eval_sharegpt_loss = float("inf")
        eval_sharegpt_perplexity = float("inf")
    
    try:
        eval_uid2hist_loss = eval_uid2hist_loss/eval_uid2hist_count
        eval_uid2hist_perplexity = math.exp(eval_uid2hist_loss)
    except:
        eval_uid2hist_loss = float("inf")
        eval_uid2hist_perplexity = float("inf")
    
    logger.info(f"count:{eval_4_count}, {eval_sharegpt_count}, {eval_uid2hist_count}")

    return perplexity, eval_loss, eval_4_loss, eval_4_perplexity, eval_sharegpt_loss, eval_sharegpt_perplexity, eval_uid2hist_loss, eval_uid2hist_perplexity #loss per sample


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment

    # when using DeepSpeed, the `gradient_accumulation_steps` is properly set from the DeepSpeed plugin/config
    # or from `accelerate launch` via `--gradient_accumulation_steps`  else
    # defaulting to the passed `args.gradient_accumulation_steps`
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)
    accelerator = (
        Accelerator(
            log_with=args.report_to,
            project_dir=args.output_dir,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            kwargs_handlers=[kwargs]
        )
        if args.with_tracking
        else Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, kwargs_handlers=[kwargs])
    )

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

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.llm_model_name_or_path,
        cache_dir=args.cache_dir,
        model_max_length=args.llm_max_length,
        use_fast=not args.use_slow_tokenizer,
    )
    # if args.task_type!="behaviour":
    tokenizer.add_special_tokens({"additional_special_tokens": ["<user>", "<item>"]})
    tokenizer.pad_token = tokenizer.unk_token
    
    llm_config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.llm_model_name_or_path,
        cache_dir=args.cache_dir,
    )
    llm_config.pad_token_id = tokenizer.pad_token_id
    model = LLM4Exp(args, llm_config)
    # if args.task_type!="behaviour":
    model.resize_token_embeddings(len(tokenizer))

    if args.use_lora:
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, 
                                    lora_dropout=0.1, fan_in_fan_out=False, modules_to_save=["lm_head", "embed_tokens"]) #["lm_head", "wte"])
        model.llm_model = get_peft_model(model.llm_model, peft_config)
        logger.info(f"model.llm_model: {model.llm_model.print_trainable_parameters()}")

    train_dataset = TrainingDataset4Exp(args, tokenizer, split="train")
    eval_dataset = TrainingDataset4Exp(args, tokenizer, split="valid")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=ExpCollator, batch_size=args.per_device_train_batch_size, num_workers=args.preprocessing_num_workers, pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=ExpCollator, batch_size=args.per_device_eval_batch_size, num_workers=args.preprocessing_num_workers, pin_memory=True
    )
    logger.info(f"args: {args}")
    logger.info(f"model: {model}")
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    # New Code #
    # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
    overrode_max_train_steps = False
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # New Code #
    # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer, total_num_steps=args.max_train_steps, warmup_num_steps=args.num_warmup_steps
        )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    # if checkpointing_steps is not None and checkpointing_steps.isdigit():
    #     checkpointing_steps = int(checkpointing_steps)

    eval_steps = args.eval_steps
    log_steps = args.log_steps

    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {accelerator.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process, ncols=80)
    completed_steps = 0
    starting_epoch = 0
    best_metric = None
    best_metric_checkpoint = None
    losses = []
    step_losses = []
    grad_accu_steps = 1 # log for every gradient accumulation steps

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
        path = os.path.basename(args.resume_from_checkpoint)
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            completed_steps = int(training_difference.replace("step_", ""))
            starting_epoch = completed_steps // num_update_steps_per_epoch
            resume_step = completed_steps - starting_epoch * num_update_steps_per_epoch
        logger.info(f"starting epoch {starting_epoch}: completed_steps: {completed_steps} resume_step: {resume_step}")
        progress_bar.update(completed_steps)

    eval_epoch_perplexity, eval_epoch_loss, eval_4_loss, eval_4_perplexity, eval_sharegpt_loss, eval_sharegpt_perplexity, eval_uid2hist_loss, eval_uid2hist_perplexity = evaluate(args, model, eval_dataloader, accelerator, eval_dataset)
    logger.info(f"Evaluate before training... epoch {starting_epoch}: step: {completed_steps} eval_epoch_perplexity: {eval_epoch_perplexity} eval_epoch_loss: {eval_epoch_loss}")
    logger.info(f"Evaluate before training... epoch {starting_epoch}: step: {completed_steps} eval_4_perplexity: {eval_4_perplexity} eval_4_loss: {eval_4_loss}")
    logger.info(f"Evaluate before training... epoch {starting_epoch}: step: {completed_steps} eval_sharegpt_perplexity: {eval_sharegpt_perplexity} eval_sharegpt_loss: {eval_sharegpt_loss}")
    logger.info(f"Evaluate before training... epoch {starting_epoch}: step: {completed_steps} eval_uid2hist_perplexity: {eval_uid2hist_perplexity} eval_uid2hist_loss: {eval_uid2hist_loss}")
    best_metric = eval_epoch_perplexity

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        cur_train_dataloader = train_dataloader
        # skip new `skip_first_batches` to skip the batches when resuming from ckpt
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            cur_train_dataloader = accelerator.skip_first_batches(train_dataloader, num_batches=resume_step*accelerator.gradient_accumulation_steps)
            logger.info(f"skip {resume_step*accelerator.gradient_accumulation_steps} batches, cur_train_dataloader: {len(cur_train_dataloader)}")
        
        train_4_loss = 0.0 # uid2next, uidiid2rank, uidiid2binary, uid2summary
        train_4_count = 0
        train_sharegpt_loss = 0.0
        train_sharegpt_count = 0
        train_uid2hist_loss = 0.0
        train_uid2hist_count = 0
        for step, batch in enumerate(cur_train_dataloader):
            # In particular, DeepSpeed handles `gradient_accumulation` via `DeepSpeedEngine`.
            # Below, we use `accelerator.accumulate` if the user
            # wants to switch to other approaches such as plain DDP, PyTorch FSDP ...
            # This avoids having to change any code as things are all handled across different distributed setups.
            data_type = batch.pop('type')
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1
                    grad_accu_steps = 0

            # We keep track of the loss at each epoch
            if data_type[0] in ['uid2next', 'uidiid2rank', 'uidiid2binary', 'uid2summary']:
                train_4_loss += loss.item()
                train_4_count += 1
            elif data_type[0] == 'sharegpt':
                train_sharegpt_loss += loss.item()
                train_sharegpt_count += 1
            elif data_type[0] == 'uid2hist':
                train_uid2hist_loss += loss.item()
                train_uid2hist_count += 1
            losses.append(accelerator.gather_for_metrics(loss.detach().clone().repeat(args.per_device_train_batch_size)).cpu())
            step_losses.append(losses[-1])
            
            if grad_accu_steps == 0:
                if completed_steps % log_steps == 0:
                    try:
                        step_loss = torch.mean(torch.cat(step_losses))
                        step_perplexity = math.exp(step_loss)
                    except OverflowError:
                        step_loss = float("inf")
                        step_perplexity = float("inf")
                    if args.with_tracking:
                        metrics = {
                            "train/step_loss": step_loss,
                            "train/step_perplexity": step_perplexity,
                            "train/learning_rate": lr_scheduler.get_last_lr()[0],
                        }
                        accelerator.log(metrics, step=completed_steps)
                    logger.info(f"epoch {epoch}: step: {completed_steps} train_step_perplexity: {step_perplexity}, train_step_loss: {step_loss}, lr: {lr_scheduler.get_last_lr()[0]}")
                    step_losses = []

                if completed_steps % eval_steps == 0:
                    eval_step_perplexity, eval_step_loss = evaluate(args, model, eval_dataloader, accelerator, eval_dataset)
                    if args.with_tracking:
                        metrics = {
                            "valid/perplexity": eval_step_perplexity,
                            "valid/loss": eval_step_loss,
                        }
                        accelerator.log(metrics, step=completed_steps)
                    logger.info(f"epoch {epoch}: step: {completed_steps} eval_step_perplexity: {eval_step_perplexity} eval_step_loss: {eval_step_loss}")
                    if best_metric is None or best_metric > eval_step_perplexity:
                        best_metric = eval_step_perplexity
                        best_metric_checkpoint = os.path.join(args.output_dir, "best_checkpoint")
                        accelerator.save_state(best_metric_checkpoint)
                        accelerator.print(f"New best metric: {best_metric} at epoch {epoch} step {completed_steps}")
                        accelerator.print(f"best_metric_checkpoint: {best_metric_checkpoint}")
                    model.train()

                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            grad_accu_steps+=1
            if completed_steps >= args.max_train_steps:
                break

        epoch_train_4_loss = accelerator.reduce(torch.tensor(train_4_loss, device=accelerator.device, dtype=torch.float32)).item()
        epoch_train_4_count = accelerator.reduce(torch.tensor(train_4_count, device=accelerator.device, dtype=torch.long)).item()
        epoch_train_sharegpt_loss = accelerator.reduce(torch.tensor(train_sharegpt_loss, device=accelerator.device, dtype=torch.float32)).item()
        epoch_train_sharegpt_count = accelerator.reduce(torch.tensor(train_sharegpt_count, device=accelerator.device, dtype=torch.long)).item()
        epoch_train_uid2hist_loss = accelerator.reduce(torch.tensor(train_uid2hist_loss, device=accelerator.device, dtype=torch.float32)).item()
        epoch_train_uid2hist_count = accelerator.reduce(torch.tensor(train_uid2hist_count, device=accelerator.device, dtype=torch.long)).item()
        try:
            epoch_train_4_loss = epoch_train_4_loss/epoch_train_4_count
            train_4_perplexity = math.exp(epoch_train_4_loss)
        except:
            epoch_train_4_loss = float("inf")
            train_4_perplexity = float("inf")

        try:
            epoch_train_sharegpt_loss = epoch_train_sharegpt_loss/epoch_train_sharegpt_count
            train_sharegpt_perplexity = math.exp(epoch_train_sharegpt_loss)
        except:
            epoch_train_sharegpt_loss = float("inf")
            train_sharegpt_perplexity = float("inf")
        
        try:
            epoch_train_uid2hist_loss = epoch_train_uid2hist_loss/epoch_train_uid2hist_count
            train_uid2hist_perplexity = math.exp(epoch_train_uid2hist_loss)
        except:
            epoch_train_uid2hist_loss = float("inf")
            train_uid2hist_perplexity = float("inf")

        eval_epoch_perplexity, eval_epoch_loss, eval_4_loss, eval_4_perplexity, eval_sharegpt_loss, eval_sharegpt_perplexity, eval_uid2hist_loss, eval_uid2hist_perplexity = evaluate(args, model, eval_dataloader, accelerator, eval_dataset)
        try:
            total_loss = torch.mean(torch.cat(losses))
            total_perplexity = math.exp(total_loss)
        except OverflowError:
            total_loss = float("inf")
            total_perplexity = float("inf")

        losses = []
        logger.info(f"epoch {epoch}: train_epoch_perplexity: {total_perplexity}, train_epoch_loss: {total_loss}, lr: {lr_scheduler.get_last_lr()[0]}")
        logger.info(f"epoch {epoch}: train_4_perplexity: {train_4_perplexity} train_4_loss: {epoch_train_4_loss}")
        logger.info(f"epoch {epoch}: train_sharegpt_perplexity: {train_sharegpt_perplexity} train_sharegpt_loss: {epoch_train_sharegpt_loss}")
        logger.info(f"epoch {epoch}: train_uid2hist_perplexity: {train_uid2hist_perplexity} train_uid2hist_loss: {epoch_train_uid2hist_loss}")
        logger.info(f"epoch {epoch}: eval_epoch_perplexity: {eval_epoch_perplexity} eval_epoch_loss: {eval_epoch_loss}")
        logger.info(f"epoch {epoch}: eval_4_perplexity: {eval_4_perplexity} eval_4_loss: {eval_4_loss}")
        logger.info(f"epoch {epoch}: eval_sharegpt_perplexity: {eval_sharegpt_perplexity} eval_sharegpt_loss: {eval_sharegpt_loss}")
        logger.info(f"epoch {epoch}: eval_uid2hist_perplexity: {eval_uid2hist_perplexity} eval_uid2hist_loss: {eval_uid2hist_loss}")
        if args.with_tracking:
            metrics = {
                "epoch": epoch,
                "train/epoch_loss": total_loss,
                "train/epoch_perplexity": total_perplexity,
                "train/learning_rate": lr_scheduler.get_last_lr()[0],
                "valid/perplexity": eval_epoch_perplexity,
                "valid/loss": eval_epoch_loss,
            }
            accelerator.log(metrics, step=completed_steps)
        # Tracks the best checkpoint and best metric
        if best_metric is None or best_metric > eval_epoch_perplexity:
            best_metric = eval_epoch_perplexity
            best_metric_checkpoint = os.path.join(args.output_dir, "best_checkpoint")
            accelerator.save_state(best_metric_checkpoint)
            accelerator.print(f"New best metric: {best_metric} at epoch {epoch} end, step {completed_steps}")
            accelerator.print(f"best_metric_checkpoint: {best_metric_checkpoint}")

        # if isinstance(checkpointing_steps, str) and checkpointing_steps == "epoch":
        accelerator.save_state(os.path.join(args.output_dir, f"step_{completed_steps}"))
      

    # New Code #
    # Loads the best checkpoint after the training is finished
    if args.load_best_model:
        accelerator.load_state(best_metric_checkpoint)

    # New Code #
    # Evaluates using the best checkpoint
    eval_perplexity, eval_loss, _, _, _, _, _, _ = evaluate(args, model, eval_dataloader, accelerator, eval_dataset)
    logger.info(f"Best model metrics: eval_perplexity: {eval_perplexity} eval_loss: {eval_loss}")
    if eval_perplexity != best_metric:
        raise AssertionError(
            f"Best metric {best_metric} does not match the metric {eval_perplexity} of the loaded best model."
        )

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        # New Code #
        # Saves the whole/unpartitioned fp16 model when in ZeRO Stage-3 to the output directory if
        # `stage3_gather_16bit_weights_on_model_save` is True in DeepSpeed Config file or
        # `zero3_save_16bit_model` is True in DeepSpeed Plugin.
        # For Zero Stages 1 and 2, models are saved as usual in the output directory.
        # The model name saved is `pytorch_model.bin`
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump({"eval_perplexity": eval_perplexity, "eval_loss": eval_loss.item()}, f)


if __name__ == "__main__":
    main()
