#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import evaluate
import torch

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from dataset.train_dataset import TrainingDataset4Exp
from dataset.data_collator import ExpCollator
from models.LLM4Exp import MistralForExp, LlamaForExp, Phi3ForExp
from peft import LoraConfig, TaskType, get_peft_model


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.36.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    lora_r: Optional[int] = field(
        default=16,
        metadata={"help": "Lora attention dimension (the 'rank')"},
    )
    lora_alpha: Optional[int] = field(
        default=32,
        metadata={"help": "The alpha parameter for Lora scaling"},
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": "The dropout probability for Lora layers"},
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={"help": ("The attention implementation to use. Options are: 'eager', 'flash_attention_2'")},
    )
    model_max_length: Optional[int] = field(
        default=None,
        metadata={"help": "The max token length of the model."},
    )
    rec_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": ("Path to the target recommender model.")},
    )
    rec_model_type: Optional[str] = field(
        default="SASRec",
        metadata={
            "help": ("The type of the target recommender model."),
            "choices": ["SASRec", "MF"],
        },
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
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
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
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
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
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
    padding_side: str = field(
        default="right",
        metadata={
            "help": (
                "The side on which the model should have padding for the attention mask. Should be selected between 'right' and 'left'."
                "For mistral and phi3, the padding side should be 'left'. For llama, the padding side should be 'right'."
            )
        },
    )


    def __post_init__(self):
        pass


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    sequential_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input sequential data file (a text file)."},
    )
    max_hist_len: Optional[int] = field(
        default=None,
        metadata={"help": "The max length of the user history."},
    )
    task_type: Optional[str] = field(
        default="both",
        metadata={"help": "The task type to use. Options are: intention, behaviour, both, none"},
    )
    template_name: Optional[str] = field(
        default="mistral",
        metadata={"help": "The template name to use. Options are: mistral, vicuna, llama-2, llama-3, phi3"},
    )
    data_type_filter: Optional[str] = field(
        default=None,
        metadata={"help": "The data type used during training."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        pass

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer_kwargs = {
        "model_max_length": model_args.model_max_length,
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
        "padding_side": model_args.padding_side,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<user>", "<item>"]})
    if tokenizer.pad_token is None:
        if data_args.template_name == "llama-3":
            tokenizer.pad_token_id = 128001
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("Tokenizer should have a pad token or an unk token or an eos token.")

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    config.pad_token_id = tokenizer.pad_token_id
    config.task_type = data_args.task_type
    config.rec_model_type = model_args.rec_model_type
    rec_cpt = torch.load(model_args.rec_model_name_or_path, map_location='cpu')
    config.rec_config = rec_cpt['config']
    del_keys = ['device', 'item_emb_path', 'text_emb_path']
    for key in del_keys:
        if key in config.rec_config:
            del config.rec_config[key]

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    if config.architectures[0] == "MistralForCausalLM":
        classForExp = MistralForExp
    elif config.architectures[0] == "LlamaForCausalLM":
        classForExp = LlamaForExp
    elif config.architectures[0] == "Phi3ForCausalLM":
        classForExp = Phi3ForExp
    else:
        raise ValueError("Model architecture not supported")

    model = classForExp.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        attn_implementation=model_args.attn_implementation,
    )

    model.rec_model.load_state_dict(rec_cpt['state_dict'], strict=False)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    peft_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","v_proj", "k_proj", "o_proj"] if config.architectures[0] != "Phi3ForCausalLM" else ["qkv_proj"],
        modules_to_save=["lm_head", "embed_tokens", "user_connector", "item_connector"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    logger.info(model)

    if training_args.do_train:
        train_dataset = TrainingDataset4Exp(data_args, tokenizer, split="train", data_type_filter=data_args.data_type_filter)

    if training_args.do_eval:
        eval_dataset = {}
        all_eval = TrainingDataset4Exp(data_args, tokenizer, split="valid")
        eval_dataset["all"] = all_eval
        for data_type in ["iid2title", "sharegpt", "uid2hist", "uid2next", "uidiid2rank", "uidiid2binary"]:
            type_eval = TrainingDataset4Exp(data_args, tokenizer, split="valid", data_type_filter=data_type)
            if len(type_eval) > 0:
                eval_dataset[data_type] = type_eval

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            filter_label_indices = (labels != -100)
            labels = labels[filter_label_indices]
            preds = preds[filter_label_indices]
            return metric.compute(predictions=preds, references=labels)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=ExpCollator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )
    trainer._signature_columns = ["input_ids", "attention_mask", "user_pos", "user_ids", "item_seq", "item_pos", "item_ids", "labels"] #ignore trainer._set_signature_columns_if_needed

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)
        try:
            perplexity = math.exp(metrics["eval_all_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["all_perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
