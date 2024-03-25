# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from transformers import SchedulerType, MODEL_MAPPING
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--data_dir", type=str, default=None, help="Path to the dataset directory."
    )
    parser.add_argument(
        "--data_names", type=str, default=None, help="json file names of the data."
    )
    parser.add_argument(
        "--sequential_file", type=str, default=None, help=""
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="Where do you want to store the cached datasets/models."
    )
    parser.add_argument(
        "--max_example_num_per_dataset", type=int, default=500000000, help=""
    )
    parser.add_argument(
        "--max_hist_len", type=int, default=512, help="the max length of the user history"
    )
    parser.add_argument(
        "--llm_max_length", type=int, default=512, help="the max token length of the LLM"
    )
    parser.add_argument(
        "--llm_model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--rec_model_name_or_path",
        type=str,
        help="Path to the target recommender model.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as llm_model_name_or_path",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as llm_model_name_or_path",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=2023, help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=0,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")

    ## training parameters
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=2, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="Run an evaluation each multiple training steps",
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=100,
        help="Log every X updates steps.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=100,
        help="Whether the various states should be saved at the end of every n steps",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--load_best_model",
        action="store_true",
        help="Whether to load the best model at the end of training",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--find_unused_parameters",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Whether or not to use lora.",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="intention",
        help="The task type to use.",
        choices=["intention", "behaviour", "both", "none"],
    )
    parser.add_argument("--data_type_filter", type=str, default=None, help="The data type not use.")

    ## inference parameters
    parser.add_argument(
        "--inference_mode",
        type=str,
        default="case study",
        help="what task to infer.",
        choices=["case study", "uid2hist", "uid2next", "uidiid2rank", "uidiid2binary"],
    )
    parser.add_argument(
        "--metadata_file", type=str, default=None, help="Path to the metainfo file."
    )
    parser.add_argument(
        "--test_top_file", type=str, default=None, help="Path to the test top file."
    )
    parser.add_argument(
        "--llm_model_ckpt_path",
        type=str,
        default=None,
        help="Path to the checkpoint of the RecExplainer model.",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether or not to use sampling ; use greedy decoding otherwise.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=20,
        help="The max number of new tokens to generate.",
    )
    parser.add_argument(
        "--min_new_tokens",
        type=int,
        default=1,
        help="The min number of new tokens to generate.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="The number of beams to use for beam search.",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="The number of beams to use for beam search.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="The repetition penalty for the generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The temperature for the generation.",
    )
 
    args = parser.parse_args()

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args