# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass, field
from typing import Optional

import torch

from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
)
from models.LLM4Exp import MistralForExp, LlamaForExp, Phi3ForExp
from peft import PeftModel


@dataclass
class Arguments:
    output_dir: Optional[str] = field(default="", metadata={"help": "the output directory"})
    cache_dir: Optional[str] = field(default="", metadata={"help": "the cache directory"})
    peft_model_name: Optional[str] = field(default="", metadata={"help": "the peft model name"})
    model_name_or_path: Optional[str] = field(default="", metadata={"help": "the model name"})
    rec_model_name_or_path: Optional[str] = field(default=None, metadata={"help": ("Path to the target recommender model.")},)
    task_type: Optional[str] = field(
        default="both",
        metadata={"help": "The task type to use. Options are: intention, behaviour, both, none"},
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={"help": ("The attention implementation to use. Options are: 'eager', 'flash_attention_2'")},
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
    rec_model_type: Optional[str] = field(
        default="SASRec",
        metadata={
            "help": ("The type of the target recommender model."),
            "choices": ["SASRec", "MF"],
        },
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
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )


def main():
    parser = HfArgumentParser(Arguments)
    args: Arguments = parser.parse_args_into_dataclasses()[0]

    tokenizer = AutoTokenizer.from_pretrained(args.peft_model_name)

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
    rec_cpt = torch.load(args.rec_model_name_or_path, map_location='cpu')
    config.rec_config = rec_cpt['config']
    del_keys = ['device', 'item_emb_path', 'text_emb_path']
    for key in del_keys:
        if key in config.rec_config:
            del config.rec_config[key]

    torch_dtype = (
        args.torch_dtype
        if args.torch_dtype in ["auto", None]
        else getattr(torch, args.torch_dtype)
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
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir,
        revision=args.model_revision,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        attn_implementation=args.attn_implementation,
    )

    model.rec_model.load_state_dict(rec_cpt['state_dict'], strict=False)

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    peft_model = PeftModel.from_pretrained(model, args.peft_model_name)
    peft_model = peft_model.merge_and_unload()

    peft_model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == '__main__':
    main()