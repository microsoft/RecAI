"""
The following code is modified from
https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/finetune/arguments.py
"""

from dataclasses import dataclass, field
from typing import Optional, List

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    sentence_pooling_method: str = field(default='cls', metadata={"help": "the pooling method, should be cls, mean, last"})
    normlized: bool = field(default=True)
    peft_model_name: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    attn_implementation: Optional[str] = field(
        default="eager", 
        metadata={
            "help": "The attention implementation to use: 'eager', 'sdpa', 'flash_attention_2'",
            "choices": ["eager", "sdpa", "flash_attention_2"],
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


@dataclass
class DataArguments:
    train_data: str = field(
        default=None, metadata={"help": "Path to train data"}
    )

    data_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the cached data datasets"}
    )

    train_group_size: int = field(default=8, metadata={"help": "1 positive and train_group_size-1 negative passages for each query"})

    query_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    passage_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    max_example_num_per_dataset: int = field(
        default=100000000, metadata={"help": "the max number of examples for each dataset"}
    )
    has_template: bool = field(default=False, metadata={"help": "whether the data has template, used only for LLM"})

@dataclass
class RetrieverTrainingArguments(TrainingArguments):
    negatives_cross_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    in_batch_negatives: bool = field(default=True, metadata={"help": "share negatives in a batch"})
    temperature: Optional[float] = field(default=0.01)
    fix_position_embedding: bool = field(default=False, metadata={"help": "Freeze the parameters of position embeddings"})
