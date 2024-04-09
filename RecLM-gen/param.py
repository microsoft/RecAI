# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import pprint


def add_args_SFT(parser):
    parser.add_argument('--share_chat_gpt_ratio', type=float, default=0.0, help='proportion of shareGPT corpus')
    parser.add_argument("--SFT_load", type=str, default=None, help='Load the SFT model params file (usually the fine-tuned model).')
    parser.add_argument('--SFT_train_tasks', type=str, default='', help='SFTSeqRec,SFTControlRec,SFTPersonalControlRec,SFTPersonalCategoryRate,SFTCategoryRate')
    parser.add_argument('--SFT_val_tasks', type=str, default='', help='SFTTestSeqRec,SFTTestSeqRanking,SFT+TestPersonalControlRec,SFT-TestPersonalControlRec,SFTTestPersonalCategoryRate,SFTTestItemCount')
    parser.add_argument("--SFT_actor_lora_r", type=int, default=16)
    parser.add_argument("--SFT_actor_lora_a", type=int, default=8)
    parser.add_argument("--full_fine_tune", action='store_true', help='full fine tune backbone.')

    return parser


def add_args_RL(parser):
    parser.add_argument('--RL_load', type=str, default=None, help='Load the RL model params file (usually the fine-tuned model).')
    parser.add_argument('--RL_train_tasks', type=str, default='', help='RLSeqRec,RLSeqRanking,RL+PersonalControlRec,RL-PersonalControlRec,RLPersonalCategoryRate')
    parser.add_argument('--RL_val_tasks', type=str, default='', help='RLSeqRec,RLSeqRanking,RL+PersonalControlRec,RL-PersonalControlRec,RLPersonalCategoryRate,RLItemCount')
    parser.add_argument("--RL_actor_lora_r", type=int, default=4)
    parser.add_argument("--RL_actor_lora_a", type=int, default=2)
    parser.add_argument("--RL_critic_lora_r", type=int, default=4)
    parser.add_argument("--RL_critic_lora_a", type=int, default=2)

    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--value_clip", type=float, default=0.4)
    parser.add_argument("--beta_s", type=float, default=0.01)
    parser.add_argument("--kl_coef", type=float, default=0.3)
    parser.add_argument("--vf_coef", type=float, default=0.1)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--entropy_weight", type=float, default=0.01)
    parser.add_argument("--sample_num", type=int, default=4)
    parser.add_argument("--fine_grain_reward", action='store_true')
    parser.add_argument("--whiten_reward", action='store_true')
    parser.add_argument("--reward_scale", action='store_true')
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--policy_kl_threshold", type=float, default=0.03)
    parser.add_argument("--lr_power", type=float, default=2.0)
    parser.add_argument("--learn_batch", type=int, default=2)
    parser.add_argument("--reward_alpha", type=float, default=0.5)
    parser.add_argument("--val_save_step", type=int, default=100)
    return parser


def add_args(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=str, default=None, help='no need to set.')
    parser.add_argument('--FA2', action='store_true', help='whether to use flash attention 2.')
    parser.add_argument('--llama2_chat_template', action='store_true', help='whether to use llama2-chat template')
    parser.add_argument('--idx', action='store_true', help='whether to add the index number. eg. 1. item1\n 2. item2')

    # Data Splits
    parser.add_argument('--data_path', type=str, default=None, help='data path')
    parser.add_argument('--train_data_file', type=str, default=None, help='train data file')
    parser.add_argument('--val_data_file', type=str, default=None, help='val data file')
    parser.add_argument('--candidate_num', type=int, default=10, help='size of candidate set')
    parser.add_argument('--max_item_length', type=int, default=10, help='max length of history')
    parser.add_argument('--max_token_length', type=int, default=512, help='max length of input tokens')
    parser.add_argument('--item_index', type=str, default='title', help='in {id, title, title64, title64_t}')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument("--val_num_per_task", type=int, default=320, help='the number of valuation samples')

    # Checkpoint
    parser.add_argument('--output_path', type=str, default='snap/', help='path to save model params file, or to save the merged model.')

    # Model Config
    parser.add_argument('--backbone', type=str, default='google/flan-t5-xl')

    # Training
    parser.add_argument('--batch_size', type=int, default=2, help='train batch size of per device')
    parser.add_argument('--val_batch_size', type=int, default=16, help='valuation batch size of per device')
    parser.add_argument('--test_batch_size', type=int, default=8, help='test batch size of per device')
    parser.add_argument('--warmup_ratio', type=float, default=0.05, help='lr warmup ration of total train stage.')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--clip_grad_norm', type=float, default=-1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--adam_eps', type=float, default=1e-5)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--val_epoch', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--quantization", action='store_true', help='whether to use QLoRA')

    # Inference
    parser.add_argument('--gen_max_length', type=int, default=512, help='the max length of output tokens in train or inference')
    parser.add_argument("--vague_mapping", action='store_true', help='whether to user vague mapping')

    # Etc.
    parser.add_argument("--dry", action='store_true', help='whether to valuate model before training')
    parser.add_argument("--train_stage", type=str, default='SFT', help='in {SFT, SFT_Merge, RL, RL_Merge}')
    parser.add_argument("--backup_ip", type=str, default='0.0.0.0', help='ip address of SASRec server')
    parser.add_argument('--teacher_port', type=int, default=12621, help='port of SASRec server, usually 12621 in movie, 12622 in steam')

    parser.add_argument("--lm_head_full_tune", action='store_true', help='whether to full fine tune the lm_head')
    parser.add_argument("--lora_module_name", type=str, default='', help='empty string for all linear layers. eg. "proj,layer" means the module names that contain "proj" or "layer"')

    parser.add_argument("--distributed", action='store_true', help='whether to train distributed')

    return parser


def get_args(add_external_args_func=None):
    parser = add_args()
    args, remain_args = parser.parse_known_args()
    if args.train_stage in ['SFT', 'SFT_Merge']:
        parser = add_args_SFT(parser)
    elif args.train_stage in ['RL', 'RL_Merge']:
        parser = add_args_RL(parser)
    if add_external_args_func:
        parser = add_external_args_func(parser)
    args = parser.parse_args(remain_args, args)
    return args


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def config_str(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += self.config_str
        return config_str


if __name__ == '__main__':
    pass
