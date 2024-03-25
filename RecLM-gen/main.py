# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import numpy as np
import torch
import transformers
from base.trainer import BaseTrainer
from rl.trainer import RLTrainer
from sft.trainer import SFTTrainer
from param import Config, get_args

if __name__ == '__main__':
    args = get_args()
    kwargs = vars(args)
    args = Config(**kwargs)
    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    transformers.set_seed(args.seed)

    if args.train_stage in ['RL', 'RL_merge']:
        if args.model_name is None:
            if args.lr > 0:
                args.model_name = f'RL_Total_train_LM-{args.lm_head_full_tune}_VM-{args.vague_mapping}_NR-20.1_SN-{args.sample_num}' \
                                  f'_Q-{args.quantization}_T{len(args.RL_train_tasks.split(","))}' \
                                  f'_FG-{args.fine_grain_reward}_LR-{args.lr}_LDO-{args.lora_dropout}_WD-{args.weight_decay}' \
                                  f'_KLC-{args.kl_coef}_EW-{args.entropy_weight}_RS-{args.reward_scale}_RW-{args.whiten_reward}' \
                                  f'_VFC-{args.vf_coef}_KLT-{args.policy_kl_threshold}_LRP-{args.lr_power}_GAMMA-{args.gamma}' \
                                  f'_GAS-{args.gradient_accumulation_steps}_LB-{args.learn_batch}_RA_{args.reward_alpha}' \
                                  f'_{args.model_name_suffix}'
            else:
                args.model_name = f'RL_Total_init_LM-{args.lm_head}_VM-{args.vague_mapping}_NR-20.1_SN-{args.sample_num}_Q-{args.quantization}_T{len(args.RL_train_tasks.split(","))}'
        args.output = f'{args.output}{args.model_name}/'

    # if args.log_to_file:
    #     log_file = open(args.output+f'{time.strftime("%Y-%m-%d %Hh_%Mm_%Ss", time.localtime())} {args.train_stage}.log', 'w')
    #     sys.stdout = log_file

    if args.train_stage == 'SFT':
        trainer = SFTTrainer(args)
        trainer.SFT_train()
    elif args.train_stage == 'RL':
        trainer = RLTrainer(args)
        trainer.RL_train()
    elif args.train_stage in ['SFT_Merge', 'RL_Merge']:
        trainer = BaseTrainer(args)
        trainer.Adapter_merge()
    else:
        raise NotImplementedError


