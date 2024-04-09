# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
import os.path
import random
import sys

import numpy as np
import torch
import transformers
from accelerate.utils import set_seed

from base.trainer import BaseTrainer
from rl.trainer import RLTrainer
from sft.trainer import SFTTrainer
from param import Config, get_args

if __name__ == '__main__':
    args = get_args()
    assert args.train_stage in ['SFT', 'RL', 'SFT_Merge', 'RL_Merge']
    assert args.output_path is not None

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    transformers.set_seed(args.seed)
    set_seed(args.seed)

    # if args.log_to_file:
    #     log_file = open(args.output_path+f'{time.strftime("%Y-%m-%d %Hh_%Mm_%Ss", time.localtime())} {args.train_stage}.log', 'w')
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


