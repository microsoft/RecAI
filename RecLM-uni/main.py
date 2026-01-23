import random
import numpy as np
import torch
import transformers
from trainer import SFTTrainer
from train_utils.param import Config, get_args


if __name__ == '__main__':
    args = get_args()
    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    transformers.set_seed(args.seed)

    if args.train_stage == 'SFT' and args.share_chat_gpt_ratio > 0.:
        args.SFT_train_tasks = args.SFT_train_tasks + ',ShareChatGPT'

    trainer = SFTTrainer(Config(**vars(args)))
    if args.train_stage == 'SFT':
        trainer.SFT_train()
    elif args.train_stage == 'SFT_Embedding':
        trainer.SFTEmbedding_train()
    elif args.train_stage in ['SFT_Test', 'SFT_Embedding_Test']:
        if args.SFT_test_task == "SFTTestSeqRec-CS-MR":
            trainer.SFT_Embedding_MR_test()
        else:
            trainer.SFT_test()
    elif args.train_stage == 'SFT_Merge':
        trainer.SFT_adapter_merge()
    else:
        raise NotImplementedError
