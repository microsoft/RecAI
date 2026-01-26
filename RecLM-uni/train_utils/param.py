import argparse
import pprint
import yaml


def add_args_SFT(parser):
    parser.add_argument('--multi_round_ratio', type=float, default=0.0, help='multi round data ratio')
    parser.add_argument('--share_chat_gpt_ratio', type=float, default=0.0, help='shareGPT ratio')
    parser.add_argument("--SFT_load", type=str, default=None, help='Load the SFT model (usually the fine-tuned model).')
    parser.add_argument('--SFT_train_tasks', type=str, default='')
    parser.add_argument('--SFT_val_tasks', type=str, default='')
    parser.add_argument('--SFT_test_task', type=str, default='')
    parser.add_argument("--SFT_actor_lora_r", type=int, default=16)
    parser.add_argument("--SFT_actor_lora_a", type=float, default=8.0)
    parser.add_argument('--use_control_symbol', action='store_true', help='use control symbol. e.g. <SOI>item<EOI>\n 2. <SOI>item2<EOI>')
    parser.add_argument('--use_scope_mask', action='store_true', help='use scope_mask in training')
    parser.add_argument('--scope_mask_type', type=int, default=3, help='scope_mask type')
    parser.add_argument('--embedding_model', type=str, default=None, help='embedding model path')
    parser.add_argument('--domain', type=str, default='', help='domain')
    parser.add_argument('--emb_alpha', type=float, default=1.0, help='the weight of emb loss in RecLM-ret')
    parser.add_argument('--teacher_port', type=float, default=2068, help='port of teacher model')
    return parser


def add_args(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=str, default=None, help='')
    parser.add_argument('--FA2', action='store_true', help='flash attention')
    parser.add_argument('--chat_template', type=str, default="in [llama-2, llama-3]")
    parser.add_argument('--idx', action='store_true', help='1. item1\n 2. item2')

    # Data Splits
    parser.add_argument('--data_path', type=str, default='data/dataset/beauty/', help='data path')
    parser.add_argument('--candidate_num', type=int, default=10, help='size of candidate set')
    parser.add_argument('--max_item_length', type=int, default=10, help='max_item_length')
    parser.add_argument('--max_token_length', type=int, default=512, help='max_token_length')

    # Checkpoint
    parser.add_argument('--output', type=str, default='snap/')

    # Model Config
    parser.add_argument('--backbone', type=str, default='google/flan-t5-xl')
    parser.add_argument('--item_index', type=str, default='title_t')

    # Training
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--val_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=8)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--clip_grad_norm', type=float, default=-1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--adam_eps', type=float, default=1e-5)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--val_epoch', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--fl_gamma', type=float, default=2.0)

    parser.add_argument("--lm_head", action='store_true', help='train lm_head')
    parser.add_argument("--token_emb", action='store_true', help='train token_emb')
    parser.add_argument("--item_emb", action='store_true', help='train item_emb in RecLM-ret')
    parser.add_argument("--only_item_emb_proj", action='store_true', help='only train proj layer in RecLM-ret')

    # Inference
    parser.add_argument('--gen_max_length', type=int, default=512, help='gen_max_length')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--use_CBS', action='store_true', help='use constrainted generation')
    parser.add_argument('--CBS_type', type=int, default=2)
    parser.add_argument('--loss_type', type=int, default=3)

    # Etc.
    parser.add_argument("--dry", action='store_true')
    parser.add_argument("--train_stage", type=str, default='SFT', help='in {SFT, SFT_Embedding, SFT_Test, SFT_Embedding_Test, SFT_Merge}')
    parser.add_argument("--backup_ip", type=str, default='0.0.0.0', help='unirec serve ip')
    parser.add_argument("--quantization", action='store_true', help='')

    # PEFT
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    return parser


def get_args(add_external_args_func=None):
    parser = add_args()
    args, remain_args = parser.parse_known_args()
    if args.train_stage in ['SFT', 'SFT_Embedding', 'SFT_Embedding_Test', 'SFT_Test', 'SFT_Merge']:
        parser = add_args_SFT(parser)
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

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            kwargs = yaml.load(f)

        return Config(**kwargs)


if __name__ == '__main__':
    pass
