import os
import re

import bitsandbytes as bnb
import torch
from peft import TaskType, LoraConfig, LoraModel
from torch import nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from .utils import eval_decorator


def param_init(params):
    if isinstance(params, dict):
        for n, p in params.items():
            # if p.ndim < 2 or "lora_B" in n:
            if p.ndim >= 2 and 'lora' in n:
                try:
                    nn.init.orthogonal_(p, gain=2 ** 0.5)
                except RuntimeError as e:
                    print(n)
                    print(e)
            if 'lora_B' in n:
                nn.init.zeros_(p)
    else:
        for p in params:
            nn.init.orthogonal_(p, gain=2 ** 0.5)


class BaseModel(nn.Module):
    def __init__(self, args, device, actor_lora_scope='actor', item_emb=None):
        super().__init__()
        self.args = args
        self.model_config = self.create_model_config()
        self.tokenizer = self.create_tokenizer()
        self.model = self.create_model(device)
        if args.use_control_symbol:
            self.resize_init_embedding(self.model, self.tokenizer)

        self.actor_lora_scope = actor_lora_scope
        if self.args.train_stage in ['SFT', 'SFT_Embedding', 'SFT_Test', 'SFT_Embedding_Test', 'SFT_Merge'] and self.args.SFT_actor_lora_r > 0:
            self.actor_lora_config = self.create_lora_config(
                self.actor_lora_scope,
                False,
                self.args.SFT_actor_lora_r,
                self.args.SFT_actor_lora_a
            )
            self.lora_model = LoraModel(self.model, self.actor_lora_config, adapter_name=self.actor_lora_scope)

            if self.args.train_stage in ['SFT_Embedding', 'SFT_Embedding_Test']:
                if self.args.embedding_model == self.args.backbone:
                    self.actor_item_proj = nn.Sequential(
                        nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
                    ).to(device).bfloat16()
                else:
                    self.actor_item_proj = nn.Sequential(
                        nn.Linear(self.model.config.hidden_size, 2048),
                        nn.GELU(),
                        nn.Linear(2048, 1024),
                    ).to(device).bfloat16()

                self.item_emb = nn.Embedding(item_emb.shape[0], item_emb.shape[1], _weight=item_emb)
                self.item_emb.requires_grad_(self.args.item_emb)

            param_init(self.actor_named_parameters)

        if self.args.lm_head:
            self.model.lm_head.requires_grad_(self.args.lm_head)
        if self.args.token_emb:
            self.model.base_model.embed_tokens.requires_grad_(self.args.token_emb)

    def save_parameters(self, name='Epoch00'):
        if not os.path.isdir(self.args.output):
            os.makedirs(self.args.output, exist_ok=True)
        state_dict = self.state_dict()
        # if self.args.use_control_symbol:
        #     param_dict.update({n: p for n, p in self.named_parameters() if "embed_" in n})

        torch.save(
            {n: state_dict[n] for n in self.actor_named_parameters},
            os.path.join(self.args.output, f"{name}_{self.args.train_stage}.pth")
        )

    def load_parameters(self, load_file):
        # self.args.load: xxx/{name}_{train_stage}
        if load_file is not None and os.path.exists(f"{load_file}.pth"):
            state_dict = torch.load(f"{load_file}.pth", map_location=self.device)
            results = self.load_state_dict(state_dict, strict=False)
            if self.args.train_stage != "SFT_Merge":
                assert len(results.unexpected_keys) == 0, results.unexpected_keys
            # if self.args.train_stage in ['SFT_Embedding', 'SFT_Embedding_Test']:
            #     assert all(["actor_item_proj" not in _ for _ in results.missing_keys])
            print(f'{self.args.train_stage} model loaded of file {load_file}')
            return int(re.findall(r'Epoch(\d+)', load_file)[-1])
        else:
            return 0

    def print_trainable_parameters(self):
        r"""
        Returns the number of trainable parameters and number of all parameters in the model.
        """
        trainable_params = {
            self.actor_lora_scope: 0,
            "base": 0
        }
        all_param = {
            self.actor_lora_scope: 0,
            "base": 0
        }
        for _, param in self.named_parameters():
            num_params = param.numel()
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            if self.actor_lora_scope in _:
                all_param[self.actor_lora_scope] += num_params
            else:
                all_param["base"] += num_params

            if param.requires_grad:
                if self.actor_lora_scope in _:
                    trainable_params[self.actor_lora_scope] += num_params
                else:
                    trainable_params["base"] += num_params

        print(f'trainable_params: {" - ".join([str(_) for _ in trainable_params.values()])} | '
              f'all_param: {" - ".join([str(_) for _ in all_param.values()])} | '
              f'percentage: {sum(trainable_params.values()) / sum(all_param.values()):.4f}')

    def resize_init_embedding(self, model, tokenizer):
        tokenizer.add_special_tokens({'additional_special_tokens': ['<SOI>', '<EOI>']})
        tokenizer.soi_token = "<SOI>"
        tokenizer.eoi_token = "<EOI>"
        tokenizer.soi_token_id = tokenizer.convert_tokens_to_ids("<SOI>")
        tokenizer.eoi_token_id = tokenizer.convert_tokens_to_ids("<EOI>")
        print(tokenizer.soi_token, tokenizer.soi_token_id)
        print(tokenizer.eoi_token, tokenizer.eoi_token_id)

        model.resize_token_embeddings(len(tokenizer))
        self.model_config.vocab_size = len(tokenizer)
        descriptions = {"<SOI>": "start of an item", "<EOI>": "end of an item"}
        with torch.no_grad():
            for token in descriptions:
                description = descriptions[token]
                token_id = tokenizer.convert_tokens_to_ids(token)
                tokenized = tokenizer.tokenize(description)
                print(tokenized)
                tokenized_ids = tokenizer.convert_tokens_to_ids(tokenized)
                new_embedding = model.get_input_embeddings().weight[tokenized_ids].mean(axis=0)
                model.get_input_embeddings().weight[token_id, :] = new_embedding.clone().detach()

    def create_model_config(self):
        config_class = AutoConfig
        config = config_class.from_pretrained(self.args.backbone)
        config.dropout_rate = self.args.dropout
        config.dropout = self.args.dropout
        config.attention_dropout = self.args.dropout
        config.activation_dropout = self.args.dropout
        return config

    def create_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.args.backbone)

        if self.args.chat_template == 'llama-2':
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        elif self.args.chat_template == 'llama-3':
            tokenizer.pad_token = '<|reserved_special_token_250|>'
            tokenizer.pad_token_id = 128255
            eot_token = "<|eot_id|>"
            eot_token_id = tokenizer.convert_tokens_to_ids(eot_token)
            tokenizer.eos_token = eot_token
            tokenizer.eos_token_id = eot_token_id

        self.model_config.pad_token_id = tokenizer.pad_token_id
        return tokenizer

    def create_model(self, device):
        model_class = AutoModelForCausalLM
        bnb_config = BitsAndBytesConfig(
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        if self.args.quantization:
            model = model_class.from_pretrained(self.args.backbone,
                                                config=self.model_config,
                                                quantization_config=bnb_config,
                                                device_map=device,
                                                torch_dtype=torch.bfloat16,
                                                use_flash_attention_2=self.args.FA2
                                                )
        else:
            model = model_class.from_pretrained(self.args.backbone,
                                                config=self.model_config,
                                                device_map=device,
                                                torch_dtype=torch.bfloat16,
                                                use_flash_attention_2=self.args.FA2
                                                )
        model.requires_grad_(False)
        return model

    def find_all_linear_names(self, scope):
        # target_name = ['q_proj', 'v_proj', 'k_proj', 'o_proj']
        target_name = ['']
        if self.args.quantization:
            cls = bnb.nn.Linear4bit
        else:
            cls = torch.nn.Linear
        lora_module_names = set()
        for name, module in self.model.named_modules():
            if 'lora' in name:
                continue
            if isinstance(module, cls) and any([_ in name for _ in target_name]):
                lora_module_names.add(name)
        if scope == self.actor_lora_scope and not self.args.lm_head:
            lora_module_names.add('lm_head')
        elif 'lm_head' in lora_module_names:
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

    def create_lora_config(self, scope, inference_mode, r, alpha):
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=inference_mode,
            r=r,
            target_modules=self.find_all_linear_names(scope),
            lora_alpha=alpha,
            lora_dropout=self.args.lora_dropout,
            init_lora_weights=True,
            bias="none",
        )
        return lora_config

    @property
    def actor_model(self):
        if not hasattr(self, 'lora_model'):
            return self.model
        self.lora_model.enable_adapter_layers()
        self.lora_model.set_adapter(self.actor_lora_scope)
        return self.lora_model

    @property
    def base_model(self):
        if not hasattr(self, 'lora_model'):
            return self.model
        self.lora_model.disable_adapter_layers()
        return self.lora_model

    @property
    def actor_parameters(self):
        return [
            p for n, p in self.named_parameters()
            if self.actor_lora_scope in n or
               (n == 'model.lm_head.weight' and self.args.lm_head) or
               (n == 'model.embed_tokens.weight' and self.args.token_emb) or
               (n == 'item_emb.weight' and self.args.item_emb)
        ]

    @property
    def actor_named_parameters(self):
        return {
            n: p
            for n, p in self.named_parameters()
            if self.actor_lora_scope in n or
               (n == 'model.lm_head.weight' and self.args.lm_head) or
               (n == 'model.embed_tokens.weight' and self.args.token_emb) or
               (n == 'item_emb.weight' and self.args.item_emb)
        }

    def trainable_parameters(self):
        return [p for n, p in self.named_parameters() if p.requires_grad]

    @property
    def device(self):
        return self.model.device

    @torch.no_grad()
    @eval_decorator
    def generate(self, scope, input_ids=None, **kwargs):
        if scope == self.actor_lora_scope:
            model = self.actor_model
        else:
            model = self.base_model
        return model.generate(input_ids=input_ids, **kwargs)

    def forward(self, scope, input_ids=None, inputs_embeds=None, **kwargs):
        if self.args.train_stage in ['SFT_Embedding', 'SFT_Embedding_Test']:
            kwargs['output_hidden_states'] = True

        if scope == self.actor_lora_scope:
            outputs = self.actor_model(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
        else:
            outputs = self.base_model(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)

        if self.args.train_stage in ['SFT_Embedding', 'SFT_Embedding_Test']:
            if self.args.only_item_emb_proj:
                hidden_states = outputs.hidden_states[-1].detach()  # [batch_size, seq_length, hidden_size]
            else:
                hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_length, hidden_size]
            soi_positions = torch.eq(input_ids, self.tokenizer.soi_token_id).nonzero().tolist()
            hidden_states = torch.stack([hidden_states[bs_idx, seq_idx-1] for [bs_idx, seq_idx] in soi_positions])
            embeddings = self.actor_item_proj(hidden_states)  # [len(soi_p), hidden_size]
            temp_embeddings = [[] for i in range(input_ids.shape[0])]
            last_eos_indices = {}
            for bs_idx, ids in enumerate(input_ids):
                last_eos_indices[bs_idx] = torch.eq(input_ids[bs_idx], self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            for [bs_idx, seq_idx], emb in zip(soi_positions, embeddings):
                for i in range(1, len(last_eos_indices[bs_idx]), 2):
                    if last_eos_indices[bs_idx][i+1] > seq_idx > last_eos_indices[bs_idx][i]:
                        temp_embeddings[bs_idx].append(emb)
            for i in range(input_ids.shape[0]):
                try:
                    temp_embeddings[i] = torch.stack(temp_embeddings[i], dim=0)
                except:
                    print(input_ids[i])
                    raise EOFError
            outputs['embeddings'] = temp_embeddings

        return outputs

    def embedding_similarity(self, embeddings):
        inner_dot = torch.matmul(embeddings, self.item_emb.weight.T)  # [n, k]
        return inner_dot

