# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import re

import bitsandbytes as bnb
import torch
from einops.layers.torch import Rearrange
from peft import TaskType, LoraConfig, inject_adapter_in_model, LoraModel
from torch import nn
from transformers import T5Config, AutoConfig, AutoTokenizer, T5ForConditionalGeneration, AutoModelForCausalLM, BitsAndBytesConfig
from utils.tools import eval_decorator, shift


def layer_init(layer, std=2**0.5):
    nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(layer.weight, gain=std)
    return layer


class ValueRewardHead(nn.Module):
    def __init__(self, hidden_size, inference):
        super().__init__()
        self.head = nn.Sequential(
            layer_init(nn.Linear(hidden_size, 64)),
            nn.GELU(),
            layer_init(nn.Linear(64, 1), std=1.0),
            Rearrange('... 1 -> ...'),
        )
        self.requires_grad_(not inference)

    def forward(self, emb: torch.Tensor):
        # emb_norm = emb / emb.square().mean(dim=1, keepdim=True)
        return self.head(emb)


def lora_param_init(named_param):
    for n, p in named_param.items():
        if p.ndim >= 2 and 'lora_A' in n:
            try:
                nn.init.orthogonal_(p, gain=2**0.5)
            except RuntimeError as e:
                print(n, e)
        if 'lora_B' in n:
            nn.init.zeros_(p)


class BaseModel(nn.Module):       # name
    def __init__(self, args, device, actor_lora_scope='actor', critic_lora_scope='critic'):
        super().__init__()
        self.args = args
        self.model_config = self.create_model_config()
        self.tokenizer = self.create_tokenizer()
        self.model = self.create_model(device)

        self.actor_lora_scope = actor_lora_scope
        self.critic_lora_scope = critic_lora_scope
        if self.args.train_stage in ['SFT', 'SFT_Merge']:
            if self.args.full_fine_tune:
                self.model.requires_grad_(True)
            elif self.args.SFT_actor_lora_r > 0:
                self.actor_lora_config = self.create_lora_config(
                    self.actor_lora_scope,
                    False,
                    self.args.SFT_actor_lora_r,
                    self.args.SFT_actor_lora_a
                )
                self.lora_model = LoraModel(self.model, self.actor_lora_config, adapter_name=self.actor_lora_scope)
                lora_param_init(self.actor_named_parameters)
            else:
                raise NotImplementedError

        if self.args.train_stage in ['RL', 'RL_Merge']:
            assert self.args.RL_actor_lora_r > 0 and self.args.RL_critic_lora_r > 0
            self.actor_lora_config = self.create_lora_config(
                self.actor_lora_scope,
                False,
                self.args.RL_actor_lora_r,
                self.args.RL_actor_lora_a
            )
            self.lora_model = LoraModel(self.model, self.actor_lora_config, adapter_name=self.actor_lora_scope)
            lora_param_init(self.actor_named_parameters)

            self.critic_lora_config = self.create_lora_config(
                self.critic_lora_scope,
                False,
                self.args.RL_critic_lora_r,
                self.args.RL_critic_lora_a
            )
            inject_adapter_in_model(self.critic_lora_config, self.model, adapter_name=self.critic_lora_scope)
            self.critic_value_head = ValueRewardHead(self.model_config.hidden_size, inference=False)
            lora_param_init(self.critic_named_parameters)

            self.critic_value_head = self.critic_value_head.to(device).bfloat16()

            if self.args.lm_head_full_tune:
                self.model.lm_head.requires_grad_(True)

    def save_parameters(self, name='Epoch00'):
        params = {}
        if self.args.train_stage in ['SFT', 'RL']:
            params.update(self.actor_named_parameters)
        if self.args.train_stage in ['RL']:
            params.update(self.critic_named_parameters)
        state_dict = {
            'params': params,
        }
        torch.save(state_dict, os.path.join(self.args.output_path, f"{name}_{self.args.train_stage}.pth"))

    def load_parameters(self, load_file):
        # self.args.load: 'xxx/Epoch{xx}_SFT' or 'xxx/{xx}step_RL'
        if load_file is not None and os.path.exists(f"{load_file}.pth"):
            state_dict = torch.load(f"{load_file}.pth", map_location=self.device)
            results = self.load_state_dict(state_dict['params'], strict=False)
            assert len(results.unexpected_keys) == 0, results.unexpected_keys
            print(f'{self.args.train_stage} model loaded of file {load_file}')
            if self.args.train_stage in ['SFT', 'SFT_Merge']:
                return int(re.findall(r'.+/Epoch(\d+)_SFT$', load_file)[0])     # return train epoch number
            elif self.args.train_stage in ['RL', 'RL_Merge']:
                return int(re.findall(r'.+/(\d+)step_RL$', load_file)[0])       # return train step number
        else:
            return 0

    def print_trainable_parameters(self):
        trainable_params = {
            self.actor_lora_scope: 0,
            self.critic_lora_scope: 0,
            "base": 0
        }
        all_param = {
            self.actor_lora_scope: 0,
            self.critic_lora_scope: 0,
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
            elif self.critic_lora_scope in _:
                all_param[self.critic_lora_scope] += num_params
            else:
                all_param["base"] += num_params

            if param.requires_grad:
                if self.actor_lora_scope in _:
                    trainable_params[self.actor_lora_scope] += num_params
                elif self.critic_lora_scope in _:
                    trainable_params[self.critic_lora_scope] += num_params
                else:
                    trainable_params["base"] += num_params

        print(f'trainable_params: {" - ".join([str(_) for _ in trainable_params.values()])} | '
              f'all_param: {" - ".join([str(_) for _ in all_param.values()])} | '
              f'percentage: {sum(trainable_params.values())/sum(all_param.values()):.4f}')

    def create_model_config(self):
        if 't5' in self.args.backbone:
            config_class = T5Config
        else:
            config_class = AutoConfig

        config = config_class.from_pretrained(self.args.backbone)
        config.dropout_rate = self.args.dropout
        config.dropout = self.args.dropout
        config.attention_dropout = self.args.dropout
        config.activation_dropout = self.args.dropout
        return config

    def create_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.args.backbone)
        # tokenizer.add_tokens(['\n'] + [f'<{i+1}>' for i in range(20)])
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
        self.model_config.pad_token_id = tokenizer.pad_token_id
        return tokenizer

    def create_model(self, device):
        if 't5' in self.args.backbone:
            model_class = T5ForConditionalGeneration
        else:
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
        # self.args.lora_module_name = 'q_proj,v_proj,k_proj,o_proj' -> target_name = ['q_proj', 'v_proj', 'k_proj', 'o_proj']
        target_name = self.args.lora_module_name.split(',')
        cls = bnb.nn.Linear4bit if self.args.quantization else torch.nn.Linear
        lora_module_names = set()
        for name, module in self.model.named_modules():
            if 'lora' in name:
                continue
            if isinstance(module, cls) and any([tgt in name for tgt in target_name]):   # at least one of target_name in the param name.
                lora_module_names.add(name)

        if scope == self.critic_lora_scope or self.args.lm_head_full_tune:
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
    def device(self):
        return self.model.device

    @property
    def actor_named_parameters(self):
        # get all trainable params of actor scope.
        if self.args.train_stage == 'SFT' and self.args.full_fine_tune:
            return {n: p for n, p in self.named_parameters() if p.requires_grad}
        else:
            return {n: p for n, p in self.named_parameters() if self.actor_lora_scope in n or (n == 'model.lm_head.weight' and self.args.lm_head_full_tune)}

    @property
    def critic_named_parameters(self):
        # get all trainable params of critic scope.
        critic_head_params = {n: p for n, p in self.critic_value_head.named_parameters()}
        critic_lora_params = {n: p for n, p in self.named_parameters() if self.critic_lora_scope in n and 'lora' in n}
        critic_head_params.update(critic_lora_params)
        return critic_lora_params

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

    @torch.no_grad()
    @eval_decorator
    def generate(self, scope, input_ids, **kwargs):
        if not hasattr(self, 'lora_model'):
            return self.model.generate(input_ids=input_ids, **kwargs)

        if scope == self.actor_lora_scope:
            self.lora_model.enable_adapter_layers()
            self.lora_model.set_adapter(scope)
            return self.lora_model.generate(input_ids=input_ids, **kwargs)
        elif scope == 'base':
            self.lora_model.disable_adapter_layers()
            return self.lora_model(input_ids=input_ids, **kwargs)
        else:
            raise NotImplementedError

    def forward(self, scope, input_ids, **kwargs):
        """
        activate specific lora adaptor according specific scope.
        :param scope: in [actor_scope, critic_scope, base]
        :param input_ids:
        :param kwargs:
        :return:
        """
        if not hasattr(self, 'lora_model'):
            return self.model(input_ids=input_ids, **kwargs)

        if scope == self.actor_lora_scope:
            self.lora_model.enable_adapter_layers()
            self.lora_model.set_adapter(scope)
            return self.lora_model(input_ids=input_ids, **kwargs)
        elif scope == self.critic_lora_scope:
            self.lora_model.enable_adapter_layers()
            self.lora_model.set_adapter(scope)
            critic_token_embed = self.lora_model(input_ids=input_ids, output_hidden_states=True, **kwargs).hidden_states[-1]
            action_value = self.critic_value_head(critic_token_embed)
            return shift(action_value, shift=1, dim=-1)
        elif scope == 'base':
            self.lora_model.disable_adapter_layers()
            return self.lora_model(input_ids=input_ids, **kwargs)
        else:
            raise NotImplementedError
