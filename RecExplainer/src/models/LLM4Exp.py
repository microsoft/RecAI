# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from accelerate.logging import get_logger
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel

from models.SASRec import SASRec

logger = get_logger(__name__)

class LLM4Exp(PreTrainedModel):
    def __init__(self, args, llm_config):
        super().__init__(llm_config)
        self.args = args
        self.llm_model = AutoModelForCausalLM.from_pretrained(
                args.llm_model_name_or_path,
                config=llm_config,
                cache_dir=args.cache_dir,
            )
        if self.args.task_type!="none":
            self.rec_model = SASRec.from_pretrained(args.rec_model_name_or_path)
            for name, param in self.rec_model.named_parameters():
                param.requires_grad = False

            self.user_connector = nn.Sequential(
                nn.Linear(self.rec_model.config['embedding_size'], self.llm_model.config.hidden_size//2),
                nn.GELU(),
                nn.Linear(self.llm_model.config.hidden_size//2, self.llm_model.config.hidden_size),
            )
            self.item_connector = nn.Sequential(
                nn.Linear(self.rec_model.config['embedding_size'], self.llm_model.config.hidden_size//2),
                nn.GELU(),
                nn.Linear(self.llm_model.config.hidden_size//2, self.llm_model.config.hidden_size),
            )


    # def gradient_checkpointing_enable(self):
    #     self.model.gradient_checkpointing_enable()

    def get_input_embeds(self, input_ids, user_pos, item_seq, item_pos, item_ids):
        if self.args.task_type!="none":
            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)
            token_embeds = self.llm_model.get_input_embeddings()(input_ids)

            blank_embeds = torch.zeros_like(token_embeds).reshape(batch_size*seq_len, -1)
            user_embeds, item_embeds = self.rec_model(item_seq=item_seq, item_ids=item_ids)
            user_embeds = self.user_connector(user_embeds)
            item_embeds = self.item_connector(item_embeds)

            filter_val = torch.tensor([[seq_len*i] for i in range(batch_size)]).to(user_embeds.device)
            filter_user = torch.all(user_pos!=filter_val, dim=0).unsqueeze(-1)
            filter_item = torch.all(item_pos!=filter_val, dim=0).unsqueeze(-1)
            blank_embeds[user_pos] += user_embeds * filter_user
            blank_embeds[item_pos] += item_embeds * filter_item

            blank_embeds = blank_embeds.reshape(batch_size, seq_len, -1)

            inputs_embeds = token_embeds + blank_embeds
        else:
            inputs_embeds = self.llm_model.get_input_embeddings()(input_ids)
        return inputs_embeds
    

    def forward(self, input_ids, attention_mask, user_pos, item_seq, item_pos, item_ids, labels=None):
        '''
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        labels: (batch_size, seq_len)
        user_pos: (u_n,)
        item_seq: (u_n, item_seq_len)
        item_pos: (i_n, )
        item_ids: (i_n, )
        '''
        inputs_embeds = self.get_input_embeds(input_ids, user_pos, item_seq, item_pos, item_ids)
        output = self.llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

        return output

    def generate(self, input_ids, attention_mask, user_pos, item_seq, item_pos, item_ids, generation_config):
        inputs_embeds = self.get_input_embeds(input_ids, user_pos, item_seq, item_pos, item_ids)
        outputs = self.llm_model.generate(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, generation_config=generation_config)
        return outputs

    def compute_transition_scores(self, **kwargs):
        return self.llm_model.compute_transition_scores(**kwargs)

    def resize_token_embeddings(self, new_num_tokens):
        self.llm_model.resize_token_embeddings(new_num_tokens)
