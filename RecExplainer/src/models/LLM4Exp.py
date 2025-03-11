# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from accelerate.logging import get_logger

import torch
from torch import nn
from transformers import MistralForCausalLM, LlamaForCausalLM, Phi3ForCausalLM

from models.SASRec import SASRec
from models.MF import MF

rec_models = {
    'SASRec': SASRec,
    'MF': MF
}

logger = get_logger(__name__)

class Phi3ForExp(Phi3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        if config.task_type!="none":
            self.rec_model = rec_models[config.rec_model_type](config.rec_config)
            # for _, param in self.rec_model.named_parameters():
            #     param.requires_grad = False

            self.user_connector = nn.Sequential(
                nn.Linear(self.rec_model.config['embedding_size'], config.hidden_size//2),
                nn.GELU(),
                nn.Linear(config.hidden_size//2, config.hidden_size),
            )
            self.item_connector = nn.Sequential(
                nn.Linear(self.rec_model.config['embedding_size'], config.hidden_size//2),
                nn.GELU(),
                nn.Linear(config.hidden_size//2, config.hidden_size),
            )

    def get_input_embeds(self, input_ids, user_pos, user_ids, item_seq, item_pos, item_ids):
        if self.config.task_type!="none":
            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)
            token_embeds = self.get_input_embeddings()(input_ids)

            blank_embeds = torch.zeros_like(token_embeds).reshape(batch_size*seq_len, -1)
            user_embeds, item_embeds = self.rec_model(user_ids=user_ids, item_seq=item_seq, item_ids=item_ids)
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
            inputs_embeds = self.get_input_embeddings()(input_ids)
        return inputs_embeds
    
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, user_pos=None, user_ids=None, item_seq=None, item_pos=None, item_ids=None):
        '''
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        labels: (batch_size, seq_len)
        user_pos: (u_n,)
        user_ids: (u_n, )
        item_seq: (u_n, item_seq_len)
        item_pos: (i_n, )
        item_ids: (i_n, )
        '''
        if user_pos is None and user_ids is None and item_seq is None and item_pos is None and item_ids is None:
            output = super().forward(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
        else:
            cur_inputs_embeds = self.get_input_embeds(input_ids, user_pos, user_ids, item_seq, item_pos, item_ids)
            output = super().forward(
                input_ids=None, 
                attention_mask=attention_mask, 
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=cur_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

        return output

    def generate(self, input_ids=None, attention_mask=None, user_pos=None, user_ids=None, item_seq=None, item_pos=None, item_ids=None, generation_config=None, **kwargs):
        if user_pos is None and user_ids is None and item_seq is None and item_pos is None and item_ids is None:
            outputs = super().generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=generation_config, **kwargs)
        else:
            inputs_embeds = self.get_input_embeds(input_ids, user_pos, user_ids, item_seq, item_pos, item_ids)
            outputs = super().generate(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, generation_config=generation_config, **kwargs)
        return outputs

class LlamaForExp(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        if config.task_type!="none":
            self.rec_model = rec_models[config.rec_model_type](config.rec_config)
            # for _, param in self.rec_model.named_parameters():
            #     param.requires_grad = False

            self.user_connector = nn.Sequential(
                nn.Linear(self.rec_model.config['embedding_size'], config.hidden_size//2),
                nn.GELU(),
                nn.Linear(config.hidden_size//2, config.hidden_size),
            )
            self.item_connector = nn.Sequential(
                nn.Linear(self.rec_model.config['embedding_size'], config.hidden_size//2),
                nn.GELU(),
                nn.Linear(config.hidden_size//2, config.hidden_size),
            )

    def get_input_embeds(self, input_ids, user_pos, user_ids, item_seq, item_pos, item_ids):
        if self.config.task_type!="none":
            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)
            token_embeds = self.get_input_embeddings()(input_ids)

            blank_embeds = torch.zeros_like(token_embeds).reshape(batch_size*seq_len, -1)
            user_embeds, item_embeds = self.rec_model(user_ids=user_ids, item_seq=item_seq, item_ids=item_ids)
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
            inputs_embeds = self.get_input_embeddings()(input_ids)
        return inputs_embeds
    
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, cache_position=None, user_pos=None, user_ids=None, item_seq=None, item_pos=None, item_ids=None):
        '''
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        labels: (batch_size, seq_len)
        user_pos: (u_n,)
        user_ids: (u_n, )
        item_seq: (u_n, item_seq_len)
        item_pos: (i_n, )
        item_ids: (i_n, )
        '''
        if user_pos is None and user_ids is None and item_seq is None and item_pos is None and item_ids is None:
            output = super().forward(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position
            )
        else:
            cur_inputs_embeds = self.get_input_embeds(input_ids, user_pos, user_ids, item_seq, item_pos, item_ids)
            output = super().forward(
                input_ids=None, 
                attention_mask=attention_mask, 
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=cur_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position
            )

        return output

    def generate(self, input_ids=None, attention_mask=None, user_pos=None, user_ids=None, item_seq=None, item_pos=None, item_ids=None, generation_config=None, **kwargs):
        if user_pos is None and user_ids is None and item_seq is None and item_pos is None and item_ids is None:
            outputs = super().generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=generation_config, **kwargs)
        else:
            inputs_embeds = self.get_input_embeds(input_ids, user_pos, user_ids, item_seq, item_pos, item_ids)
            outputs = super().generate(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, generation_config=generation_config, **kwargs)
        return outputs



class MistralForExp(MistralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        if config.task_type!="none":
            self.rec_model = rec_models[config.rec_model_type](config.rec_config)
            # for _, param in self.rec_model.named_parameters():
            #     param.requires_grad = False

            self.user_connector = nn.Sequential(
                nn.Linear(self.rec_model.config['embedding_size'], config.hidden_size//2),
                nn.GELU(),
                nn.Linear(config.hidden_size//2, config.hidden_size),
            )
            self.item_connector = nn.Sequential(
                nn.Linear(self.rec_model.config['embedding_size'], config.hidden_size//2),
                nn.GELU(),
                nn.Linear(config.hidden_size//2, config.hidden_size),
            )

    def get_input_embeds(self, input_ids, user_pos, user_ids, item_seq, item_pos, item_ids):
        if self.config.task_type!="none":
            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)
            token_embeds = self.get_input_embeddings()(input_ids)

            blank_embeds = torch.zeros_like(token_embeds).reshape(batch_size*seq_len, -1)
            user_embeds, item_embeds = self.rec_model(user_ids=user_ids, item_seq=item_seq, item_ids=item_ids)
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
            inputs_embeds = self.get_input_embeddings()(input_ids)
        return inputs_embeds
    
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, user_pos=None, user_ids=None, item_seq=None, item_pos=None, item_ids=None):
        '''
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        labels: (batch_size, seq_len)
        user_pos: (u_n,)
        user_ids: (u_n, )
        item_seq: (u_n, item_seq_len)
        item_pos: (i_n, )
        item_ids: (i_n, )
        '''
        if user_pos is None and user_ids is None and item_seq is None and item_pos is None and item_ids is None:
            output = super().forward(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
        else:
            cur_inputs_embeds = self.get_input_embeds(input_ids, user_pos, user_ids, item_seq, item_pos, item_ids)
            output = super().forward(
                input_ids=None, 
                attention_mask=attention_mask, 
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=cur_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

        return output

    def generate(self, input_ids=None, attention_mask=None, user_pos=None, user_ids=None, item_seq=None, item_pos=None, item_ids=None, generation_config=None, **kwargs):
        if user_pos is None and user_ids is None and item_seq is None and item_pos is None and item_ids is None:
            outputs = super().generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=generation_config, **kwargs)
        else:
            inputs_embeds = self.get_input_embeds(input_ids, user_pos, user_ids, item_seq, item_pos, item_ids)
            outputs = super().generate(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, generation_config=generation_config, **kwargs)
        return outputs