import os
import math
import torch

import numpy as np
from FlagEmbedding import BGEM3FlagModel
from Levenshtein import distance
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_from_disk
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_polynomial_decay_schedule_with_warmup

from train_utils.dataset import SFTDataset, Train_task_group_mapping, Val_task_group_mapping, Test_task_group_mapping
from train_utils.loss import CrossEntropyLoss_e
from train_utils.metrics import Metrics
from train_utils.model import BaseModel
from train_utils.processor import FastPrefixConstrainedLogitsProcessor
from train_utils.utils import *


class SFTTrainer(nn.Module):
    def __init__(self, args):
        super(SFTTrainer, self).__init__()
        self.get_scope_mask = None
        self.args = args
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps
        )
        self.args.process_index = self.accelerator.process_index
        self.args.num_processes = self.accelerator.num_processes
        set_seed(self.args.seed)
        self.args.gpu = self.args.gpu or self.accelerator.device
        self.args.is_main_process = self.accelerator.is_main_process

        self.data = {
            'category': load_json(args.data_path + 'category.jsonl'),
            'metas': load_json(args.data_path + 'metas.jsonl'),
            'sequential': load_json(args.data_path + 'sequential.jsonl'),
            'share_chat_gpt': load_pickle('data/share_chat_gpt2.pickle'),
        }

        code_info = load_item_code_mapping(self.args.data_path)
        self.args.item_code_tokens = []
        if code_info:
            self.args.item_code_tokens = code_info.get('token_vocab', [])
            for item_id, token_seq in code_info.get('item_seq', {}).items():
                if item_id in self.data['metas']:
                    self.data['metas'][item_id][ITEM_CODE_FIELD] = token_seq
            if self.args.is_main_process:
                covered_items = sum(1 for _ in self.data['metas'].values() if ITEM_CODE_FIELD in _)
                print(f"Loaded item code mapping for {covered_items} items "
                      f"({len(self.args.item_code_tokens)} unique tokens, "
                      f"{code_info.get('missing', 0)} missing indices).")
                
        self.item_emb = self.create_embeddings() if self.args.train_stage in ['SFT_Embedding', 'SFT_Embedding_Test'] else None
        self.actor = BaseModel(args=self.args, device=self.args.gpu, item_emb=self.item_emb)

        print(self.args.gpu)
        if self.accelerator.is_main_process:
            print(args)
            self.actor.print_trainable_parameters()

        self.start_epoch = self.actor.load_parameters(self.args.SFT_load)

        self.sft_loss_fct = CrossEntropyLoss(reduction='none')
        self.sft_loss_e = CrossEntropyLoss_e(gamma=self.args.fl_gamma)

    def get_optimizer(self, params, filter_by_requires_grad=True, group_wd_params=True):
        def separate_weight_decay_params(p):
            wd_p, no_wd_p = [], []
            for _ in p:
                param_list = no_wd_p if _.ndim < 2 else wd_p
                param_list.append(_)
            return wd_p, no_wd_p

        if filter_by_requires_grad:
            params = list(filter(lambda _: _.requires_grad, params))

        if group_wd_params and self.args.weight_decay > 0:
            wd_params, no_wd_params = separate_weight_decay_params(params)

            params = [
                {'params': wd_params},
                {'params': no_wd_params, 'weight_decay': 0},
            ]

        if self.args.weight_decay == 0:
            return Adam(params,
                        lr=self.args.lr,
                        betas=(self.args.adam_beta1, self.args.adam_beta2),
                        eps=self.args.adam_eps)

        return AdamW(params,
                     lr=self.args.lr,
                     weight_decay=self.args.weight_decay,
                     betas=(self.args.adam_beta1, self.args.adam_beta2),
                     eps=self.args.adam_eps)

    @property
    def device(self):
        return self.args.gpu

    @property
    def tokenizer(self):
        return self.actor.tokenizer

    @torch.no_grad()
    @eval_decorator
    def create_embeddings(self, batch_size=256, max_length=1024):
        if self.args.embedding_model is None:
            self.args.embedding_model = self.args.backbone
            return self.create_embeddings_()

        field_name = f'{self.args.embedding_model}_emb'
        model = None
        has_updated = False

        item_ids = [_ for _ in self.data['metas']]
        for i in range(0, len(item_ids), batch_size):
            batch_item_ids = item_ids[i:i + batch_size]
            if all([field_name in self.data['metas'][_] for _ in batch_item_ids]):
                continue

            model = BGEM3FlagModel(self.args.embedding_model, use_fp16=True,
                                   device=self.args.gpu) if model is None else model
            has_updated = True
            print(f'embedding: Batch {i // batch_size + 1}/{(len(item_ids) + batch_size - 1) // batch_size}')
            batch_item_titles = [self.data['metas'][_id][self.args.item_index] for _id in batch_item_ids]
            batch_item_descriptions = [self.data['metas'][_id]['description'] for _id in batch_item_ids]
            batch_embeddings = model.encode(
                [f"{t}\n{des}" for t, des in zip(batch_item_titles, batch_item_descriptions)],
                batch_size=batch_size,
                max_length=max_length
            )['dense_vecs'].tolist()

            for _id, emb in zip(batch_item_ids, batch_embeddings):
                self.data['metas'][_id][field_name] = emb

        del model
        if has_updated:
            save_json(self.data['metas'], os.path.join(self.args.data_path, 'metas.jsonl'))

        item_emb = np.stack([v[field_name] for k, v in self.data['metas'].items()])
        item_emb = torch.tensor(item_emb, dtype=torch.bfloat16, device=self.device)
        for idx, _ in enumerate(self.data['metas']):
            self.data['metas'][_]['emb_idx'] = idx
        return item_emb

    def create_embeddings_(self, batch_size=8, max_length=1024):
        field_name = f'{self.args.backbone}_emb'
        model = None
        tokenizer = AutoTokenizer.from_pretrained(self.args.backbone)
        tokenizer.pad_token = '<|reserved_special_token_250|>'
        tokenizer.pad_token_id = 128255
        eot_token = "<|eot_id|>"
        eot_token_id = tokenizer.convert_tokens_to_ids(eot_token)
        tokenizer.eos_token = eot_token
        tokenizer.eos_token_id = eot_token_id
        has_updated = False

        item_ids = [_ for _ in self.data['metas']]
        for i in range(0, len(item_ids), batch_size):
            batch_item_ids = item_ids[i:i + batch_size]
            if all([field_name in self.data['metas'][_] for _ in batch_item_ids]):
                continue

            model = AutoModelForCausalLM.from_pretrained(self.args.backbone, torch_dtype=torch.bfloat16,
                                                         device_map=self.args.gpu) if model is None else model
            has_updated = True
            print(f'embedding: Batch {i // batch_size + 1}/{(len(item_ids) + batch_size - 1) // batch_size}')
            batch_item_titles = [self.data['metas'][_id][self.args.item_index] for _id in batch_item_ids]
            batch_item_descriptions = [self.data['metas'][_id]['description'] for _id in batch_item_ids]

            input_data = side_tokenizer([f"{t}\n{des}" for t, des in zip(batch_item_titles, batch_item_descriptions)],
                                        'left', tokenizer, padding=True, truncation=True,
                                        max_length=max_length, return_tensors='pt').to(self.args.gpu).data

            outputs = model.forward(output_hidden_states=True, **input_data)
            batch_embeddings = outputs.hidden_states[-1].to(dtype=torch.float32).mean(dim=1).detach().cpu().numpy()

            for _id, emb in zip(batch_item_ids, batch_embeddings):
                self.data['metas'][_id][field_name] = emb

        del model
        if has_updated:
            save_json(self.data['metas'], os.path.join(self.args.data_path, 'metas.jsonl'))

        item_emb = np.stack([v[field_name] for k, v in self.data['metas'].items()])
        item_emb = torch.tensor(item_emb, dtype=torch.bfloat16, device=self.device)
        for idx, _ in enumerate(self.data['metas']):
            self.data['metas'][_]['emb_idx'] = idx
        return item_emb

    def SFT_loss_edit(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        if self.args.use_scope_mask:
            scope_mask = self.get_scope_mask(labels)
            neg_inf = float('-inf')
            shift_scope_mask = scope_mask[..., 1:, :].contiguous()
            shift_logits[shift_scope_mask] = neg_inf
            if self.args.loss_type == 1:
                loss = self.sft_loss_e.forward_1(shift_logits.view(-1, self.actor.model_config.vocab_size),
                                                 shift_labels.view(-1), shift_scope_mask.view(-1, self.actor.model_config.vocab_size))
            elif self.args.loss_type == 2:
                loss = self.sft_loss_e.forward_2(shift_logits.view(-1, self.actor.model_config.vocab_size),
                                                 shift_labels.view(-1), shift_scope_mask.view(-1, self.actor.model_config.vocab_size))
            elif self.args.loss_type == 3:
                loss = self.sft_loss_e.forward_3(shift_logits.view(-1, self.actor.model_config.vocab_size),
                                                 shift_labels.view(-1))

            else:
                raise NotImplementedError
            loss = loss.view(-1)
        else:
            loss = self.sft_loss_fct(shift_logits.view(-1, self.actor.model_config.vocab_size), shift_labels.view(-1))
            loss = loss.view(labels.shape[0], -1)
            loss = loss.sum(dim=1) / (shift_labels != -100).sum(dim=1)  # [bs]

        return loss

    def SFTEmbedding_train(self):
        TaskTemplate = {_: Train_task_group_mapping[_] for _ in self.args.SFT_train_tasks.split(',')}
        TaskNum = {_: 1 for _ in self.args.SFT_train_tasks.split(',')}
        ValTaskTemplate = {_: Val_task_group_mapping[_.split('_')[0]] for _ in self.args.SFT_val_tasks.split(',')}
        ValTaskNum = {_: 1 for _ in self.args.SFT_val_tasks.split(',')}

        with self.accelerator.main_process_first():
            train_data = SFTDataset(self.args, TaskTemplate, TaskNum, self.data, self.tokenizer, 'train')
            val_data = SFTDataset(self.args, ValTaskTemplate, ValTaskNum, self.data, self.tokenizer, 'val')

        train_loader = DataLoader(train_data, batch_size=self.args.batch_size, shuffle=True, collate_fn=train_data.collate_fn)
        val_loader = DataLoader(val_data, batch_size=self.args.val_batch_size, shuffle=False, collate_fn=val_data.collate_fn, drop_last=False)

        SFT_optim = self.get_optimizer(self.actor.actor_parameters)
        batch_per_epoch = len(train_loader)
        step_total = batch_per_epoch * (self.args.epoch - self.start_epoch) // self.args.gradient_accumulation_steps
        warmup_iters = int(step_total * self.args.warmup_ratio)
        SFT_lr_scheduler = get_polynomial_decay_schedule_with_warmup(SFT_optim, warmup_iters, step_total, power=2.0)

        warp_actor_critic, SFT_optim, train_loader, val_loader, SFT_lr_scheduler = self.accelerator.prepare(
            self.actor, SFT_optim, train_loader, val_loader, SFT_lr_scheduler
        )

        writer = None
        if self.accelerator.is_main_process:
            name = self.args.output.split('snap/')[-1]
            writer = SummaryWriter(log_dir=f'logs/SFT_train/{self.args.SFT_train_tasks}/{name}', flush_secs=30)

        best_val_loss = float('inf')
        for epoch in range(self.start_epoch + 1, self.args.epoch + 1):
            task_loss = {_: 0.0 for _ in train_data.task_num}
            task_emb_loss = {_: 0.0 for _ in train_data.task_num}
            task_count = {_: 1e-10 for _ in train_data.task_num}

            warp_actor_critic.train()
            pbar = tqdm(total=len(train_loader), ncols=210, disable=not self.accelerator.is_local_main_process)

            for step_i, batch in enumerate(train_loader):
                with self.accelerator.accumulate(warp_actor_critic):
                    input_data = batch['complete_text_data']
                    if self.accelerator.is_main_process and step_i % 10000 == 0:
                        print(batch['infer_text'][0])
                        print(batch['output_texts'][0])
                        print(input_data['input_ids'][0])
                        print(batch['complete_label_ids'][0])

                    results = warp_actor_critic.forward(scope='actor', **input_data)
                    emb_labels = torch.tensor(
                        [__ for i, _ in enumerate(batch['output_field_data']) for __ in _['emb_idx_list'][:results['embeddings'][i].shape[0]]],
                        dtype=torch.long, device=self.device
                    )  # [n]
                    embeddings = torch.concat(results['embeddings'])  # [n, d]
                    similarities = self.actor.embedding_similarity(embeddings)
                    emb_loss = self.sft_loss_fct(similarities, emb_labels).mean()

                    lm_labels = batch['complete_label_ids']
                    lm_loss = self.SFT_loss_edit(results.logits, lm_labels)
                    loss = self.args.emb_alpha * emb_loss + 1 * lm_loss

                    self.accelerator.backward(loss.mean())

                    if self.accelerator.sync_gradients:
                        # print({n: p.grad.abs().max().item() for n, p in self.actor_critic.actor_named_parameters.items()})
                        if self.args.clip_grad_norm > 0:
                            total_norm = self.accelerator.clip_grad_norm_(SFT_optim.param_groups[0]['params'], self.args.clip_grad_norm)

                    SFT_optim.step()
                    SFT_lr_scheduler.step()
                    SFT_optim.zero_grad()

                    for idx, task in enumerate(batch['task']):
                        task_loss[task] += float(lm_loss.item())
                        task_emb_loss[task] += float(emb_loss.item())
                        task_count[task] += 1

                if self.accelerator.sync_gradients:
                    lm_losses = torch.tensor([_ for _ in task_loss.values()], device=self.accelerator.device)
                    emb_losses = torch.tensor([_ for _ in task_emb_loss.values()], device=self.accelerator.device)
                    counts = torch.tensor([_ for _ in task_count.values()], device=self.accelerator.device)
                    lm_losses = self.accelerator.reduce(lm_losses)  # [task_num]
                    emb_losses = self.accelerator.reduce(emb_losses)  # [task_num]
                    counts = self.accelerator.reduce(counts)  # [task_num]
                    if self.accelerator.is_main_process:
                        for idx, task in enumerate(list(task_loss.keys())):
                            writer.add_scalars(f'training/{task}_LM_Loss', {f'epoch{epoch}': lm_losses[idx] / counts[idx]}, counts[idx])
                            writer.add_scalars(f'training/{task}_EMB_Loss', {f'epoch{epoch}': emb_losses[idx] / counts[idx]}, counts[idx])
                        ShareChatGPT_mask = torch.tensor(
                            [1.0 if _ != 'ShareChatGPT' else 0.0 for _ in task_loss.keys()],
                            device=self.accelerator.device
                        )
                        writer.add_scalars('training/All_Loss',
                                           {f'epoch{epoch}': float(masked_mean((lm_losses + emb_losses) / counts, ShareChatGPT_mask))}, step_i)
                        desc_str = f'E{epoch} | ShareGPT_Idx {train_data.share_chat_gpt_idx} | LR {SFT_lr_scheduler.get_lr()[0]:.4f}' \
                                   f' | {" | ".join([f"{task}_LM: {lm_losses[idx] / counts[idx]:.4f}" for idx, task in enumerate(list(task_loss.keys()))])}' \
                                   f' | {" | ".join([f"{task}_EMB: {emb_losses[idx] / counts[idx]:.4f}" for idx, task in enumerate(list(task_loss.keys()))])}'
                        pbar.set_description(desc_str, refresh=False)
                        pbar.update(self.args.gradient_accumulation_steps)

            pbar.close()
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self.actor.save_parameters("Epoch%02d" % epoch)
            if epoch < self.args.val_epoch:
                continue
            val_loss = self.SFT_evl_inference(epoch, val_loader, writer)
            if self.accelerator.is_main_process and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.actor.save_parameters(f"BEST_EVAL_LOSS_E{epoch}")

    def SFT_train(self):
        TaskTemplate = {_: Train_task_group_mapping[_] for _ in self.args.SFT_train_tasks.split(',')}
        TaskNum = {_: 1 for _ in self.args.SFT_train_tasks.split(',')}
        ValTaskTemplate = {_: Val_task_group_mapping[_.split('_')[0]] for _ in self.args.SFT_val_tasks.split(',')}
        ValTaskNum = {_: 1 for _ in self.args.SFT_val_tasks.split(',')}
        with self.accelerator.main_process_first():
            train_data = SFTDataset(self.args, TaskTemplate, TaskNum, self.data, self.tokenizer, 'train')
            val_data = SFTDataset(self.args, ValTaskTemplate, ValTaskNum, self.data, self.tokenizer, 'val')
            self.get_scope_mask = train_data.get_scope_mask

        train_loader = DataLoader(train_data, batch_size=self.args.batch_size, shuffle=True, collate_fn=train_data.collate_fn)
        val_loader = DataLoader(val_data, batch_size=self.args.val_batch_size, shuffle=False, collate_fn=val_data.collate_fn, drop_last=False)
        SFT_optim = self.get_optimizer(self.actor.actor_parameters)
        batch_per_epoch = len(train_loader)
        step_total = batch_per_epoch * (self.args.epoch - self.start_epoch) // self.args.gradient_accumulation_steps
        warmup_iters = int(step_total * self.args.warmup_ratio)
        # SFT_lr_scheduler = get_linear_schedule_with_warmup(SFT_optim, warmup_iters, step_total)
        SFT_lr_scheduler = get_polynomial_decay_schedule_with_warmup(SFT_optim, warmup_iters, step_total, power=2.0)
        warp_actor_critic, SFT_optim, train_loader, val_loader, SFT_lr_scheduler = self.accelerator.prepare(
            self.actor, SFT_optim, train_loader, val_loader, SFT_lr_scheduler
        )
        # print(SFT_actor_parameters)
        writer = None
        if self.accelerator.is_main_process:
            name = self.args.output.split('snap/')[-1]
            writer = SummaryWriter(log_dir=f'logs/SFT_train/{self.args.SFT_train_tasks}/{name}', flush_secs=30)
        if self.args.dry:
            self.SFT_evl_inference(self.start_epoch, val_loader, writer)
        best_val_loss = 1e10
        for epoch in range(self.start_epoch + 1, self.args.epoch + 1):
            # Train
            task_loss = {_: 0.0 for _ in train_data.task_num}
            task_count = {_: 1e-10 for _ in train_data.task_num}
            pbar = tqdm(total=len(train_loader), ncols=210, disable=not self.accelerator.is_local_main_process)
            self.train()
            for step_i, batch in enumerate(train_loader):
                with self.accelerator.accumulate(warp_actor_critic):
                    # print(self.actor_critic.actor_named_parameters)
                    # print(f'parameter {step_i}: ', self.actor_critic.actor_parameters[0].data.abs().max())
                    # self.accelerator.wait_for_everyone()
                    input_data = batch['complete_text_data']
                    if self.accelerator.is_main_process and step_i % 10000 == 0:
                        print(batch['infer_text'][0])
                        print(batch['output_texts'][0])
                        print(input_data['input_ids'][0])
                    labels = batch['complete_label_ids']
                    results = warp_actor_critic.forward(scope='actor', **input_data)
                    # loss = self.SFT_Loss(results.logits, labels, input_data, batch['task'])
                    loss = self.SFT_loss_edit(results.logits, labels)

                    for idx, task in enumerate(batch['task']):
                        task_loss[task] += float(loss[idx])
                        task_count[task] += 1
                    self.accelerator.backward(loss.mean())  # auto divide accumulate step, sync grad if arrive accumulate step
                    # print(f'grad {step_i}: ', self.actor_critic.actor_parameters[0].grad.abs().max())
                    # self.accelerator.wait_for_everyone()

                    if self.accelerator.sync_gradients:
                        # print(f'sync grad {step_i}: ', self.actor_critic.actor_parameters[0].grad.abs().max())
                        # self.accelerator.wait_for_everyone()
                        if self.args.clip_grad_norm > 0:
                            total_norm = self.accelerator.clip_grad_norm_(SFT_optim.param_groups[0]['params'], self.args.clip_grad_norm)
                            # writer.add_scalars('training/total_norm', {f'epoch{epoch}': float(total_norm)}, step_i)
                    SFT_optim.step()
                    SFT_lr_scheduler.step()
                    SFT_optim.zero_grad()

                if self.accelerator.sync_gradients:
                    # print(f'parameter {step_i}: ', {n: p.data.abs().max() for n, p in self.actor_critic.actor_named_parameters.items()})
                    # self.accelerator.wait_for_everyone()
                    losses = torch.tensor([_ for _ in task_loss.values()], device=self.accelerator.device)
                    counts = torch.tensor([_ for _ in task_count.values()], device=self.accelerator.device)
                    losses = self.accelerator.reduce(losses)  # [task_num]
                    counts = self.accelerator.reduce(counts)  # [task_num]
                    if self.accelerator.is_main_process:
                        for idx, task in enumerate(list(task_loss.keys())):
                            writer.add_scalars(f'training/{task}_Loss', {f'epoch{epoch}': losses[idx] / counts[idx]}, counts[idx])
                        ShareChatGPT_mask = torch.tensor(
                            [1.0 if _ != 'ShareChatGPT' else 0.0 for _ in task_loss.keys()],
                            device=self.accelerator.device
                        )
                        writer.add_scalars('training/All_Loss', {f'epoch{epoch}': float(masked_mean(losses / counts, ShareChatGPT_mask))}, step_i)
                        # desc_str = f'E{epoch} | LR {SFT_lr_scheduler.get_lr()[0]:.4f} | GN {total_norm.item():.4f}' \
                        desc_str = f'E{epoch} | ShareGPT_Idx {train_data.share_chat_gpt_idx} | LR {SFT_lr_scheduler.get_lr()[0]:.5f}' \
                                   f' | {" | ".join([f"{task}: {losses[idx] / counts[idx]:.4f}" for idx, task in enumerate(list(task_loss.keys()))])}'
                        pbar.set_description(desc_str, refresh=False)
                        pbar.update(self.args.gradient_accumulation_steps)

            pbar.close()
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self.actor.save_parameters("Epoch%02d" % epoch)
            if epoch < self.args.val_epoch:
                continue
            val_loss = self.SFT_evl_inference(epoch, val_loader, writer)
            if self.accelerator.is_main_process and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.actor.save_parameters(f"BEST_EVAL_LOSS_E{epoch}")

    @torch.no_grad()
    def SFT_evl_inference(self, epoch, val_loader, writer):
        torch.cuda.empty_cache()
        self.eval()
        metrics_dict = Metrics(self.args.SFT_val_tasks.split(','), self.args.topk, val_loader.dataset.category2item,
                               val_loader.dataset.title2item)
        logits_processors = [
            FastPrefixConstrainedLogitsProcessor(val_loader.dataset.item_prefix_tree.constrain_search_list, 1)
        ] if self.args.use_CBS else None
        pbar = tqdm(total=len(val_loader), ncols=200, disable=not self.accelerator.is_local_main_process)
        for step_i, batch in enumerate(val_loader):
            bs = len(batch['task'])
            input_data = batch['input_data']
            if self.accelerator.is_main_process and step_i % 1000 == 0:
                print(batch['infer_text'][0])
                print(batch['output_texts'][0])
                print(input_data['input_ids'][0])
            input_ids_length = input_data['input_ids'].shape[1]

            output_labels = [[__ for __ in _[-1].strip().split('\n')] for _ in batch['output_texts']]
            with torch.no_grad():
                if epoch == 0:
                    if self.args.train_stage in ['SFT_Embedding', 'SFT_Embedding_Test']:
                        output_title, _ = self.SFT_Embedding_generate('base', input_data['input_ids'], val_loader.dataset.idx2token_ids)
                    else:
                        output_ids = self.actor.generate(
                            scope='base',
                            **input_data,
                            logits_processor=logits_processors if self.args.use_CBS else None,
                            max_length=self.args.max_token_length + self.args.gen_max_length,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                        output_title = self.tokenizer.batch_decode(output_ids[:, input_ids_length:], skip_special_tokens=False if self.args.use_control_symbol else True)
                else:
                    if self.args.train_stage in ['SFT_Embedding', 'SFT_Embedding_Test']:
                        output_title, _ = self.SFT_Embedding_generate('actor', input_data['input_ids'], val_loader.dataset.idx2token_ids)
                    else:
                        output_ids = self.actor.generate(
                            scope='actor',
                            **input_data,
                            logits_processor=logits_processors if self.args.use_CBS else None,
                            max_length=self.args.max_token_length + self.args.gen_max_length,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                        output_title = self.tokenizer.batch_decode(output_ids[:, input_ids_length:], skip_special_tokens=False if self.args.use_control_symbol else True)

                if self.args.use_control_symbol:
                    output_title_list = [get_ctrl_item(_.strip()) for _ in output_title]
                else:
                    output_title_list = [
                        [__.strip() for __ in _.strip().split('\n')] for _ in output_title
                    ]
                    if self.args.idx:
                        output_title_list = [[rm_idx(__) for __ in _] for _ in output_title_list]
                if step_i % 10000 == 0:
                    print(output_title[0])
                    print(output_title_list[0])
            for i in range(bs):
                task = batch['task'][i]
                metrics_dict.add_sample(task, batch['input_field_data'][i], output_title_list[i], output_labels[i])
            pbar.update(1)
        pbar.close()

        _ndcg, _non_exist_rate, _repeat_rate, _correct_count = 0.0, 0.0, 0.0, 0.0
        for task in metrics_dict.metrics_dict:
            task_count = metrics_dict[task]['Count']
            recall = metrics_dict[task][f'Recall@{metrics_dict.topk}']
            ndcg = metrics_dict[task][f'NDCG@{metrics_dict.topk}']
            non_exist_rate = metrics_dict[task][f'NonExistRate@{metrics_dict.topk}']
            repeat_rate = metrics_dict[task][f'RepeatRate@{metrics_dict.topk}']
            correct_count = metrics_dict[task][f'CorrectCount@{metrics_dict.topk}']

            if task == 'SFTTestPersonalCategoryRate':
                category_rate_correct = metrics_dict[task][f'CategoryRateCorrect@{metrics_dict.topk}']
                log_d = torch.tensor(
                    [task_count, recall, ndcg, non_exist_rate, repeat_rate, correct_count, category_rate_correct],
                    device=self.accelerator.device)
            else:
                log_d = torch.tensor(
                    [task_count, recall, ndcg, non_exist_rate, repeat_rate, correct_count],
                    device=self.accelerator.device)
            log_d = self.accelerator.reduce(log_d)
            with self.accelerator.main_process_first():
                print(log_d)

            _ndcg += log_d[2] / log_d[0]
            _non_exist_rate += log_d[3] / log_d[0]
            _repeat_rate += log_d[4] / log_d[0]
            _correct_count += log_d[5] / log_d[0]

            if self.accelerator.is_main_process:
                writer.add_scalar(f'valuating/{task}_Recall', log_d[1] / log_d[0], epoch)
                writer.add_scalar(f'valuating/{task}_NDCG', log_d[2] / log_d[0], epoch)
                writer.add_scalar(f'valuating/{task}_NonExist_rate', log_d[3] / log_d[0], epoch)
                writer.add_scalar(f'valuating/{task}_Repeat_rate', log_d[4] / log_d[0], epoch)
                writer.add_scalar(f'valuating/{task}_Correct_count', log_d[5] / log_d[0], epoch)
                if task == 'RLHFPersonalCategoryRate':
                    writer.add_scalar(f'valuating/{task}_Category_rate_correct', log_d[6] / log_d[0], epoch)
        if self.accelerator.is_main_process:
            val_task_num = len(val_loader.dataset.task_num)
            writer.add_scalar(f'valuating/Total_NDCG', _ndcg / val_task_num, epoch)
            writer.add_scalar(f'valuating/Total_NonExist_rate', _non_exist_rate / val_task_num, epoch)
            writer.add_scalar(f'valuating/Total_Repeat_rate', _repeat_rate / val_task_num, epoch)
            writer.add_scalar(f'valuating/Total_Correct_count', _correct_count / val_task_num, epoch)
            metrics_dict.print()
            print(f'Epoch {epoch} | SFT_Val_NDCG: {_ndcg:.4f}\n')
        self.train()
        return -1 * _ndcg

    @torch.no_grad()
    def SFT_test(self):
        torch.cuda.empty_cache()
        self.eval()
        TestTaskTemplate = {self.args.SFT_test_task: Test_task_group_mapping[self.args.SFT_test_task.split('_')[0]]}
        TestTaskNum = {self.args.SFT_test_task: 1}
        test_data = SFTDataset(self.args, TestTaskTemplate, TestTaskNum, self.data, self.tokenizer, 'test')
        test_loader = DataLoader(test_data, batch_size=self.args.test_batch_size, shuffle=False,
                                 collate_fn=test_data.collate_fn, drop_last=False)

        metrics_dict = Metrics([self.args.SFT_test_task], self.args.topk, test_data.category2item, test_data.title2item)
        result_file = self.args.backbone + f'Result_{self.args.SFT_test_task}{"_CBS" if self.args.use_CBS else ""}_Top{self.args.topk}.jsonl'
        if self.args.SFT_load:
            result_file = self.args.SFT_load + \
                          f'_Result_{self.args.SFT_test_task}' + \
                          (f'_CBS{f"-{self.args.CBS_type}" if self.args.CBS_type > 1 else ""}' if self.args.use_CBS else '') + \
                          f'_Top{10}' + \
                          (f'_process_{str(self.accelerator.local_process_index)}' if self.accelerator.num_processes > 1 else '') + \
                          '.jsonl'

        num_beams = 1
        logits_processors = [
            FastPrefixConstrainedLogitsProcessor(test_data.item_prefix_tree.constrain_search_list, num_beams)
        ] if self.args.use_CBS else None
        with torch.no_grad():
            result = load_json(result_file) or []
            result = result[:-(len(result) % self.args.batch_size) if (len(result) % self.args.batch_size) != 0 else None]
            pbar = tqdm(total=len(test_loader), ncols=150, disable=not self.accelerator.is_local_main_process)
            for step_i, batch in enumerate(test_loader):
                bs = len(batch['task'])
                input_data = batch['input_data']
                if self.accelerator.is_main_process and step_i % 10000 == 0:
                    print(batch['infer_text'][0])
                    print(batch['output_texts'][0])
                    print(input_data['input_ids'][0])

                output_labels = [[__.strip() for __ in _[-1].strip().split('\n')] for _ in batch['output_texts']]
                if (step_i + 1) * self.args.test_batch_size <= len(result):
                    output_title_list = [_[1] for _ in result[step_i * self.args.test_batch_size: (step_i + 1) * self.args.test_batch_size]]
                else:
                    if self.args.train_stage in ['SFT_Embedding', 'SFT_Embedding_Test']:
                        output_title, _ = self.SFT_Embedding_generate('actor', input_data['input_ids'], test_data.idx2token_ids)
                    else:
                        input_ids_length = input_data['input_ids'].shape[1]
                        output_ids = self.actor.generate(
                            scope='actor',
                            **input_data,
                            logits_processor=logits_processors if self.args.use_CBS else None,
                            max_length=self.args.max_token_length + self.args.gen_max_length,
                            num_beams=num_beams,
                            num_return_sequences=1,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                        output_title = self.tokenizer.batch_decode(output_ids[:, input_ids_length:],
                                                                   skip_special_tokens=False if self.args.use_control_symbol else True)

                    try:
                        if self.args.use_control_symbol:
                            output_title_list = [get_ctrl_item(_) for _ in output_title]
                        else:
                            output_title_list = [[__.strip() for __ in _.strip().split('\n')] for _ in output_title]
                            if self.args.idx:
                                output_title_list = [[rm_idx(__) for __ in _] for _ in output_title_list]
                    except:
                        print(output_title_list)
                        raise Exception()

                    if step_i % 10000 == 0:
                        print(output_title[0])
                        print(output_title_list[0])

                    result += [d for d in zip(output_labels, output_title_list, output_title)]
                    if (step_i + 1) % 100 == 0 or (step_i + 1) == len(test_loader):
                        save_json(result, result_file)

                for i in range(bs):
                    metrics_dict.add_sample(batch['task'][i], batch['input_field_data'][i], output_title_list[i], output_labels[i])
                metrics_dict.print()
                pbar.update(1)
            pbar.close()
        self.train()

    def SFT_Embedding_generate(self, scope, input_ids, idx2tokens_ids):
        all_embed = [[] for i in range(input_ids.shape[0])]
        unfinished_sequence_flags = torch.ones(input_ids.shape[0], dtype=torch.bool, device=self.device)
        _input_ids = [_[_ != self.tokenizer.pad_token_id] for _ in input_ids]
        _input_ids_len = [_.shape[0] for _ in _input_ids]
        repeat_mask = torch.zeros((input_ids.shape[0], self.actor.item_emb.num_embeddings), dtype=torch.bool, device=self.device)
        while unfinished_sequence_flags.any():
            unfinished_sequence_indices = unfinished_sequence_flags.nonzero(as_tuple=True)[0]
            temp_input_ids = torch.nn.utils.rnn.pad_sequence(
                [_input_ids[i].flip(dims=[0]) for i in unfinished_sequence_indices],
                batch_first=True, padding_value=self.tokenizer.pad_token_id
            ).flip(dims=[1])
            bs, temp_input_ids_len = temp_input_ids.shape[0], temp_input_ids.shape[1]
            attention_mask = temp_input_ids.ne(self.tokenizer.pad_token_id).long()
            output = self.actor.generate(scope=scope, input_ids=temp_input_ids, attention_mask=attention_mask,
                                         return_dict_in_generate=True, output_hidden_states=True,
                                         max_length=self.args.max_token_length + self.args.gen_max_length,
                                         num_beams=1, num_return_sequences=1,
                                         eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.soi_token_id])
            sequences = output.sequences
            for _ in sequences:
                if len(_) >= self.args.max_token_length + self.args.gen_max_length:
                    _[-1] = self.tokenizer.eos_token_id
            _unfinished_sequence_indices_in_batch = (~(torch.eq(sequences[:, temp_input_ids_len:], self.tokenizer.eos_token_id).any(dim=1))).nonzero(as_tuple=True)[0]
            _finished_sequence_indices_in_batch = torch.eq(sequences[:, temp_input_ids_len:], self.tokenizer.eos_token_id).any(dim=1).nonzero(as_tuple=True)[0]
            _unfinished_sequence_indices = unfinished_sequence_indices[_unfinished_sequence_indices_in_batch]
            _finished_sequence_indices = unfinished_sequence_indices[_finished_sequence_indices_in_batch]

            hidden_states = output.hidden_states
            soi_positions = torch.eq(
                sequences[_unfinished_sequence_indices_in_batch, temp_input_ids_len:],
                self.tokenizer.soi_token_id
            ).nonzero(as_tuple=True)
            soi_indices = torch.ones_like(soi_positions[0], dtype=torch.long)
            soi_indices[:-1] = soi_positions[0][1:] - soi_positions[0][:-1]
            soi_indices = soi_indices.nonzero(as_tuple=True)[0]
            bs_indices = [_unfinished_sequence_indices_in_batch[i] for i in soi_positions[0][soi_indices].tolist()]
            seq_indices = soi_positions[1][soi_indices].tolist()
            last_hidden_states = [
                hidden_states[seq_idx][-1][bs_idx, -1:]
                for seq_idx, bs_idx in zip(seq_indices, bs_indices)
            ]
            if len(last_hidden_states) > 0:
                last_hidden_states = torch.concat(last_hidden_states, dim=0)
                embeddings = self.actor.actor_item_proj(last_hidden_states)  # [bs, hidden_size]
                similarities = self.actor.embedding_similarity(embeddings)
                _repeat_mask = torch.index_select(repeat_mask, dim=0, index=_unfinished_sequence_indices)
                similarities[_repeat_mask] = -torch.inf
                item_indices = torch.topk(similarities, k=1).indices.squeeze(dim=1).tolist()
                for bs_idx, unf_idx, item_idx, emb in zip(_unfinished_sequence_indices_in_batch, _unfinished_sequence_indices, item_indices, embeddings):
                    temp_sequence = sequences[bs_idx][temp_input_ids_len:]
                    temp_sequence = temp_sequence[temp_sequence != self.tokenizer.pad_token_id]
                    _input_ids[unf_idx] = torch.cat(
                        (_input_ids[unf_idx], temp_sequence, idx2tokens_ids[item_idx][1:])
                    )
                    repeat_mask[unf_idx, item_idx] = True
                    all_embed[unf_idx].append(emb)

            for bs_idx, f_idx in zip(_finished_sequence_indices_in_batch, _finished_sequence_indices):
                _input_ids[f_idx] = torch.cat(
                    (_input_ids[f_idx], sequences[bs_idx][temp_input_ids_len:])
                )
                unfinished_sequence_flags[f_idx] = False
        output_ids = [_[l:].tolist() for _, l in zip(_input_ids, _input_ids_len)]

        return self.tokenizer.batch_decode(output_ids), [torch.stack(_) if len(_) > 0 else [] for _ in all_embed]

    @torch.no_grad()
    def SFT_Embedding_indexing_test(self):
        torch.cuda.empty_cache()
        self.eval()
        TestTaskTemplate = {self.args.SFT_test_task: Test_task_group_mapping[self.args.SFT_test_task.split('_')[0]]}
        TestTaskNum = {self.args.SFT_test_task: 1}
        test_data = SFTDataset(self.args, TestTaskTemplate, TestTaskNum, self.data, self.tokenizer, 'test')
        test_loader = DataLoader(test_data, batch_size=self.args.test_batch_size, shuffle=False,
                                 collate_fn=test_data.collate_fn, drop_last=False)

        avg_similarities = None
        recall_10, ndcg_10 = torch.zeros(10, dtype=torch.float32), torch.zeros(10, dtype=torch.float32)
        recall_5, ndcg_5 = torch.zeros(10, dtype=torch.float32), torch.zeros(10, dtype=torch.float32)
        recall_10_gt, ndcg_10_gt = torch.zeros(10, dtype=torch.float32), torch.zeros(10, dtype=torch.float32)
        recall_5_gt, ndcg_5_gt = torch.zeros(10, dtype=torch.float32), torch.zeros(10, dtype=torch.float32)
        origin_recall_10, origin_ndcg_10 = 0.0, 0.0
        count = 0
        for step_i, batch in tqdm(enumerate(test_loader)):
            if count >= 1024:
                break
            if self.accelerator.is_main_process and step_i % 10000 == 0:
                print(batch['infer_text'][0])
                print(batch['output_texts'][0])
                print(batch['complete_text_data']['input_ids'][0])
                print(batch['complete_label_ids'][0])

            for _ in batch['output_field_data']:
                _['emb_idx_list'] = [test_data.metas[_].get('emb_idx') for _ in _['item_list']]

            results = self.actor.forward(scope='actor', **batch['complete_text_data'])
            embeddings = results['embeddings']
            # output_title, embeddings_g = self.SFT_Embedding_generate('actor', batch['input_data']['input_ids'], test_data.idx2token_ids)

            emb_labels = torch.tensor(
                [[__ for __ in _['emb_idx_list']] for i, _ in enumerate(batch['output_field_data']) if len(embeddings[i]) == self.args.topk],
                dtype=torch.long, device=self.device
            )  # [bs, top_k]
            embeddings = torch.stack([_ for i, _ in enumerate(embeddings) if len(embeddings[i]) == self.args.topk]) # [bs, top_k, d]
            embeddings_f32 = embeddings.to(dtype=torch.float32)
            inner_dot = torch.matmul(embeddings_f32, embeddings_f32.transpose(1, 2))  # [bs, top_k, top_k]
            norm1 = torch.norm(embeddings_f32, p=2, keepdim=True, dim=2)    # [bs, top_k, 1]
            emb_sim = ((inner_dot / norm1).transpose(1, 2) / norm1).transpose(1, 2)  # [bs, top_k, top_k]
            if avg_similarities is not None:
                avg_similarities = avg_similarities + torch.sum(emb_sim, dim=0)
            else:
                avg_similarities = torch.sum(emb_sim, dim=0)
            count += emb_sim.shape[0]

            item_sim = self.actor.embedding_similarity(embeddings)   # [bs, top_k, item_num]
            topk_idx = torch.topk(item_sim, k=10, dim=-1).indices       # [bs, top_k, 10]

            for _, ls in zip(topk_idx, emb_labels):
                for i, (__, l) in enumerate(zip(_, ls)):
                    indices = torch.nonzero(__ == l)
                    if len(indices) > 0:
                        recall_10[i] += 1
                        ndcg_10[i] += 1.0/math.log2(indices[0]+2)

                    indices5 = torch.nonzero(__[:5] == l)
                    if len(indices5) > 0:
                        recall_5[i] += 1
                        ndcg_5[i] += 1.0/math.log2(indices5[0]+2)

            for _, ls in zip(topk_idx, emb_labels):
                for i, __ in enumerate(_):
                    indices = torch.nonzero(__ == ls[0])
                    if len(indices) > 0:
                        recall_10_gt[i] += 1
                        ndcg_10_gt[i] += 1.0/math.log2(indices[0]+2)

                    indices5 = torch.nonzero(__[:5] == ls[0])
                    if len(indices5) > 0:
                        recall_5_gt[i] += 1
                        ndcg_5_gt[i] += 1.0/math.log2(indices5[0]+2)

            for _, ls in zip(topk_idx, emb_labels):
                for i, __ in enumerate(_):
                    indices1 = torch.nonzero(__[:1] == ls[0])
                    if len(indices1) > 0:
                        origin_recall_10 += 1
                        origin_ndcg_10 += 1.0 / math.log2(indices1[0]+2)
                        break


        print('recall_10: ', recall_10/count)
        print('ndcg_10: ', ndcg_10/count)
        print('recall_5: ', recall_5/count)
        print('ndcg_5: ', ndcg_5/count)

        print('recall_10_gt: ', recall_10_gt/count)
        print('ndcg_10_gt: ', ndcg_10_gt/count)
        print('recall_5_gt: ', recall_5_gt/count)
        print('ndcg_5_gt: ', ndcg_5_gt/count)
        print('origin_recall_10: ', origin_recall_10/count)
        print('origin_ndcg_10: ', origin_ndcg_10/count)

        matrix = (avg_similarities/count).detach().cpu().numpy()

        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis", linewidths=.5)

        plt.title("Heatmap of a k*k Tensor Matrix")
        plt.xlabel("Columns")
        plt.ylabel("Rows")

        plt.show()

    @torch.no_grad()
    def SFT_Embedding_MR_test(self):
        GSM8K_Q1 = '''Question: In 2004, there were 60 kids at a cookout. In 2005, half the number of kids came to the cookout as compared to 2004. In 2006, 2/3 as many kids came to the cookout as in 2005. How many kids came to the cookout in 2006?'''
        GSM8K_A1 = '''Let's think step by step.
        In 2005, 60/2=30 kids came to the cookout.
        In 2006, 30/3*2=20 kids came to the cookout.
        The answer is 20'''

        def vague_mapping(ts):
            for idx, __ in enumerate(ts):
                if __ in test_data.title2item:
                    continue
                for ___ in test_data.title2item:
                    if distance(__, ___) <= 3:
                        ts[idx] = ___
                        break

        def process_api_output(d):
            if f'{self.args.SFT_load}_output' not in d:
                return d
            if d[f'{self.args.SFT_load}_output'] == "":
                d[f'{self.args.SFT_test_task}_output_title_list'] = []
                return d
            if f'{self.args.SFT_test_task}_output_title_list' in d:
                return d

            raw_output = d[f'{self.args.SFT_load}_output']
            if self.args.use_control_symbol:
                ts = get_ctrl_item(raw_output)
            else:
                ts = [_.strip() for _ in raw_output.strip().split('\n')]
                ts = [rm_idx(_) if self.args.idx else _ for _ in ts]

            vague_mapping(ts)
            d[f'{self.args.SFT_test_task}_output_title_list'] = ts

            return d

        def process_dataset_hf(data_list):
            if len(data_list) == 0:
                return

            bs = self.args.test_batch_size
            for i in tqdm(range(0, len(data_list), bs)):
                input_texts = [
                    d['input_text'] if 'input_text' in d else
                    process_train_sample([GSM8K_Q1] + d['input_texts'], [GSM8K_A1] + d['output_texts'], self.tokenizer)[3]
                    for d in data_list[i: i + bs]
                ]
                input_data = side_tokenizer(input_texts, 'left', self.tokenizer, padding=True, truncation=True,
                                            max_length=self.args.max_token_length, return_tensors='pt').to(device=self.args.gpu).data
                output_texts, _ = self.SFT_Embedding_generate(
                    scope='actor',
                    input_ids=input_data['input_ids'],
                    idx2tokens_ids=test_data.idx2token_ids
                )

                for d, o in zip(data_list[i: i + bs], output_texts):
                    d[f'{self.args.SFT_load}_output'] = o.split(self.tokenizer.eos_token)[0]

                if i == 0:
                    print(output_texts[0])

                input_texts_gsm8k = [
                    d['input_text'] if 'input_text' in d else
                    process_train_sample(
                        [GSM8K_Q1] + d['input_texts'] + [d['gsm8k_question']],
                        [GSM8K_A1] + d['output_texts'][:-1] + d[f'{self.args.SFT_load}_output'].split(self.tokenizer.eos_token)[:1] + [d['gsm8k_answer']],
                        self.tokenizer
                    )[3]
                    for d in data_list[i: i + bs]
                ]
                if i == 0:
                    print(input_texts_gsm8k[0])

                input_data = side_tokenizer(input_texts_gsm8k, 'left', self.tokenizer, padding=True, truncation=True,
                                            max_length=self.args.max_token_length, return_tensors='pt').to(device=self.args.gpu).data
                output_texts_gsm8k, _ = self.SFT_Embedding_generate(
                    scope='actor',
                    input_ids=input_data['input_ids'],
                    idx2tokens_ids=test_data.idx2token_ids
                )
                for d, o in zip(data_list[i: i + bs], output_texts_gsm8k):
                    d[f'{self.args.SFT_load}_output_gsm8k'] = o.split(self.tokenizer.eos_token)[0]

                if i == 0:
                    print(output_texts_gsm8k[0])

        torch.cuda.empty_cache()
        self.eval()
        TestTaskTemplate = {self.args.SFT_test_task: Test_task_group_mapping[self.args.SFT_test_task.split('_')[0]]}
        TestTaskNum = {self.args.SFT_test_task: 1}
        test_data = SFTDataset(self.args, TestTaskTemplate, TestTaskNum, self.data, self.tokenizer, 'test')
        metrics_dict = Metrics([self.args.SFT_test_task], self.args.topk, test_data.category2item, test_data.title2item)
        GSM8K_dataset = load_from_disk('data/dataset/general_task/GSM8K/')
        _test_data_list = [_ for _ in test_data][:len(GSM8K_dataset["test"])]
        for d, gsm8k_d in zip(_test_data_list, GSM8K_dataset["test"]):
            d['gsm8k_question'] = gsm8k_d['question']
            d['gsm8k_answer'] = gsm8k_d['answer']

        dataset = self.args.data_path.strip('/').split('/')[-1]
        print(_test_data_list[1]['input_texts'] if 'input_texts' in _test_data_list[1] else _test_data_list[1]['input_text'])
        result_file = os.path.join(self.args.SFT_load+f'_{dataset}_{self.args.SFT_test_task}_Top10{f"_CBS{self.args.CBS_type}" if self.args.use_CBS else ""}_test_MR_Result.jsonl')
        test_data_list = (load_json(result_file) or [])[:len(GSM8K_dataset["test"])]
        if test_data_list and len(test_data_list) == len(_test_data_list):
            for _, __ in zip(test_data_list, _test_data_list):
                _.update(__)
        else:
            test_data_list = _test_data_list

        remain_test_data_list = [_ for _ in test_data_list if f'{self.args.SFT_load}_output' not in _ or f'{self.args.SFT_load}_output_gsm8k' not in _][:]
        process_dataset_hf(remain_test_data_list)

        if len(remain_test_data_list) > 0:
            save_json(test_data_list, result_file)

        test_data_list = [process_api_output(_) for _ in test_data_list]

        for step_i, example in tqdm(enumerate(test_data_list)):
            if f'{self.args.SFT_test_task}_output_title_list' not in example or len(example[f'{self.args.SFT_test_task}_output_title_list']) == 0:
                continue
            if self.args.use_control_symbol:
                output_label = [example['output_texts'][-1]]
            else:
                output_label = [_.strip() for _ in example['output_texts'][-1].strip().split('\n')]
                output_label = [rm_idx(_) if self.args.idx else _ for _ in output_label]
            metrics_dict.add_sample(example['task'], example['input_field_data'], example[f'{self.args.SFT_test_task}_output_title_list'], output_label, vague_mapping=False)

        metrics_dict.print()

        acc_res = []
        ctl_res = []
        for example in test_data_list:
            pred = gsm8K_clean_answer(example[f'{self.args.SFT_load}_output_gsm8k'])
            label = example["gsm8k_answer"]
            acc = gsm8K_is_correct(pred, label)
            acc_res.append(acc)
            ctl_res.append(example[f'{self.args.SFT_load}_output_gsm8k'].count('<SOI>') + example[f'{self.args.SFT_load}_output_gsm8k'].count('<EOI>'))

        print("GSM8K acc: ", np.mean(acc_res))
        print("CSN_{R2}^{n=0}: ", (len(ctl_res) - len(np.nonzero(ctl_res)[0])) / len(ctl_res))

        item_count1_res = [1 if len(example[f'{self.args.SFT_test_task}_output_title_list']) == self.args.topk else 0 for example in test_data_list]
        print(f"CSN_{{R1}}^{{n={self.args.topk}}}: ", sum(item_count1_res) / len(item_count1_res))

        if len(remain_test_data_list) > 0:
            save_json(test_data_list, result_file)

    def SFT_adapter_merge(self):
        model = self.actor.lora_model.merge_and_unload(progressbar=True)
        model.save_pretrained(f'{self.args.output}SFT_Epoch{self.start_epoch:02d}', safe_serialization=True)
        self.tokenizer.save_pretrained(f'{self.args.output}SFT_Epoch{self.start_epoch:02d}')


if __name__ == "__main__":
    pass
