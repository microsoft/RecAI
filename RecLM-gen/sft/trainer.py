# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os.path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import StoppingCriteriaList, MaxLengthCriteria
from sft.dataset import SFTDataset, Train_task_group_mapping, Val_task_group_mapping
from base.trainer import BaseTrainer
from base.dataset import BaseDataset
from utils.metrics import Metrics
from utils.tools import load_pickle, sync_dict, eval_decorator, vague_map, rm_idx


class SFTTrainer(BaseTrainer):
    def __init__(self, args):
        super(SFTTrainer, self).__init__(args)
        self.title2item = None
        self.item2category = None
        self.metas = None
        self.category2item = None

        self.writer = None
        if self.accelerator.is_main_process:
            self.writer = SummaryWriter(log_dir=os.path.join('logs', self.args.output_path), flush_secs=30)

        self.start_epoch = self.actor_critic.load_parameters(self.args.SFT_load)
        self.dataset_prepare()

    def dataset_prepare(self):
        self.category2item = load_pickle(self.args.data_path + 'category.pickle')
        self.metas = load_pickle(self.args.data_path + 'meta.pickle')
        self.item2category = {}
        for c in self.category2item:
            for i in self.category2item[c]:
                if self.item2category.get(i) is None:
                    self.item2category[i] = []
                self.item2category[i].append(c)
        self.title2item = {}
        for _ in self.metas:
            if self.title2item.get(self.metas[_][self.args.item_index]) is None:
                self.title2item[self.metas[_][self.args.item_index]] = []
            self.title2item[self.metas[_][self.args.item_index]].append(_)
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                print(f'computing SFT train, val datum info')
            if self.args.train_data_file:             # from generated static dataset file
                self.train_data = BaseDataset(self.args, self.tokenizer, 'train')
            if self.args.val_data_file:
                self.val_data = BaseDataset(self.args, self.tokenizer, 'val')
            if self.args.data_path and not self.train_data and not self.val_data:           # dynamic fetched from raw dataset
                TaskTemplate = {_: Train_task_group_mapping[_] for _ in self.args.SFT_train_tasks.split(',')}
                TaskNum = {_: 1 for _ in self.args.SFT_train_tasks.split(',')}
                ValTaskTemplate = {_: Val_task_group_mapping[_.split('_')[0]] for _ in self.args.SFT_val_tasks.split(',')}
                ValTaskNum = {_: 1 for _ in self.args.SFT_val_tasks.split(',')}
                data = {
                    'category2item': self.category2item,
                    'item2category': self.item2category,
                    'metas': self.metas,
                    'title2item': self.title2item,
                    'sequential': load_pickle(self.args.data_path + 'sequential.pickle'),
                    'ranking_candidate': load_pickle(self.args.data_path + 'ranking_candidate.pickle'),
                    'share_chat_gpt': load_pickle('data/dataset/share_chat_gpt2.pickle'),
                }
                self.train_data = SFTDataset(self.args, TaskTemplate, TaskNum, data, self.tokenizer, 'train')
                self.val_data = SFTDataset(self.args, ValTaskTemplate, ValTaskNum, data, self.tokenizer, 'val')

        self.train_loader = DataLoader(self.train_data, batch_size=self.args.batch_size, shuffle=True, collate_fn=self.train_data.collate_fn)
        self.val_loader = DataLoader(self.val_data, batch_size=self.args.val_batch_size, shuffle=False, collate_fn=self.val_data.collate_fn, drop_last=False)

        self.prepare(self.train_loader, self.val_loader)

    def SFT_train(self):
        best_val_loss = float('inf')
        if self.args.dry:
            best_val_loss = self.SFT_val_inference(self.start_epoch)
        for epoch in range(self.start_epoch+1, self.args.epoch+1):
            if hasattr(self.train_data, "set_epoch"):
                self.train_data.set_epoch(epoch-1)
            task_loss = {_: 0.0 for _ in self.args.SFT_train_tasks.split(',')}
            task_count = {_: 1e-10 for _ in self.args.SFT_train_tasks.split(',')}
            pbar = tqdm(total=len(self.train_loader), ncols=210, disable=not self.accelerator.is_local_main_process)
            self.train()
            for step_i, batch in enumerate(self.train_loader):
                if self.accelerator.is_main_process and step_i % 10000 == 0:
                    print(batch['complete_text'][0])
                    print(batch['complete_text_data']['input_ids'][0])
                batch = self.SFT_train_batch(batch)

                # record
                for idx, task in enumerate(batch['task']):
                    task_loss[task] += float(batch['loss'][idx])
                    task_count[task] += 1

                # logging
                if self.accelerator.sync_gradients:
                    _task_loss = sync_dict(self.accelerator, task_loss)
                    _task_count = sync_dict(self.accelerator, task_count)
                    _task_loss = {task: _task_loss[task] / _task_count[task] for task in _task_loss}
                    if self.accelerator.is_main_process:
                        for task in _task_loss:
                            self.writer.add_scalars(f'training/{task}_Loss', {f'epoch{epoch}': _task_loss[task]}, _task_count[task])
                        desc_str = f'E{epoch} | LR {self.lr_scheduler.get_lr()[0]:.4f}' \
                                   f' | {" | ".join([f"{task}: {_task_loss[task]:.4f}" for task in _task_loss])}'
                        _task_loss['ShareChatGPT'], _task_count['ShareChatGPT'] = 0, 0
                        self.writer.add_scalars('training/All_Loss', {f'epoch{epoch}': sum(list(_task_loss.values()))/len(_task_loss)}, step_i)
                        pbar.set_description(desc_str, refresh=False)
                        pbar.update(self.args.gradient_accumulation_steps)

            pbar.close()
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self.actor_critic.save_parameters(f"Epoch{epoch:02d}")
            if epoch < self.args.val_epoch:
                continue
            val_loss = self.SFT_val_inference(epoch)
            if val_loss < best_val_loss and self.accelerator.is_main_process:
                best_val_loss = val_loss
                self.actor_critic.save_parameters("BEST_EVAL_LOSS")

    @torch.no_grad()
    @eval_decorator
    def SFT_val_loss(self, epoch: int):
        torch.cuda.empty_cache()
        task_loss = {_: 0.0 for _ in self.args.SFT_val_tasks.split(',')}
        task_count = {_: 1e-10 for _ in self.args.SFT_val_tasks.split(',')}
        pbar = tqdm(total=len(self.val_loader), ncols=200, disable=not self.accelerator.is_main_process)
        for step_i, batch in enumerate(self.val_loader):
            if self.accelerator.is_main_process and step_i % 10000 == 0:
                print(batch['complete_text'][0])
                print(batch['complete_text_data']['input_ids'][0])
            labels = batch['complete_label_ids']
            results = self.actor_critic.forward(scope=self.actor_critic.actor_lora_scope, **batch['complete_text_data'])
            loss = self.SFT_Loss(results.logits, labels).detach()

            # record
            for idx, task in enumerate(batch['task']):
                task_loss[task] += (float(loss[idx]))
                task_count[task] += 1
            pbar.update(1)

        # logging
        task_loss = sync_dict(self.accelerator, task_loss)
        task_count = sync_dict(self.accelerator, task_count)
        task_loss = {task: task_loss[task]/task_count[task] for task in task_loss}
        val_loss = sum(list(task_loss.values()))/len(task_loss)
        if self.accelerator.is_main_process:
            print(f'Epoch {epoch} | {" | ".join([f"Val_{task}_Loss: {task_loss[task]:.4f}" for task in task_loss])}')
            print(f'Epoch {epoch} | SFT_Val_Loss: {val_loss:.4f}\n')
            self.writer.add_scalars(f'valuating', {f'{task}_Loss': task_loss[task] for task in task_loss}, epoch)
            self.writer.add_scalars(f'valuating', {'total_Loss': val_loss}, epoch)
        return val_loss

    @torch.no_grad()
    @eval_decorator
    def SFT_val_inference(self, epoch: int):
        torch.cuda.empty_cache()
        stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=self.args.max_token_length + self.args.gen_max_length)])
        metrics_dict = Metrics(self.args.SFT_val_tasks.split(','), self.args.topk, self.category2item, self.title2item, self.accelerator)
        pbar = tqdm(total=len(self.val_loader), ncols=200, disable=not self.accelerator.is_main_process)
        for step_i, batch in enumerate(self.val_loader):
            if self.accelerator.is_main_process and step_i % 1000 == 0:
                print(batch['input_text'][0])
                print(batch['input_data']['input_ids'][0])
            bs = len(batch['task'])
            input_ids_length = batch['input_data']['input_ids'].shape[1]
            if epoch == 0:
                output_ids = self.actor_critic.base_model.greedy_search(**batch['input_data'], stopping_criteria=stopping_criteria)
            else:
                output_ids = self.actor_critic.actor_model.greedy_search(**batch['input_data'], stopping_criteria=stopping_criteria)

            # process output
            output_text = self.tokenizer.batch_decode(output_ids[:, input_ids_length:], skip_special_tokens=True)
            output_title_list = [
                [__.strip() for __ in _.strip().split('\n')] for _ in output_text
            ]
            output_labels = [[__ for __ in _.strip().split('\n')] for _ in batch['output_text']]
            if self.args.idx:
                output_title_list = [[rm_idx(__) for __ in _] for _ in output_title_list]
            if self.args.vague_mapping:
                output_title_list = [vague_map(_, self.title2item) for _ in output_title_list]

            # record
            for i in range(bs):
                metrics_dict.add_sample(batch['task'][i], batch['input_field_data'][i], output_title_list[i], output_labels[i])
            pbar.update(1)

        # logging
        self.accelerator.wait_for_everyone()
        sync_metrics_dict = metrics_dict.get_sync_metrics()
        metrics_dict.print(sync_metrics_dict)
        _ndcg, _non_exist_rate, _repeat_rate, _correct_count = 0.0, 0.0, 0.0, 0.0
        if self.accelerator.is_main_process:
            for task in sync_metrics_dict:
                task_count = sync_metrics_dict[task]['Count']
                self.writer.add_scalar(f'valuating/{task}_Recall', sync_metrics_dict[task]['Recall'] / task_count, epoch)
                self.writer.add_scalar(f'valuating/{task}_NDCG', sync_metrics_dict[task]['NDCG'] / task_count, epoch)
                self.writer.add_scalar(f'valuating/{task}_NonExist_rate', sync_metrics_dict[task]['NonExistRate'] / task_count, epoch)
                self.writer.add_scalar(f'valuating/{task}_Repeat_rate', sync_metrics_dict[task]['RepeatRate'] / task_count, epoch)
                self.writer.add_scalar(f'valuating/{task}_Correct_count', sync_metrics_dict[task]['CorrectCount'] / task_count, epoch)
                if task == 'SFTPersonalCategoryRate':
                    self.writer.add_scalar(f'valuating/{task}_Category_rate_correct', sync_metrics_dict[task]['CategoryRateCorrect'] / task_count, epoch)

                _ndcg += sync_metrics_dict[task]['NDCG'] / task_count
                _non_exist_rate += sync_metrics_dict[task]['NonExistRate'] / task_count
                _repeat_rate += sync_metrics_dict[task]['RepeatRate'] / task_count
                _correct_count += sync_metrics_dict[task]['CorrectCount'] / task_count

            self.writer.add_scalar(f'valuating/Total_NDCG', _ndcg / len(sync_metrics_dict), epoch)
            self.writer.add_scalar(f'valuating/Total_NonExist_rate', _non_exist_rate / len(sync_metrics_dict), epoch)
            self.writer.add_scalar(f'valuating/Total_Repeat_rate', _repeat_rate / len(sync_metrics_dict), epoch)
            self.writer.add_scalar(f'valuating/Total_Correct_count', _correct_count / len(sync_metrics_dict), epoch)
            print(f'Epoch {epoch} | SFT_Val_Total_NDCG: {_ndcg/len(sync_metrics_dict):.4f}\n')
        return 0.0 - _ndcg


if __name__ == "__main__":
    pass
