# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import re

from torch.utils.tensorboard import SummaryWriter
from rl.dataset import *
from base.trainer import BaseTrainer
from utils.metrics import Metrics
from base.dataset import BaseDataset
from utils.tools import rm_idx, vague_map, sync_dict, eval_decorator


# RL trainer
class RLTrainer(BaseTrainer):
    def __init__(self, args):
        super(RLTrainer, self).__init__(args)
        self.title2item = None
        self.item2category = None
        self.metas = None
        self.category2item = None

        self.writer = None
        if self.accelerator.is_main_process:
            self.writer = SummaryWriter(log_dir=os.path.join('logs', self.args.output_path), flush_secs=30)

        self.actor_critic.load_parameters(self.args.RL_load)
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
                print(f'computing RL train, val datum info')
            if self.args.train_data_file:  # from generated static dataset file
                self.train_data = BaseDataset(self.args, self.tokenizer, 'train')
            if self.args.val_data_file:
                self.val_data = BaseDataset(self.args, self.tokenizer, 'val')
            if self.args.data_path and not self.train_data and not self.val_data:  # dynamic fetched from raw dataset
                TaskTemplate = {_: Train_task_group_mapping[_] for _ in self.args.RL_train_tasks.split(',')}
                TaskNum = {_: 1 for _ in self.args.RL_train_tasks.split(',')}
                ValTaskTemplate = {_: Val_task_group_mapping[_.split('_')[0]] for _ in self.args.RL_val_tasks.split(',')}
                ValTaskNum = {_: 1 for _ in self.args.RL_val_tasks.split(',')}
                data = {
                    'category2item': self.category2item,
                    'item2category': self.item2category,
                    'metas': self.metas,
                    'title2item': self.title2item,
                    'sequential': load_pickle(self.args.data_path + 'sequential.pickle'),
                    'ranking_candidate': load_pickle(self.args.data_path + 'ranking_candidate.pickle'),
                }
                self.train_data = RLDataset(self.args, TaskTemplate, TaskNum, data, self.tokenizer, 'train')
                self.val_data = RLDataset(self.args, ValTaskTemplate, ValTaskNum, data, self.tokenizer, 'val')

        self.train_loader = DataLoader(self.train_data, batch_size=self.args.batch_size, shuffle=True, collate_fn=self.train_data.collate_fn)
        self.val_loader = DataLoader(self.val_data, batch_size=self.args.val_batch_size, shuffle=False, collate_fn=self.val_data.collate_fn, drop_last=False)

        self.prepare(self.train_loader, self.val_loader)

    def RL_train(self):
        best_val_reward = -float('inf')
        metrics_dict = Metrics(['RLTotal']+self.args.RL_train_tasks.split(','), self.args.topk, self.category2item, self.title2item, self.accelerator)
        if self.args.dry and self.args.lr > 0:
            best_val_reward = self.RL_val(0)
        for eps in range(self.args.num_episodes):
            if hasattr(self.train_data, "set_epoch"):
                self.train_data.set_epoch(eps)
            pbar = tqdm(total=len(self.train_loader), ncols=150, disable=not self.accelerator.is_main_process)
            for batch in self.train_loader:
                self.accelerator.wait_for_everyone()
                self.sample_batch += 1
                if self.accelerator.is_main_process and self.sample_batch % 100 == 1:
                    print(batch['input_text'][0])
                    print(batch['input_data']['input_ids'][0])

                # sampling train data
                output_text = self.RL_sample_batch(batch)

                # process output
                output_title_list = [[__.strip() for __ in _.strip().split('\n')] for _ in output_text]
                if self.args.idx:
                    output_title_list = [[rm_idx(__) for __ in _] for _ in output_title_list]
                if self.args.vague_mapping:
                    output_title_list = [vague_map(_, self.title2item) for _ in output_title_list]

                # get reward
                output_reward = self.reward_model.get_reward(batch, output_title_list)
                (
                    complete_data,
                    action_mask,
                    total_reward,
                    list_reward
                ) = (
                    output_reward.complete_data,
                    output_reward.action_mask,
                    output_reward.total_reward,
                    output_reward.reward
                )
                # process ppo data
                self.PPO_process_batch(complete_data, action_mask, total_reward)

                # record
                output_labels = [[__ for __ in _.strip().split('\n')] for _ in batch['output_text']]
                for i in range(len(batch['task'])):
                    metrics_dict.add_sample(batch['task'][i], batch['input_field_data'][i], output_title_list[i], output_labels[i], list_reward[i])
                    metrics_dict.add_sample('RLTotal', batch['input_field_data'][i], output_title_list[i], output_labels[i], list_reward[i])

                # logging
                sync_metrics_dict = metrics_dict.get_sync_metrics()
                if self.accelerator.is_main_process:
                    for t in sync_metrics_dict:
                        self.writer.add_scalar(f'sampling/{t}_reward', metrics_dict[t]['RewardSum']/metrics_dict[t]['Count'], metrics_dict[t]['Count'])
                        self.writer.add_scalar(f'sampling/{t}_ndcg', metrics_dict[t][f'NDCG']/metrics_dict[t]['Count'], metrics_dict[t]['Count'])
                    total_count = sync_metrics_dict['RLTotal']['Count']
                    total_reward_sum = sync_metrics_dict['RLTotal']['RewardSum']
                    total_non_exist_rate = sync_metrics_dict['RLTotal'][f'NonExistRate']
                    total_repeat_rate = sync_metrics_dict['RLTotal'][f'RepeatRate']
                    total_correct_count = sync_metrics_dict['RLTotal'][f'CorrectCount']
                    total_ndcg = sync_metrics_dict['RLTotal'][f'NDCG']
                    self.writer.add_scalar('sampling/total_reward', total_reward_sum / total_count, total_count)
                    self.writer.add_scalar('sampling/total_non_exist_rate', total_non_exist_rate / total_count, total_count)
                    self.writer.add_scalar('sampling/total_repeat_rate', total_repeat_rate / total_count, total_count)
                    self.writer.add_scalar('sampling/total_correct_count', total_correct_count / total_count, total_count)
                    self.writer.add_scalar('sampling/total_ndcg', total_ndcg / total_count, total_count)

                pbar.set_description(f'RL learning in episode: {eps} | sample_batch: {self.sample_batch} | example: {len(self.memories)} | '
                                     f'max_length: {complete_data["input_ids"].shape[1]}')
                pbar.update(1)

                if self.sample_batch % self.args.learn_batch == 0 and self.args.lr > 0:
                    # ppo learning
                    self.lr_scheduler.step()
                    ppo_stat = self.learn_PPO()
                    # logging
                    ppo_stat = sync_dict(self.accelerator, ppo_stat)
                    if self.accelerator.is_main_process:
                        for k, v in ppo_stat.items():
                            self.writer.add_scalar(k, v / ppo_stat['training/backward_step'], self.sample_batch)
                    self.memories.clear()

                if self.sample_batch % self.args.val_save_step == 0:
                    metrics_dict.print(sync_metrics_dict)
                    if self.args.lr > 0:
                        if self.accelerator.is_main_process:
                            self.actor_critic.save_parameters(f'{self.sample_batch}step')
                        val_reward = self.RL_val(self.sample_batch)
                        if val_reward > best_val_reward and self.accelerator.is_main_process:
                            best_val_reward = val_reward
                            self.actor_critic.save_parameters("BEST_EVAL_REWARD")

            pbar.close()
        print('RL training complete')

    @torch.no_grad()
    @eval_decorator
    def RL_val(self, step: int):
        metrics_dict = Metrics(self.args.RL_val_tasks.split(','), self.args.topk, self.category2item, self.title2item, self.accelerator)
        pbar = tqdm(total=len(self.val_loader), ncols=150, disable=not self.accelerator.is_main_process)
        for step_i, batch in enumerate(self.val_loader):
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process and step_i % 1000 == 0:
                print(batch['input_text'][0])
                print(batch['input_data']['input_ids'][0])
            input_ids_length = batch['input_data']['input_ids'].shape[1]
            val_model = self.actor_critic.base_model if step == 0 else self.actor_critic.actor_model
            output_ids = val_model.greedy_search(**batch['input_data'], stopping_criteria=self.stopping_criteria)

            # process output
            output_title = self.tokenizer.batch_decode(output_ids[:, input_ids_length:], skip_special_tokens=True)
            output_title_list = [[__.strip() for __ in _.strip().split('\n')] for _ in output_title]
            if self.args.idx:
                output_title_list = [[rm_idx(__) for __ in _] for _ in output_title_list]
            if self.args.vague_mapping:
                output_title_list = [vague_map(_, self.title2item) for _ in output_title_list]

            # record
            output_labels = [[__ for __ in _.strip().split('\n')] for _ in batch['output_text']]
            output_reward = self.reward_model.get_reward(batch, output_title_list, only_reward=True)
            list_reward = output_reward.reward
            for i, task in enumerate(batch['task']):
                metrics_dict.add_sample(task, batch['input_field_data'][i], output_title_list[i], output_labels[i], list_reward[i])
            pbar.update(1)

        # logging
        self.accelerator.wait_for_everyone()
        sync_metrics_dict = metrics_dict.get_sync_metrics()
        metrics_dict.print(sync_metrics_dict)
        _reward_sum, _ndcg, _non_exist_rate, _repeat_rate, _correct_count = 0.0, 0.0, 0.0, 0.0, 0.0
        if self.accelerator.is_main_process:
            for t in sync_metrics_dict:
                task_count = sync_metrics_dict[t]['Count']
                reward_sum = sync_metrics_dict[t]['RewardSum'] / task_count
                recall = sync_metrics_dict[t][f'Recall'] / task_count
                ndcg = sync_metrics_dict[t][f'NDCG'] / task_count
                non_exist_rate = sync_metrics_dict[t][f'NonExistRate'] / task_count
                repeat_rate = sync_metrics_dict[t][f'RepeatRate'] / task_count
                correct_count = sync_metrics_dict[t][f'CorrectCount'] / task_count
                if t == 'RLPersonalCategoryRate':
                    category_rate_correct = sync_metrics_dict[t][f'CategoryRateCorrect'] / task_count
                    self.writer.add_scalar(f'valuating/{t}_Category_rate_correct', category_rate_correct, step)
                elif t == 'RLSeqRanking':
                    non_in_candidate_rate = sync_metrics_dict[t][f'NotInCandidateRatio'] / task_count
                    self.writer.add_scalar(f'valuating/{t}_NotInCandidate_rate', non_in_candidate_rate, step)

                _reward_sum += reward_sum
                _ndcg += ndcg
                _non_exist_rate += non_exist_rate
                _repeat_rate += repeat_rate
                _correct_count += correct_count
                self.writer.add_scalar(f'valuating/{t}_Reward_mean', reward_sum, step)
                self.writer.add_scalar(f'valuating/{t}_Recall', recall, step)
                self.writer.add_scalar(f'valuating/{t}_NDCG', ndcg, step)
                self.writer.add_scalar(f'valuating/{t}_NonExist_rate', non_exist_rate, step)
                self.writer.add_scalar(f'valuating/{t}_Repeat_rate', repeat_rate, step)
                self.writer.add_scalar(f'valuating/{t}_Correct_count', correct_count, step)

            self.writer.add_scalar(f'valuating/Total_Reward_mean', _reward_sum / len(sync_metrics_dict), step)
            self.writer.add_scalar(f'valuating/Total_NDCG', _ndcg / len(sync_metrics_dict), step)
            self.writer.add_scalar(f'valuating/Total_NonExist_rate', _non_exist_rate / len(sync_metrics_dict), step)
            self.writer.add_scalar(f'valuating/Total_Repeat_rate', _repeat_rate / len(sync_metrics_dict), step)
            self.writer.add_scalar(f'valuating/Total_Correct_count', _correct_count / len(sync_metrics_dict), step)
        return _reward_sum/len(sync_metrics_dict)

    def RL_val_path(self):
        val_steps = {}
        for params_file in os.listdir(self.args.output_path):
            step = re.findall(r'^(\d+)step_RL\.pth', params_file)   # matching the train step from file name
            if len(step) > 0:
                val_steps[step[0]] = os.path.join(self.args.output_path, params_file[:-4])
        if self.args.dry:
            val_steps[0] = None
        val_steps = {_: val_steps[_] for _ in sorted(val_steps, key=lambda k: k) if _ >= 0}
        if self.accelerator.is_main_process:
            print(val_steps.keys())

        for train_step, model_file in val_steps.items():
            assert train_step == self.actor_critic.load_parameters(model_file)
            self.accelerator.wait_for_everyone()
            self.RL_val(train_step)


if __name__ == '__main__':
    pass
