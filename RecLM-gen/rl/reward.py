# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import numpy as np
import torch
from base.reward import BaseRewardModel, RewardOutput
from utils.tools import load_pickle, get_item_ranking, RunningMoments, side_tokenizer, get_output_text, get_complete_text


class RewardModel(BaseRewardModel):
    def __init__(self, args, tokenizer):
        super().__init__()
        assert args.data_path is not None
        self.args = args
        self.teacher_port = self.args.teacher_port
        self.tokenizer = tokenizer
        self.metas = load_pickle(self.args.data_path + 'meta.pickle')
        self.title2item = {}
        for _ in self.metas:
            if self.title2item.get(self.metas[_][self.args.item_index]) is None:
                self.title2item[self.metas[_][self.args.item_index]] = []
            self.title2item[self.metas[_][self.args.item_index]].append(_)
        self.category2item = load_pickle(self.args.data_path + 'category.pickle')
        self.best_ranking_score = [0]
        for i in range(0, self.args.topk*2):
            self.best_ranking_score.append(self.best_ranking_score[i]+self.ranking_score_func(i)/math.log2(i+2))

        self.running = {
            'item': {_: RunningMoments() for _ in self.args.RL_train_tasks.split(',')},
            'list': RunningMoments()
        }

    @staticmethod
    def ranking_score_func(idx):
        return 1.0/math.log2(idx+2)         # NR-8

    def reward_calculate(self, task, input_field_data, title_list):
        ranking_score_frac, task_score_frac = self.args.reward_alpha, 1.0-self.args.reward_alpha            # NR-13
        item_count = input_field_data['item_count']
        repeat_score, exceed_score, not_exist_score, in_history_score, target_score = -1.0, -1.0, -1.0, -1.0, 1.0
        candidates = input_field_data.get('candidate_titles') or self.title2item
        user = input_field_data['user']
        sub_sequential = input_field_data['sub_sequential']
        item_list = [self.title2item[_][0] if _ in self.title2item else list(self.metas.keys())[0] for _ in title_list]

        target_item = input_field_data['target_item']
        item_ranking = get_item_ranking(self.args.backup_ip, [user], [sub_sequential], [item_list], port=self.teacher_port)
        if task == 'RLSeqRanking':
            item_ranking = np.argsort(item_ranking)
            item_ranking = np.argsort(item_ranking).tolist()

        rank_score = []
        rank_corrected_score = []
        task_score = []
        in_category_count, out_category_count = 0, 0
        target_category = input_field_data['target_category']
        for idx, (_, __) in enumerate(zip(item_list, item_ranking)):
            if idx >= item_count:
                rank_score.append(exceed_score)
                rank_corrected_score.append(exceed_score)
                task_score.append(exceed_score)
            elif title_list[idx] not in candidates:
                rank_score.append(not_exist_score)
                rank_corrected_score.append(not_exist_score)
                task_score.append(not_exist_score)
            elif _ in item_list[:idx]:
                rank_score.append(repeat_score)
                rank_corrected_score.append(repeat_score)
                task_score.append(repeat_score)
            elif _ in input_field_data['sub_sequential']:
                rank_score.append(in_history_score)
                rank_corrected_score.append(in_history_score)
                task_score.append(in_history_score)
            else:
                temp_score = 1.0 if _ == target_item else self.ranking_score_func(__+1)       # NR-13
                rank_score.append(temp_score)
                rank_corrected_score.append(temp_score / math.log2(idx + 2))
                if _ in self.category2item[target_category]:
                    in_category_count += 1
                else:
                    out_category_count += 1
                if task in ['RLSeqRec', 'RLItemCount', 'RLSeqRanking']:
                    pass
                elif task in ['RL+PersonalControlRec', 'RL-PersonalControlRec']:
                    if _ in self.category2item[target_category]:
                        if '+' in task:
                            task_score.append(1.0)
                        else:
                            task_score.append(0.0)
                    elif _ not in self.category2item[target_category]:
                        if '-' in task:
                            task_score.append(1.0)
                        else:
                            task_score.append(0.0)
                elif task.startswith('RLPersonalCategoryRate'):
                    if 'RateLP' in task:
                        # if in_category_count <= input_field_data['category_count']:
                        #     task_score.append(0.5)
                        # else:
                        #     if _ in self.category2item[target_category]:
                        #         task_score.append(0.0)
                        #     else:
                        #         task_score.append(1.0)
                        # NR-19
                        if out_category_count > (input_field_data['item_count']-input_field_data['category_count']):
                            task_score.append(0.5)
                        else:
                            if _ not in self.category2item[target_category]:
                                task_score.append(1.0)
                            elif in_category_count < input_field_data['category_count']:
                                task_score.append(0.5)
                            else:
                                task_score.append(0.0)

                    elif 'RateMP' in task:
                        # if out_category_count < (input_field_data['item_count'] - input_field_data['category_count']):
                        #     task_score.append(0.5)
                        # else:
                        #     if _ in self.category2item[target_category]:
                        #         task_score.append(1.0)
                        #     else:
                        #         task_score.append(0.0)
                        # NR-19
                        if in_category_count > input_field_data['category_count']:
                            task_score.append(0.5)
                        else:
                            if _ in self.category2item[target_category]:
                                task_score.append(1.0)
                            elif out_category_count < (input_field_data['item_count']-input_field_data['category_count']):
                                task_score.append(0.5)
                            else:
                                task_score.append(0.0)

                    elif 'RateEP' in task:
                        if _ in self.category2item[target_category]:
                            if in_category_count <= input_field_data['category_count']:
                                task_score.append(1.0)
                            else:
                                task_score.append(0.0)
                        else:
                            if in_category_count >= input_field_data['category_count']:
                                task_score.append(1.0)
                            else:
                                if out_category_count <= (input_field_data['item_count']-input_field_data['category_count']):
                                    task_score.append(0.5)
                                else:
                                    task_score.append(0.0)
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
        rank_score = torch.tensor(rank_score, device=self.args.gpu)
        rank_corrected_score = torch.tensor(rank_corrected_score, device=self.args.gpu)
        task_score = torch.tensor(task_score, device=self.args.gpu)
        if task in ['RLSeqRec', 'RLItemCount', 'RLSeqRanking']:
            item_reward = rank_score
            # list_task_reward = rank_corrected_score.sum()/self.best_ranking_score[item_count]       # NR-18
            list_task_reward = 1.0/math.log2(item_list.index(target_item)+2) if target_item in item_list else 0.0
        elif task in ['RL+PersonalControlRec', 'RL-PersonalControlRec']:
            item_reward = rank_score*ranking_score_frac + task_score*task_score_frac
            target_count = min(item_count, len(self.category2item[target_category])) if '+' in task else item_count
            # list_task_reward = rank_corrected_score.sum()/self.best_ranking_score[item_count]*ranking_score_frac + task_score.sum()/target_count*task_score_frac       # NR-18
            if '+' in task:
                # list_task_reward = 1.0 / math.log2(target_count-in_category_count+2)        # NR-15, 17
                list_task_reward = 1.0/(target_count-in_category_count+1)        # NR-18
                # list_task_reward = 2*in_category_count/target_count-1
            else:
                # list_task_reward = 1.0 / math.log2(target_count-out_category_count+2)        # NR-15, 17
                list_task_reward = 1.0/(target_count-out_category_count+1)        # NR-18
                # list_task_reward = 2*out_category_count/target_count-1
        elif task.startswith('RLPersonalCategoryRate'):
            item_reward = rank_score*ranking_score_frac + task_score*task_score_frac
            # list_task_reward = rank_corrected_score.sum()/self.best_ranking_score[item_count]*ranking_score_frac + task_score.sum()/target_count*task_score_frac       # NR-18
            if 'RateLP' in task and in_category_count <= input_field_data['category_count']:
                list_task_reward = 1.0
            elif 'RateMP' in task and in_category_count >= input_field_data['category_count']:
                list_task_reward = 1.0
            elif 'RateEP' in task and in_category_count == input_field_data['category_count']:
                list_task_reward = 1.0
            else:
                # list_task_reward = 1.0/(math.log2(abs(in_category_count-input_field_data['category_count'])+2))   # NR-15, 17
                list_task_reward = 1.0/(abs(in_category_count-input_field_data['category_count'])+1)   # NR-18
                # list_task_reward = -abs(in_category_count-input_field_data['category_count'])/input_field_data['item_count']
        else:
            raise NotImplementedError

        # list_reward = list_task_reward        # NR-14, 15
        list_reward = rank_corrected_score.sum()/self.best_ranking_score[item_count]*ranking_score_frac + list_task_reward*task_score_frac        # NR-17-21
        # res = [list_reward, item_reward]        # NR-10, 17, 18, 19
        # res = [list_reward, item_reward*0.1]    # NR-11
        # res = [list_reward, item_reward*0.3]    # NR-12
        # res = [list_reward, item_reward*0.5]    # NR-13
        # res = [list_reward, item_reward*0.3]    # NR-14, 15
        # res = [list_reward*100, item_reward]    # NR-16
        res = [item_reward, float(list_reward*10)]    # NR-20
        # res = [list_reward*10, item_reward*2]    # NR-21
        return res

    def get_reward(self, batch, output_title_list, only_reward=False) -> RewardOutput:
        """
        :param batch: Got by collate_fn, which contains everything of 'bs' train samples useful for reward calculation.
        :param output_title_list: The type is list[str] with number of 'sample_num*bs'.
        :param only_reward: whether return only rewards in RewardOutput.
        :return: RewardOutput
        """
        task, input_text, input_field_data = batch['task']*self.args.sample_num, batch['input_text']*self.args.sample_num, batch['input_field_data']*self.args.sample_num
        reward_data = [self.reward_calculate(t, ifd, tl) for t, ifd, tl in zip(task, input_field_data, output_title_list)]
        item_reward, list_reward = [rd[0] for rd in reward_data], [rd[1] for rd in reward_data]
        if only_reward:
            return RewardOutput(reward=list_reward)
        _output_text = [get_output_text(tl, '\n'+self.tokenizer.eos_token, idx=self.args.idx) for tl in output_title_list]
        complete_text = [get_complete_text(it, ot) for it, ot in zip(input_text, _output_text)]
        complete_data = side_tokenizer(complete_text,
                                       'left', self.tokenizer,
                                       padding=True, truncation=True,
                                       max_length=self.args.max_token_length + self.args.gen_max_length,
                                       return_tensors='pt').to(self.args.gpu).data
        output_data = side_tokenizer(_output_text, 'right', self.tokenizer)
        action_mask = torch.zeros_like(complete_data['input_ids'], dtype=torch.bool)
        for idx in range(len(task)):
            output_length = len(output_data['input_ids'][idx])
            action_mask[idx][-output_length+1:] = True

        total_reward = torch.zeros_like(complete_data['input_ids'], dtype=torch.float, device=self.args.gpu)
        _list_reward = torch.tensor(list_reward, device=self.args.gpu)
        if self.args.reward_scale:
            self.running['list'].update(_list_reward)
            score_scaling_factor = self.running['list'].std + torch.finfo(_list_reward.dtype).eps
            _list_reward /= score_scaling_factor
        total_reward[:, -1] += _list_reward

        if self.args.fine_grain_reward:
            _item_reward = [torch.concat([torch.tensor([0.0] * 4, device=self.args.gpu), _]) for _ in item_reward]
            for idx in range(len(task)):
                if self.args.reward_scale:
                    self.running['item'][task[idx]].update(_item_reward[idx])
                    score_scaling_factor = self.running['item'][task[idx]].std + torch.finfo(_item_reward[idx].dtype).eps
                    _item_reward[idx] /= score_scaling_factor
                total_reward[idx][complete_data['input_ids'][idx] == 13] += _item_reward[idx]

        return RewardOutput(complete_data=complete_data, action_mask=action_mask, total_reward=total_reward, reward=list_reward)
