# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import random
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from sft.templates import *
from utils.tools import load_pickle, save_pickle, get_item_list, get_history_text, get_output_text, side_tokenizer, get_complete_text


class SFTDataset(Dataset):
    def __init__(self, args, task_template, task_num, data, tokenizer, mode='train', saving=False, immediately=True):
        self.args = args
        self.task_template = task_template
        self.task_num = task_num
        self.mode = mode
        self.tokenizer = tokenizer
        self.teacher_port = self.args.teacher_port
        self.saving = saving
        self.immediately = immediately

        self.item2category = data['item2category']
        self.category2item = data['category2item']
        self.metas = data['metas']
        self.title2item = data['title2item']
        self.sequential = data['sequential']
        self.share_chat_gpt = data['share_chat_gpt']
        self.ranking_candidate = data['ranking_candidate']
        if self.args.llama2_chat_template:
            self.chat_gpt_conv = get_conversation_template("llama-2")
            self.chat_gpt_conv.set_system_message("You are a helpful, respectful and honest assistant.")
            self.chat_gpt_conv.append_message(self.chat_gpt_conv.roles[0], '')
            self.chat_gpt_conv.append_message(self.chat_gpt_conv.roles[1], '')
        if 'SFTTestSeqRec_Result' in data:
            self.SFTTestSeqRec_Result = {u: data['SFTTestSeqRec_Result'][idx].get('SFTTestSeqRec_output_title_list') or [] for idx, u in enumerate(self.sequential) if idx < len(data['SFTTestSeqRec_Result'])}
        if 'SFTTestSeqRec_Candidate' in data:
            self.SFTTestSeqRec_Candidate = data['SFTTestSeqRec_Candidate']

        self.datum_info = []
        self.complete_datum_info = []
        self.compute_datum_info()

    def find_maximum_category(self, item_list, target_item):
        category_count = {c: 0 for c in self.category2item if target_item not in self.category2item[c]}
        for o_i in item_list:
            for c in self.item2category.get(o_i) or []:
                if c in category_count:
                    category_count[c] += 1
        max_count = max(list(category_count.values()))
        category = [c for c in list(category_count.keys()) if category_count[c] == max_count]
        return category

    def compute_datum_info(self):
        val_num = self.args.val_num_per_task if hasattr(self.args, 'val_num_per_task') else 0
        val_task_num = 0
        for task, num in self.task_num.items():
            if task in ["SFTSeqRec", "SFTPersonalControlRec", "SFTPersonalCategoryRate", "SFTSeqRecPALR"]:
                for _ in range(num):
                    self.datum_info += [[task, u] for u in self.sequential]
            elif task in ["SFTControlRec", "SFTControlRec_re"]:
                for _ in range(num):
                    self.datum_info += [[task, i] for i in self.metas if self.item2category.get(i)]
            elif task == "SFTCategoryRate":
                for _ in range(num):
                    self.datum_info += [[task, c, 'CategoryRate-LP'] for c in self.category2item]
                    self.datum_info += [[task, c, 'CategoryRate-MP'] for c in self.category2item]
                    self.datum_info += [[task, c, 'CategoryRate-LC'] for c in self.category2item]
                    self.datum_info += [[task, c, 'CategoryRate-MC'] for c in self.category2item]

            elif task in ["SFTTestSeqRec", "SFTTestSeqRanking", "SFT+TestPersonalControlRec", "SFT-TestPersonalControlRec",
                          "SFTTestItemCount"] or task.startswith("SFTTestPersonalCategoryRate"):
                for _ in range(num):
                    if self.mode == 'test':
                        self.datum_info += [[task, u] for u in self.sequential]
                    elif self.mode == 'val':
                        self.datum_info += [[task, u] for u in list(self.sequential.keys())[val_num*val_task_num: val_num*(val_task_num+1)]]
                        val_task_num += 1

            elif task == 'ShareChatGPT' and self.mode == 'train':
                share_chat_gpt_count = int(self.args.share_chat_gpt_ratio * len(self.datum_info))
                self.datum_info += [['ShareChatGPT'] for _ in range(share_chat_gpt_count)]
            else:
                raise NotImplementedError

        complete_datum_info_path = None
        if self.mode == 'val':
            complete_datum_info_path = self.args.data_path + f'SFT_datum_info_{self.mode}_{self.args.SFT_val_tasks}{"_LCT" if self.args.llama2_chat_template else ""}_Top{self.args.topk}.pickle'
        if self.mode == 'train':
            complete_datum_info_path = self.args.data_path + f'SFT_datum_info_{self.mode}_{self.args.SFT_train_tasks}{"_LCT" if self.args.llama2_chat_template else ""}_Top{self.args.topk}.pickle'

        if self.saving:
            self.complete_datum_info = []
            for idx in tqdm(range(len(self.complete_datum_info), len(self.datum_info)), desc=f'computing {self.mode} datum info'):
                self.complete_datum_info.append(self.getitem(idx))
            save_pickle(self.complete_datum_info, complete_datum_info_path)
        else:
            self.complete_datum_info = load_pickle(complete_datum_info_path) or []
            if len(self.complete_datum_info) != len(self.datum_info):
                self.complete_datum_info = []

    def __len__(self):
        return len(self.datum_info)

    def get_item_index(self, item):
        if self.args.item_index == 'title':
            return self.metas[item]['title']
        elif self.args.item_index == 'title32':
            return self.metas[item]['title32']
        elif self.args.item_index == 'title64':
            return self.metas[item]['title64']
        elif self.args.item_index == 'item':
            return item
        else:
            return self.metas[item][self.args.item_index]

    def get_sub_sequential(self, user):
        if self.mode == 'train':
            sequential = self.sequential[user][:-2]
            target_item_index = random.choice(range(1, len(sequential)))
            min_start_item_index = max(0, target_item_index-self.args.max_item_length)
            start_item_index = random.choice(range(min_start_item_index, target_item_index))
            sub_sequential = sequential[start_item_index: target_item_index]
            target_item = sequential[target_item_index]
        elif self.mode == 'val':
            sub_sequential = self.sequential[user][-self.args.max_item_length-2:-2]
            target_item = self.sequential[user][-2]
            target_item_index = len(self.sequential[user])-2
        elif self.mode == 'test':
            sub_sequential = self.sequential[user][-self.args.max_item_length-1:-1]
            target_item = self.sequential[user][-1]
            target_item_index = len(self.sequential[user])-1
        else:
            raise NotImplementedError
        return sub_sequential, target_item, target_item_index

    def get_output_item_list(self, task, user=None, sub_sequential=None, target_item=None, target_category=None, direction=None, item_count=None, category_item_count=None, has_candidate=False):
        output_items, candidate_items = [], []
        if task in ['SFTSeqRec']:
            output_items = get_item_list(self.args.backup_ip, [user], [sub_sequential], item_count, port=self.teacher_port, immediately=self.immediately)
            if target_item in output_items:
                output_items.remove(target_item)
            output_items = ([target_item] + output_items)[:item_count]
        elif task in ["SFTControlRec", "SFTControlRec_re"]:
            if direction == '+':
                output_items = copy.deepcopy(self.category2item[target_category])
            else:
                output_items = list(set(list(self.metas.keys()))-set(self.category2item[target_category]))
            random.shuffle(output_items)
            if target_item in output_items:
                output_items.remove(target_item)
            output_items = ([target_item] + output_items)[:item_count]
        elif task in ["SFTPersonalControlRec"]:
            output_items = get_item_list(self.args.backup_ip, [user], [sub_sequential], item_count,
                                         target_category=[direction+target_category], port=self.teacher_port, immediately=self.immediately)
            if target_item in output_items:
                output_items.remove(target_item)
                output_items = ([target_item] + output_items)
        elif task in ["SFTCategoryRate"]:
            output_items = random.sample(self.category2item[target_category], category_item_count)
            output_items = output_items + random.sample(list(set(self.metas.keys()) - set(self.category2item[target_category])),
                                                        item_count-category_item_count)
            random.shuffle(output_items)
        elif task in ['SFTPersonalCategoryRate']:
            in_category_items = get_item_list(self.args.backup_ip, [user], [sub_sequential], category_item_count,
                                              target_category=['+'+target_category], port=self.teacher_port, immediately=self.immediately)
            out_category_items = get_item_list(self.args.backup_ip, [user], [sub_sequential], item_count-category_item_count,
                                               target_category=['-'+target_category], port=self.teacher_port, immediately=self.immediately)
            output_items = in_category_items + out_category_items
            random.shuffle(output_items)
        else:
            raise NotImplementedError

        assert len(output_items) == item_count
        if has_candidate:
            candidate_num = random.choice(range(self.args.topk, self.args.candidate_num+1))
            candidate_items = output_items + random.choices(list(set(self.metas.keys())-set(output_items)), k=candidate_num-item_count)
            random.shuffle(candidate_items)
        return output_items, candidate_items

    def getitem(self, idx):
        task = self.datum_info[idx][0]
        template_id = random.choice(list(self.task_template[task].keys()))
        template_selected = self.task_template[task][template_id]
        input_field_data, output_field_data = {}, {}
        if task in ["SFTSeqRec", "SFTTestSeqRec", "SFTSeqRanking", "SFTTestSeqRanking", "SFTPersonalControlRec",
                    "SFT+TestPersonalControlRec", "SFT-TestPersonalControlRec", "SFTPersonalCategoryRate",
                    "SFTTestItemCount", "SFTSeqRecPALR"] or task.startswith("SFTTestPersonalCategoryRate"):
            user = self.datum_info[idx][1]
            sub_sequential, target_item, target_item_index = self.get_sub_sequential(user)
            input_field_data.update({
                'user': user,
                'target_item': target_item,
                'target_item_title': self.get_item_index(target_item),
                'sub_sequential': sub_sequential,
                'history': get_history_text([f"'{self.get_item_index(_)}'" for _ in sub_sequential]),
            })

            if task in ["SFTSeqRec"]:
                item_count = random.choice(range(self.args.topk))+1
                output_items, candidate_items = self.get_output_item_list(task, user=user, sub_sequential=sub_sequential,
                                                                          target_item=target_item, item_count=item_count,
                                                                          has_candidate='candidate_titles' in template_selected.input_fields)
                input_field_data.update({
                    'target_category': self.item2category.get(target_item)[-1],
                    'item_count': item_count,
                    'candidate_titles': ', '.join([f"'{self.get_item_index(_)}'" for _ in candidate_items]),
                    'candidate_items': candidate_items
                })
                output_field_data.update({
                    'item_list': get_output_text([self.get_item_index(_) for _ in output_items], '\n'+self.tokenizer.eos_token, self.args.idx)
                })

            if task in ['SFTSeqRecPALR']:
                remain_sequential_length = len(self.sequential[user]) - target_item_index - 2
                item_count = random.choice(range(min(remain_sequential_length, self.args.topk)))+1
                output_items = self.sequential[user][target_item_index: target_item_index+item_count]
                candidate_items = []
                if 'candidate_titles' in template_selected.input_fields:
                    candidate_num = random.choice(range(self.args.topk, self.args.candidate_num+1))
                    candidate_items = output_items + random.choices(list(set(self.metas.keys())-set(output_items)), k=candidate_num-item_count)
                    random.shuffle(candidate_items)
                input_field_data.update({
                    'target_category': self.item2category.get(target_item)[-1],
                    'item_count': item_count,
                    'candidate_titles': ', '.join([f"'{self.get_item_index(_)}'" for _ in candidate_items]),
                    'candidate_items': candidate_items
                })
                output_field_data.update({
                    'item_list': get_output_text([self.get_item_index(_) for _ in output_items], '\n'+self.tokenizer.eos_token, self.args.idx)
                })

            elif task in ["SFTTestSeqRec"]:
                item_count = self.args.topk
                input_field_data.update({
                    'target_category': self.item2category.get(target_item)[-1],
                    'item_count': item_count
                })
                output_field_data.update({
                    'item_list': get_output_text([self.get_item_index(target_item)], '', idx=False)
                })
            elif task in ['SFTTestSeqRanking']:
                item_count = self.args.topk
                ranking_candidate = self.ranking_candidate[user][:self.args.candidate_num-1]
                insert_idx = idx % self.args.candidate_num
                ranking_candidate.insert(insert_idx, target_item)
                input_field_data.update({
                    'target_category': self.item2category.get(target_item)[-1],
                    'item_count': item_count,
                    'candidate_titles': ', '.join([f"'{self.get_item_index(_)}'" for _ in ranking_candidate]),
                    'candidate_items': ranking_candidate
                })
                output_field_data.update({
                    'item_list': get_output_text([self.get_item_index(target_item)], '', idx=False)
                })
            elif task in ['SFTTestItemCount']:
                item_count = self.args.topk + 1 + idx % 5
                input_field_data.update({
                    'target_category': self.item2category.get(target_item)[-1],
                    'item_count': item_count
                })
                output_field_data.update({
                    'item_list': get_output_text([self.get_item_index(target_item)], '', idx=False)
                })
            elif task in ["SFTPersonalControlRec"]:
                if random.random() > 0.5:
                    intention_group, d = Intention_plus_group, '+'
                    target_category = random.choice(list(self.category2item.keys()))
                    max_count = min(self.args.topk, len(self.category2item[target_category]))
                else:
                    intention_group, d = Intention_minus_group, '-'
                    SASRec_output = get_item_list(self.args.backup_ip, [user], [sub_sequential], self.args.topk, port=self.teacher_port, immediately=self.immediately)
                    target_category = random.choice(self.find_maximum_category(SASRec_output, target_item))
                    max_count = min(self.args.topk, len(self.metas) - len(self.category2item[target_category]))
                item_count = random.choice(range(max_count))+1 if task == "SFTPersonalControlRec" else 1
                output_items, candidate_items = self.get_output_item_list(task, user=user, sub_sequential=sub_sequential,
                                                                          target_item=target_item, item_count=item_count,
                                                                          target_category=target_category, direction=d,
                                                                          has_candidate='candidate_titles' in template_selected.input_fields)
                input_field_data.update({'target_category': target_category})
                intention_template_key = random.choice(list(intention_group.keys()))
                intention = intention_group[intention_template_key].get_input_text(input_field_data)
                input_field_data.update({
                    'synthetic_intention': intention,
                    'item_count': item_count,
                    'candidate_titles': ', '.join([f"'{self.get_item_index(_)}'" for _ in candidate_items]),
                    'candidate_items': candidate_items
                })
                output_field_data.update({
                    'item_list': get_output_text([self.get_item_index(_) for _ in output_items], '\n'+self.tokenizer.eos_token, self.args.idx)
                })
            elif task in ["SFT+TestPersonalControlRec", "SFT-TestPersonalControlRec"]:
                if self.mode == 'test':
                    item_list = [self.title2item[_][0] if _ in self.title2item else 'None' for _ in self.SFTTestSeqRec_Result[user]]
                else:
                    item_list = get_item_list(self.args.backup_ip, [user], [sub_sequential], self.args.topk, port=self.teacher_port, immediately=self.immediately)
                if task == "SFT+TestPersonalControlRec":
                    target_category = self.item2category.get(target_item)[-1]
                    intention_group = Intention_plus_group
                else:
                    target_category = self.find_maximum_category(item_list, target_item)[-1]
                    intention_group = Intention_minus_group
                item_count = self.args.topk
                input_field_data.update({'target_category': target_category})
                intention_template_key = random.choice(list(intention_group.keys()))
                intention = intention_group[intention_template_key].get_input_text(input_field_data)
                input_field_data.update({
                    'synthetic_intention': intention,
                    'item_count': item_count,
                    'SeqRec_Result': item_list
                })
                output_field_data.update({
                    'item_list': get_output_text([self.get_item_index(target_item)], '', idx=False)
                })

            elif task == "SFTPersonalCategoryRate":
                template_id = random.choice(list(self.task_template[task].keys()))
                template_selected = self.task_template[task][template_id]
                item_count = 10
                target_category = random.choice(list(self.category2item.keys()))
                category_item_count = min(len(self.category2item[target_category]), item_count)
                target_category_item_count = random.choice(range(category_item_count)) + 1 if template_id[-2] == 'L' else random.choice(range(category_item_count))
                output_category_item_count = target_category_item_count + random.choice([-1 if template_id[-2] == 'L' else 1, 0])
                output_items, candidate_items = self.get_output_item_list(task, user=user, sub_sequential=sub_sequential,
                                                                          item_count=item_count, target_category=target_category,
                                                                          category_item_count=output_category_item_count,
                                                                          has_candidate='candidate_titles' in template_selected.input_fields)
                input_field_data.update({
                    'target_category': target_category,
                    'item_count': item_count,
                    'category_proportion': f"{target_category_item_count}0%",
                    'category_count': target_category_item_count,
                    'candidate_titles': ', '.join([f"'{self.get_item_index(_)}'" for _ in candidate_items]),
                    'candidate_items': candidate_items
                })
                output_field_data.update({
                    'item_list': get_output_text([self.get_item_index(_) for _ in output_items], '\n' + self.tokenizer.eos_token, self.args.idx)
                })

            elif task.startswith("SFTTestPersonalCategoryRate"):
                if self.mode == 'test':
                    item_list = [self.title2item[_][0] if _ in self.title2item else 'None' for _ in self.SFTTestSeqRec_Result[user]]
                else:
                    item_list = get_item_list(self.args.backup_ip, [user], [sub_sequential], self.args.topk, port=self.teacher_port, immediately=self.immediately)
                if 'LP1' in task or 'LP' in task:
                    target_category = self.find_maximum_category(item_list, target_item)[-1]
                else:
                    target_category = self.item2category.get(target_item)[-1]
                template_selected = self.task_template[task]['PersonalCategoryRate']
                item_count = self.args.topk
                p = int(task.split('_')[-1])
                input_field_data.update({
                    'target_category': target_category,
                    'item_count': item_count,
                    'category_proportion': f'{p}%',
                    'category_count': int(p*item_count/100),
                    'SeqRec_Result': item_list,
                })
                output_field_data.update({
                    'item_list': get_output_text([self.get_item_index(target_item)], '', idx=False)
                })

        elif task in ["SFTControlRec", "SFTControlRec_re"]:
            target_item = self.datum_info[idx][1]
            if 'reverse' in template_id:
                input_field_data.update({
                    'item': self.get_item_index(target_item),
                })
                output_field_data.update({
                    'item': self.get_item_index(target_item),
                    'target_category': random.choice(self.item2category[target_item])+self.tokenizer.eos_token,
                })
            else:
                if random.random() > 0.5:
                    target_category = random.choice(self.item2category[target_item])
                    max_count = min(self.args.topk, len(self.category2item[target_category]))
                    intention_group, d = Intention_plus_group, '+'
                else:
                    categories = [c for c in self.category2item if target_item not in self.category2item[c]]
                    target_category = random.choice(categories)
                    max_count = min(self.args.topk, len(self.metas)-len(self.category2item[target_category]))
                    intention_group, d = Intention_minus_group, '-'
                item_count = random.choice(range(max_count))+1 if task == "SFTControlRec" else 1
                output_items, candidate_items = self.get_output_item_list(task, target_item=target_item, item_count=item_count,
                                                                          target_category=target_category, direction=d,
                                                                          has_candidate='candidate_titles' in template_selected.input_fields)

                input_field_data.update({'target_category': target_category})
                intention_template_key = random.choice(list(intention_group.keys()))
                intention = intention_group[intention_template_key].get_input_text(input_field_data)
                input_field_data.update({
                    'synthetic_intention': intention,
                    'item_count': item_count,
                    'candidate_titles': ', '.join([f"'{self.get_item_index(_)}'" for _ in candidate_items]),
                    'candidate_items': candidate_items
                })
                output_field_data.update({
                    'item_list': get_output_text([self.get_item_index(_) for _ in output_items], '\n'+self.tokenizer.eos_token, self.args.idx)
                })
        elif task == "SFTCategoryRate":
            target_category = self.datum_info[idx][1]
            template_id = self.datum_info[idx][2]
            template_selected = self.task_template[task][template_id]
            item_count = 10
            category_item_count = min(len(self.category2item[target_category]), item_count)
            target_category_item_count = random.choice(range(category_item_count))+1 if template_id[-2] == 'L' else random.choice(range(category_item_count))
            output_category_item_count = target_category_item_count + random.choice([-1 if template_id[-2] == 'L' else 1, 0])
            output_items, candidate_items = self.get_output_item_list(task, item_count=item_count, target_category=target_category,
                                                                      category_item_count=output_category_item_count,
                                                                      has_candidate='candidate_titles' in template_selected.input_fields)
            input_field_data.update({
                'target_category': target_category,
                'item_count': item_count,
                'category_proportion': f"{target_category_item_count}0%",
                'category_count': target_category_item_count,
                'candidate_titles': ', '.join([f"'{self.get_item_index(_)}'" for _ in candidate_items]),
                'candidate_items': candidate_items
            })
            output_field_data.update({
                'item_list': get_output_text([self.get_item_index(_) for _ in output_items], '\n'+self.tokenizer.eos_token, self.args.idx)
            })
        elif task == "ShareChatGPT":
            scg_data = random.choice(self.share_chat_gpt)
            chat_end_idx = random.choice([idx for idx, c in enumerate(scg_data) if c['from'] == 'human'])
            chat_start_idxes = [chat_end_idx]
            chat_start_idx = chat_start_idxes[-1]-2
            while chat_start_idx > 0:
                chat_start_idx = chat_start_idx-2
                pre_length = scg_data[chat_start_idx-1]['pre_length'] if chat_start_idx > 0 else 0
                if scg_data[chat_start_idx]['pre_length'] - pre_length < self.args.max_token_length-64:
                    chat_start_idxes.append(chat_start_idx)
                else:
                    break
            chat_start_idx_selected = random.choice(chat_start_idxes)
            if self.args.llama2_chat_template:
                chat_gpt_conv = get_conversation_template("llama-2")
                chat_gpt_conv.set_system_message("You are a helpful, respectful and honest assistant.")
                input_data = scg_data[chat_start_idx_selected:chat_end_idx+1]+[{'from': 'gpt', 'value': None}]
                for idx in range(0, len(input_data), 2):
                    assert input_data[idx]['from'] == 'human'
                    chat_gpt_conv.append_message(chat_gpt_conv.roles[0], input_data[idx]['value'])
                    assert input_data[idx+1]['from'] == 'gpt'
                    chat_gpt_conv.append_message(chat_gpt_conv.roles[1], input_data[idx+1]['value'])
                input_text = chat_gpt_conv.get_prompt()
                output_text = scg_data[chat_end_idx+1]['value']
            else:
                raise NotImplementedError
            out_dict = {
                'input_text': input_text,
                'output_text': output_text+self.tokenizer.eos_token,
                'task': task,
                'input_field_data': input_field_data,
            }
            return out_dict
        else:
            raise NotImplementedError

        input_text = template_selected.get_input_text(input_field_data, llama2_chat_template=self.args.llama2_chat_template).strip()
        output_text = template_selected.get_output_text(output_field_data).strip()
        out_dict = {
            'input_text': input_text,
            'output_text': output_text,
            'task': task,
            'input_field_data': input_field_data,
        }
        return out_dict

    def __getitem__(self, idx):
        if len(self.complete_datum_info) > 0:
            return self.complete_datum_info[idx]
        return self.getitem(idx)

    def collate_fn(self, batch):
        batch_entry = {}

        tasks = []
        input_text = []
        output_text = []
        complete_text = []
        input_field_data = []
        for i, entry in enumerate(batch):
            if 'task' in entry:
                tasks.append(entry['task'])
            if 'input_field_data' in entry:
                input_field_data.append(entry['input_field_data'])
            if 'input_text' in entry:
                input_text.append(entry['input_text'])
            if 'output_text' in entry:
                output_text.append(entry['output_text'])
            complete_text.append(get_complete_text(entry['input_text'], entry['output_text']))
        batch_entry['input_text'] = input_text
        batch_entry['output_text'] = output_text
        batch_entry['complete_text'] = complete_text

        batch_entry['input_data'] = side_tokenizer(batch_entry['input_text'],
                                                   'left', self.tokenizer,
                                                   padding=True, truncation=True,
                                                   max_length=self.args.max_token_length,
                                                   return_tensors='pt').to(self.args.gpu).data
        batch_entry['output_data'] = side_tokenizer(batch_entry['output_text'],
                                                    'right', self.tokenizer,
                                                    padding=True, truncation=True,
                                                    max_length=self.args.gen_max_length,
                                                    return_tensors='pt').to(self.args.gpu).data
        batch_entry['complete_text_data'] = {
            'input_ids':
                torch.cat([batch_entry['input_data']['input_ids'], batch_entry['output_data']['input_ids'][:, 1:]], dim=-1),
            'attention_mask':
                torch.cat([batch_entry['input_data']['attention_mask'], batch_entry['output_data']['attention_mask'][:, 1:]], dim=-1)
        }
        prompt_length = batch_entry['input_data']['input_ids'].shape[-1]
        batch_entry['complete_label_ids'] = copy.deepcopy(batch_entry['complete_text_data']['input_ids'])
        batch_entry['complete_label_ids'][..., :prompt_length] = -100
        batch_entry['complete_label_ids'][batch_entry['complete_label_ids'] == self.tokenizer.pad_token_id] = -100

        if len(input_field_data) > 0:
            batch_entry['input_field_data'] = input_field_data
        if len(tasks) > 0:
            batch_entry['task'] = tasks
        return batch_entry


Train_task_group_mapping = {
    "SFTSeqRec": SeqRec_group,                                  # 1). Sequential recommendation task, label generated by SASRec------------------I_0
    "SFTSeqRecPALR": SeqRec_group,                              # 2). Sequential recommendation task, label generated from history---------------PALR
    "SFTControlRec": ControlRec_group,                          # 3). Category control task(without user's history). Similar to product search---I_3
    "SFTControlRec_re": ControlRec_re_group,                    # 4). Similar to 3), but including item-category reversed task additionally.-----I_3_Re
    "SFTPersonalControlRec": PersonalControlRec_group,          # 5). Personal category control task(within user's history).---------------------I_1
    "SFTCategoryRate": CategoryRate_group,                      # 6). Category proportion control task(without user's history).
    "SFTPersonalCategoryRate": PersonalCategoryRate_group,      # 7). Personal category proportion control task(within user's history).----------I_2
    "ShareChatGPT": {'ShareChatGPT-1': ''},
}

Val_task_group_mapping = {
    "SFTTestSeqRec": ValSeqRec_group,                                       # I_{0}, the requirement of length is {self.args.topk}.
    "SFTTestSeqRanking": ValSeqRanking_group,
    "SFT+TestPersonalControlRec": ValPersonalControlRec_group,              # I_{1}^{+C}
    "SFT-TestPersonalControlRec": ValPersonalControlRec_group,              # I_{1}^{-C}
    "SFTTestPersonalCategoryRateLP": TestPersonalCategoryRateLP_group,      # I_{2}^{<=50%}: SFTTestPersonalCategoryRateLP_50, target category comes from target item.
    "SFTTestPersonalCategoryRateLP1": TestPersonalCategoryRateLP1_group,    # I_{2}^{<=50%}: SFTTestPersonalCategoryRateLP1_50, similar to I_{1}^{-C}, target category is the max proportion of SeqRec.
    "SFTTestPersonalCategoryRateMP": TestPersonalCategoryRateMP_group,      # I_{2}^{>=50%}: SFTTestPersonalCategoryRateMP_50, target category comes from target item.
    "SFTTestPersonalCategoryRateEP": TestPersonalCategoryRateEP_group,      # I_{2}^{==50%}: SFTTestPersonalCategoryRateEP_50, target category comes from target item.
    'SFTTestItemCount': ValSeqRec_group,                                    # Similar to I_{0}, the requirement of length is range from {self.args.topk+1} to {self.args.topk+5}.
}

Test_task_group_mapping = {                                                 # Similar to val task.
    "SFTTestSeqRec": ValSeqRec_group,
    "SFTTestSeqRanking": ValSeqRanking_group,
    "SFT+TestPersonalControlRec": ValPersonalControlRec_group,
    "SFT-TestPersonalControlRec": ValPersonalControlRec_group,
    "SFTTestPersonalCategoryRateLP": TestPersonalCategoryRateLP_group,
    "SFTTestPersonalCategoryRateLP1": TestPersonalCategoryRateLP1_group,
    "SFTTestPersonalCategoryRateMP": TestPersonalCategoryRateMP_group,
    "SFTTestPersonalCategoryRateEP": TestPersonalCategoryRateEP_group,
    'SFTTestItemCount': ValSeqRec_group,
}
