# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from rl.template import *
from utils.tools import load_pickle, save_pickle, get_history_text, get_output_text, get_item_list, side_tokenizer


class ExperienceDataset(Dataset):
    def __init__(self, data: List[torch.Tensor], device=None):
        super().__init__()
        self.data = data
        self.device = device

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, ind):
        return tuple(map(lambda t: t[ind].to(self.device), self.data))


def create_dataloader(data, batch_size, shuffle=True, device=None, **kwargs):
    ds = ExperienceDataset(data, device=device)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, **kwargs)


class RLDataset(Dataset):
    def __init__(self, args, task_template, task_num, data, tokenizer, mode='train', saving=False, immediately=True):
        self.args = args
        self.task_template = task_template
        self.task_num = task_num
        self.mode = mode
        self.tokenizer = tokenizer
        self.teacher_port = self.args.teacher_port
        self.saving = saving
        self.immediately = immediately

        self.metas = data['metas']
        self.title2item = data['title2item']
        self.sequential = data['sequential']
        self.category2item = data['category2item']
        self.item2category = data['item2category']
        self.ranking_candidate = data['ranking_candidate']
        if 'RLSeqRec_Result' in data:
            self.RLSeqRec_Result = {u: data['RLSeqRec_Result'][idx][1] for idx, u in enumerate(self.sequential)}

        # self.title = [__[self.args.item_index] for _, __ in self.metas.items()]
        # self.decoder_start_token_id = self.tokenizer.bos_token_id
        # self.item_prefix_tree = self.create_item_prefix_tree()

        self.datum_info = []
        self.complete_datum_info = []
        self.compute_datum_info()

    def find_maximum_category(self, item_list, target_item, max_count=99999):
        category_count = {c: 0 for c in self.category2item if target_item not in self.category2item[c]}
        for o_i in item_list:
            for c in category_count:
                if o_i in self.category2item[c]:
                    category_count[c] += 1
        max_count = min(max_count, max(list(category_count.values())))
        category = [c for c in list(category_count.keys()) if category_count[c] >= max_count]
        return category

    # def create_item_prefix_tree(self):
    #     title_index = [self.get_item_index(self.metas[_]['asin']) for _ in self.metas]
    #     item_list = [self.metas[_]['asin'] for _ in self.metas]
    #     title_ids = self.tokenizer.batch_encode_plus(title_index,
    #                                                  padding=True, truncation=True,
    #                                                  max_length=self.args.gen_max_length,
    #                                                  return_tensors='pt').data['input_ids']
    #     if torch.all(torch.eq(title_ids[:, 0], self.decoder_start_token_id)):
    #         title_ids = title_ids[:, 1:]
    #     item_prefix_tree = {str(self.decoder_start_token_id): []}
    #     for i, ids in enumerate(title_ids):
    #         temp = str(self.decoder_start_token_id)
    #         for token in ids:
    #             _next = int(token)
    #             if token == self.tokenizer.pad_token_id or token == self.tokenizer.eos_token:
    #                 break
    #             if item_prefix_tree.get(temp) is None:
    #                 item_prefix_tree[temp] = []
    #             if _next not in item_prefix_tree[temp]:
    #                 item_prefix_tree[temp].append(_next)
    #             temp = temp + ' ' + str(_next)
    #         if item_prefix_tree.get(temp) is None:
    #             item_prefix_tree[temp] = []
    #         item_prefix_tree[temp].append(item_list[i])
    #     return item_prefix_tree

    def compute_datum_info(self):
        val_num = self.args.val_num_per_task
        for task, num in self.task_num.items():
            if task == "RLSeqRec":
                for _ in range(num):
                    if self.mode in ['train', 'test']:
                        self.datum_info += [[task, u] for u in self.sequential]
                    elif self.mode == 'val':
                        self.datum_info += [[task, u] for u in list(self.sequential.keys())[val_num*0:val_num*1]]
            elif task == "RL+PersonalControlRec":
                for _ in range(num):
                    if self.mode in ['train', 'test']:
                        self.datum_info += [[task, u] for u in self.sequential]
                    elif self.mode == 'val':
                        self.datum_info += [[task, u] for u in list(self.sequential.keys())[val_num*1:val_num*2]]
            elif task == "RL-PersonalControlRec":
                for _ in range(num):
                    if self.mode in ['train', 'test']:
                        self.datum_info += [[task, u] for u in self.sequential]
                    elif self.mode == 'val':
                        self.datum_info += [[task, u] for u in list(self.sequential.keys())[val_num*2:val_num*3]]
            elif task == 'RLPersonalCategoryRate':
                for _ in range(num):
                    if self.mode in ['train', 'test']:
                        self.datum_info += [[task, u] for u in self.sequential]
                    elif self.mode == 'val':
                        self.datum_info += [[task, u] for u in list(self.sequential.keys())[val_num*3:val_num*4]]
            elif task.startswith('RLPersonalCategoryRate'):
                for _ in range(num):
                    if self.mode in ['train', 'test']:
                        self.datum_info += [[task, u] for u in self.sequential]
                    elif self.mode == 'val':
                        self.datum_info += [[task, u] for u in list(self.sequential.keys())[val_num*3:val_num*4]]
            elif task == "RLSeqRanking":
                for _ in range(num):
                    if self.mode in ['train', 'test']:
                        self.datum_info += [[task, u] for u in self.sequential]
                    elif self.mode == 'val':
                        self.datum_info += [[task, u] for u in list(self.sequential.keys())[val_num*4:val_num*5]]
            elif task == "RLItemCount":
                for _ in range(num):
                    if self.mode in ['test']:
                        self.datum_info += [[task, u] for u in self.sequential]
                    elif self.mode == 'val':
                        self.datum_info += [[task, u] for u in list(self.sequential.keys())[val_num*5:val_num*6]]
            else:
                raise NotImplementedError

        complete_datum_info_path = None
        if self.mode == 'val':
            complete_datum_info_path = self.args.data_path + f'RL_datum_info_{self.mode}_{self.args.RL_val_tasks}_Top{self.args.topk}.pickle'
        elif self.mode == 'train':
            complete_datum_info_path = self.args.data_path + f'RL_datum_info_{self.mode}_{self.args.RL_train_tasks}_Top{self.args.topk}.pickle'

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
        elif self.mode == 'test':
            sub_sequential = self.sequential[user][-self.args.max_item_length-1:-1]
            target_item = self.sequential[user][-1]
        else:
            raise NotImplementedError
        return sub_sequential, target_item

    def getitem(self, idx):
        task = self.datum_info[idx][0]
        template_id = random.choice(list(self.task_template[task].keys()))
        template_selected = self.task_template[task][template_id]
        item_count = random.choice(range(self.args.topk//2, self.args.topk))+1 if self.mode == 'train' else self.args.topk
        input_field_data, output_field_data = {'item_count': item_count}, {}
        if task in ["RLSeqRec", 'RL+PersonalControlRec', 'RL-PersonalControlRec', 'RLSeqRanking',
                    'RLItemCount'] or task.startswith('RLPersonalCategoryRate'):
            user = self.datum_info[idx][1]
            sub_sequential, target_item = self.get_sub_sequential(user)
            input_field_data.update({
                'history': get_history_text([f"'{self.get_item_index(_)}'" for _ in sub_sequential]),
                'user': user,
                'sub_sequential': sub_sequential,
                'target_item': target_item,
                'target_item_title': self.get_item_index(target_item),
                'target_category': self.item2category[target_item][-1]
            })
            output_field_data.update({
                'item_list': get_output_text([self.get_item_index(target_item)], '', idx=False)
            })

            if task in ["RL+PersonalControlRec"]:
                if self.mode in ['train', 'val']:
                    item_list = get_item_list(self.args.backup_ip, [user], [sub_sequential], item_count, port=self.teacher_port, immediately=self.immediately)
                else:
                    item_list = [self.title2item[_][0] if _ in self.title2item else 'None' for _ in self.RLSeqRec_Result[user]]

                if self.mode == 'train':
                    target_category = random.choice(self.item2category[target_item])
                else:
                    target_category = self.item2category[target_item][-1]
                input_field_data.update({
                    'target_category': target_category,
                    'SeqRec_Result': item_list
                })
                intention_template_key = random.choice(list(Intention_plus_group.keys()))
                intention = Intention_plus_group[intention_template_key].get_input_text(input_field_data)
                input_field_data.update({
                    'synthetic_intention': intention,
                })
            elif task in ["RL-PersonalControlRec"]:
                if self.mode in ['train', 'val']:
                    item_list = get_item_list(self.args.backup_ip, [user], [sub_sequential], item_count, port=self.teacher_port, immediately=self.immediately)
                else:
                    item_list = [self.title2item[_][0] if _ in self.title2item else 'None' for _ in self.RLSeqRec_Result[user]]

                if self.mode in ['train']:
                    target_category = random.choice(self.find_maximum_category(item_list, target_item, max_count=2))
                else:
                    target_category = self.find_maximum_category(item_list, target_item)[-1]

                input_field_data.update({
                    'target_category': target_category,
                    'SeqRec_Result': item_list
                })
                intention_template_key = random.choice(list(Intention_minus_group.keys()))
                intention = Intention_minus_group[intention_template_key].get_input_text(input_field_data)
                input_field_data.update({
                    'synthetic_intention': intention,
                })
            elif task.startswith("RLPersonalCategoryRate"):
                if self.mode in ['val', 'test']:
                    item_count = self.args.topk
                if self.mode in ['train', 'val']:
                    item_list = get_item_list(self.args.backup_ip, [user], [sub_sequential], item_count, port=self.teacher_port, immediately=self.immediately)
                else:
                    item_list = [self.title2item[_][0] if _ in self.title2item else 'None' for _ in self.RLSeqRec_Result[user]]

                if self.mode in ['train']:
                    if 'RateMP' in task or 'RateEP' in task:
                        target_category = random.choice(self.item2category[target_item])
                    elif 'LP' in task:
                        target_category = random.choice(self.find_maximum_category(item_list, target_item, max_count=2))
                    else:
                        raise NotImplementedError
                else:
                    if 'RateMP' in task or 'RateEP' in task:
                        target_category = self.item2category[target_item][-1]
                    elif 'RateLP' in task:
                        target_category = self.find_maximum_category(item_list, target_item)[-1]
                    else:
                        raise NotImplementedError
                input_field_data.update({
                    'target_category': target_category,
                    'SeqRec_Result': item_list
                })

                if self.mode != 'train':
                    p = int(task.split('_')[-1])
                    output_category_item_count = int(p*item_count / 100)
                else:
                    category_item_count = min(len(self.category2item[target_category]), item_count)
                    if 'LP_' in task:
                        target_category_item_count = random.choice(range(category_item_count))+1
                        output_category_item_count = target_category_item_count-1
                    elif 'MP_' in task:
                        target_category_item_count = random.choice(range(category_item_count))
                        output_category_item_count = target_category_item_count
                    else:
                        target_category_item_count = random.choice(range(category_item_count+1))
                        output_category_item_count = target_category_item_count
                    p = int(target_category_item_count/item_count*10)*10
                input_field_data.update({
                    'item_count': item_count,
                    'category_proportion': f"{p}%",
                    'category_count': output_category_item_count,
                })
            elif task in ["RLSeqRanking"]:
                if self.mode == 'train':
                    candidate_num = random.choice(range(item_count, self.args.candidate_num)) + 1
                    output_items = [target_item]
                    ranking_candidate = output_items + random.choices(list(set(self.metas.keys()) - set(output_items)), k=candidate_num-1)
                    random.shuffle(ranking_candidate)
                else:
                    ranking_candidate = self.ranking_candidate[user][:self.args.candidate_num - 1]
                    insert_idx = idx % self.args.candidate_num
                    ranking_candidate.insert(insert_idx, target_item)
                input_field_data.update({
                    'candidate_titles': ', '.join([f"'{self.get_item_index(_)}'" for _ in ranking_candidate]),
                    'candidate_items': ranking_candidate
                })
            elif task in ["RLItemCount"]:
                item_count = self.args.topk + 1 + idx % 5
                input_field_data.update({'item_count': item_count})

        input_text = template_selected.get_input_text(input_field_data, llama2_chat_template=self.args.llama2_chat_template).strip()
        output_text = template_selected.get_output_text(output_field_data).strip()
        out_dict = {
            'task': task,
            'input_text': input_text,
            'output_text': output_text,
            'input_field_data': input_field_data
        }
        return out_dict

    def __getitem__(self, idx):
        if len(self.complete_datum_info) > 0:
            return self.complete_datum_info[idx]
        return self.getitem(idx)

    def collate_fn(self, batch):
        batch_entry = {}
        tasks = []
        output_text = []
        input_text = []
        input_field_data = []
        for i, entry in enumerate(batch):
            if 'task' in entry:
                tasks.append(entry['task'])
            if 'input_text' in entry:
                input_text.append(entry['input_text'])
            if 'input_field_data' in entry:
                input_field_data.append(entry['input_field_data'])
            if 'output_text' in entry:
                output_text.append(entry['output_text'])
        batch_entry['input_text'] = input_text
        batch_entry['output_text'] = output_text

        batch_entry['input_data'] = side_tokenizer(batch_entry['input_text'],
                                                   'left', self.tokenizer,
                                                   padding=True, truncation=True,
                                                   max_length=self.args.max_token_length,
                                                   return_tensors='pt').to(self.args.gpu).data

        if len(input_field_data) > 0:
            batch_entry['input_field_data'] = input_field_data
        if len(tasks) > 0:
            batch_entry['task'] = tasks
        return batch_entry


Train_task_group_mapping = {
    "RLSeqRec": SeqRec_group,
    "RLSeqRanking": SeqRanking_group,
    "RL+PersonalControlRec": PersonalControlRec_group,
    "RL-PersonalControlRec": PersonalControlRec_group,
    "RLPersonalCategoryRate": PersonalCategoryRate_group,
    "RLPersonalCategoryRateLP": PersonalCategoryRateLP1_group,
    "RLPersonalCategoryRateMP": PersonalCategoryRateMP_group,
    "RLPersonalCategoryRateEP": PersonalCategoryRateEP_group,
}

Val_task_group_mapping = {
    "RLSeqRec": SeqRec_group,
    "RLSeqRanking": SeqRanking_group,
    "RLItemCount": SeqRec_group,
    "RL+PersonalControlRec": PersonalControlRec_group,
    "RL-PersonalControlRec": PersonalControlRec_group,
    "RLPersonalCategoryRate": PersonalCategoryRate_group,
    "RLPersonalCategoryRateLP": PersonalCategoryRateLP_group,
    "RLPersonalCategoryRateMP": PersonalCategoryRateMP_group,
    "RLPersonalCategoryRateEP": PersonalCategoryRateEP_group,
}

Test_task_group_mapping = {
    "RLSeqRec": SeqRec_group,
    "RLSeqRanking": SeqRanking_group,
    "RLItemCount": SeqRec_group,
    "RL+PersonalControlRec": PersonalControlRec_group,
    "RL-PersonalControlRec": PersonalControlRec_group,
    "RLPersonalCategoryRate": PersonalCategoryRate_group
}
