import os
import random

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .processor import Trie_link
from .template import *
from .utils import get_item_list, get_output_text, get_history_text, \
    process_train_sample, process_train_sample_llama2, save_json, load_json, pad_sequence


class SFTDataset(Dataset):
    def __init__(self, args, task_template, task_num, data, tokenizer, mode, immediately=True):
        if not hasattr(args, 'process_index'):
            args.process_index = 0
        self.args = args
        self.task_template = task_template
        self.task_num = task_num
        self.mode = mode
        self.tokenizer = tokenizer
        self.immediately = immediately

        self.category2item = data['category']
        self.metas = data['metas']
        self.sequential = data['sequential']
        self.share_chat_gpt = data['share_chat_gpt']
        self.ranking_candidate = None
        self.share_chat_gpt_idx = self.args.process_index
        if args.use_control_symbol:
            self.item_prefix_tree, self.idx2token_ids = self.create_item_prefix_tree()

        self.item2category = {}
        for c in self.category2item:
            for i in self.category2item[c]:
                if self.item2category.get(i) is None:
                    self.item2category[i] = []
                self.item2category[i].append(c)

        self.idx2item = {self.metas[_]['emb_idx']: _ for _ in self.metas if 'emb_idx' in self.metas[_]}
        self.title2item = {}
        for _ in self.metas:
            title = self.get_item_index(_)
            if self.title2item.get(title) is None:
                self.title2item[title] = []
            self.title2item[title].append(_)

        if args.is_main_process:
            print(f'compute_{self.mode}_datum_info')
        self.datum_info = []
        self.complete_datum_info = []
        self.compute_datum_info()

    def create_item_prefix_tree(self):
        item_list = [self.get_item_index(_) for _ in self.metas]
        item_ids = self.tokenizer.batch_encode_plus(item_list, add_special_tokens=False).data['input_ids']

        idx2token_ids = {
            self.metas[_]['emb_idx']: torch.tensor([self.tokenizer.soi_token_id] + item_ids[i] + [self.tokenizer.eoi_token_id],
                                                   device=self.args.gpu, dtype=torch.long)
            for i, _ in enumerate(self.metas) if 'emb_idx' in self.metas[_]
        }
        if self.args.CBS_type == 1:
            return Trie_link(item_ids, self.tokenizer), idx2token_ids
        elif self.args.CBS_type == 2:
            return Trie_link(item_ids, self.tokenizer), idx2token_ids
        else:
            raise NotImplementedError

    def get_scope_mask(self, labels):
        batch_size = labels.size()[0]
        step_size = labels.size()[1]

        scope_mask = torch.zeros(batch_size, step_size, len(self.tokenizer), dtype=torch.bool)
        start_positions = torch.eq(labels, self.tokenizer.soi_token_id).nonzero()
        end_positions = torch.eq(labels, self.tokenizer.eoi_token_id).nonzero()

        for idx, end_pos in enumerate(end_positions):
            batch_idx, end_idx = end_pos[0], end_pos[1]

            start_pos = start_positions[idx]
            start_batch_idx, start_idx = start_pos[0], start_pos[1]

            assert batch_idx == start_batch_idx

            if end_idx <= start_idx:
                raise ValueError("End index must be greater than start index")

            for step_idx in range(start_idx + 1, end_idx):
                scope_list = self.item_prefix_tree.next_tokens(labels[batch_idx, start_idx:step_idx].tolist())
                if self.args.scope_mask_type == 1:
                    if len(scope_list) < 2:
                        continue
                elif self.args.scope_mask_type == 2:
                    if len(scope_list) < 2 and self.tokenizer.eoi_token_id in scope_list:
                        continue
                elif self.args.scope_mask_type == 3:  # navi
                    pass
                else:
                    raise NotImplementedError
                scope_mask[batch_idx, step_idx] = True
                scope_mask[batch_idx, step_idx, scope_list] = False

        scope_mask = scope_mask.to(labels.device)

        return scope_mask

    def compute_datum_info(self):
        val_num = 320
        val_task_num = 0
        for task, num in self.task_num.items():
            if task in ["SFTSeqRec", "SFTSeqRec-CS", "SFTSeqRec-MR", "SFTSeqRec-CS-MR"]:
                for _ in range(num):
                    self.datum_info += [[task, u] for u in self.sequential]

            elif task in ["SFTTestSeqRec", "SFTTestSeqRec-CS", "SFTTestSeqRec-MR", "SFTTestSeqRec-CS-MR", "SFTTestSeqRec-MR-same",
                          "SFTTestSeqRanking", "SFTTestItemCount"]:
                for _ in range(num):
                    if self.mode == 'test':
                        self.datum_info += [[task, u] for u in self.sequential]
                    elif self.mode == 'val':
                        self.datum_info += [[task, u] for u in list(self.sequential.keys())[val_num * val_task_num: val_num * (val_task_num + 1)]]
                        val_task_num += 1
            elif task == 'ShareChatGPT' and self.mode == 'train':
                share_chat_gpt_count = int(self.args.share_chat_gpt_ratio * len(self.datum_info))
                self.datum_info += [['ShareChatGPT'] for _ in range(share_chat_gpt_count)]

            elif task == "SFTTestEmbedding":
                self.datum_info += [[task, u] for u in list(self.sequential.keys())[:]]

            else:
                raise NotImplementedError

        complete_datum_info_path = os.path.join(self.args.data_path, f'SFT_datum_info_{self.mode}_{self.args.SFT_test_task}_Top{self.args.topk}.jsonl')
        self.complete_datum_info = load_json(complete_datum_info_path) or []
        if len(self.complete_datum_info) != len(self.datum_info) and self.mode in ['val', 'test']:
            self.complete_datum_info = [self.getitem(idx) for idx in tqdm(range(len(self.datum_info)), desc=f'computing {self.mode} datum info')]
            if self.args.SFT_test_task == "SFTTestEmbedding":
                save_json(self.complete_datum_info, complete_datum_info_path)

        if self.mode == 'train':
            self.shuffle()

    def shuffle(self):
        random.shuffle(self.datum_info)

    def __len__(self):
        return len(self.datum_info)

    def get_item_index(self, item):
        return self.metas[item][self.args.item_index]

    def get_sub_sequential(self, user):
        if self.mode == 'train':
            sequential = self.sequential[user][:-2]
            target_item_index = random.choice(range(1, len(sequential)))
            min_start_item_index = max(0, target_item_index - self.args.max_item_length)
            start_item_index = random.choice(range(min_start_item_index, target_item_index))
            sub_sequential = sequential[start_item_index: target_item_index]
            target_item = sequential[target_item_index]
        elif self.mode == 'val':
            sub_sequential = self.sequential[user][-self.args.max_item_length - 2:-2]
            target_item = self.sequential[user][-2]
        elif self.mode == 'test':
            sub_sequential = self.sequential[user][-self.args.max_item_length - 1:-1]
            target_item = self.sequential[user][-1]
        else:
            raise NotImplementedError
        return sub_sequential, target_item

    def get_output_item_list(self, task, user=None, sub_sequential=None, target_item=None, item_count=None, has_candidate=False):
        output_items, candidate_items = [], []
        if task in ['SFTSeqRec', 'SFTSeqRec-CS', 'SFTSeqRec-MR', 'SFTSeqRec-CS-MR', 'SFTTestSeqRanking', 'SFTTestEmbedding']:
            output_items = get_item_list(self.args.backup_ip, [user], [sub_sequential], item_count, port=self.args.teacher_port, immediately=self.immediately)
            if target_item in output_items:
                output_items.remove(target_item)
            output_items = ([target_item] + output_items)[:item_count]
        else:
            raise NotImplementedError

        assert len(output_items) == item_count
        if has_candidate:
            candidate_num = random.choice(range(self.args.topk, self.args.candidate_num + 1))
            candidate_items = output_items + random.choices(list(set(self.metas.keys()) - set(output_items)), k=candidate_num - item_count)
            random.shuffle(candidate_items)
        return output_items, candidate_items

    def getitem(self, idx):
        task = self.datum_info[idx][0]
        template_id = random.choice(list(self.task_template[task].keys()))
        template_selected = self.task_template[task][template_id]
        input_field_data, output_field_data = {}, {}

        if task in ["SFTSeqRec", "SFTSeqRec-CS", "SFTSeqRec-MR", "SFTSeqRec-CS-MR",
                    "SFTTestSeqRec", "SFTTestSeqRec-CS", "SFTTestSeqRec-MR", "SFTTestSeqRec-CS-MR", "SFTTestSeqRec-MR-same",
                    "SFTTestSeqRanking", "SFTTestItemCount", "SFTTestEmbedding"]:
            user = self.datum_info[idx][1]
            sub_sequential, target_item = self.get_sub_sequential(user)
            input_field_data.update({
                'user': user,
                'target_item': target_item,
                'sub_sequential': sub_sequential,
                'history': get_history_text([f"'{self.get_item_index(_)}'" for _ in sub_sequential]),
            })

            if task in ["SFTSeqRec", "SFTSeqRec-CS", "SFTSeqRec-MR", "SFTSeqRec-CS-MR"]:
                item_count = random.choice(range(self.args.topk)) + 1
                output_items, candidate_items = self.get_output_item_list(task, user=user, sub_sequential=sub_sequential,
                                                                          target_item=target_item, item_count=item_count,
                                                                          has_candidate='candidate_titles' in template_selected.input_fields)
                input_field_dict = {
                    'target_category': self.item2category.get(target_item)[-1],
                    'item_count': item_count,
                    'candidate_titles': ', '.join([f"'{self.get_item_index(_)}'" for _ in candidate_items]),
                    'candidate_items': candidate_items,
                }
                input_field_data.update(input_field_dict)

                output_field_data.update({
                    'item_list': output_items,
                    'item_title_list': get_output_text(
                        output_titles=[self.get_item_index(_) for _ in output_items],
                        idx=self.args.idx,
                        user_control_symbol=self.args.use_control_symbol
                    ) + '\n',
                })
            elif task in ["SFTTestSeqRec", "SFTTestSeqRec-CS", "SFTTestSeqRec-MR", "SFTTestSeqRec-CS-MR", "SFTTestSeqRec-MR-same"]:
                item_count = self.args.topk
                input_field_dict = {
                    'target_category': self.item2category.get(target_item)[-1],
                    'item_count': item_count
                }
                input_field_data.update(input_field_dict)
                output_field_data.update({
                    'item_list': [target_item],
                    'item_title_list': get_output_text([self.get_item_index(target_item)])
                })
            elif task in ['SFTTestSeqRanking']:
                item_count = self.args.topk
                ranking_candidate = self.ranking_candidate[user][:self.args.candidate_num - 1]
                insert_idx = idx % self.args.candidate_num
                ranking_candidate.insert(insert_idx, target_item)
                input_field_data.update({
                    'target_category': self.item2category.get(target_item)[-1],
                    'item_count': item_count,
                    'candidate_titles': ', '.join([f"'{self.get_item_index(_)}'" for _ in ranking_candidate]),
                    'candidate_items': ranking_candidate
                })
                output_field_data.update({
                    'item_list': [target_item],
                    'item_title_list': get_output_text([self.get_item_index(target_item)])
                })
            elif task in ['SFTTestItemCount']:
                item_count = self.args.topk + 1 + idx % 5
                input_field_data.update({
                    'target_category': self.item2category.get(target_item)[-1],
                    'item_count': item_count
                })
                output_field_data.update({
                    'item_list': [target_item],
                    'item_title_list': get_output_text([self.get_item_index(target_item)])
                })
            elif task in ['SFTTestEmbedding']:
                item_count = self.args.topk
                output_items, candidate_items = self.get_output_item_list(task, user=user, sub_sequential=sub_sequential,
                                                                          target_item=target_item, item_count=item_count,
                                                                          has_candidate='candidate_titles' in template_selected.input_fields)

                input_field_dict = {
                    'target_category': self.item2category.get(target_item)[-1],
                    'item_count': item_count,
                    'candidate_titles': ', '.join([f"'{self.get_item_index(_)}'" for _ in candidate_items]),
                    'candidate_items': candidate_items,
                }
                input_field_data.update(input_field_dict)

                output_field_data.update({
                    'item_list': output_items,
                    'item_title_list': get_output_text(
                        output_titles=[self.get_item_index(_) for _ in output_items],
                        idx=self.args.idx,
                        user_control_symbol=self.args.use_control_symbol
                    ) + '\n',
                })

        elif task == "ShareChatGPT":
            scg_data = self.share_chat_gpt[self.share_chat_gpt_idx % len(self.share_chat_gpt)]
            self.share_chat_gpt_idx += self.args.num_processes
            chat_end_idx = random.choice([idx for idx, c in enumerate(scg_data) if c['from'] == 'human'])
            chat_start_idxes = [chat_end_idx]
            chat_start_idx = chat_start_idxes[-1] - 2
            while chat_start_idx > 0:
                chat_start_idx = chat_start_idx - 2
                pre_length = scg_data[chat_start_idx - 1]['pre_length'] if chat_start_idx > 0 else 0
                if (scg_data[chat_start_idx]['pre_length'] - pre_length) < (512 - 64):
                    chat_start_idxes.append(chat_start_idx)
                else:
                    break
            input_texts, output_texts = [], []
            input_texts.append(scg_data[0]['value'])
            output_texts.append(scg_data[1]['value'])
            out_dict = {
                'input_texts': input_texts,
                'output_texts': output_texts,
                'task': task,
            }
            return out_dict

        else:
            raise NotImplementedError

        output_field_data['emb_idx_list'] = [self.metas[_].get('emb_idx') for _ in output_field_data['item_list']]
        input_texts = template_selected.get_input_text(input_field_data)
        output_texts = template_selected.get_output_text(output_field_data)

        if task in ["SFTSeqRec-CS-MR", "SFTSeqRec-MR"] and random.random() < self.args.multi_round_ratio:
            scg_data = self.share_chat_gpt[self.share_chat_gpt_idx % len(self.share_chat_gpt)]
            self.share_chat_gpt_idx += self.args.num_processes
            # chat_start_idx = random.choice([idx for idx, c in enumerate(scg_data) if c['from'] == 'human'])
            chat_start_idx = 0
            insert_idx = random.choice(range(len(input_texts) + 1))
            assert scg_data[chat_start_idx]['from'] == 'human'
            input_texts.insert(insert_idx, scg_data[chat_start_idx]['value'])
            assert scg_data[chat_start_idx + 1]['from'] == 'gpt'
            output_texts.insert(insert_idx, scg_data[chat_start_idx + 1]['value'])

        out_dict = {
            'task': task,
            'input_field_data': input_field_data,
            'output_field_data': output_field_data,
            'input_texts': input_texts,
            'output_texts': output_texts,
        }
        return out_dict

    def __getitem__(self, idx):
        if self.mode in ['train']:
            return self.getitem(idx)
        return self.complete_datum_info[idx]

    def collate_fn(self, batch):
        batch_entry = {}
        tasks = []
        input_texts = []
        infer_text = []
        output_texts = []
        input_ids = []
        complete_ids = []
        labels = []
        input_field_data = []
        output_field_data = []
        for i, entry in enumerate(batch):
            if 'task' in entry:
                tasks.append(entry['task'])
            if 'input_texts' in entry:
                input_texts.append(entry['input_texts'])
            if 'output_texts' in entry:
                output_texts.append(entry['output_texts'])
            if 'input_field_data' in entry:
                input_field_data.append(entry['input_field_data'])
            if 'output_field_data' in entry:
                output_field_data.append(entry['output_field_data'])

            if self.args.chat_template == 'llama-3':
                i_ids, c_ids, lab, i_text = process_train_sample(entry['input_texts'], entry['output_texts'], self.tokenizer)
            elif self.args.chat_template == 'llama-2':
                i_ids, c_ids, lab, i_text = process_train_sample_llama2(entry['input_texts'], entry['output_texts'], self.tokenizer)
            else:
                raise NotImplementedError
            input_ids.append(i_ids)
            complete_ids.append(c_ids)
            labels.append(lab)
            infer_text.append(i_text)

        input_ids = pad_sequence(input_ids, self.tokenizer.pad_token_id, self.args.gpu, pad_side='left')
        complete_ids = pad_sequence(complete_ids, self.tokenizer.pad_token_id, self.args.gpu, pad_side='left')
        labels = pad_sequence(labels, self.tokenizer.pad_token_id, self.args.gpu, pad_side='left')

        batch_entry['task'] = tasks
        batch_entry['input_texts'] = input_texts
        batch_entry['output_texts'] = output_texts
        batch_entry['infer_text'] = infer_text
        batch_entry['input_field_data'] = input_field_data
        batch_entry['output_field_data'] = output_field_data

        batch_entry['input_data'] = {
            'input_ids': input_ids,
            'attention_mask': torch.where(torch.eq(input_ids, self.tokenizer.pad_token_id), 0, 1)
        }

        batch_entry['complete_text_data'] = {
            'input_ids': complete_ids,
            'attention_mask': torch.where(torch.eq(complete_ids, self.tokenizer.pad_token_id), 0, 1)
        }
        batch_entry['complete_label_ids'] = labels

        if self.args.train_stage in ['SFT_Embedding', 'SFT_Embedding_Test']:
            lm_mask = torch.zeros_like(batch_entry['complete_label_ids'], dtype=torch.long)
            lm_mask[torch.eq(batch_entry['complete_label_ids'], self.tokenizer.soi_token_id)] = 1
            lm_mask[torch.eq(batch_entry['complete_label_ids'], self.tokenizer.eoi_token_id)] = -1
            lm_mask = (torch.cumsum(lm_mask, dim=1) - lm_mask).to(dtype=torch.bool)
            batch_entry['complete_label_ids'][lm_mask] = -100
            batch_entry['lm_mask'] = lm_mask

        return batch_entry


Train_task_group_mapping = {
    # "SFTSeqRec": SeqRec_group,
    # "SFTSeqRec-CS": SeqRec_CS_group,
    "SFTSeqRec-MR": SeqRec_MR_group,        # MR1
    "SFTSeqRec-CS-MR": SeqRec_CS_MR_group,
    "ShareChatGPT": {'ShareChatGPT-1': ''},
}

Val_task_group_mapping = {
    "SFTTestSeqRec": ValSeqRec_group,
    "SFTTestSeqRec-MR": ValSeqRec_MR_group,
    "SFTTestSeqRec-MR-same": ValSeqRec_MR_same_group,
    # "SFTTestSeqRec-CS": ValSeqRec_CS_group,
    # "SFTTestSeqRec-CS-MR": ValSeqRec_CS_MR_group,
    # "SFTTestSeqRanking": ValSeqRanking_group,
    # 'SFTTestItemCount': ValSeqRec_group,
}

Test_task_group_mapping = {
    "SFTTestSeqRec": ValSeqRec_group,
    "SFTTestSeqRec-MR": ValSeqRec_MR_group,
    "SFTTestSeqRec-MR-same": ValSeqRec_MR_same_group,
    # "SFTTestSeqRec-CS": ValSeqRec_CS_group,
    "SFTTestSeqRec-CS-MR": ValSeqRec_CS_MR_group,
    "SFTTestEmbedding": ValSeqRec_MR_same_group,
    # "SFTTestSeqRanking": ValSeqRanking_group,
    # 'SFTTestItemCount': ValSeqRec_group,
}


if __name__ == '__main__':
    pass
