import copy
import math
from utils.tools import sync_dict, vague_map


def task_register(task):
    metrics_dict = {
        'NonExistRate': 0.0,
        'RepeatRate': 0.0,
        'CorrectCount': 0.0,
        'Count': 1e-24,
        'Recall': 0.0,
        'MRR': 0.0,
        'NDCG': 0.0,
        'TargetCategoryRate': 0.0,
        'InHistoryRate': 0.0,
    }
    if task in ['SFT+TestPersonalControlRec', 'SFT-TestPersonalControlRec', 'RL+PersonalControlRec',
                'RL-PersonalControlRec'] or task.startswith('SFTTestPersonalCategoryRate') or task.startswith('RLPersonalCategoryRate'):
        metrics_dict.update({
            'SRTargetCategoryRate': 0.0,
        })
    if task in ['RLSeqRec', 'RL+PersonalControlRec', 'RL-PersonalControlRec', 'RLSeqRanking', 'RLItemCount',
                'RLTotal'] or task.startswith('RLPersonalCategoryRate'):
        metrics_dict.update({
            'RewardSum': 0.0,
        })
    if task.startswith('SFTTestPersonalCategoryRate') or task.startswith('RLPersonalCategoryRate'):
        metrics_dict.update({
            'CategoryRateCorrect': 0.0,
            'SRCategoryRateCorrect': 0.0,
        })
    if task in ['SFTTestSeqRanking', 'RLSeqRanking']:
        metrics_dict.update({
            'NotInCandidateRate': 0.0,
        })
    return metrics_dict


class Metrics:
    def __init__(self, tasks, topk, category2item, title2item, accelerator=None):
        self.tasks = tasks
        self.topk = topk
        self.category2item = category2item
        self.title2item = title2item
        self.accelerator = accelerator

        self.metrics_dict = {_: task_register(_) for _ in self.tasks}

    def add_sample(self, task, input_field_data, output_titles, target_title, list_reward=0.0, vague_mapping=True):
        CorrectCount = 1 if len(output_titles) == input_field_data['item_count'] else 0
        _output_titles = output_titles[:input_field_data['item_count']]
        if vague_mapping:
            _output_titles = vague_map(_output_titles, self.title2item)
        NonExistRate = sum([1 if _ not in self.title2item else 0 for _ in _output_titles])
        RepeatRate = sum([1 if _ in _output_titles[:idx] else 0 for idx, _ in enumerate(_output_titles)])
        output_items = [self.title2item[_][0] if self.title2item.get(_) else 'None' for _ in _output_titles]
        InHistoryRate = sum([1 if _ in input_field_data['sub_sequential'] else 0 for idx, _ in enumerate(output_items)]) if 'sub_sequential' in input_field_data else -1.0

        self.metrics_dict[task]['NonExistRate'] += NonExistRate
        self.metrics_dict[task]['RepeatRate'] += RepeatRate
        self.metrics_dict[task]['InHistoryRate'] += InHistoryRate
        self.metrics_dict[task]['CorrectCount'] += CorrectCount
        self.metrics_dict[task]['Count'] += 1

        if input_field_data['target_item_title'] in _output_titles:
            idx = _output_titles.index(input_field_data['target_item_title'])
            self.metrics_dict[task]['Recall'] += 1
            self.metrics_dict[task]['MRR'] += 1/(idx+1)
            self.metrics_dict[task]['NDCG'] += 1/math.log2(idx+2)

        target_category = input_field_data['target_category']
        TargetCategoryRate = sum([1 if _ in self.category2item[target_category] else 0 for _ in output_items])
        self.metrics_dict[task]['TargetCategoryRate'] += TargetCategoryRate

        if 'SRTargetCategoryRate' in self.metrics_dict[task] and 'SeqRec_Result' in input_field_data:
            SeqRec_item_list = input_field_data['SeqRec_Result'][:input_field_data['item_count']]
            # SeqRec_item_list = [self.title2item[_][0] if self.title2item.get(_) else 'None' for _ in _SeqRec_output_titles]
            SRTargetCategoryRate = sum([1 if _ in self.category2item[target_category] else 0 for _ in SeqRec_item_list])
            self.metrics_dict[task]['SRTargetCategoryRate'] += SRTargetCategoryRate
            if 'SRCategoryRateCorrect' in self.metrics_dict[task]:
                if 'RateMP' in task or 'RateMC' in task:
                    SRCategoryRateCorrect = 1 if SRTargetCategoryRate >= input_field_data['category_count'] else 0
                elif 'RateEP' in task or 'RateEC' in task:
                    SRCategoryRateCorrect = 1 if abs(SRTargetCategoryRate - input_field_data['category_count']) <= 1 else 0
                elif 'RateLP' in task or 'RateLC' in task:
                    SRCategoryRateCorrect = 1 if (SRTargetCategoryRate+NonExistRate) <= input_field_data['category_count'] else 0
                else:
                    raise NotImplementedError
                self.metrics_dict[task]['SRCategoryRateCorrect'] += SRCategoryRateCorrect

        if 'RewardSum' in self.metrics_dict[task]:
            self.metrics_dict[task]['RewardSum'] += list_reward

        if 'CategoryRateCorrect' in self.metrics_dict[task]:
            if 'RateMP' in task or 'RateMC' in task:
                CategoryRateCorrect = 1 if TargetCategoryRate >= input_field_data['category_count'] else 0
            elif 'RateEP' in task or 'RateEC' in task:
                CategoryRateCorrect = 1 if abs(TargetCategoryRate - input_field_data['category_count']) <= 1 else 0
            elif 'RateLP' in task or 'RateLC' in task:
                CategoryRateCorrect = 1 if (TargetCategoryRate+NonExistRate) <= input_field_data['category_count'] else 0
            else:
                raise NotImplementedError
            self.metrics_dict[task]['CategoryRateCorrect'] += CategoryRateCorrect

        if 'candidate_items' in input_field_data and 'NotInCandidateRate' in self.metrics_dict[task]:
            NotInCandidateRate = sum([1 if _ not in input_field_data['candidate_titles'] else 0 for _ in _output_titles])
            self.metrics_dict[task]['NotInCandidateRate'] += NotInCandidateRate

    def __getitem__(self, item):
        return self.metrics_dict[item]

    def __iter__(self):
        return iter(self.metrics_dict.keys())

    def get_sync_metrics(self):
        """
        get the synchronized metrics dict cross all processes while using multi gpus in training.
        :return:
        """
        temp = copy.deepcopy(self.metrics_dict)
        if self.accelerator:
            temp = {t: sync_dict(self.accelerator, temp[t]) for t in temp}
        return temp

    def print(self, temp=None):
        if self.accelerator and not self.accelerator.is_main_process:
            return
        if temp is None:
            temp = copy.deepcopy(self.metrics_dict)
        temp = {_: {__: f'{temp[_][__]/temp[_]["Count"]:.4f}' if __ != 'Count' else f'{int(temp[_][__])}' for __ in temp[_]} for _ in temp}
        tasks = [_ for _ in temp]
        metrics = ['NonExistRate',
                   'RepeatRate',
                   'InHistoryRate',
                   'CorrectCount',
                   'Count',
                   'Recall',
                   'MRR',
                   'NDCG',
                   'TargetCategoryRate',
                   'SRTargetCategoryRate',
                   'CategoryRateCorrect',
                   'SRCategoryRateCorrect',
                   'NotInCandidateRate',
                   'Loss',
                   'RewardSum']
        table_rows = [f"|{_.center(24)}|{'|'.join([str(temp[__][_]).center(len(__)+4) if _ in temp[__] else '/'.center(len(__)+4) for __ in tasks])}|" for _ in metrics]
        table_rows_str = '\n'.join(table_rows)
        print(f'''
-{('tasks'+'@'+str(self.topk)).center(24, '-')}-{'-'.join([_.center(len(_)+4, '-') for _ in tasks])}-
{table_rows_str}
-{'-' * 24}-{'-'.join(['-' * (len(_)+4) for _ in tasks])}-''')
