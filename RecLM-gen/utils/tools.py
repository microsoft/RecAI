import copy
import os.path
import pickle
import re
import requests
import torch
import torch.nn.functional as F
from Levenshtein import distance
from einops import rearrange
from openai import OpenAI
from openai.lib.azure import AzureOpenAI
from torch.nn.utils.rnn import pad_sequence
from collections import namedtuple


def rm_idx(s):
    return re.sub(r'^(\d+)\. *', '', s, count=1)


def match_idx(s):
    return re.match(r'^(\d+)\. *', s)


def vague_map(titles, all_titles):
    temp = copy.deepcopy(titles)
    for idx, title in enumerate(temp):
        if title in all_titles:
            continue
        for _title in all_titles:
            if distance(title, _title) <= 3:
                temp[idx] = _title
                break
    return temp


def load_pickle(filename):
    if filename is None or not os.path.exists(filename):
        return None
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    if filename is None:
        return
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def side_tokenizer(text: list[str] | list[list[str]], padding_side, tokenizer, **kwargs):
    """
    :param text:
    :param padding_side: in ['left', 'right']
    :param tokenizer:
    :param kwargs:
    :return:
    """
    tokenizer.padding_side = padding_side
    tokenizer.truncation_side = padding_side
    return tokenizer.batch_encode_plus(text, **kwargs)


def sync_dict(accelerator, data: dict):
    """
    get the synchronized dict cross all processes while use multi gpus
    :param accelerator:
    :param data:
    :return:
    """
    temp = copy.deepcopy(data)
    data_tensor = torch.tensor([v for k, v in data.items()], device=accelerator.device)
    data_tensor = accelerator.reduce(data_tensor)
    for k, v in zip(temp, data_tensor):
        temp[k] = float(v)
    return temp


def get_item_list(ip, users, sub_sequential, k, candidate_item_list=None, target_category=None, port=12621, immediately=True):
    """
    :param ip:
    :param users: user id
    :param sub_sequential: user history, [[item_1, ..., item_n]]
    :param k: top_k recommendation
    :param candidate_item_list: candidate items, [[item_1, ..., item_m]]
    :param target_category: '+C': only keep items in C. '-C': exclude items in C.
    :param port:
    :param immediately: don't wait for batching
    :return: return the recommendation list with k items that complying the params.
    """
    url = f"http://{ip}:{port}/inference"
    data = {
        "users": users,
        "item_lengths": [len(_) for _ in sub_sequential],
        "k": k,
        "item_lists": sub_sequential,
        "immediately": 1 if immediately else 0
    }
    if candidate_item_list is not None:
        data['candidate_item_lists'] = candidate_item_list
    if target_category is not None:
        data['target_category'] = target_category
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36 Edg/83.0.478.45",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive"
    }

    response = requests.post(url, json=data, headers=headers)
    assert response.status_code == 200
    return response.json()['inference'][0]


def get_item_ranking(ip, users, sub_sequential, candidate_item_list, port=12621, immediately=True):
    """
    :param ip:
    :param users: user id
    :param sub_sequential: user history
    :param candidate_item_list:
    :param port:
    :param immediately:
    :return: return the rank of each item in candidate_item_list among total item corpus.
    """
    url = f"http://{ip}:{port}/ranking"
    data = {
        "users": users,
        "item_lengths": [len(_) for _ in sub_sequential],
        "item_lists": sub_sequential,
        "candidate_item_lists": candidate_item_list,
        "immediately": 1 if immediately else 0
    }
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36 Edg/83.0.478.45",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive"
    }

    response = requests.post(url, json=data, headers=headers)
    assert response.status_code == 200
    return response.json()['ranking'][0]


def masked_mean(seq, mask, dim=None):
    return (seq * mask).sum(dim=dim) / mask.sum(dim=dim)


def masked_var(values, mask, unbiased=True):
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def whiten(values, masks, shift_mean=True, dim=None):
    if shift_mean:
        mean, var = masked_mean(values, masks, dim=dim), masked_var(values, masks)
        whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    else:
        var = masked_var(values, masks)
        whitened = values * torch.rsqrt(var + 1e-8)
    return whitened


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def log_prob(prob, indices):
    assert prob.shape[
           :2] == indices.shape, f'preceding shapes of prob {prob.shape[:2]} and indices {indices.shape} must match'
    return log(prob.gather(-1, indices[..., None])).squeeze(-1)


def shift(t, value=0, shift=1, dim=-1):
    zeros = (0, 0) * (-dim - 1)
    return F.pad(t, (*zeros, shift, -shift), value=value)


def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out

    return inner


def get_history_text(output_titles: list[str]):
    history_text = ' → '.join(output_titles)
    return history_text


def get_output_text(output_titles: list[str], eos_token='', idx=False):
    if not idx:
        output_text = '\n '.join(output_titles)
    else:
        output_text = '\n '.join([f'{i+1}. {t}' for i, t in enumerate(output_titles)])
    return output_text + eos_token


# <s>input_text item1\n item2\n item3\n</s>
def get_complete_text(input_text: str, output_titles: str):
    return input_text + ' ' + output_titles


class GPT:
    def __init__(self) -> None:
        self.client = None
        self.max_wrong_time = 2
        self.api_base = os.environ['OPENAI_API_BASE'] if 'OPENAI_API_BASE' in os.environ else None
        self.api_version = os.environ['OPENAI_API_VERSION'] if 'OPENAI_API_VERSION' in os.environ else None
        self.api_type = os.environ['OPENAI_API_TYPE'] if 'OPENAI_API_TYPE' in os.environ else None
        self.api_key = os.environ['OPENAI_API_KEY'] if 'OPENAI_API_KEY' in os.environ else 'Empty'
        self.engine = os.environ['ENGINE'] if 'ENGINE' in os.environ else None
        self.init_client()
        print(f'use model of {self.engine}')

    def init_client(self):
        if self.api_type == "azure":
            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.api_base,
                max_retries=self.max_wrong_time,
            )
        else:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                max_retries=self.max_wrong_time,
            )

    def call(self, content, t=0.0):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            temperature=t,
            # top_p=0.2,
            max_tokens=2048,
            model=self.engine,
        )
        response = chat_completion.choices[0].message.content
        return response

    def test(self):
        try:
            print(self.call('Hello.'))
        except Exception as e:
            print(e)


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = torch.zeros(shape)
        self.S = torch.zeros(shape)
        self.std = torch.sqrt(self.S)

    def update(self, x):
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.clone()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = torch.sqrt(self.S / self.n)


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = torch.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = (x-self.running_ms.mean) / (self.running_ms.std + 1e-8)  # Only divided std
        return x


class RunningMoments:
    def __init__(self):
        """
        Calculates the running mean and standard deviation of a data stream. Reference:
        https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/utils.py#L75
        """
        self.mean = 0
        self.std = 1
        self.var = 1
        self.count = 1e-24

    @torch.no_grad()
    def update(self, xs: torch.Tensor):
        """
        Updates running moments from batch's moments computed across ranks
        """
        xs_count = xs.numel()
        xs_var, xs_mean = torch.var_mean(xs, unbiased=False)
        xs_mean, xs_var = xs_mean.float(), xs_var.float()

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta**2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += delta * xs_count / tot_count
        self.var = tot_sum / tot_count
        self.std = (self.var * tot_count / (tot_count - 1)).float().sqrt()
        self.count = tot_count

        return xs_mean.item(), (xs_var * xs_count / (xs_count - 1)).float().sqrt().item()


Memory = namedtuple('Memory', [
    'sequence',
    'action_mask',
    'old_action_value',
    'old_sequence_log_probs_shifted',
    'ref_sequence_log_probs_shifted',
    'whitened_advantages',
    'returns'
])
if __name__ == '__main__':
    pass
