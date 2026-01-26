import math
import torch

from typing import Callable, List
from transformers import LogitsProcessor, add_start_docstrings
from transformers.generation.logits_process import LOGITS_PROCESSOR_INPUTS_DOCSTRING


class Node:
    def __init__(self, token_id):
        self.token_id = token_id
        self.children = {}
        self.count = 0

    def add_child(self, child_token_id):
        if child_token_id not in self.children:
            self.children[child_token_id] = Node(child_token_id)
        self.children[child_token_id].count += 1

    def __getitem__(self, child_token_id):
        if child_token_id not in self.children:
            return None
        return self.children[child_token_id]


class Trie_link:
    def __init__(self, item_ids, tokenizer):
        self.tokenizer = tokenizer
        self.all_token_ids = list(range(len(self.tokenizer)))
        self.max_height = max([len(one) for one in item_ids])
        self.trie = Node(-1)
        for _, token_ids in enumerate(item_ids):
            temp_node = self.trie
            token_ids_wcs = [self.soi_token_id] + token_ids + [self.eoi_token_id]
            for child_token_id in token_ids_wcs:
                temp_node.add_child(child_token_id)
                temp_node = temp_node[child_token_id]

    @property
    def soi_token_id(self):
        return self.tokenizer.soi_token_id

    @property
    def eoi_token_id(self):
        return self.tokenizer.eoi_token_id

    def next_tokens(self, current_seq):
        if len(current_seq) == 0:
            return None

        temp_node = self.trie
        try:
            for child_token_id in current_seq:
                temp_node = temp_node[child_token_id]
            return list(temp_node.children.keys())
        except:
            return None

    def constrain_search_list(self, batch_id, input_ids):
        """
        Args:
            batch_id: int
            input_ids: [seq_len]
        Returns:
        """
        try:
            last_eos_indices = torch.eq(input_ids, self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            last_eos_index = last_eos_indices[-1].item() if last_eos_indices.shape[0] > 0 else -1
            soi_indices = torch.eq(input_ids, self.soi_token_id).nonzero(as_tuple=True)[0]
            soi_indices = soi_indices[soi_indices > last_eos_index]
            eoi_indices = torch.eq(input_ids, self.eoi_token_id).nonzero(as_tuple=True)[0]
            eoi_indices = eoi_indices[eoi_indices > last_eos_index]
            last_eoi_index = eoi_indices[-1].item() if len(eoi_indices) > 0 else -1
            last_soi_index = soi_indices[-1].item() if len(soi_indices) > 0 else -1
            if last_soi_index <= last_eoi_index:
                return None

            visited_count = {}
            temp_node = None
            for s_idx in soi_indices:
                temp_node = self.trie
                temp_idx = s_idx.item()
                while temp_idx < input_ids.shape[0]:
                    temp_token = input_ids[temp_idx].item()
                    temp_node = temp_node[temp_token]
                    if temp_node is None:
                        break

                    temp_node_addr = id(temp_node)
                    if temp_node_addr not in visited_count:
                        visited_count[temp_node_addr] = 0
                    visited_count[temp_node_addr] += 1

                    if temp_token == self.eoi_token_id:
                        break
                    temp_idx += 1

            if temp_node is not None:
                return [
                    next_token_id
                    for next_token_id, child in temp_node.children.items()
                    if (id(child) not in visited_count or visited_count[id(child)] < child.count) and isinstance(next_token_id, int)
                ]
            else:
                return None
        except Exception as e:
            print("batch id: {}".format(batch_id))
            print(input_ids)
            print(self.tokenizer.batch_decode(input_ids, skip_special_tokens=False))
            print(e)
            return None


class FastPrefixConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]], num_beams: int):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, sent)
                if prefix_allowed_tokens is None:
                    mask[batch_id * self._num_beams + beam_id] = 0
                else:
                    if len(prefix_allowed_tokens) == 0:
                        raise ValueError(
                            f"`prefix_allowed_tokens_fn` returned an empty list for batch ID {batch_id}."
                            f"This means that the constraint is unsatisfiable. Please check your implementation"
                            f"of `prefix_allowed_tokens_fn` "
                        )
                    mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0

        return scores + mask


if __name__ == '__main__':
    from transformers import AutoTokenizer
    _item_ids = [
        [32, 426, 356, 423],
        [32, 426, 356, 423, 469, 435],
        [32, 426, 356, 469, 435],
        [32, 426, 469, 435]
    ]
    _tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
    _tokenizer.soi_token_id = 128257
    _tokenizer.eoi_token_id = 128256
    tire = Trie_link(_item_ids, _tokenizer)

    res = tire.constrain_search_list(0, torch.tensor([128257, 32, 426, 356, 423, 128256, 123,
                                                      128257, 32, 426, 356, 423, 469, 128256, 123,
                                                      128257, 32, 426, 356]))
    print(res)
