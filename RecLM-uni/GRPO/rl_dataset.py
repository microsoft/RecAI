import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

from train_utils.template import SeqRec_MR_group
from train_utils.utils import load_json, get_history_text, get_output_text


SYSTEM_PROMPT = "You are an expert recommender engine as well as a helpful, respectful and honest assistant."
DEFAULT_TEMPLATE_ID = next(iter(SeqRec_MR_group.keys()))


@dataclass
class RLSample:
    prompt: str
    reference_response: str
    target_title: str
    target_item: str
    history_text: str


def build_llama3_prompt(user_turn: str) -> str:
    """Create a Meta-Llama-3 style prompt with a single user turn."""
    parts = [
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
        SYSTEM_PROMPT,
        "<|eot_id|>",
        "<|start_header_id|>user<|end_header_id|>\n\n",
        user_turn,
        "<|eot_id|>",
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
    ]
    return "".join(parts)


def _sample_history(sequential: List[str], max_item_length: int, rng: random.Random) -> Tuple[List[str], str]:
    """Replicate SFT sub sequence sampling strategy."""
    if len(sequential) <= 2:
        return [], None
    prefix = sequential[:-1]
    target_idx = rng.randrange(1, len(prefix))
    min_start = max(0, target_idx - max_item_length)
    start_idx = rng.randrange(min_start, target_idx + 1)
    history = prefix[start_idx:target_idx]
    target = prefix[target_idx]
    return history, target


def build_rl_samples(
    data_path: str,
    item_index_field: str,
    max_samples: int,
    seed: int,
    topk: int,
    max_item_length: int,
    use_control_symbol: bool,
    use_idx: bool,
) -> List[RLSample]:
    """Create RL samples directly from sequential logs."""
    metas: Dict[str, Dict] = load_json(f"{data_path}metas.jsonl")
    sequential: Dict[str, List[str]] = load_json(f"{data_path}sequential.jsonl")
    template = SeqRec_MR_group[DEFAULT_TEMPLATE_ID]
    rng = random.Random(seed)
    valid_items = [iid for iid, meta in metas.items() if meta.get(item_index_field)]
    samples: List[RLSample] = []

    for _, seq in sequential.items():
        if len(samples) >= max_samples:
            break
        history, target_item = _sample_history(seq, max_item_length, rng)
        if not history or target_item is None:
            continue
        if target_item not in metas:
            continue
        target_title = metas[target_item].get(item_index_field)
        if not target_title:
            continue
        history_titles = [
            metas[item_id].get(item_index_field)
            for item_id in history
            if item_id in metas and metas[item_id].get(item_index_field)
        ]
        if len(history_titles) < 1:
            continue
        history_text = get_history_text([f"'{title}'" for title in history_titles])

        # Build synthetic recommendation list: target + random negatives (order shuffled later)
        negatives = []
        pool = list(valid_items)
        rng.shuffle(pool)
        for candidate in pool:
            if candidate == target_item:
                continue
            negatives.append(candidate)
            if len(negatives) >= max(topk - 1, 0):
                break
        output_items = [target_item] + negatives[: max(topk - 1, 0)]
        output_titles = [
            metas[item_id].get(item_index_field, "")
            for item_id in output_items
            if metas[item_id].get(item_index_field)
        ]
        if not output_titles:
            continue
        recommendation_text = get_output_text(
            output_titles, idx=use_idx, user_control_symbol=use_control_symbol
        ) + "\n"

        instruction_fields = {"history": history_text, "item_count": topk}
        user_turn = " ".join(template.get_input_text(instruction_fields))
        assistant_turn = " ".join(
            template.get_output_text({"item_title_list": recommendation_text})
        )

        prompt = build_llama3_prompt(user_turn)
        samples.append(
            RLSample(
                prompt=prompt,
                reference_response=assistant_turn,
                target_title=target_title,
                target_item=target_item,
                history_text=history_text,
            )
        )

    return samples
