# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Built-ins & stdlib
import os
from typing import Optional  # ← for Python <3.10 compatibility

# Third-party deps
from tqdm import tqdm
import json
from transformers import AutoConfig, AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import vllm
import functools
import inspect

def map_gpu_count(m):
    """Return the nearest supported tensor parallel size (1, 2, 4 or 8)
    that does not exceed the available GPU count. vLLM requires the tensor
    parallel size to be a power-of-two within this set.
    """
    n_values = [1, 2, 4, 8]
    for n in reversed(n_values):
        if m >= n:
            return n
    return None  
def env(var_name):
    """Tiny wrapper around :pyfunc:`os.getenv` for brevity."""
    return os.getenv(var_name)
if not env("VLLM_TENSOR_PARALLEL_SIZE"):
    gpu_count = torch.cuda.device_count()
    VLLM_TENSOR_PARALLEL_SIZE = map_gpu_count(gpu_count)
    print(f"set VLLM_TENSOR_PARALLEL_SIZE == {VLLM_TENSOR_PARALLEL_SIZE}")
else:
    VLLM_TENSOR_PARALLEL_SIZE = env("VLLM_TENSOR_PARALLEL_SIZE")

if not env("VLLM_GPU_MEMORY_UTILIZATION"):
    VLLM_GPU_MEMORY_UTILIZATION = 0.9
    print(f"set VLLM_GPU_MEMORY_UTILIZATION == {VLLM_GPU_MEMORY_UTILIZATION}")
else:
    VLLM_GPU_MEMORY_UTILIZATION = env("VLLM_GPU_MEMORY_UTILIZATION")
    
DEFAULT_SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."

class vllmModel:
    """Lightweight convenience wrapper around :pyclass:`vllm.LLM`.

    The class infers a safe ``max_model_len`` from the model configuration,
    instantiates an internal ``vllm.LLM`` object accordingly and exposes a
    higher-level :pyfunc:`batch_predict` method for batch generation.
    """

    def __init__(self, model_name: str, max_model_len: Optional[int] = None, dtype: str = "bfloat16"):
        self.model_name = model_name

        # Build argument dict for vLLM; only include ``max_model_len`` when it
        # is explicitly specified so that ``None`` falls back to the model
        # default.
        llm_kwargs = dict(
            tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
            trust_remote_code=True,
            dtype=dtype,
            enforce_eager=False,
        )
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        # Instantiate vLLM
        self.llm = vllm.LLM(self.model_name, **llm_kwargs)

        # Cache tokenizer instance for later reuse
        self.tokenizer = self.llm.get_tokenizer()

    def batch_predict(self, prompts: list[str], max_new_tokens: int = 1000) -> list[str]:
        """Generate one answer for every prompt in *prompts*.

        Parameters
        ----------
        prompts : list[str]
            A batch of plain text prompts or chat-formatted strings.
        max_new_tokens : int, default=1000
            Maximum number of tokens to generate for each prompt.

        Returns
        -------
        list[str]
            Model outputs, order-aligned with the input *prompts*.
        """
        responses = self.llm.generate(
            prompts,
            vllm.SamplingParams(
                n=1,
                top_p=0.9,
                temperature=0.3,
                skip_special_tokens=True,
                max_tokens=max_new_tokens,
            ),
            use_tqdm=False
        )

        batch_response = [response.outputs[0].text.strip() for response in responses]
        return batch_response

class ChatDataset(Dataset):
    def __init__(self, test_dataset, tokenizer, max_seq_len, system_prompt) -> None:
        super().__init__()
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.system_prompt = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT
        # Check whether tokenizer.apply_chat_template supports enable_thinking
        self.support_enable_thinking = (
            "enable_thinking" in inspect.signature(self.tokenizer.apply_chat_template).parameters
        )
            
    def __len__(self):
        return len(self.test_dataset)
    
    def __getitem__(self, idx):
        data = self.test_dataset[idx]
        
        # Always use apply_chat_template; dynamically pass enable_thinking if supported
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if self.support_enable_thinking:
            kwargs["enable_thinking"] = False  # Disable thinking mode (if supported)

        if isinstance(data["prompt"], str):
            conv = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": data["prompt"]},
            ]
            inputs = self.tokenizer.apply_chat_template(conv, **kwargs)
        else:
            # Ensure the first message is the system prompt
            data["prompt"].insert(0, {"role": "system", "content": self.system_prompt})
            inputs = self.tokenizer.apply_chat_template(data["prompt"], **kwargs)
        inputs = inputs[-self.max_seq_len:]
        return inputs

def get_batches(data, batch_size):

    batches = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batches.append(batch)
    return batches

@functools.lru_cache(maxsize=1)
def get_cached_model(model_path_or_name: str):
    """Load the model once (LRU-cached) and automatically infer an appropriate
    context length.
    """

    # Read model config; if ``max_position_embeddings`` exists use it as
    # the detected length.
    try:
        cfg = AutoConfig.from_pretrained(model_path_or_name, trust_remote_code=True)
        detected_len = getattr(cfg, "max_position_embeddings", None)
        # Hard upper bound to avoid ridiculous values (e.g. 131 k) exploding
        # memory usage on a 40 GB GPU. Adjust if you have more memory.
        MAX_ALLOWED_LEN = 8192
        if detected_len is not None and detected_len > MAX_ALLOWED_LEN:
            print(f"[vllm_models] Detected max_position_embeddings={detected_len} > {MAX_ALLOWED_LEN}, cap to {MAX_ALLOWED_LEN} to save memory.")
            max_len = MAX_ALLOWED_LEN
        else:
            max_len = detected_len
    except Exception:
        # Fallback: if config loading fails, let vLLM decide on its own.
        max_len = None

    return vllmModel(model_path_or_name, max_model_len=max_len)



def run_chat(model_path_or_name, question_file, answer_file, args, system_prompt):
    # load tokenizer
    model = get_cached_model(model_path_or_name)
    # load test dataset
    test_data = []
    for line in open(question_file):
        test_data.append(json.loads(line))

    model_config = AutoConfig.from_pretrained(args.model_path_or_name, trust_remote_code=True)
    max_position_embeddings = getattr(model_config, 'max_position_embeddings', 2048)

    # vLLM may cap the usable context length (see get_cached_model). To avoid
    # ``ValueError: decoder prompt length > max model length``, we conservatively
    # enforce an upper bound of 8192 **before** reserving space for generation.
    MAX_CONTEXT = 8192
    max_position_embeddings = min(max_position_embeddings, MAX_CONTEXT)

    test_dataset = ChatDataset(
        test_data,
        model.tokenizer,
        max_position_embeddings - args.max_new_tokens - 100,
        system_prompt,
    )

    dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    result_lists = []
    
    with torch.no_grad():
        for batch in dataloader:
            responses = model.batch_predict(
                batch,
                max_new_tokens=args.max_new_tokens
            )
            result_lists.extend(responses)

    # --- Post-processing: exact -> canonical mapping & cleanup ---
    import unicodedata, re, random

    def _canonical(text: str) -> str:
        """Lowercase, remove non-alnum to create a canonical form."""
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode()
        return re.sub(r'[^a-z0-9]', '', text.lower())

    def _map_titles(answer_line: str, candidates: list[str]) -> list[str]:
        cand_set = set(candidates)
        canon_map = {_canonical(c): c for c in candidates}
        titles = [t for t in answer_line.split('<end>') if t.strip()]
        mapped = []
        for t in titles:
            t_strip = t.strip()
            # remove wrapper tokens if model accidentally copied them
            if t_strip.startswith('<SOI>') and t_strip.endswith('<EOI>'):
                t_strip = t_strip[5:-5].strip()
            elif t_strip.startswith('<SOI>'):
                t_strip = t_strip[5:].strip()
            elif t_strip.endswith('<EOI>'):
                t_strip = t_strip[:-5].strip()
            if t_strip in cand_set:
                chosen = t_strip
            else:
                chosen = canon_map.get(_canonical(t_strip))
            if chosen and chosen not in mapped:
                mapped.append(chosen)
        return mapped

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "w", encoding='utf-8') as fd:
        for data, raw in zip(test_data, result_lists):
            candidates = data.get('candidate', [])
            history = set(data.get('history', []))

            task_type = data.get('task', '').lower()

            # Only apply strict candidate-based filtering for *ranking* (or when candidates list is non-empty).
            if task_type == 'ranking' or candidates:
                mapped_titles = _map_titles(raw, candidates)
            else:
                # retrieval task: keep model-generated titles as-is
                mapped_titles = [t.strip() for t in raw.split('<end>') if t.strip()]
            # filter history
            mapped_titles = [t for t in mapped_titles if t not in history]
            # Pad / truncate to ``top_k`` so evaluation metrics match expectation.
            top_k = getattr(args, "top_k", 20)
            if len(mapped_titles) < top_k:
                pool = [c for c in candidates if c not in mapped_titles and c not in history]
                random.shuffle(pool)
                mapped_titles.extend(pool[:top_k - len(mapped_titles)])
            mapped_titles = mapped_titles[:top_k]
            fixed_answer = '<end>'.join(mapped_titles) + '<end>'
            data["answer"] = fixed_answer
            fd.write(json.dumps(data, ensure_ascii=False) + '\n')

def gen_model_chat_answer(model_path_or_name, question_file, answer_file, args, system_prompt):
    run_chat(model_path_or_name, question_file, answer_file, args, system_prompt)
