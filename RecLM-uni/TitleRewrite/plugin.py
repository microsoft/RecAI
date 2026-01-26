import os
import random
import openai
import json
import math
import requests
import faiss

import numpy as np
from typing import List, Dict, Any
from transformers import AutoTokenizer
from swift.plugin import ORM, orms
from swift.utils import get_logger
from scipy.stats import spearmanr

from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

logger = get_logger()

MODEL_PATH = os.getenv("MODEL_PATH", "meta-llama/Meta-Llama-3-8B-Instruct")
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

# Embedding service configuration
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://localhost:8010/v1")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "not-needed")
EMBEDDING_CLIENT = openai.OpenAI(base_url=EMBEDDING_API_URL, api_key=EMBEDDING_API_KEY)

MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))  # Adjust concurrency based on your system and network conditions

_embedding_file_cache = {}

def read_embedding_file_cached(embedding_path):
    if embedding_path in _embedding_file_cache:
        return _embedding_file_cache[embedding_path]

    item2embedding = {}
    with open(embedding_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            item2embedding[data['title']] = data['info']['embedding']
    _embedding_file_cache[embedding_path] = item2embedding
    return item2embedding

# --- Optimization Core: Centralized resource manager for Faiss indexes and Embeddings ---
class ResourceManager:
    """
    Handles loading, building, and caching heavy resources (such as Faiss indexes and item embeddings)
    to avoid repetitive I/O and computation during batch processing.
    """
    def __init__(self):
        self._faiss_indexes: Dict[str, Any] = {}
        self._item_names: Dict[str, List[str]] = {}
        self._item_embeddings: Dict[str, Dict[str, List[float]]] = {}

    def _load_and_build(self, source: str, embedding_file: str):
        # If the index for this data source already exists, return directly
        if source in self._faiss_indexes:
            return

        logger.info(f"Building Faiss index for data source: {source}...")
        item2embedding = read_embedding_file_cached(embedding_file)
        self._item_embeddings[source] = item2embedding

        item_names = list(item2embedding.keys())
        # Specify dtype as float32 for Faiss compatibility
        embeddings_matrix = np.vstack([np.array(v, dtype=np.float32) for v in item2embedding.values()])
        faiss.normalize_L2(embeddings_matrix)

        index = faiss.IndexFlatIP(embeddings_matrix.shape[1])
        index.add(embeddings_matrix)

        self._item_names[source] = item_names
        self._faiss_indexes[source] = index
        logger.info(f"Faiss index construction completed for data source {source}.")

    def get_faiss_resources(self, source: str, embedding_file: str):
        self._load_and_build(source, embedding_file)
        return self._faiss_indexes[source], self._item_names[source]

    def get_item_embeddings(self, source: str, embedding_file: str):
        self._load_and_build(source, embedding_file)
        return self._item_embeddings[source]

# Create global instance of resource manager
resource_manager = ResourceManager()


# --- Refactored network request utility functions ---
def _get_vllm_endpoints(source: str) -> List[str]:
    # Centralize endpoint selection logic for easier maintenance
    # Configure vLLM endpoints via environment variables or use defaults
    default_endpoints = os.getenv("VLLM_ENDPOINTS", "").split(",") if os.getenv("VLLM_ENDPOINTS") else []

    if default_endpoints:
        return default_endpoints

    # Default endpoints for different data sources (replace with your own)
    endpoints = {
        "steam": os.getenv("VLLM_ENDPOINTS_STEAM", "http://localhost:8020/v1").split(","),
        "movies": os.getenv("VLLM_ENDPOINTS_MOVIES", "http://localhost:8020/v1").split(","),
        "toys": os.getenv("VLLM_ENDPOINTS_TOYS", "http://localhost:8020/v1").split(","),
    }
    return endpoints.get(source, endpoints["steam"])  # Default to steam endpoints if source is unknown

def try_vllm_chat(prompt: str, source: str, max_tokens: int = 1, temperature: float = 0.0) -> str:
    for url in _get_vllm_endpoints(source):
        try:
            client = openai.OpenAI(base_url=url, api_key="not-needed")
            response = client.chat.completions.create(
                model="llama3-8b",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"[vLLM Chat] Request failed at {url}: {e}")
    logger.error("All vLLM chat endpoints failed.")
    raise RuntimeError(f"All vLLM chat endpoints failed for source '{source}'.")


def try_vllm_completion(prompt: str, source: str) -> object:
    for url in _get_vllm_endpoints(source):
        try:
            client = openai.OpenAI(base_url=url, api_key="not-needed")
            response = client.completions.create(
                model="llama3-8b",
                prompt=prompt,
                max_tokens=0,
                logprobs=True,
                echo=True,
            )
            return response
        except Exception as e:
            logger.warning(f"[vLLM Completion] Request failed at {url}: {e}")
    logger.error("All vLLM completion endpoints failed.")
    return None

@lru_cache(maxsize=10000)
def cached_generate_embedding(text: str) -> tuple:
    response = EMBEDDING_CLIENT.embeddings.create(input=text, model='bge-m3')
    return tuple(response.data[0].embedding)

def _get_request_embedding_url(source: str) -> str:
    # Configure embedding service URLs via environment variables
    base_url = os.getenv("EMBEDDING_SERVICE_BASE_URL", "http://localhost:5000")
    port_map = {
        "steam": os.getenv("EMBEDDING_PORT_STEAM", "5003"),
        "movies": os.getenv("EMBEDDING_PORT_MOVIES", "5004"),
        "toys": os.getenv("EMBEDDING_PORT_TOYS", "5005")
    }
    port = port_map.get(source, "5000")
    return f"{base_url.replace(':5000', '')}:{port}/embedding"

@lru_cache(maxsize=10000)
def cached_request_embedding(text: str, source: str) -> tuple:
    url = _get_request_embedding_url(source)
    try:
        response = requests.post(url, json={"text": text}, timeout=30)
        response.raise_for_status()  # Raise exception if request fails (e.g., 4xx or 5xx)
        return tuple(response.json()["embedding"])
    except requests.exceptions.RequestException as e:
        logger.error(f"[Embedding] Request failed for source '{source}': {e}")
        raise RuntimeError(f"Failed to get embedding for source '{source}': {e}") from e

# --- Helper functions ---
def spearman_rank_correlation(top10_positions, top10_items):
    original_ranks = list(range(1, len(top10_items) + 1))
    correlation, _ = spearmanr(original_ranks, top10_positions)
    # Map Spearman correlation coefficient from [-1, 1] range to [0, 1] range
    return (correlation + 1) / 2


# ========= 1. ConditionalPPL (并行化处理) =========
class ConditionalPPL(ORM):
    def _process_item(self, args):
        content, sol = args
        try:
            prompt = sol["recommend_prompt"]
            full_text = prompt + content
            
            input_tokens = TOKENIZER(prompt, add_special_tokens=False)
            output_tokens = TOKENIZER(content, add_special_tokens=False)
            split_index = len(input_tokens["input_ids"])
            output_len = len(output_tokens["input_ids"])

            response = try_vllm_completion(full_text, sol["source"])
            if response is None: return 0.0

            logprobs = response.choices[0].logprobs.token_logprobs[split_index : split_index + output_len]

            if not logprobs or len(logprobs) != output_len:
                logger.warning("[ConditionalPPL] Logprob length mismatch.")
                return 0.0

            cross_entropy = -np.mean(logprobs)
            ppl = np.exp(cross_entropy)
            score = np.exp(-0.02 * ppl)
            return score
        except Exception as e:
            logger.warning(f"[ConditionalPPL] Processing failed: {e}")
            return 0.0
            
    def __call__(self, completions, task, solution, **kwargs) -> List[float]:
        # Filter tasks that need processing
        single_tasks_args = [(c, s) for c, t, s in zip(completions, task, solution) if t == "single"]

        # Use thread pool to execute network requests in parallel
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results_iter = executor.map(self._process_item, single_tasks_args)

        results = list(results_iter)

        # Reassemble results in original order
        final_rewards = []
        result_idx = 0
        for t in task:
            if t == "single":
                final_rewards.append(results[result_idx])
                result_idx += 1
            else:
                final_rewards.append(None)
        return final_rewards

# ========= 2. LengthReward (This function is fast, minimal code cleanup) =========
class LengthReward(ORM):
    def __call__(self, completions, task, solution, **kwargs) -> List[float]:
        rewards = []
        for content, sol, t in zip(completions, solution, task):
            if t == "single":
                try:
                    input_len = len(TOKENIZER(sol["recommend_item"], add_special_tokens=False)["input_ids"])
                    output_len = len(TOKENIZER(content, add_special_tokens=False)["input_ids"])
                    if input_len == 0:
                        rewards.append(0.0)
                        continue
                    ratio = output_len / input_len
                    rewards.append(1.0 / (1.0 + ratio ** 2))
                except Exception as e:
                    logger.warning(f"[LengthReward] Calculation failed: {e}")
                    rewards.append(0.0)
            else:
                rewards.append(None)
        return rewards

# ========= 3. DiscriminativeReward (并行化处理) =========
class DiscriminativeReward(ORM):
    def _process_item(self, args):
        content, sol = args
        try:
            target_item = sol["recommend_item"]
            all_items = list(set(sol["top3"] + [target_item]))
            random.shuffle(all_items)
            options = all_items[:4]
            
            option_labels = ["A", "B", "C", "D"]
            labeled_options = {label: item for label, item in zip(option_labels, options)}

            prompt_parts = [
                "I have a rewritten result that is derived from one of the following four options. "
                "Please tell me which option it corresponds to by answering with A, B, C, or D only."
            ]
            for label, item in labeled_options.items():
                prompt_parts.append(f"{label}. {item}")
            prompt_parts.append(f"\nThe rewritten result is:\n{content}\nPlease respond with the correct option letter.")
            prompt = "\n".join(prompt_parts)

            answer = try_vllm_chat(prompt, sol["source"], max_tokens=1)
            if answer and labeled_options.get(answer.strip().upper()) == target_item:
                return 1.0
            return 0.0
        except Exception as e:
            logger.warning(f"[DiscriminativeReward] Processing failed: {e}")
            return 0.0

    def __call__(self, completions, task, solution, **kwargs) -> List[float]:
        single_tasks_args = [(c, s) for c, t, s in zip(completions, task, solution) if t == "single"]
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results_iter = executor.map(self._process_item, single_tasks_args)

        results = list(results_iter)
        
        final_rewards = []
        result_idx = 0
        for t in task:
            if t == "single":
                final_rewards.append(results[result_idx])
                result_idx += 1
            else:
                final_rewards.append(None)
        return final_rewards

# ========= 4. item2item (使用批量Faiss搜索进行优化) =========
class Item2ItemReward(ORM):
    def __call__(self, completions, task, solution, **kwargs) -> List[float]:
        # Group tasks by data source to reuse Faiss indexes
        grouped_tasks = defaultdict(list)
        for i, (c, t, s) in enumerate(zip(completions, task, solution)):
            if t == "single":
                grouped_tasks[s['source']].append({'original_idx': i, 'completion': c, 'solution': s})

        results = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for source, tasks in grouped_tasks.items():
                # Path to item embedding files (configure via environment variable)
                embedding_base_dir = os.getenv("EMBEDDING_DATA_DIR", "./data/embeddings")
                embedding_file = f"{embedding_base_dir}/{source}_all_item_embedding.jsonl"
                faiss_index, all_item_names = resource_manager.get_faiss_resources(source, embedding_file)

                # Step 1: Parallel generation of all query embeddings
                def get_embedding(task_item):
                    new_item_info = (f"The item's title : {task_item['completion']}\n"
                                     f"The item's description: {task_item['solution']['title_desc']}\n"
                                     f"The item's category: {task_item['solution']['title_category']}")
                    return cached_generate_embedding(new_item_info)

                embedding_futures = [executor.submit(get_embedding, task_item) for task_item in tasks]
                content_embeddings = np.array([future.result() for future in embedding_futures], dtype=np.float32)
                faiss.normalize_L2(content_embeddings)

                # Step 2: Execute one-time batch search
                _, all_indices = faiss_index.search(content_embeddings, len(all_item_names))

                # Step 3: Calculate scores for each item in the batch
                for i, task_item in enumerate(tasks):
                    # Get current query ranking results
                    sorted_items = [all_item_names[idx] for idx in all_indices[i]]

                    # Remove the query item itself from the ranking results for a fairer ranking
                    original_title = task_item['solution']['recommend_item']
                    if original_title in sorted_items:
                        sorted_items.remove(original_title)

                    top10_items = task_item['solution']['similarity_top10']
                    try:
                        top10_positions = [sorted_items.index(item) + 1 for item in top10_items]
                        score = spearman_rank_correlation(top10_positions, top10_items)
                    except ValueError: # If a top10 item is not found in the new ranking
                        score = 0.0
                    results[task_item['original_idx']] = score

        # Reassemble the final reward list in original order
        final_rewards = [results.get(i) for i in range(len(completions))]
        return final_rewards

# ========= 5. User2Item (使用批量Faiss搜索进行优化) =========
class User2ItemReward(ORM):
    def __call__(self, completions, task, solution, **kwargs) -> List[float]:
        # Group by data source
        grouped_tasks = defaultdict(list)
        for i, (c, t, s) in enumerate(zip(completions, task, solution)):
            if t == "group":
                grouped_tasks[s['source']].append({'original_idx': i, 'completion': c, 'solution': s})

        results = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for source, tasks in grouped_tasks.items():
                # Path to user embedding files (configure via environment variable)
                embedding_base_dir = os.getenv("EMBEDDING_DATA_DIR", "./data/embeddings")
                embedding_file = f"{embedding_base_dir}/{source}_all_item_embedding_t-desc.jsonl"
                faiss_index, item_names = resource_manager.get_faiss_resources(source, embedding_file)

                # Step 1: Parallel generation of all embeddings
                def get_embedding(task_item):
                    prompt = (f"You need to generate a recommendation list considering user's preference from "
                              f"historical interactions. The historical interactions are provided as follows: "
                              f"{task_item['completion']}. Please generate a recommendation list with 1 different "
                              f"items. Each item should be enclosed by <SOI> and <EOI>. <SOI> should be generated "
                              f"before item title, and <EOI> should be generated after item title.")
                    return cached_request_embedding(prompt, source)
                
                embedding_futures = [executor.submit(get_embedding, task_item) for task_item in tasks]
                prompt_embeddings = np.array([future.result() for future in embedding_futures], dtype=np.float32)
                faiss.normalize_L2(prompt_embeddings)

                # Step 2: Execute one-time batch search
                _, all_indices = faiss_index.search(prompt_embeddings, len(item_names))

                # Step 3: Calculate scores
                for i, task_item in enumerate(tasks):
                    sorted_items = [item_names[idx] for idx in all_indices[i]]
                    item_title = task_item['solution']['target_item']
                    try:
                        item_rank = sorted_items.index(item_title) + 1
                        score = math.exp(-(item_rank - 1) / 2000)
                    except ValueError: # Target item not found in ranking
                        score = 0.0
                    results[task_item['original_idx']] = score

        # Reassemble results in original order
        final_rewards = [results.get(i) for i in range(len(completions))]
        return final_rewards

# ========= Register all reward functions =========
orms['average_ppl'] = ConditionalPPL
orms['length'] = LengthReward
orms['discriminative'] = DiscriminativeReward
orms['item2item'] = Item2ItemReward
orms['user2item'] = User2ItemReward