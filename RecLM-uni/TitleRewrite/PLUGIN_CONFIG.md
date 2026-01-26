# Plugin Configuration Guide

This document explains how to configure the reward functions in `plugin.py` for item title rewrite training.

## Environment Variables

The plugin uses environment variables for configuration. Copy `plugin.env.example` to `plugin.env` and fill in your actual values:

```bash
cp plugin.env.example plugin.env
# Edit plugin.env with your configuration
source plugin.env  # Load variables before running training
```

## Configuration Options

### 1. Model Configuration

```bash
export MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
```
- Path to the LLaMA model used for tokenization

### 2. Embedding API Service

```bash
export EMBEDDING_API_URL="http://localhost:8010/v1"
export EMBEDDING_API_KEY="not-needed"
```
- URL and API key for the embedding service (e.g., BGE-M3)
- Used by `Item2ItemReward` for semantic similarity computation

### 3. vLLM Endpoints

```bash
# Option 1: Single set of endpoints for all data sources
export VLLM_ENDPOINTS="http://localhost:8020/v1,http://localhost:8021/v1"

# Option 2: Separate endpoints per data source
export VLLM_ENDPOINTS_STEAM="http://localhost:8020/v1"
export VLLM_ENDPOINTS_MOVIES="http://localhost:8021/v1"
export VLLM_ENDPOINTS_TOYS="http://localhost:8022/v1"
```
- vLLM server endpoints for LLM inference
- Multiple endpoints provide automatic failover
- Used by `ConditionalPPL` and `DiscriminativeReward`

### 4. Embedding Service Ports

```bash
export EMBEDDING_SERVICE_BASE_URL="http://localhost:5000"
export EMBEDDING_PORT_STEAM="5003"
export EMBEDDING_PORT_MOVIES="5004"
export EMBEDDING_PORT_TOYS="5005"
```
- Different ports for different dataset embedding services
- Used by `User2ItemReward` for user preference embedding

### 5. Data Directory

```bash
export EMBEDDING_DATA_DIR="./data/embeddings"
```
- Directory containing pre-computed item embeddings in JSONL format
- Expected files:
  - `{source}_all_item_embedding.jsonl` - Item description embeddings
  - `{source}_all_item_embedding_t-desc.jsonl` - User history embeddings

### 6. Performance Settings

```bash
export MAX_WORKERS="8"
```
- Number of parallel threads for API requests
- Adjust based on your system capabilities

## Setting Up Services

### vLLM Server Setup

Start vLLM servers for inference:

```bash
# Example for movies dataset
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --port 8020 \
  --host 0.0.0.0 \
  --tensor-parallel-size 1
```

### Embedding Service Setup

Start embedding service (e.g., using FlagAI or similar):

```bash
# Example for BGE-M3
python -m flagai.embedding_server \
  --model_name BAAI/bge-m3 \
  --port 8010
```

## Preparing Embedding Files

Generate embedding files in JSONL format:

```jsonl
{"title": "Item 1", "info": {"embedding": [0.1, 0.2, ...]}}
{"title": "Item 2", "info": {"embedding": [0.3, 0.4, ...]}}
```

Example script to generate embeddings:

```python
import json
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-m3')
items = load_items_from_metas_jsonl()

with open('./data/embeddings/movies_all_item_embedding.jsonl', 'w') as f:
    for item in items:
        text = f"{item['title']} {item.get('description', '')}"
        embedding = model.encode(text).tolist()
        f.write(json.dumps({"title": item['title'], "info": {"embedding": embedding}}))
        f.write('\n')
```

## Quick Start

1. Configure environment variables:
```bash
source plugin.env
```

2. Start required services (vLLM, embedding service)

3. Run training:
```bash
swift rl \
  --model_type llama3-8b \
  --dataset path/to/rewrite_data.jsonl \
  --plugin_module plugin \
  --reward_functions average_ppl length discriminative item2item user2item
```

## Troubleshooting

### Connection Errors

If you see connection errors:
- Verify all services are running: `curl http://localhost:8020/v1/models`
- Check firewall rules allow connections between services
- Ensure environment variables are correctly set

### Missing Embedding Files

If you get "embedding file not found" errors:
- Verify `EMBEDDING_DATA_DIR` points to the correct location
- Check that embedding files exist: `{source}_all_item_embedding.jsonl`
- Ensure file format matches the expected JSONL structure

### Performance Issues

If training is slow:
- Increase `MAX_WORKERS` (but be mindful of resource limits)
- Use multiple vLLM endpoints with `VLLM_ENDPOINTS`
- Pre-generate and cache embeddings to reduce API calls
