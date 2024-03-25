# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

DIR="/path/to/LLM4CRS"

EVAL_DIR=$DIR"/eval"

cd $EVAL_DIR

DOMAIN="game" \
AGENT_ENGINE="xxx" \
OPENAI_API_KEY="xxx" \
OPENAI_API_BASE="xxx" \
OPENAI_API_VERSION="xxx" \
OPENAI_API_TYPE="xx" \
PYTHONPATH="$DIR/llm4crs"
TOKENIZERS_PARALLELISM=false \
python one_turn_eval.py \
    --bot_type=chat \
    --timeout=20 \
    --enable_shorten=0 \
    --demo_mode="dynamic" \
    --demo_dir=path_to_demo \
    --num_demos=3 \
    --enable_reflection=1 \
    --plan_first=1 \
    --langchain=0 \
    --data=path_test_data.jsonl \
    --agent=recbot
