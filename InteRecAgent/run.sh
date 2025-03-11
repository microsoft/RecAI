# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash

# API related info
API_KEY="" # your api key
API_BASE="" # [https://xxxx.azure.com, https://api.openai.com/v1]
API_VERSION="2023-03-15-preview"
API_TYPE="open_ai" # ['open_ai', 'azure']
engine="gpt-4"   # model name for OpenAI or deployment name for Azure OpenAI. GPT-4 is recommended.
bot_type="chat" # model type, ["chat", "completetion"]. For gpt-3.5-turbo and gpt-4, it should be "chat". For text-davinci-003, it should be "completetion" 


# We load variables above from a shell script. It could be disabled if you do not need to load variables from file
OAI_FILE="oai.sh"
if [ -f "$OAI_FILE" ]; then
    # check if the file exists
    source $OAI_FILE
    echo "File $OAI_FILE loaded."
else
    echo "File $OAI_FILE does not exist."
fi


domain="game"   # item domain
enable_shorten=0    # whether to enable shorten chat history

# demonstration mode. 
# 1. zero: Zero-shot setting. No demonstration would be inserted into prompt.
# 2. fixed: Fixed demonstrations are used for all cases. It would use the first n demonstrations in the `demo_dir_or_file`, where n is `num_demos`.
# 3. dynamic: Retrieval the most n relevant demonstrations.
demo_mode="dynamic"  # ["zero", "fixed", "dynamic"]

num_demos=5 # number of demonstrations to use, when demo_mode=="fixed" and num_demos<0, all demonstrations would be used.

# folder or file path of demonstrations. If folder, all jsonl files would be used. 
# If demo_mode=="zero", the argument does not function.
demo_dir_or_file="./demonstration/seed_demos_placeholder.jsonl"   


enable_reflection=0 # whether to use reflection. Reflection would increase the token usage and the response delay.

plan_first=1 # whether to use plan-first agent. Recommend 1

# Whether to use langchain in plan-first agent. Langchain may bring more time delay in API calling. 
# The argument only functions when "plan_first=1". Recommend 0.
langchain=0 


DOMAIN=$domain \
OPENAI_API_KEY=$API_KEY \
OPENAI_API_BASE=$API_BASE \
OPENAI_API_VERSION=$API_VERSION \
OPENAI_API_TYPE=$API_TYPE \
TOKENIZERS_PARALLELISM=false \
PYTHONPATH=$(pwd) \
python ./app.py \
    --engine=$engine \
    --bot_type=$bot_type \
    --enable_shorten=$enable_shorten \
    --demo_mode=$demo_mode \
    --num_demos=$num_demos \
    --enable_reflection=$enable_reflection \
    --plan_first=$plan_first \
    --langchain=$langchain \
    --demo_dir_or_file=$demo_dir_or_file
