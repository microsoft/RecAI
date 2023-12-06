# Recommender AI Agent: Integrating Large Language Models for Interactive Recommendations
This is the Repo for InteRecAgent, a interactive recommender agent, which applies Large Language Model(LLM) to bridge the gap between traditional recommender systems and  conversational recommender system(CRS). 


## Table of Contents

- [Recommender AI Agent: Integrating Large Language Models for Interactive Recommendations](#recommender-ai-agent-integrating-large-language-models-for-interactive-recommendations)
  - [Table of Contents](#table-of-contents)
  - [InteRecAgent Framework](#interecagent-framework)
  - [Usage](#usage)
  - [License](#license)
  - [Citation](#citation)
  - [Contributing](#contributing)
  - [Trademarks](#trademarks)
  - [Acknowledge](#acknowledge)

## InteRecAgent Framework
<p align="center">
  <img src="./assets/framework.png" alt="InteRecAgnet Framework" width="100%">
  <br>
  <b>Figure 1</b>: InteRecAgent Framework
</p>

InteRecAgent (**Inte**ractive **Rec**ommender **Agent**) is a framework to utilize pre-trained domain-specific recommendation tools (such as SQL tools, id-based recommendation models) and large language models (LLM) to implement an interactive, conversational recommendation agent. In this framework, the LLM primarily engages in user interaction and parses user interests as input for the recommendation tools, which are responsible for finding suitable items.   
  
Within the InteRecAgent framework, recommendation tools are divided into three main categories: query, retrieval, and ranking. You need to provide the API for the LLM and the pre-configured domain-specific recommendation tools to build an interactive recommendation agent using the InteRecAgent framework. Neither the LLM nor the recommendation tools will be updated or modified within InteRecAgent.
  
This repository mainly implements the right-hand side of the figure, i.e., the communication between the LLM and the recommendation tools. For more details, please refer to our paper [*Recommender AI Agent: Integrating Large Language Models for Interactive Recommendations*](https://arxiv.org/abs/2308.16505).

InteRecAgent consists of 4 necessary components:

1. LLM: A large language model, which serves as conversational agent.
 
2. Item profile table: A table containing item informations, whose columns consists of id, title, tag, description, price, release_date, visited_num, et al.


3. Query module: A SQL module to query item information in the item profile table.
    - Input: SQL command
    - Output: information queried with SQL


4. Retrieval module: The module aims to retrieve item candidates from the all item corups according to user's intention(requirements). Note that the module does not function in deal with user's personal profile, like user history, user age, et al. Instead, it focuses on what user wants, like "give me some sports games", "I want some popular games". The module should consist of at least two kinds of retrieval tools:
    - SQL tool: The tools is used to deal with complex search condition, which is related to item information. For example, "I want some popular sports games". Then the tool would use SQL command to search in the item profile table.
        - Input: SQL command
        - Output: execution log
    
    - Item similarity tool: The tools aims to retrieve items according to item similarity. Sometimes, user's intention is not clear enough to organized as SQL command, for example, "I want some games similar with Call of Duty", where the requirements is expressed implicitly through item similarity instead of explicit item features. 
        - Input: item name list as string. 
        - Output: execution log


5. Ranking module: Refine the rank of item candidates according to schema (popularity, similarity, preference). User prefernece comprises `prefer` and `unwanted`. The module could be a traditional recommender model, which inputs user and item features and outputs relevant score.
    - Input: ranking schema, user's profile (prefer, unwanted)
    - Output: execution log


## Usage

1. Environments

    
    First, install other packages listed in `requirements.txt`

       ```bash
       cd LLM4CRS
       conda create -n llm4crs python==3.9
       conda activate llm4crs
       pip install -r requirements.txt
       ```

2. Copy resource

   Copy all files in "InteRecAgent/{domain}" in [OneDrive](https://1drv.ms/f/s!Asn0lWVfky4FpF3saprFClmobx8g?e=lgeT0M) / [RecDrive](https://rec.ustc.edu.cn/share/baa4d930-48e1-11ee-b20c-3fee0ba82bbd) to  your local "./LLM4CRS/resources/{domain}".
   If you cannot access those links, please contact <a href="mailto:jialia@microsoft.com">jialia@microsoft.com</a> or <a href="mailto:xuhuangcs@mail.ustc.edu.cn">xuhuangcs@mail.ustc.edu.cn</a>.

3. Run

    - Personal OpenAI API key:

        ```bash
        cd LLM4CRS
        DOMAIN="game" OPENAI_API_KEY="xxxx" python app.py
        ```

    - Azure OpenAI API key:

        ```bash
        cd LLM4CRS
        DOMAIN="game" OPENAI_API_KEY="xxx" OPENAI_API_BASE="xxx" OPENAI_API_VERSION="xxx" OPENAI_API_TYPE="xxx" python app.py
        ```
    
    Note `DOMAIN` represents the item domain, supported `game`, `movie` and `beauty_product` now.

    We support two types of OpenAI API: [Chat](https://platform.openai.com/docs/api-reference/chat) and [Completions](https://platform.openai.com/docs/api-reference/completions). Here are commands for running RecBot with GPT-3.5-turbo and text-davinci-003.

    ```bash
    cd LLMCRS

    # Note that the engine should be your deployment id
    # completion type: text-davinci-003
    OPENAI_API_KEY="xxx" [OPENAI_API_BASE="xxx" OPENAI_API_VERSION="xxx" OPENAI_API_TYPE="xxx"] python app.py --engine text-davinci-003 --bot_type completion

    # chat type: gpt-3.5-turbo, gpt-4 (Recommended)
    OPENAI_API_KEY="xxx" [OPENAI_API_BASE="xxx" OPENAI_API_VERSION="xxx" OPENAI_API_TYPE="xxx"] python app.py --engine gpt-3.5-turbo/gpt-4 --bot_type chat
    ```

    We also provide a shell script `run.sh`, where commonly used arguments are given. You could directly set the API related information in `run.sh`, or create a new shell script `oai.sh` that would be loaded in `run.sh`. GPT-4 API is highly recommended for the InteRecAgent since it has remarkable instruction-following capability.

    Here is an example of the `oai.sh` script:

    ```bash
    API_KEY="xxxxxx" # your api key
    API_BASE="https://xxxx.azure.com/" # [https://xxxx.azure.com, https://api.openai.com/v1]
    API_VERSION="2023-03-15-preview"
    API_TYPE="azure" # ['open_ai', 'azure']
    engine="gpt4"   # model name for OpenAI or deployment name for Azure OpenAI. GPT-4 is recommended.
    bot_type="chat" # model type, ["chat", "completetion"]. For gpt-3.5-turbo and gpt-4, it should be "chat". For text-davinci-003, it should be "completetion" 
    ```


4. Features    

    There are several optional features to enhance the agent.

    1. History Shortening

        - `enable_shorten`: if true, enable shortening chat history by LLM; else use all chat history

    2. Demonstrations Selector
        
        - `demo_mode`: mode to choose demonstration. Optional values: [`zero`, `fixed`, `dynamic`]. If `zero`, no demonstration would be used (zero-shot). If `fixed`, the first `demo_nums` examples in `demo_dir_or_file` would be used. If `dynamic`, most `demo_nums` related examples would be selected for in-context learning. 
        - `demo_dir_or_file`: the directory or file path of demostrations. If a directory path is given, all `.jsonl` file in the folder would be loaded. If `demo_mode` is `zero`, the argument is invalid. Default None.
        - `demo_nums`: number of demonstrations used in prompt for in-context learning. The argument is invalid when `demo_dir` is not given. Default 3. 

    3. Reflection

        - `enable_reflection`: if true, enable reflection for better plan making
        - `reflection_limits`: maximum times of reflection

        Note that reflection happens after the plan is finished. That means the conversational agent would generate an answer first and then do reflection.

    4. Plan First

        - `plan_first`: if true, the agent would make tool using plan first. There would only a tool executor for LLM to call, where the plan is input. Default true.

        Additionally, we have implemented a version without using the black-box API calling in langchain.
        To enable it, use the following arguments.

        - `langchain`: if true, use langchain in plan-first strategy. Otherwise, the API calls would be made by directly using openai. Default false.


## License
InteRecAgent uses [MIT](./LICENSE) license. All data and code in this project can only be used for academic purposes.


## Citation
Please cite the following paper as the reference if you use our code or data[![Paper](https://img.shields.io/badge/arxiv-PDF-red)](https://arxiv.org/abs/2308.16505).

```
@misc{huang2023recommender,
      title={Recommender AI Agent: Integrating Large Language Models for Interactive Recommendations}, 
      author={Xu Huang and Jianxun Lian and Yuxuan Lei and Jing Yao and Defu Lian and Xing Xie},
      year={2023},
      eprint={2308.16505},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.


## Acknowledge

Thanks to the open source codes of the following projects:

[UniRec](https://github.com/microsoft/UniRec) &#8194;
[VisualChatGPT](https://github.com/microsoft/TaskMatrix/blob/main/visual_chatgpt.py) &#8194;
[JARVIS](https://github.com/microsoft/JARVIS) &#8194;
[LangChain](https://github.com/langchain-ai/langchain) &#8194;
[guidance](https://github.com/microsoft/guidance) &#8194;

## Responsible AI FAQ

Please refer to [RecAI: Responsible AI FAQ](./RAI_FAQ.md) for document on the purposes, capabilities, and limitations of the RecAI systems. 