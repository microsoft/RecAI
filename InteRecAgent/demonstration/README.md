# Demonstrations


Demonstration plays an import role in in-context learning for LLM. However, we could not put too many demonstrations in the prompt due to the limit 
of tokens and the cost. Therefore, we want to build a demonstration memory and retrieve the most relevant demos from the memory when dealing with 
user request. 

To build such a memory, manually contruction is time-consuming and challenging for few developers. Language is diverse, and different people express 
the same idea differently. So we want to generate demonstrations with the help of LLM.

Here we provide demonstration generating scripts and filtering scripts.

- `generator.py`: the scripts is used to generate some demonstrations and save them into jsonl file
- `filter.py`: the scripts is used to filter out some repetitive and unqualitied demons


## Usage

### Generator
 
1. You should have your LLM API, like personal OpenAI API, Azure API or some open-source LLM deployed locally, such as Vicuna.

2. Add API-related info to environment and run the scripy. Just like:

    - Personal OpenAI API key:

        ```bash
        cd LLM4CRS/demonstration
        OPENAI_API_KEY="xxxx" python generator.py --engine [YOUR_LLM_DEPLOY_ID] -m [LLM_MODEL_NAME]  -n 20 --mode input-first --verbose
        ```

    - Azure OpenAI API key:

        ```bash
        cd LLM4CRS/demonstration
        OPENAI_API_KEY="xxx" OPENAI_API_BASE="xxx" OPENAI_API_VERSION="xxx" OPENAI_API_TYPE="xxx" python generator.py --engine [YOUR_LLM_DEPLOY_ID] -m [LLM_MODEL_NAME] -n 20 --mode input-first --verbose
        ```

    For details about arguments, you could run: `python generator.py -h`.

    > Note:
    > There are two modes supported in `generator.py`: input-first and output-first. 
    > - input-first: LLM would generate user request first and then make plans for those requests. Here `n` controls the number of demos to be generated.
    > 
    >     `python generator.py --engine [YOUR_LLM_DEPLOY_ID] -m [LLM_MODEL_NAME] -n 20 --mode input-first --verbose`
    > - output-first: LLM would receive a plan first and then generate several requests according to the plan. Then LLM would make plan for the request. 
    Here, there is an optional argument `check_consistency`, which could be used to check the consistency of generated plan and the given plan. If set true, only the consisten demos would be saved. In output-first mode, `n` represents number of request generated for each plan.
    > 
    >     Don't check consistency:  `python generator.py --engine [YOUR_LLM_DEPLOY_ID] -m [LLM_MODEL_NAME] -n 3 --mode output-first --verbose`
    > 
    >     Check consistency: `python generator.py --engine [YOUR_LLM_DEPLOY_ID] -m [LLM_MODEL_NAME] -n 3 --mode output-first --check_consistency --verbose`

    The generated would be saved in the `./gen_demos` folder by default. Also you can change the saved folder with `--dir [YOU_FOLDER]`.


*Tips*

*If you want to use open-source LLM to complete the task, I recommend Vicuna due to the consistent API to OpenAI API. Here is a [guideline](https://github.com/lm-sys/FastChat/blob/main/docs/langchain_integration.md) that would help you deploy a local LLM. But the demonstrations generated with Vicuna is poor-quality, even using the latest Vicuna-v1.3.*


---
### Filter

Upon we get amount of demonstrations, we need to filter out some poor-quality demonstrations. The `filter.py` script would filter out some examples based on two rules below:

- tool-using priority rule: the priority of tools should be fixed, which is 'candidate store tool' > 'look up tool' > 'hard candidate filter tool' > 'soft condition filter tool' > 'ranking tool' > 'map tool'. 
- diversity rule: to make the demonstrations more diverse, we use RougeL score to measure the similarity between requests. Only when the similarity of the request is below `thres`, the example would be kept.

Here is the command to run filter:

```bash
python filter.py --demo_dir ./gen_demos --seed_demo_file ./seed_demos.jsonl --rougeL_thres 0.8
```

The filtered demos would be saved in `./filtered` folder.