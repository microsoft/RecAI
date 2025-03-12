'''
This script is adopted from https://github.com/QwenLM/Qwen/blob/main/eval/evaluate_gsm8k.py
'''
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

import datasets
import numpy as np
import requests
from datasets import load_from_disk, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

Q1 = '''Question: In 2004, there were 60 kids at a cookout. In 2005, half the number of kids came to the cookout as compared to 2004. In 2006, 2/3 as many kids came to the cookout as in 2005. How many kids came to the cookout in 2006?'''
A1 = '''Let's think step by step.
In 2005, 60/2=30 kids came to the cookout.
In 2006, 30/3*2=20 kids came to the cookout.
The answer is 20'''

Q2 = '''Question: Zilla spent 7% of her monthly earnings on rent, half of it on her other monthly expenses, and put the rest in her savings. If she spent $133 on her rent, how much does she deposit into her savings account in a month?'''
A2 = '''Let's think step by step.
Since $133 is equal to 7% of her earnings, then 1% is equal to $133/7 = $19.
The total monthly earning of Zilla is represented by 100%, so $19 x 100 = $1900 is her monthly earnings.
So, $1900/2 = $950 is spent on her other monthly expenses.
The total amount spent on the rent and other monthly expenses is $133 + $950 = $1083.
Hence, she saves $1900 - $1083 = $817 per month.
The answer is 817'''

Q3 = '''Question: If Buzz bought a pizza with 78 slices at a restaurant and then decided to share it with the waiter in the ratio of 5:8, with Buzz's ratio being 5, what's twenty less the number of slices of pizza that the waiter ate?'''
A3 = '''Let's think step by step.
The total ratio representing the slices of pizza that Buzz bought is 5+8=13.
If he shared the slices of pizza with the waiter, the waiter received a fraction of 8/13 of the total number of slices, which totals 8/13 * 78 = 48 slices.
Twenty less the number of slices of pizza that the waiter ate is 48-20 = 28.
The answer is 28'''

Q4 = '''Question: Jame gets a raise to $20 per hour and works 40 hours a week.  His old job was $16 an hour for 25 hours per week.  How much more money does he make per year in his new job than the old job if he works 52 weeks a year?'''
A4 = '''Let's think step by step.
He makes 20*40=$800 per week.
He used to make 16*25=$400 per week.
So his raise was 800-400=$400 per week.
So he makes 400*52=$20,800 per year more.
The answer is 20800'''

Q5 = '''Question: Mr. Gardner bakes 20 cookies, 25 cupcakes, and 35 brownies for his second-grade class of 20 students. If he wants to give each student an equal amount of sweet treats, how many sweet treats will each student receive?'''
A5 = '''Let's think step by step.
Mr. Gardner bakes a total of 20 + 25 + 35 = 80 sweet treats.
Each student will receive 80 / 20 = 4 sweet treats.
The answer is 4'''

Q6 = '''Question: A used car lot has 24 cars and motorcycles (in total) for sale. A third of the vehicles are motorcycles, and a quarter of the cars have a spare tire included. How many tires are on the used car lot's vehicles in all?'''
A6 = '''Let's think step by step.
The used car lot has 24 / 3 = 8 motorcycles with 2 tires each.
The lot has 24 - 8 = 16 cars for sale.
There are 16 / 4 = 4 cars with a spare tire with 5 tires each.
The lot has 16 - 4 = 12 cars with 4 tires each.
Thus, the used car lot's vehicles have 8 * 2 + 4 * 5 + 12 * 4 = 16 + 20 + 48 = 84 tires in all.
The answer is 84'''

Q7 = '''Question: Norma takes her clothes to the laundry. She leaves 9 T-shirts and twice as many sweaters as T-shirts in the washer. When she returns she finds 3 sweaters and triple the number of T-shirts. How many items are missing?'''
A7 = '''Let's think step by step.
Norma left 9 T-shirts And twice as many sweaters, she took 9 * 2= 18 sweaters.
Adding the T-shirts and sweaters, Norma left 9 + 18 = 27 clothes.
When she came back, she found 3 sweaters And triple the number of T-shirts, she found 3 * 3 = 9 T-shirts.
Adding the T-shirts and sweaters, Norma found 3 + 9 = 12 clothes.
Subtracting the clothes she left from the clothes she found, 27 - 12 = 15 clothes are missing.
The answer is 15'''

Q8 = '''Question: Adam has an orchard. Every day for 30 days he picks 4 apples from his orchard. After a month, Adam has collected all the remaining apples, which were 230. How many apples in total has Adam collected from his orchard?'''
A8 = '''Let's think step by step.
During 30 days Adam picked 4 * 30 = 120 apples.
So in total with all the remaining apples, he picked 120 + 230 = 350 apples from his orchard.
The answer is 350'''

shots = [(Q1, A1), (Q2, A2), (Q3, A3), (Q4, A4),
         (Q5, A5), (Q6, A6), (Q7, A7), (Q8, A8)]


def get_prompt(doc):
    messages = [
        {'role': 'system', 'content': "You are a helpful, respectful and honest assistant. Please provide a detailed step-by-step solution for the following grade school math question. "},
    ]
    for q, a in shots[:args.shots]:
        messages.append({'role': 'user', 'content': q})
        messages.append({'role': 'assistant', 'content': a})
    messages.append({'role': 'user', 'content': f"Question: {doc['question']}"})
    return tokenizer.apply_chat_template(messages, tokenize=False)


def clean_answer(text):
    text = text.split("Question:")[0]
    return text


def sample_process(doc):
    prompt = get_prompt(doc)
    pload = {
        "model": args.model_name,
        "prompt": prompt,
        "n": 1,
        "temperature": 0.0,
        "top_k": 1,
        "max_tokens": args.gen_max_length,
        "skip_special_tokens": False,
        'stop': ["<|eot_id|>", "<|end_of_text|>"]
    }
    response = requests.post(f'http://127.0.0.1:{args.port}/v1/completions', json=pload, stream=False)
    output_data = json.loads(response.content)
    output_text = output_data["choices"][0]['text']
    output = output_text
    doc["raw_output"] = output
    pred = clean_answer(output)
    label = doc["answer"]
    acc = is_correct(pred, label)
    doc["completion"] = pred
    doc["acc"] = acc
    if doc['idx'] == 0:
        print("PROMPT: ", prompt)
        print("OUTPUT: ", output)
        print("PREDICTION: ", pred)
    return doc


def extract_answer_hf(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return eval(match_str)
    else:
        return INVALID_ANS


def extract_answer(completion):
    try:
        last_number = re.findall(r'-?[0-9]+[0-9.,]?', completion)[-1].rstrip('.').replace(',', '')
        return eval(last_number)
    except:
        return INVALID_ANS


def is_correct(completion, answer):
    gold = extract_answer_hf(answer)
    assert gold != INVALID_ANS, "No ground truth answer found in the document."
    return extract_answer(completion) == gold


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--sample-input-file", type=str, default='./')
    parser.add_argument("--shots", type=int, default=8)
    parser.add_argument("--gen_max_length", type=int, default=2048)
    parser.add_argument("--port", type=int, default=13579)
    parser.add_argument("--model_name", type=str, default="snap/.../SFT_Epoch20/")
    args = parser.parse_args()
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.sample_input_file is not None:
        dataset = load_from_disk(args.sample_input_file)
    else:
        config = datasets.DownloadConfig(resume_download=True, max_retries=100)
        dataset = load_dataset("gsm8k", 'main', download_config=config)
        dataset.save_to_disk("gsm8k")

    test = dataset["test"]
    test = [{
        'idx': i,
        'question': _['question'],
        'answer': _['answer'],
    } for i, _ in enumerate(test)]

    acc_res = []

    with ThreadPoolExecutor(max_workers=40) as executor:
        result = list(tqdm(executor.map(sample_process, test), total=len(test)))

    for idx, d in enumerate(result):
        acc_res.append(d["acc"])
    print(np.mean(acc_res))

    with open(args.model_name.rstrip('/') + f'_GSM8K_{args.shots}shot.txt', 'w') as f:
        f.write(str(np.mean(acc_res)))


'''prompt
[INST] <<SYS>>
You are a helpful, respectful and honest assistant.
<</SYS>>

Please provide a detailed step-by-step solution for the following grade school math question.  [/INST] Ok. </s><s>[INST] Question: In 2004, there were 60 kids at a cookout. In 2005, half the number of kids came to the cookout as compared to 2004. In 2006, 2/3 as many kids came to the cookout as in 2005. How many kids came to the cookout in 2006? [/INST] Let's think step by step.
In 2005, 60/2=30 kids came to the cookout.
In 2006, 30/3*2=20 kids came to the cookout.
The answer is 20 </s><s>[INST] Question: Zilla spent 7% of her monthly earnings on rent, half of it on her other monthly expenses, and put the rest in her savings. If she spent $133 on her rent, how much does she deposit into her savings account in a month? [/INST] Let's think step by step.
Since $133 is equal to 7% of her earnings, then 1% is equal to $133/7 = $19.
The total monthly earning of Zilla is represented by 100%, so $19 x 100 = $1900 is her monthly earnings.
So, $1900/2 = $950 is spent on her other monthly expenses.
The total amount spent on the rent and other monthly expenses is $133 + $950 = $1083.
Hence, she saves $1900 - $1083 = $817 per month.
The answer is 817 </s><s>[INST] Question: If Buzz bought a pizza with 78 slices at a restaurant and then decided to share it with the waiter in the ratio of 5:8, with Buzz's ratio being 5, what's twenty less the number of slices of pizza that the waiter ate? [/INST] Let's think step by step.
The total ratio representing the slices of pizza that Buzz bought is 5+8=13.
If he shared the slices of pizza with the waiter, the waiter received a fraction of 8/13 of the total number of slices, which totals 8/13 * 78 = 48 slices.
Twenty less the number of slices of pizza that the waiter ate is 48-20 = 28.
The answer is 28 </s><s>[INST] Question: Jame gets a raise to $20 per hour and works 40 hours a week.  His old job was $16 an hour for 25 hours per week.  How much more money does he make per year in his new job than the old job if he works 52 weeks a year? [/INST] Let's think step by step.
He makes 20*40=$800 per week.
He used to make 16*25=$400 per week.
So his raise was 800-400=$400 per week.
So he makes 400*52=$20,800 per year more.
The answer is 20800 </s><s>[INST] Question: Mr. Gardner bakes 20 cookies, 25 cupcakes, and 35 brownies for his second-grade class of 20 students. If he wants to give each student an equal amount of sweet treats, how many sweet treats will each student receive? [/INST] Let's think step by step.
Mr. Gardner bakes a total of 20 + 25 + 35 = 80 sweet treats.
Each student will receive 80 / 20 = 4 sweet treats.
The answer is 4 </s><s>[INST] Question: A used car lot has 24 cars and motorcycles (in total) for sale. A third of the vehicles are motorcycles, and a quarter of the cars have a spare tire included. How many tires are on the used car lot's vehicles in all? [/INST] Let's think step by step.
The used car lot has 24 / 3 = 8 motorcycles with 2 tires each.
The lot has 24 - 8 = 16 cars for sale.
There are 16 / 4 = 4 cars with a spare tire with 5 tires each.
The lot has 16 - 4 = 12 cars with 4 tires each.
Thus, the used car lot's vehicles have 8 * 2 + 4 * 5 + 12 * 4 = 16 + 20 + 48 = 84 tires in all.
The answer is 84 </s><s>[INST] Question: Norma takes her clothes to the laundry. She leaves 9 T-shirts and twice as many sweaters as T-shirts in the washer. When she returns she finds 3 sweaters and triple the number of T-shirts. How many items are missing? [/INST] Let's think step by step.
Norma left 9 T-shirts And twice as many sweaters, she took 9 * 2= 18 sweaters.
Adding the T-shirts and sweaters, Norma left 9 + 18 = 27 clothes.
When she came back, she found 3 sweaters And triple the number of T-shirts, she found 3 * 3 = 9 T-shirts.
Adding the T-shirts and sweaters, Norma found 3 + 9 = 12 clothes.
Subtracting the clothes she left from the clothes she found, 27 - 12 = 15 clothes are missing.
The answer is 15 </s><s>[INST] Question: Adam has an orchard. Every day for 30 days he picks 4 apples from his orchard. After a month, Adam has collected all the remaining apples, which were 230. How many apples in total has Adam collected from his orchard? [/INST] Let's think step by step.
During 30 days Adam picked 4 * 30 = 120 apples.
So in total with all the remaining apples, he picked 120 + 230 = 350 apples from his orchard.
The answer is 350 </s><s>[INST] 
Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? [/INST]
'''
