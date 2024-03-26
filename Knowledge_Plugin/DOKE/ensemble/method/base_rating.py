import json
import random
import re
def gen_data(user, history, candidate, target, item2title, dataset):
    dataset2words = {
        "ml1m": {"prefix":"","watch":"watch", "watched":"watched", "watching":"watching", "movie":"movie", "movies":"movies"},
        "online_retail": {"prefix":"online retail ", "watch":"purchase", "watched":"purchased", "watching":"purchasing", "movie":"product", "movies":"products"},
        "beauty": {"prefix":"beauty ", "watch":"purchase", "watched":"purchased", "watching":"purchasing", "movie":"product", "movies":"products"},
    }
    words = dataset2words[dataset]

    prompt = f"I've {words['watched']} the following {words['prefix']}{words['movies']} in the past in order:\n"
    prompt += "["+", ".join([f"{idx}. {item2title[x]}" for idx, x in enumerate(history)])+"]\n"
    if random.random() > 0.5:
        new_item = target
        ans = 1
    else:
        new_item = random.choice(list(set(candidate)-set([target])))
        ans = 0
    prompt += f"Now here is a candidate {words['movies']} that I can {words['watch']} next: {item2title[new_item]}\n"

    prompt += f"In order for you to make better recommendations, I will give you some {words['movies']} that I may be interested in in the candidate list according to my history, and ranked in descending order.\n"

    I2I_data = json.loads(open(f"../data/{dataset}/MF_CF_candidate_pop.json").readline())
    text = ""
    for idx, item in enumerate(history[-10:]):
        most_co_click_ids = [int(x[0]) for x in I2I_data[user][str(item)]][:3]
        most_co_click_titles = ", ".join([item2title[x] for x in most_co_click_ids])
        text += f"Users who {words['watched']} {item2title[item]}, their most frequently {words['watched']} {words['movies']} in descending order are: {most_co_click_titles}.\n"
    prompt += text
    
    prompt += f"Please predict the probability that I will {words['watch']} the new {words['movie']}: {item2title[new_item]}, according to my {words['watching']} history and the interested {words['movies']}. Please think step by step. Start your answer with a short explanation and then give the final result. Your answer only allowed to a number between 0 and 1.\n"
    prompt += f"Please give the results strictly following this format: [[probability]]"

    data = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "target": ans
    }
    return data

def gen_answer(data, gen_openai_response_func):
    response = gen_openai_response_func(data["messages"])
    data["response"] = response
    pattern = f"\[\[([\d\.]+)\]\]"
    match = re.search(pattern, response)
    if match:
        result = float(match.group(1))
    else:
        result = 0
    data['result'] = result
    return data