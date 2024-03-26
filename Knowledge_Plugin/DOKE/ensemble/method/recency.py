import json

def gen_data(user, history, candidate, target, item2title, dataset):
    dataset2words = {
        "ml1m": {"prefix":"","watch":"watch", "watched":"watched", "watching":"watching", "movie":"movie", "movies":"movies"},
        "online_retail": {"prefix":"online retail ", "watch":"purchase", "watched":"purchased", "watching":"purchasing", "movie":"product", "movies":"products"},
        "beauty": {"prefix":"beauty ", "watch":"purchase", "watched":"purchased", "watching":"purchasing", "movie":"product", "movies":"products"},
    }
    words = dataset2words[dataset]

    prompt = f"I've {words['watched']} the following {words['prefix']}{words['movies']} in the past in order:\n"
    prompt += "["+", ".join([f"{idx}. {item2title[x]}" for idx, x in enumerate(history)])+"]\n"
    prompt += f"Now there are 20 candidate {words['movies']} that I can {words['watch']} next:\n"
    prompt += "["+", ".join([f"{idx}. {item2title[x]}" for idx, x in enumerate(candidate)])+"]\n"

    prompt += f"Please rank the 20 candidate {words['movies']} by measuring the possibilities that I would like to {words['watch']} next most, according to my {words['watching']} history and the interested {words['movies']}. Please think step by step.\n"
    prompt += f"Note that my most recently {words['watched']} {words['movie']} is {item2title[history[-1]]}.\n"
    prompt += f"Your output is only allowed to be a rerank of the candidate list. Do not add {words['movies']} out of the candidate list.\n"
    prompt += f"Please give the results as JSON array (from highest to lowest priority, {words['movie']} names only): "

    data = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "target": item2title[target].strip("\"")
    }
    return data

def gen_answer(data, gen_openai_response_func):
    response = gen_openai_response_func(data["messages"])
    data["response"] = response
    try:
        response = ' '.join(response.strip().split('\n'))
        response = response.split('[', 1)[1]
        response = ''.join(response.split(']')[:-1])
        result = json.loads("["+response+"]")
        if len(result) == 0:
            result = ["1"]
    except:
        result = ["1"]
    data['result'] = result
    return data