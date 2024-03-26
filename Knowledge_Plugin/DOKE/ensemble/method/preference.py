import json

def gen_data(user, history, candidate, target, item2title, dataset):
    dataset2words = {
        "ml1m": {"prefix":"","watch":"watch", "watched":"watched", "watching":"watching", "movie":"movie", "movies":"movies"},
        "online_retail": {"prefix":"online retail ", "watch":"purchase", "watched":"purchased", "watching":"purchasing", "movie":"product", "movies":"products"},
        "beauty": {"prefix":"beauty ", "watch":"purchase", "watched":"purchased", "watching":"purchasing", "movie":"product", "movies":"products"},
    }
    words = dataset2words[dataset]

    prompt1 = f"I've {words['watched']} the following {words['prefix']}{words['movies']} in the past in order:\n"
    prompt1 += "["+", ".join([f"{idx}. {item2title[x]}" for idx, x in enumerate(history)])+"]\n"
    prompt1 += f"What features are most important to me when selecting {words['movies']}(Summarize my preferences briefly)?"
    prompt2 =  f"You will select the {words['movies']}(at most 5 {words['movies']}) that appeal to me the most from the list of {words['movies']} I have {words['watched']}, based on my personal preferences. The selected {words['movies']} will be presented in descending order of preference.(Format: no. a {words['watched']} {words['movie']})."
    prompt3 = f"Now there are 20 candidate {words['movies']} that I can {words['watch']} next:\n"
    prompt3 += "["+", ".join([f"{idx}. {item2title[x]}" for idx, x in enumerate(candidate)])+"]\n"
    prompt3 += f"Please rank the 20 candidate {words['movies']} by measuring the possibilities that I would like to {words['watch']} next most, according to the similarity to the selected {words['movies']} I've {words['watched']}. Please think step by step.\n"
    prompt3 += f"Your output is only allowed to be a rerank of the candidate list. Do not add {words['movies']} out of the candidate list.\n"
    prompt3 += f"Please give the results as JSON array (from highest to lowest priority, {words['movie']} names only): "

    data = {
        "messages1": [
            {
                "role": "user",
                "content": prompt1
            }
        ],
        "messages2": [
            {
                "role": "user",
                "content": prompt2
            }
        ],
        "messages3": [
            {
                "role": "user",
                "content": prompt3
            }
        ],
        "target": item2title[target].strip("\"")
    }
    return data

def gen_answer(data, gen_openai_response_func):
    messages1 = data["messages1"]
    response1 = gen_openai_response_func(messages1)
    data["response1"] = response1
    messages2 = data["messages1"] + [{"role":"assistant", "content":response1}] + data["messages2"]
    response2 = gen_openai_response_func(messages2)
    data["response2"] = response2
    messages3 = data["messages1"] + [{"role":"assistant", "content":response1}] + data["messages2"] + [{"role":"assistant", "content":response2}] + data["messages3"]
    response3 = gen_openai_response_func(messages3)
    data["response3"] = response3
    try:
        response3 = ' '.join(response3.strip().split('\n'))
        response3 = response3.split('[', 1)[1]
        response3 = ''.join(response3.split(']')[:-1])
        result = json.loads("["+response3+"]")
        if len(result) == 0:
            result = ["1"]
    except:
        result = ["1"]
    data['result'] = result
    return data