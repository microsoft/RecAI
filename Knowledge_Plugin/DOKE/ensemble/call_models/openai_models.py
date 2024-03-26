import os
import json
import time
import yaml
from tqdm import tqdm
import openai

import queue
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

api_config = yaml.safe_load(open("api_config.yaml"))
openai.api_base = api_config["API_BASE"]
if api_config["API_TYPE"] == "azure":
    openai.api_version = api_config["API_VERSION"]
    openai.api_type = api_config["API_TYPE"]

api_keys = api_config["API_KEYS"]
MAX_THREADS = len(api_keys)
MAX_RETRIES = 3
INTERVAL = 30

def call_openai_chat(model, api_key, messages, temp=0.0):
    # initialize OpenAI GPT-3 with your api_key
    openai.api_key = api_key
    for i in range(MAX_RETRIES):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                # engine=model,
                messages=messages,
                temperature=temp,
                top_p=0.95,
                request_timeout=30,
            )
            content = response['choices'][0]['message']['content']
            return content
        except Exception as e:
            print(f"{api_key}\nError occurred: {e}. Retrying...")
            # time.sleep(INTERVAL)  # 重试之间的休眠时间
    print(f"Failed to get response for messages: {messages} after {MAX_RETRIES} retries.")
    return "None"

def worker_gen_answer(i, model, gen_answer_func):
    while not prompts_queue.empty():
        index, data = prompts_queue.get()
        api_key = api_keys[i % len(api_keys)]
        def gen_openai_response_func(messages):
            return call_openai_chat(model, api_key, messages)
        result = gen_answer_func(data, gen_openai_response_func)
        results.put((index, result))
        # if model == "gpt-3.5-turbo":
        # time.sleep(INTERVAL)  # 控制调用频率
        with num_completed.get_lock():
            num_completed.value += 1  # 更新完成任务数量

# 创建进度条更新进程
def progress_monitor(total):
    with tqdm(total=total) as pbar:
        while True:
            completed = num_completed.value
            pbar.n = completed
            pbar.refresh()
            if completed >= total:
                break
            time.sleep(0.1)

def gen_openai_answer(model, question_file, answer_file, gen_answer_func):
    all_data = []
    for line in open(question_file):
        data = json.loads(line)
        all_data.append(data)

    global results, num_completed, prompts_queue
    results = queue.PriorityQueue()
    num_completed = multiprocessing.Value('i', 0)

    prompts_queue = queue.Queue()
    for i, (data) in enumerate(all_data):
        prompts_queue.put((i, data))

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        progress_process = multiprocessing.Process(target=progress_monitor, args=(len(all_data),))
        progress_process.start()

        for i in range(MAX_THREADS):
            executor.submit(worker_gen_answer, i, model, gen_answer_func)

    # 现在我们的结果队列中应该有所有的 API 调用结果，可以像这样处理它们：
    final_results = []
    while not results.empty():
        final_results.append(results.get())

    # 对结果按照索引进行排序
    final_results.sort()
    
    # 等待进度条进程完成
    progress_process.join()

    # 打印排序后的结果
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    fd = open(answer_file, "w", encoding='utf-8')
    for index, result in final_results:
        fd.write(json.dumps(result, ensure_ascii=False) + '\n')
    fd.close()