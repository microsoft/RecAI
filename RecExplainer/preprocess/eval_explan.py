# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pandas as pd
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="data process")
    parser.add_argument(
        "--model_names", type=str, help=""
    )
    parser.add_argument(
        "--model_response_files", type=str, help=""
    )
    parser.add_argument(
        "--judge_response_file", type=str, help=""
    )
    parser.add_argument(
        "--judge_query_file", type=str, help="", default=None
    )
    args = parser.parse_args()
    return args

def eval_data_gen(args):
    data_files = {}
    model_names = args.model_names.split(',')
    for model, file in zip(args.model_names.split(','), args.model_response_files.split(',')):
        data_files[model] = file

    template = "Please act as an impartial judge and evaluate the AI assistant's recommendation decision as well as decision explanation based on the user's purchase history, target item, and ground truth label. Assign a score according to the following four levels:\n \
    A. <score>A</score>: Incorrect classification - The assistant failed to generate a correct recommendation decision.\n \
    B. <score>B</score>: Correct classification, insufficient explanation - The assistant correctly classified the recommendation but provided no, few, or irrelevant explanations, or provided explanations with hallucination, some of which do not conform to the actual situation.\n \
    C. <score>C</score>: Correct classification, acceptable explanation - The assistant correctly classified the recommendation and provided an explanation that is logically consistent and aligns with the user's history and target item, but has minor imperfections such as lack of persuasiveness or informativeness.\n \
    D. <score>D</score>: Correct classification, satisfying explanation - The assistant correctly classified the recommendation and provided a satisfactory explanation, including a summary of the user's historical behavior patterns and characteristics, as well as a thorough analysis of the consistency or inconsistency between user preferences and the target item.\n \
    Please give your score in the form of <score>label</score>, for example: if the score is C, output <score>C</score>. Do not allow the length of the explanation to influence your evaluation. Be as objective as possible.\n \
    Known information: User history: {0}, Target item: {1}, Label: {2}. Assistant's output: {3}"

    # final df columns : model, label, history, target item, question
    all_dfs = {}

    for model, file in data_files.items():
        df = pd.read_csv(file, header=0, sep=',')
        all_dfs[model] = df

    output_df = pd.DataFrame(columns=['model', 'label', 'history', 'target item', 'question'])

    for i in range(len(all_dfs[model_names[0]])):
        label = all_dfs[model_names[0]].loc[i, 'label']
        history = eval(all_dfs[model_names[0]].loc[i, 'history'])
        target_item = all_dfs[model_names[0]].loc[i, 'target item']

        for model in model_names:
            output_df = output_df._append({'model': model, 'label': label, 'history': history, 'target item': target_item, 'question': template.format(', '.join(history), target_item, label, all_dfs[model].loc[i, 'answer'])}, ignore_index=True)

    output_df.to_csv(args.judge_query_file, sep=',', index=False)
        

def eval_metric(args):
    metric_file = args.judge_response_file
    df = pd.read_csv(metric_file, header=0, sep=',')
    results = {}
    for model in args.model_names.split(','):
        results[model] = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    
    for i in range(len(df)):
        model = df.loc[i, 'model']
        answer = df.loc[i, 'score'].strip().lower()
        if answer == '<score>a</score>':
            results[model]['A'] += 1
        elif answer == '<score>b</score>':
            results[model]['B'] += 1
        elif answer == '<score>c</score>':
            results[model]['C'] += 1
        elif answer == '<score>d</score>':
            results[model]['D'] += 1
        else:
            print('error')
    
    print(results)

if __name__ == '__main__':
    args = parse_args()
    if args.judge_query_file is not None:
        os.makedirs(os.path.dirname(args.judge_query_file), exist_ok=True)
        eval_data_gen(args)
    else:
        eval_metric(args)