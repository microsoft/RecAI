# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
from call_models.huggingface_models import gen_model_embedding_answer
from call_models.vllm_models import gen_model_chat_answer
from call_models.openai_models import gen_api_chat_answer, gen_api_embedding_answer
from evaluates.evaluate import compute_metrics_on_multi_choices, compute_metrics_on_title_recommend, compute_errors_on_title_ranking, compute_metrics_on_id_recommend
from evaluates.TFIDF_model import TFIDF_model
from utils import *

DEFAULT_JUDGE_SYSTEM_PROMPT = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any positional biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie."
DEFAULT_JUDGE_PROMPT_TEMPLATE = "[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"
allow_regenerate = os.getenv("ALLOW_REGENERATE", "False").lower() == "true"


## If you use customerized deployment names, don't forget to add them to this list
OPENAI_MODELS = ["gpt-35-turbo", "gpt-4o", "gpt-4o-mini", "gpt-4.1"]


if __name__ == "__main__":
    args = parse_args()

    # ------------------------------------------------------------------
    # Support *multiple* benchmark datasets in a single command.
    # We launch a new Python subprocess of the same script for every
    # additional dataset beyond the first.  This keeps the original
    # evaluation logic untouched while providing the desired behaviour.
    # ------------------------------------------------------------------
    if len(getattr(args, "bench_names", [args.bench_name])) > 1:
        import subprocess, sys

        def _strip_bench_args(argv):
            """Remove --bench-name(s) flag and *all* of its positional values.

            Because the number of dataset names is unknown (nargs="+"), we
            keep consuming tokens until we encounter the next option that
            starts with "--" or reach the end of argv.
            """
            cleaned = []
            i = 0
            while i < len(argv):
                tok = argv[i]
                if tok in ("--bench-name", "--bench-names"):
                    i += 1  # skip the flag itself
                    # skip all following non-option tokens (dataset names)
                    while i < len(argv) and not argv[i].startswith("--"):
                        i += 1
                    continue  # continue outer while loop without increment
                cleaned.append(tok)
                i += 1
            return cleaned

        # The arguments that are *not* benchmark specific remain the same
        base_argv = _strip_bench_args(sys.argv[1:])

        # Kick off a separate subprocess for each benchmark
        for bench in args.bench_names:
            cmd = [sys.executable, sys.argv[0], "--bench-name", bench] + base_argv
            print(f"\n[Multi-bench] Launching evaluation for dataset '{bench}' → {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

        # All work delegated; terminate parent process successfully.
        sys.exit(0)

    meta_data_file = f"data/{args.bench_name}/metadata.json"
    all_prompt_config = load_prompt_config()

    for task_name in args.task_names:
        if task_name in all_prompt_config:
            prompt_config = all_prompt_config[task_name]
        else:
            prompt_config = {}

        if task_name in ["retrieval", "ranking"]:
            question_file = f"data/{args.bench_name}/{task_name}.jsonl"
            answer_file = f"output/{args.bench_name}/{parse_model_name_to_dirname(args.model_path_or_name)}/{task_name}.jsonl"

            if allow_regenerate or not os.path.exists(answer_file):
                if args.model_path_or_name not in OPENAI_MODELS:
                    system_prompt = prompt_config.get("local_model_system_prompt", None)
                    gen_model_chat_answer(args.model_path_or_name, question_file, answer_file, args, system_prompt)
                else:
                    system_prompt = prompt_config.get("api_system_prompt", None)
                    gen_api_chat_answer(args.model_path_or_name, question_file, answer_file, args, system_prompt)
            all_metric = compute_metrics_on_title_recommend(answer_file, meta_data_file)
            error_metric = compute_errors_on_title_ranking(answer_file, meta_data_file, 0.8)
            all_metric = {**all_metric, **error_metric}
            # --- save metrics ---
            record_metrics(args.bench_name, args.model_path_or_name, task_name, all_metric)
            
        if task_name == "multi_choices":
            question_file = f"data/{args.bench_name}/{task_name}.jsonl"
            answer_file = f"output/{args.bench_name}/{parse_model_name_to_dirname(args.model_path_or_name)}/{task_name}.jsonl"
            if allow_regenerate or not os.path.exists(answer_file):
                if args.model_path_or_name not in OPENAI_MODELS:
                    system_prompt = prompt_config.get("local_model_system_prompt", None)
                    gen_model_chat_answer(args.model_path_or_name, question_file, answer_file, args, system_prompt)
                else:
                    system_prompt = prompt_config.get("api_system_prompt", None)
                    gen_api_chat_answer(args.model_path_or_name, question_file, answer_file, args, system_prompt)
            # Compute accuracy@1 for multiple-choice questions
            acc1, none_ratio = compute_metrics_on_multi_choices(answer_file)
            # --- save metrics ---
            record_metrics(args.bench_name, args.model_path_or_name, task_name, {"acc@1": acc1, "none_ratio": none_ratio})
            print(f"Model's multi-choices acc@1: {acc1:.3f}, none_ratio: {none_ratio:.3f}")
            
        # === CF / Sequential  ===
        #   cf_ranking_mc: reuse the retrieval metric (title-similarity + threshold)
        if task_name == "cf_ranking_mc":
            question_file = f"data/{args.bench_name}/{task_name}.jsonl"
            answer_file = f"output/{args.bench_name}/{parse_model_name_to_dirname(args.model_path_or_name)}/{task_name}.jsonl"
            if allow_regenerate or not os.path.exists(answer_file):
                if args.model_path_or_name not in OPENAI_MODELS:
                    system_prompt = prompt_config.get("local_model_system_prompt", None)
                    gen_model_chat_answer(args.model_path_or_name, question_file, answer_file, args, system_prompt)
                else:
                    system_prompt = prompt_config.get("api_system_prompt", None)
                    gen_api_chat_answer(args.model_path_or_name, question_file, answer_file, args, system_prompt)
            # Use multiple-choice accuracy as the metric
            cf_acc1, cf_none = compute_metrics_on_multi_choices(answer_file)
            record_metrics(args.bench_name, args.model_path_or_name, task_name, {"acc@1": cf_acc1, "none_ratio": cf_none})
            print(f"Model's cf_ranking_mc acc@1: {cf_acc1:.3f}, none_ratio: {cf_none:.3f}")

        elif task_name == "seq_ranking_mc":
            question_file = f"data/{args.bench_name}/{task_name}.jsonl"
            answer_file = f"output/{args.bench_name}/{parse_model_name_to_dirname(args.model_path_or_name)}/{task_name}.jsonl"
            if allow_regenerate or not os.path.exists(answer_file):
                if args.model_path_or_name not in OPENAI_MODELS:
                    system_prompt = prompt_config.get("local_model_system_prompt", None)
                    gen_model_chat_answer(args.model_path_or_name, question_file, answer_file, args, system_prompt)
                else:
                    system_prompt = prompt_config.get("api_system_prompt", None)
                    gen_api_chat_answer(args.model_path_or_name, question_file, answer_file, args, system_prompt)

            # Sequential recommendation also evaluated as multiple-choice
            seq_acc1, seq_none = compute_metrics_on_multi_choices(answer_file)
            record_metrics(args.bench_name, args.model_path_or_name, task_name, {"acc@1": seq_acc1, "none_ratio": seq_none})
            print(f"Model's seq_ranking_mc acc@1: {seq_acc1:.3f}, none_ratio: {seq_none:.3f}")

        if task_name == "chatbot":
            question_file = f"data/{args.bench_name}/chatbot.jsonl" 
            # generate response of each model
            for model in [args.model_path_or_name, args.baseline_model]:
                answer_file = f"output/{args.bench_name}/{parse_model_name_to_dirname(model)}.jsonl"
                if allow_regenerate or not os.path.exists(answer_file):
                    if model not in OPENAI_MODELS:
                        system_prompt = prompt_config.get("local_model_system_prompt", None)
                        gen_model_chat_answer(model, question_file, answer_file, args, system_prompt)
                    else:
                        system_prompt = prompt_config.get("api_system_prompt", None)
                        gen_api_chat_answer(model, question_file, answer_file, args, system_prompt)
            
            # construct pairwise judge prompt
            eval_model_name = parse_model_name_to_dirname(args.model_path_or_name)
            base_model_name = parse_model_name_to_dirname(args.baseline_model)
            prompt_template = prompt_config.get("judge_prompt_template", DEFAULT_JUDGE_PROMPT_TEMPLATE)
            gen_judge_prompts(
                f"output/{args.bench_name}/{eval_model_name}.jsonl",
                f"output/{args.bench_name}/{base_model_name}.jsonl",
                f"output/{args.bench_name}/{eval_model_name}/{base_model_name}/question.jsonl",
                prompt_template
            )

            # generate judge response
            question_file = f"output/{args.bench_name}/{eval_model_name}/{base_model_name}/question.jsonl"
            answer_file = f"output/{args.bench_name}/{eval_model_name}/{base_model_name}/answer.jsonl"
            system_prompt = prompt_config.get("judge_system_prompt", DEFAULT_JUDGE_SYSTEM_PROMPT)
            gen_api_chat_answer(args.judge_model, question_file, answer_file, args, system_prompt)
            
            loss = 0
            win = 0
            tie = 0
            for line in open(answer_file):
                data = json.loads(line)
                if "[[A]]" in data["answer"][0] and "[[B]]" in data["answer"][1]:
                    win += 1
                elif "[[B]]" in data["answer"][0] and "[[A]]" in data["answer"][1]:
                    loss += 1
                else:
                    tie += 1
            if win>loss:
                print(f"{eval_model_name} is better")
            elif win==loss:
                print(f"draw!")
            else:
                print(f"{base_model_name} is better")
            print(f"win:{win}, loss:{loss}, tie/error:{tie}")
            print(f"win_rate: {win/(win+loss+tie)}")
            print(f"loss_rate: {loss/(win+loss+tie)}")

        if task_name == "explanation":
            question_file = f"data/{args.bench_name}/explanation.jsonl"

            for model in [args.model_path_or_name, args.baseline_model]:
                answer_file = f"output/{args.bench_name}/{parse_model_name_to_dirname(model)}/explanation.jsonl"
                if allow_regenerate or not os.path.exists(answer_file):
                    if model not in OPENAI_MODELS:
                        system_prompt = prompt_config.get("local_model_system_prompt", None)
                        gen_model_chat_answer(model, question_file, answer_file, args, system_prompt)
                    else:
                        system_prompt = prompt_config.get("api_system_prompt", None)
                        gen_api_chat_answer(model, question_file, answer_file, args, system_prompt)

            eval_model_name = parse_model_name_to_dirname(args.model_path_or_name)
            base_model_name = parse_model_name_to_dirname(args.baseline_model)
            prompt_template = prompt_config.get("judge_prompt_template", DEFAULT_JUDGE_PROMPT_TEMPLATE)
            gen_judge_prompts(
                f"output/{args.bench_name}/{parse_model_name_to_dirname(eval_model_name)}/explanation.jsonl",
                f"output/{args.bench_name}/{parse_model_name_to_dirname(base_model_name)}/explanation.jsonl",
                f"output/{args.bench_name}/{eval_model_name}/explanation/{base_model_name}/question.jsonl",
                prompt_template
            )

            question_file = f"output/{args.bench_name}/{eval_model_name}/explanation/{base_model_name}/question.jsonl"
            answer_file = f"output/{args.bench_name}/{eval_model_name}/explanation/{base_model_name}/answer.jsonl"
            system_prompt = prompt_config.get("judge_system_prompt", DEFAULT_JUDGE_SYSTEM_PROMPT)
            if allow_regenerate or not os.path.exists(answer_file):
                gen_api_chat_answer(args.judge_model, question_file, answer_file, args, system_prompt)
            
            for aspect in ["informativeness", "persuasiveness", "helpfulness"]:
                print(f"evaluate result of {aspect}")
                eval_model_score = 0
                base_model_score = 0
                cnt = 0
                for line in open(answer_file):
                    data = json.loads(line)
                    try:
                        result = '{'+'}'.join(data["answer"][0].split('{', 1)[1].split('}')[:-1])+'}'
                        result = json.loads(result.lower())
                        eval_model_score += result["a"][aspect]
                        base_model_score += result["b"][aspect]
                        cnt += 1
                        result = '{'+'}'.join(data["answer"][1].split('{', 1)[1].split('}')[:-1])+'}'
                        result = json.loads(result.lower())
                        eval_model_score += result["b"][aspect]
                        base_model_score += result["a"][aspect]
                        cnt += 1
                    except:
                        pass
                print(f"eval_model: {eval_model_score/cnt:.3f}")
                print(f"base_model: {base_model_score/cnt:.3f}")
            loss = 0
            win = 0
            tie = 0
            for line in open(answer_file):
                data = json.loads(line)
                if "[[A]]" in data["answer"][0] and "[[B]]" in data["answer"][1]:
                    win += 1
                elif "[[B]]" in data["answer"][0] and "[[A]]" in data["answer"][1]:
                    loss += 1
                else:
                    tie += 1
            print(f"win:{win}, loss:{loss}, tie/error:{tie}")
            print(f"win_rate: {win/(win+loss+tie)}")
            print(f"loss_rate: {loss/(win+loss+tie)}")

        if task_name == "conversation":
            eval_model_name = parse_model_name_to_dirname(args.model_path_or_name)
            eval_model_answer_file = f"data/{args.bench_name}/conversation.jsonl"
            for turn in range(args.max_turn):
                simulator_question_file = f"output/{args.bench_name}/{eval_model_name}/conversation/{args.simulator_model}/simulator_question.jsonl"
                simulator_answer_file = f"output/{args.bench_name}/{eval_model_name}/conversation/{args.simulator_model}/simulator_answer.jsonl"
                gen_user_simulator_prompts(eval_model_answer_file, simulator_question_file)
                gen_api_chat_answer(args.simulator_model, simulator_question_file, simulator_answer_file, args, system_prompt=None,
                    verify=True,
                    verify_prompt_config=prompt_config["verify_prompt_config"]
                )

                eval_model_question_file = f"output/{args.bench_name}/{eval_model_name}/conversation/{args.simulator_model}/eval_model_question.jsonl"
                eval_model_answer_file = f"output/{args.bench_name}/{eval_model_name}/conversation/{args.simulator_model}/eval_model_answer.jsonl"
                gen_eval_model_conversation_prompts(simulator_answer_file, eval_model_question_file)
                if allow_regenerate or not os.path.exists(eval_model_answer_file):
                    if args.model_path_or_name not in OPENAI_MODELS:
                        system_prompt = prompt_config.get("local_model_system_prompt", None)
                        gen_model_chat_answer(args.model_path_or_name, eval_model_question_file, eval_model_answer_file, args, system_prompt)
                    else:
                        system_prompt = prompt_config.get("api_system_prompt", None)
                        gen_api_chat_answer(args.model_path_or_name, eval_model_question_file, eval_model_answer_file, args, system_prompt)

            item_list = []
            for line in open(meta_data_file):
                line = json.loads(line)
                if 'TitleName' in line:
                    item_list.append(line['TitleName'])
                elif 'app_name' in line:
                    item_list.append(line['app_name'])
                elif 'title' in line:
                    item_list.append(line['title'])

            tfidf_model = TFIDF_model(item_list)
            recall = 0
            all_turns = 0
            cnt = 0
            for line in open(eval_model_answer_file):
                data = json.loads(line)
                responses = [x["content"] for x in data["prompt"] if x["role"] == "assistant"] + [data["answer"]]
                for idx, response in enumerate(responses):
                    all_turns += 1
                    if fuzzy_substring_matching(data["target"], response, tfidf_model, 0.8):
                        recall += 1
                        break
                cnt += 1
            print(f"recall: {recall/cnt:.3f}")
            print(f"AT: {all_turns/cnt:.3f}")
        if task_name in ["embedding_ranking", "embedding_retrieval"]:
            # infer embedding
            eval_model_name = parse_model_name_to_dirname(args.model_path_or_name)
            item_embedding_path = f"output/{args.bench_name}/{eval_model_name}/item_embedding_{args.item_emb_type}.pkl"
            if not os.path.exists(item_embedding_path):
                item_embedding_prompt_path = f"output/{args.bench_name}/item_embedding_prompt_{args.item_emb_type}.jsonl"
                item_embedding_answer_path = f"output/{args.bench_name}/{eval_model_name}/item_embedding_answer_{args.item_emb_type}.jsonl"
                gen_item_embedding_prompt(args.bench_name, args.item_emb_type, item_embedding_prompt_path)
                if allow_regenerate or not os.path.exists(item_embedding_answer_path):
                    if args.model_path_or_name in ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]:
                        gen_api_embedding_answer(args.model_path_or_name, item_embedding_prompt_path, item_embedding_answer_path)
                    else:
                        gen_model_embedding_answer(args.model_path_or_name, item_embedding_prompt_path, item_embedding_answer_path, args)
                extract_embedding(item_embedding_answer_path, item_embedding_path)
            
            user_embedding_path = f"output/{args.bench_name}/{eval_model_name}/user_embedding_{args.user_emb_type}.pkl"
            if not os.path.exists(user_embedding_path):    
                user_embedding_prompt_path = f"output/{args.bench_name}/user_embedding_prompt_{args.user_emb_type}.jsonl"
                user_embedding_answer_path = f"output/{args.bench_name}/{eval_model_name}/user_embedding_answer_{args.user_emb_type}.jsonl"
                gen_user_embedding_prompt(args.bench_name, args.user_emb_type, user_embedding_prompt_path, prompt_config)
                if allow_regenerate or not os.path.exists(user_embedding_answer_path):
                    if args.user_emb_type == "summary":
                        gen_api_chat_answer(args.summary_model, user_embedding_prompt_path, user_embedding_answer_path, args, "")
                        extract_summary(user_embedding_answer_path, user_embedding_prompt_path)
                    if args.model_path_or_name in ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]:
                        gen_api_embedding_answer(args.model_path_or_name, user_embedding_prompt_path, user_embedding_answer_path)
                    else:
                        gen_model_embedding_answer(args.model_path_or_name, user_embedding_prompt_path, user_embedding_answer_path, args)
                extract_embedding(user_embedding_answer_path, user_embedding_path)
                
            answer_file = f"output/{args.bench_name}/embedding_methods/{eval_model_name}/{task_name}_user_{args.user_emb_type}_item_{args.item_emb_type}.jsonl"
            negative_path = f"data/{args.bench_name}/negative_samples.txt"
            sequential_path = f"data/{args.bench_name}/sequential_data.txt"
            if "ranking" in task_name:
                gen_ranking_result(user_embedding_path, item_embedding_path, sequential_path, negative_path, answer_file)
            elif "retrieval" in task_name:
                gen_retrieval_result(user_embedding_path, item_embedding_path, sequential_path, answer_file)
            all_metric = compute_metrics_on_id_recommend(answer_file)
            # --- save metrics ---
            record_metrics(args.bench_name, args.model_path_or_name, task_name, all_metric)