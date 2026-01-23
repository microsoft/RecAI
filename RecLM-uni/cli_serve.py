import argparse
import html
import os.path

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from train_utils.processor import FastPrefixConstrainedLogitsProcessor, Trie_link
from train_utils.utils import load_json

domains = ["steam", "movies", "toys"]
system_message = {
    'role': "system",
    'content': 'You are an expert recommender engine as well as a helpful, respectful and honest assistant. When you generate an item, each item should be enclosed by <SOI> and <EOI>. <SOI> should be generated before item title, and <EOI> should be generated after item title.'
}
chat_history = [system_message]


def create_PCLP(data_path):
    metas = load_json(os.path.join(data_path, 'metas.jsonl'))
    item_list = [metas[_]['title_t'] for _ in metas]
    item_ids = tokenizer.batch_encode_plus(item_list, add_special_tokens=False).data['input_ids']
    item_prefix_tree = Trie_link(item_ids, tokenizer)
    return FastPrefixConstrainedLogitsProcessor(item_prefix_tree.constrain_search_list, num_beams)


def chat(user_message, history, target_domain):
    global chat_history
    if history is None:
        history = []

    chat_history.append({'role': "user", 'content': user_message})
    prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
    print('prompt: ', prompt)
    input_data = tokenizer.batch_encode_plus([prompt], add_special_tokens=False, return_tensors='pt').to(device=args.gpu).data
    input_ids_length = input_data['input_ids'].shape[1]

    output_ids = model.generate(
        **input_data,
        logits_processor=domain_PCLP.get(target_domain),
        max_length=1024,
        num_beams=num_beams,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    output_text = tokenizer.batch_decode(output_ids[:, input_ids_length:], skip_special_tokens=False)[0]
    print('output_text: ', output_text)
    if output_text.endswith(tokenizer.eos_token):
        output_text = output_text.replace(tokenizer.eos_token, '')
    chat_history.append({'role': "assistant", 'content': output_text})

    user_message = html.escape(user_message)
    output_text = html.escape(output_text)
    history.append([user_message, output_text])
    return "", history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='Llama-2-7b-hf-chat', help="openai model")
    parser.add_argument("--gpu", type=str, default='cuda:0')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = '<|reserved_special_token_250|>'
    tokenizer.pad_token_id = 128255
    tokenizer.soi_token = "<SOI>"
    tokenizer.eoi_token = "<EOI>"
    tokenizer.soi_token_id = tokenizer.convert_tokens_to_ids(tokenizer.soi_token)
    tokenizer.eoi_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eoi_token)
    tokenizer.eos_token = "<|eot_id|>"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map=args.gpu).eval()
    # model = None

    num_beams = 1
    domain_PCLP = {d: [create_PCLP(f'data/dataset/{d}/')] for d in domains if os.path.exists(f'data/dataset/{d}/')}

    with gr.Blocks() as demo:
        gr.Markdown("# RecLM-cgen")

        chatbot = gr.Chatbot(label="Chat history", height=500, show_copy_button=True)
        clear_btn = gr.Button("Clear")

        with gr.Row():
            msg = gr.Textbox(label="input", placeholder="Input instruction...", lines=2)
            domain_choice = gr.Radio(choices=domains + ["None"], label="Choice the target domain", value="None")

        def clear_history():
            global chat_history
            chat_history = [system_message]
            return "", []

        clear_btn.click(fn=clear_history, inputs=[], outputs=[msg, chatbot])
        msg.submit(chat, inputs=[msg, chatbot, domain_choice], outputs=[msg, chatbot])

    demo.launch(server_name="0.0.0.0", server_port=7861)

