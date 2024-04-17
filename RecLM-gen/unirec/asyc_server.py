# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pickle
import sys
import asyncio
import functools
from sanic import Sanic
from sanic.log import logger

import sanic
import torch
from sanic.worker.manager import WorkerManager
from unirec.utils import argument_parser, general


WorkerManager.THRESHOLD = 600


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


app = Sanic(__name__)
# we only run 1 inference run at any time (one could schedule between several runners if desired)
MAX_QUEUE_SIZE = 204800
MAX_BATCH_SIZE = 20480
MAX_WAIT = 1


class HandlingError(Exception):
    def __init__(self, msg, code=500):
        super().__init__()
        self.handling_code = code
        self.handling_msg = msg


model_path = {
    'sub_movie': "unirec/output/sub_movie/SASRec/train/checkpoint_2024-03-17_014803_35/SASRec-SASRec-sub_movie.pth",
    'steam': "unirec/output/steam/SASRec/train/checkpoint_2024-03-17_014033_93/SASRec-SASRec-steam.pth",
}
print(' '.join(sys.argv))
config = argument_parser.parse_arguments()
config['port'] = [_.split('=')[1] for _ in sys.argv if _.startswith('--port=')][0]
config['device'] = torch.device('cuda:0')
dataset = config['dataset']
config['dataset_path'] = f'data/{dataset}'

category2item = load_pickle(f'unirec/{config["dataset_path"]}/category.pickle')
map_dict = load_pickle(f'unirec/{config["dataset_path"]}/map.pkl')


def process_output(batched_data):
    if 'k' in batched_data:
        candidate_item_list = batched_data['candidate_item_list']
        target_category = batched_data['target_category']
        item_list = batched_data['item_list']
        score = batched_data['score']
        if candidate_item_list is None and target_category is not None:
            candidate_item_list = list(set(category2item[target_category[1:]]) - set(item_list)) \
                if target_category[0] == '+' else \
                list(set(map_dict['item2id'].keys()) - set(category2item[target_category[1:]]) - {'[PAD]'} - set(item_list))
        elif candidate_item_list is None:
            candidate_item_list = list(set(map_dict['item2id'].keys()) - set(item_list) - {'[PAD]'})
        if candidate_item_list is not None:
            candidate_id_list = torch.tensor(
                [map_dict['item2id'][_] for _ in candidate_item_list if _ in map_dict['item2id'].keys()], dtype=torch.int64).to(config["device"])

            score[candidate_id_list] += 1
            score -= 1
            score[..., map_dict['item2id']['[PAD]']] = -torch.inf
        top_k_indices = torch.topk(score, batched_data['k']).indices
        top_k = [[map_dict['id2item'][_] for _ in top_k_indices.tolist()]]
        batched_data['top_k'] = top_k

    else:
        candidate_item_list = batched_data['candidate_item_list']
        score = batched_data['score']
        assert candidate_item_list is not None
        candidate_id_lists = torch.tensor(
            [map_dict['item2id'][_] for _ in candidate_item_list if _ in map_dict['item2id']],
            dtype=torch.int64, device=config["device"]
        )
        score[map_dict['item2id']['[PAD]']] = -torch.inf

        ranked_indices = torch.argsort(score, descending=True)
        ranked_indices = torch.argsort(ranked_indices)
        batched_data['ranking'] = [ranked_indices[candidate_id_lists].tolist()]


def process_input(data):
    users = data.get('users')
    item_length = data.get('item_lengths')
    k = data.get('k')
    item_lists = data.get('item_lists')
    candidate_item_lists = data.get('candidate_item_lists')
    target_category = data.get('target_category')
    max_len = 10
    for idx, _ in enumerate(item_lists):
        if len(_) < max_len:
            item_lists[idx] = ['[PAD]'] * (max_len - len(_)) + _

    batched_data = {
        'user_id': torch.tensor(map_dict['user2id'][users[0]], dtype=torch.int64, device=config["device"]),
        'item_length': torch.tensor(item_length[0], dtype=torch.int64, device=config["device"]),
        'item_id_list': torch.tensor([map_dict['item2id'][_] for _ in item_lists[0]], dtype=torch.int64, device=config["device"]),
        'item_list': item_lists[0],
        'candidate_item_list': candidate_item_lists[0] if candidate_item_lists else None,
        'target_category': target_category[0] if target_category else None,
        "time": app.loop.time(),
        "done_event": asyncio.Event(loop=app.loop)
    }
    if k is not None:
        batched_data['k'] = k
    # print(batched_data)
    return batched_data


class ModelRunner:
    def __init__(self, model_name):
        self.model_name = model_name
        self.queue = []
        self.queue_lock = None

        self.model = general.get_class_instance(model_name, 'unirec/model')(config).to(config['device'])
        checkpoint = torch.load(model_path[dataset], map_location=config["device"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.model.requires_grad_(False)

        self.needs_processing = None
        self.needs_processing_timer = None

        self.all_item_id = torch.arange(len(map_dict['id2item']), device=config['device'], dtype=torch.int64)
        _, scores, _, _ = self.model.forward(item_seq=torch.tensor([[0, 0, 0, 1, 2, 3, 4]], device=config['device']), item_id=self.all_item_id)

    def schedule_processing_if_needed(self):
        if len(self.queue) >= MAX_BATCH_SIZE:
            logger.debug("next batch ready when processing a batch")
            self.needs_processing.set()
        elif self.queue:
            logger.debug("queue nonempty when processing a batch, setting next timer")
            self.needs_processing_timer = app.loop.call_at(self.queue[0]["time"] + MAX_WAIT, self.needs_processing.set)

    async def push_input(self, data):
        batched_data = process_input(data)
        async with self.queue_lock:
            if len(self.queue) >= MAX_QUEUE_SIZE:
                raise HandlingError("I'm too busy", code=503)
            self.queue.append(batched_data)
            logger.debug("enqueued task. new queue size {}".format(len(self.queue)))
            self.schedule_processing_if_needed()
        await batched_data["done_event"].wait()
        process_output(batched_data)
        return batched_data["top_k"]

    def run_model(self, batch_data):
        _, scores, _, _ = self.model.forward(**batch_data, item_id=self.all_item_id)
        # return self.model.forward_user_emb(batch_data).softmax(dim=1)
        return scores.softmax(dim=1)

    async def model_runner(self):
        self.queue_lock = asyncio.Lock(loop=app.loop)
        self.needs_processing = asyncio.Event(loop=app.loop)
        logger.info("started model runner for {}".format(self.model_name))
        # while True: Infinite loop, the program will be in a listening state
        while True:
            # Waiting for a task to come
            await self.needs_processing.wait()
            self.needs_processing.clear()
            # Clear timer
            if self.needs_processing_timer is not None:
                self.needs_processing_timer.cancel()
                self.needs_processing_timer = None
            # All processing queues are locked
            async with self.queue_lock:
                # If the queue is not empty, set the maximum waiting time
                if self.queue:
                    longest_wait = app.loop.time() - self.queue[0]["time"]
                else:  # oops
                    longest_wait = None
                # Logger start processing
                logger.debug("launching processing. queue size: {}. longest wait: {}".format(len(self.queue), longest_wait))
                # Get a batch of data
                to_process = self.queue[:MAX_BATCH_SIZE]
                # delete these data from the task queue
                del self.queue[:len(to_process)]
                self.schedule_processing_if_needed()
            # Generate batch data
            # print(to_process)
            if len(to_process) == 0:
                continue
            batch_data = {
                # 'user_id': torch.stack([t["user_id"] for t in to_process], dim=0),
                # 'item_length': torch.stack([t["item_length"] for t in to_process], dim=0),
                'item_seq': torch.stack([t["item_id_list"] for t in to_process], dim=0),
            }
            # print(batch_data)
            # Run the model in a separate thread and return the results
            scores = await app.loop.run_in_executor(
                None, functools.partial(self.run_model, batch_data)
            )
            # Log the results and set a completion event
            for t, s in zip(to_process, scores):
                t["score"] = s
                t["done_event"].set()
            del to_process


@app.route('/inference', methods=['POST'])
async def inference(request):
    try:
        data = request.json
        if data.get('immediately'):
            batched_data = process_input(data)
            input_data = {
                # 'user_id': torch.unsqueeze(batched_data["user_id"], dim=0),
                # 'item_length': torch.unsqueeze(batched_data["item_length"], dim=0),
                'item_seq': torch.unsqueeze(batched_data["item_id_list"], dim=0),
            }
            scores = style_transfer_runner.run_model(input_data)
            batched_data['score'] = scores[0]
            process_output(batched_data)
            top_k = batched_data['top_k']
        else:
            top_k = await style_transfer_runner.push_input(data)
        return sanic.response.json({'inference': top_k}, status=200)
    except HandlingError as e:
        # we don't want these to be logged...
        return sanic.response.text(e.handling_msg, status=e.handling_code)


@app.route('/ranking', methods=['POST'])
async def ranking(request):
    try:
        data = request.json
        if data.get('immediately'):
            batched_data = process_input(data)
            input_data = {
                # 'user_id': torch.unsqueeze(batched_data["user_id"], dim=0),
                # 'item_length': torch.unsqueeze(batched_data["item_length"], dim=0),
                'item_seq': torch.unsqueeze(batched_data["item_id_list"], dim=0),
            }
            scores = style_transfer_runner.run_model(input_data)
            batched_data['score'] = scores[0]
            process_output(batched_data)
            _ranking = batched_data['ranking']
        else:
            _ranking = await style_transfer_runner.push_input(data)
        return sanic.response.json({'ranking': _ranking}, status=200)
    except HandlingError as e:
        # we don't want these to be logged...
        return sanic.response.text(e.handling_msg, status=e.handling_code)


style_transfer_runner = ModelRunner(config['model'])
app.add_task(style_transfer_runner.model_runner())

if __name__ == '__main__':
    # mp.set_start_method(None, force=True)
    app.run(host="0.0.0.0", port=int(config['port']), debug=True, workers=int(config['num_workers']))
