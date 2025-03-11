# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import setproctitle
import logging
import copy
import argparse
from tqdm import tqdm
import random
import inspect
from accelerate import Accelerator
### import modules defined in this project
from unirec.utils import logger, general
from unirec.constants.protocols import *
from torch.utils.data import Dataset
from collections import defaultdict


class InferDataset(Dataset):
    def __init__(self, config, data_ids, user_history=None, is_seqrec=True):
        self.config = config
        self.data_ids = data_ids
        self.user_history = user_history
        self.empty_history = np.zeros((1,), dtype=np.int32)
        self.is_seqrec = is_seqrec
        self.max_seq_len = config.get('max_seq_len', 0)
        self.set_return_column_index()

    def set_return_column_index(self):
        self.return_key_2_index = {}
        self.return_key_2_index['user_id'] = len(self.return_key_2_index)
        self.return_key_2_index['item_id'] = len(self.return_key_2_index)
        if self.is_seqrec:
            self.return_key_2_index['item_seq'] = len(self.return_key_2_index)
            self.return_key_2_index['item_seq_len'] = len(self.return_key_2_index)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        sample = self.data_ids[idx]
        user_id = sample[0]
        item_id = sample[1]
        return_tup = (user_id, item_id)
        if self.is_seqrec:
            res = np.zeros((self.max_seq_len,), dtype=np.int32)
            hist = self.user_history[user_id]
            if hist is None:
                hist = self.empty_history
            
            k=len(hist)
            for i, item in enumerate(hist):
                if item == item_id:
                    k = i
                    break
            hist = hist[:k]

            n = len(hist)
            if n > self.max_seq_len:
                res[:] = hist[n-self.max_seq_len:]
            else:
                res[self.max_seq_len-n:] = hist[:]
            res_len = min(n, self.max_seq_len)
            return_tup = return_tup + (res, res_len)
        return return_tup

@torch.no_grad()
def _get_tok_recommendations(test_data_loader, model, user_history, topk, test_config, accelerator):
    model.eval()
    model = accelerator.unwrap_model(model)
    # n_user = len(test_data_loader.data)
    # res_reco = np.zeros((n_user, topk), dtype=np.int32) 
    item_embeddings = model.forward_all_item_emb()
    if model.has_item_bias:
        item_bias = model.get_all_item_bias()
        item_bias = item_bias.reshape((1, -1))
    
    iter_data = (
        tqdm(
            enumerate(test_data_loader),
            total=len(test_data_loader),
            desc="TopK inference",
            dynamic_ncols=True,
            disable=not accelerator.is_local_main_process
        )
    )

    f = open(test_config['output_path'], 'w')
    for batch_idx, inter_data in iter_data:
        samples = {k:inter_data[v] for k,v in test_data_loader.dataset.return_key_2_index.items()}
        inputs = {k: v for k, v in samples.items() if k in inspect.signature(model.forward_user_emb).parameters}
        user_embeddings = model.forward_user_emb(**inputs).detach().cpu().numpy() if model.__optimized_by_SGD__ else model.forward_user_emb(**inputs)
        # samples['user_id'] = samples['user_id'].cpu().numpy() if isinstance(samples['user_id'], torch.Tensor) else samples['user_id']
        
        batch_scores = np.matmul(user_embeddings, item_embeddings.T) if model.__optimized_by_SGD__ else np.array(model.sparse_matrix_mul(user_embeddings, item_embeddings))

        if model.has_item_bias:
            batch_scores += item_bias
        if model.has_user_bias:
            user_bias = model.get_user_bias(samples)
            batch_scores += user_bias.reshape(-1, 1)
        batch_scores = batch_scores / test_config['tau']
        
        user_ids = accelerator.gather_for_metrics(samples['user_id']).cpu().numpy()
        item_ids = accelerator.gather_for_metrics(samples['item_id']).cpu().numpy()
        past_hists = []
        for idx in range(len(user_ids)):
            userid = user_ids[idx]
            itemid = item_ids[idx]
            # if userid < len(user_history) and user_history[userid] is not None:
            history = user_history[userid]
            index = np.where(history==itemid)[0][0]
            past_hist = history[:index]
            past_hists.append(set(past_hist))
            # target_score = batch_scores[idx][itemid]
            batch_scores[idx][past_hist] = -np.Inf
            batch_scores[idx][0] = -np.Inf
            # batch_scores[idx][itemid] = target_score

        sorted_index = np.argsort(-batch_scores, axis=1)
        sorted_index = accelerator.gather_for_metrics(torch.tensor(sorted_index, device=accelerator.device)).cpu().numpy()
        for idx, userid in enumerate(user_ids):
            itemid = item_ids[idx]
            reco_items = sorted_index[idx]
            past_hist = past_hists[idx]
            filter_items = []
            for item in reco_items:
                if item in past_hist or item==0:
                    continue
                filter_items.append(item)
            tops = [str(k) for k in filter_items[:test_config['times']]]
            sample_orders = []
            for i in range(test_config['times']):
                indices = sorted(random.sample(range(len(filter_items)), 5))  
                sample_orders.append([str(filter_items[i]) for i in indices]) 

            scores = batch_scores[idx]
            pos_thred=scores[reco_items[test_config['n_items']//5]] # top 20%
            neg_thred=scores[reco_items[-test_config['n_items']//2]] # bottom 50%
            
            pos_indices = np.where(scores>pos_thred)[0] # 1.0
            pos_indice = np.random.choice(pos_indices, test_config['times'], replace=False)

            neg_indices = np.where((scores<neg_thred) & (scores!=-np.Inf))[0] #0.0
            neg_indice = np.random.choice(neg_indices, test_config['times'], replace=False)

            for i in range(test_config['times']):
                f.write(str(userid) + '\t' + str(itemid) + '\t' + tops[i] + '\t' + ','.join(sample_orders[i]) + '\t' + str(pos_indice[i]) + '\t' + str(neg_indice[i]) + '\t' + str(scores[pos_indice[i]]) + '\t' + str(scores[neg_indice[i]]) +'\n')

    f.close()

    ## item similarity
    if 'sim_item_file' in test_config:
        item2sim = defaultdict(list)
        for i in range(1, item_embeddings.shape[0]-1):
            for j in range(i+1, item_embeddings.shape[0]):
                sim = np.dot(item_embeddings[i], item_embeddings[j])
                item2sim[i].append((j, sim))
                item2sim[j].append((i, sim))
        for i in range(1, item_embeddings.shape[0]):
            item2sim[i] = sorted(item2sim[i], key=lambda x:x[1], reverse=True)[0][0]
        with open(test_config['sim_item_file'], 'w') as f:
            for i in range(1, item_embeddings.shape[0]):
                f.write(str(i) + '\t' + str(item2sim[i]) + '\n')
    

                
        
   
def do_topk_reco(config, accelerator):
    logger = logging.getLogger(config['exp_name']) 
    logger.info(str(config))
    
    model_path = config['model_file']
    test_data_path = config['dataset_path']
    test_data_name = config['dataset_name'] ## should be a user id file

    user_history_2_mask = config['user_history_filename']
    user_history_format = config.get('user_history_file_format', None)
    outpath = config['output_path']
    
    logger.info('Loading model from {0}'.format(model_path))
    model, cpk_config = general.load_model_freely(model_path, config['device'])
    cpk_config.update(config)
    config = cpk_config 
    test_config = copy.deepcopy(config)

    if 'test_batch_size' in test_config:
        test_config['batch_size'] = test_config['test_batch_size']
        
    test_data = pd.read_csv(os.path.join(test_data_path, test_data_name), sep='\t', header=0)
    test_data = test_data.values.astype(object)
    logger.info('#. users for recommendations: {0}'.format(len(test_data)))
    
    logger.info('loading user history...')
    user_history, user_history_time = general.load_user_history(test_data_path, user_history_2_mask, format=user_history_format)   
    logger.info('finished loading user history.') 
  
    logger.info('loading dataset...')
    is_seqrec = False if 'SeqRecBase' not in model.annotations else True
    test_data = InferDataset(test_config, test_data, user_history, is_seqrec)
    
    test_data_loader = DataLoader(
        dataset=test_data, 
        batch_size=test_config['batch_size'], 
        shuffle=False,  
        num_workers=0, 
    )

    model, test_data_loader = accelerator.prepare(model, test_data_loader) 
    _get_tok_recommendations(test_data_loader, model, user_history, test_config['topk'], test_config, accelerator)
  
 
def parse_cmd_arguments():
    parser = argparse.ArgumentParser()
      
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--model_file", type=str)   ## rename from model_path to model_file
    parser.add_argument("--test_batch_size", type=int)  
    parser.add_argument("--user_history_filename", type=str)  ## rename from user_history_2_mask to user_history_filename
    parser.add_argument("--user_history_file_format", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--sim_item_file", type=str)
    parser.add_argument("--topk", type=int, default=100, help='topk for recommendation')
    parser.add_argument("--seed", type=int, default=2023, help='random seed')
    parser.add_argument("--times", type=int, default=1, help='number of data points for one user')
        
    (args, unknown) = parser.parse_known_args()  
    # print(args)
    parsed_results = {}
    for arg in sorted(vars(args)):
        value = getattr(args, arg)
        if value is not None and value not in ['none', 'None']:
            parsed_results[arg] = value
    
    return parsed_results

def init_seed(seed):
    r""" init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True # could improve efficiency
    torch.backends.cudnn.deterministic = True # fix random seed


if __name__ == '__main__':
    config = parse_cmd_arguments()
    init_seed(config['seed'])
    exp_name = 'reco_topk'
    setproctitle.setproctitle(exp_name)  
    config['exp_name'] = exp_name
    logger_dir = 'output' if 'output_path' not in config else os.path.dirname(config['output_path'])
    logger_time_str = general.get_local_time_str().replace(':', '')
    logger_rand = random.randint(0, 100)

    accelerator = Accelerator()
    config['device'] = accelerator.device

    config['logger_time_str']=logger_time_str
    config['logger_rand']=logger_rand
    mylog = logger.Logger(logger_dir, exp_name, time_str=logger_time_str, rand=logger_rand, is_main_process=accelerator.is_local_main_process) 
    do_topk_reco(config, accelerator)
