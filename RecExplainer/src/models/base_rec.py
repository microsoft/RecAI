"""
The following code is modified from
https://github.com/microsoft/UniRec/blob/main/unirec/model/base/reco_abc.py
"""

from typing import *
import numpy as np
import pandas as pd
import inspect
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_
from enum import Enum
import random
from accelerate.logging import get_logger

from models import modules

class LossFuncType(Enum):
    BCE = 'bce'
    BPR = 'bpr'
    SOFTMAX = 'softmax'
    CCL = 'ccl'
    FULLSOFTMAX = 'fullsoftmax'

class DistanceType(Enum):
    DOT = 'dot'
    COSINE = 'cosine'
    MLP = 'mlp'

EPS = 1e-8
VALID_TRIGGER_P = 0.1 ## The probability to trigger compliance validation.

def _transfer_emb(x):
    x = x.split(',')
    new_x = [float(x_) for x_ in x]
    return new_x

def load_pre_item_emb(file_path, logger):
    logger.info('loading pretrained item embeddings...')

    item_emb_data = pd.read_csv(file_path, names=['id', 'emb'], sep='\t')

    item_emb_data['emb'] = item_emb_data['emb'].apply(lambda x: _transfer_emb(x))
    item_emb_ = item_emb_data['emb'].values

    if 0 in item_emb_data['id'].values:
        item_emb_ = item_emb_[1:]

    item_emb = []
    for ie in item_emb_:
        item_emb.append(ie)
    item_emb = np.array(item_emb)

    return item_emb

def xavier_normal_initialization(module):
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
        if module.padding_idx is not None:
            constant_(module.weight.data[module.padding_idx], 0.)
    elif isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

def normal_initialization(mean, std):
    def normal_init(module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=mean, std=std)
            if module.padding_idx is not None:
                constant_(module.weight.data[module.padding_idx], 0.)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=mean, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    return normal_init

def xavier_uniform_initialization(module):
    if isinstance(module, nn.Embedding):
        xavier_uniform_(module.weight.data)
        if module.padding_idx is not None:
            constant_(module.weight.data[module.padding_idx], 0.)
    elif isinstance(module, nn.Linear):
        xavier_uniform_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

class AbstractRecommender(nn.Module):    
    def __init__(self, config):
        super(AbstractRecommender, self).__init__()
        self.logger = get_logger(__name__)  
        self.__optimized_by_SGD__ = True
        self.config = config        
        self._init_attributes()
        self._init_modules() 
        
        self.annotations = []
        self.add_annotation()

        self._parameter_validity_check() 

    ## -------------------------------
    ## Basic functions you need to pay attention to.
    ## In most cases, you will need to override them.
    ## -------------------------------  
    def _parameter_validity_check(self):
        # if self.loss_type in [LossFuncType.BPR.value, LossFuncType.CCL.value]:            
        #     if self.config['train_file_format'] not in [DataFileFormat.T1.value, DataFileFormat.T2.value, DataFileFormat.T5.value, DataFileFormat.T6.value]:
        #         raise ValueError(r'''
        #                 for efficiency concern, we put the limitation in implementation:
        #                 if you want to use BPR or CCL as the loss function, please make sure only the first item in one group is positive item.  
        #                 and the data format is T1 or T4 
        #         ''')
        
        pass
        
    def _define_model_layers(self):
        raise NotImplementedError

    def forward(self, user_id):
        raise NotImplementedError

    def forward_user_emb(self, interaction):
        raise NotImplementedError

    def forward_item_emb(self, interaction):
        raise NotImplementedError

    ## -------------------------------
    ## More functions you may need to override.
    ## -------------------------------
    def _predict_layer(self, user_emb, items_emb, interaction):  
        raise NotImplementedError  
    
    def predict(self, interaction):
        raise NotImplementedError  
    
    def add_annotation(self):
        self.annotations.append('AbstractRecommender')


    ## -------------------------------
    ## Belowing functions can fit most scenarios so you don't need to override.
    ## ------------------------------- 
    
    def _init_attributes(self):
        config = self.config
        self.n_users = config['n_users']
        self.n_items = config['n_items']
        # self.device = config['device']
        self.loss_type = config.get('loss_type', 'bce')
        self.embedding_size = config.get('embedding_size', 0)
        self.hidden_size = self.embedding_size
        self.dropout_prob = config.get('dropout_prob', 0.0)
        self.use_pre_item_emb = config.get('use_pre_item_emb', 0)
        self.use_text_emb = config.get('use_text_emb', 0)
        self.text_emb_size = config.get('text_emb_size', 768)
        self.init_method = config.get('init_method', 'normal')
        self.use_features = config.get('use_features', 0)
        # if self.use_features:
        #     self.feature_emb_size = self.embedding_size #config.get('feature_emb_size', 40)
        #     self.features_shape = eval(config.get('features_shape', '[]'))
        #     self.item2features = file_io.load_features(self.config['features_filepath'], self.n_items, len(self.features_shape))
        if 'group_size' in config:
            self.group_size = config['group_size']
        else:
            self.group_size = -1
        ## clip the score to avoid loss being nan
        ## usually this is not necessary, so you don't need to set up score_clip_value in config
        self.SCORE_CLIP = -1
        if 'score_clip_value' in self.config:
            self.SCORE_CLIP = self.config['score_clip_value']
        self.has_user_bias = False
        self.has_item_bias = False
        if 'has_user_bias' in config:
            self.has_user_bias = config['has_user_bias']
        if 'has_item_bias' in config:
            self.has_item_bias = config['has_item_bias']
        self.tau = config.get('tau', 1.0)

    def _init_modules(self):
        # define layers and loss
        # TODO: remove user_embedding when user_id is not needed to save memory. Like in VAE.
        if self.has_user_bias:
            self.user_bias = nn.Parameter(torch.normal(0, 0.1, size=(self.n_users,)))
        if self.has_item_bias:
            self.item_bias = nn.Parameter(torch.normal(0, 0.1, size=(self.n_items,)))

        if self.config['has_user_emb']:
            self.user_embedding = nn.Embedding(self.n_users, self.embedding_size, padding_idx=0)
        
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0) #if padding_idx is not set, the embedding vector of padding_idx will change during training
        if self.use_text_emb:
            # We load the pretrained text embedding, and freeze it during training.
            # But text_mlp is trainable to map the text embedding to the same space as item embedding.
            # Architecture of text_mlp is fixed as a simple 2-layer MLP.
            self.text_embedding = nn.Embedding(self.n_items, self.text_emb_size, padding_idx=0)
            self.text_embedding.weight.requires_grad_(False)
            self.text_mlp = nn.Sequential(
                    nn.Linear(self.text_emb_size, 2*self.hidden_size),
                    nn.GELU(),
                    nn.Linear(2*self.hidden_size, self.hidden_size),
                )
        if self.use_features:
            # we merge all features into one embedding layer, for example, if we have 2 features, and each feature has 10 categories, 
            # the feature1 index is from 0 to 9, and feature2 index is from 10 to 19.
            self.features_embedding = nn.Embedding(sum(self.features_shape), self.feature_emb_size, padding_idx=0)
        
        # model layers
        self._define_model_layers()
        self._init_params()

        if self.use_pre_item_emb and 'item_emb_path' in self.config: #check for item_emb_path is to ensure that in infer/test task (or other cases that need to load model ckpt), we don't load pre_item_emb which will be overwritten by model ckpt.
            pre_item_emb = load_pre_item_emb(self.config['item_emb_path'], self.logger) 
            pre_item_emb = torch.from_numpy(pre_item_emb)
            pad_emb = torch.zeros([1, self.embedding_size])
            pre_item_emb = torch.cat([pad_emb, pre_item_emb], dim=-2).to(torch.float32)
            self.logger.info('{0}={1}'.format('self.n_items', self.n_items))
            self.logger.info('{0}={1}'.format('pre_item_emb', pre_item_emb))
            self.logger.info('{0}={1}'.format('pre_item_emb.size', pre_item_emb.size())) 
            self.item_embedding.weight = nn.Parameter(pre_item_emb)
        if self.use_text_emb and 'text_emb_path' in self.config:
            text_emb = load_pre_item_emb(self.config['text_emb_path'], self.logger)
            text_emb = torch.from_numpy(text_emb)
            pad_emb = torch.zeros([1, self.text_emb_size])
            text_emb = torch.cat([pad_emb, text_emb], dim=-2).to(torch.float32)
            self.logger.info('{0}={1}'.format('self.n_items', self.n_items))
            self.logger.info('{0}={1}'.format('text_emb', text_emb))
            self.logger.info('{0}={1}'.format('text_emb.size', text_emb.size())) 
            self.text_embedding = nn.Embedding.from_pretrained(text_emb, freeze=True, padding_idx=0)  

    def _init_params(self):
        init_methods = {
            'xavier_normal': xavier_normal_initialization,
            'xavier_uniform': xavier_uniform_initialization,
            'normal': normal_initialization(self.config['init_mean'], self.config['init_std']),
        }
        for name, module in self.named_children():
            init_method = init_methods[self.init_method]
            module.apply(init_method)
    
    def _cal_loss(self, scores, labels=None, reduction=True):
        r""" Calculate loss with scores and labels.
        
        Args:
            scores (torch.Tensor): scores of positive or negative items in batch.
            labels (torch.Tensor): labels of items. labels are not required in BPR and CCL loss since 
                they are pairwise loss, in which the first item is positive and the residual is negative.
            reduction (bool): whether to reduce on the batch number dimension (usually the first dim). 
                If True, a loss value would be returned. Otherwise, a tensor with shape (B,) would be returned. 

        Return:
            loss ():
        """
        if self.group_size > 0:
            scores = scores.view(-1, self.group_size)
            if labels is not None:
                labels = labels.view(-1, self.group_size)

        ## trigger compliance validation
        if self.loss_type in [LossFuncType.BPR.value, LossFuncType.CCL.value]: 
            if labels is not None and VALID_TRIGGER_P > random.random():
                has_pos = torch.gt(torch.sum(labels[:, 1:], dtype=torch.float32), 0.5) 
                if has_pos.item():
                    raise ValueError(r'''
                            For efficiency concern, we put the limitation in implementation:
                            if you want to use BPR or CCL as the loss function, please make sure only the first item in one group is positive item.  
                        ''')
        
        if self.loss_type == LossFuncType.BCE.value: 
            logits = torch.clamp(nn.Sigmoid()(scores), min=-1*EPS, max=1-EPS)
            labels = labels.float()
            loss = nn.BCELoss(reduction='mean' if reduction else 'none')(logits, labels).mean(dim=-1)
        elif self.loss_type == LossFuncType.BPR.value: 
            neg_score = scores[:, 1:]  ##-- currently only support one positive in item list.  
            pos_score = scores[:, 0].unsqueeze(1).expand_as(neg_score)
            loss = modules.bpr_loss(pos_score, neg_score, reduction) 
        elif self.loss_type == LossFuncType.CCL.value:
            neg_score = scores[:, 1:]  ##-- currently only support one positive in item list.  
            pos_score = scores[:, 0]
            loss = modules.ccl_loss(pos_score, neg_score, self.config['ccl_w'], self.config['ccl_m'], reduction)
        elif self.loss_type == LossFuncType.SOFTMAX.value:
            scores = -nn.functional.log_softmax(scores, dim=-1) #-torch.log(nn.Softmax(dim=-1)(scores) + EPS)
            # loss = scores[:, 0].mean()
            loss = scores[labels>0] # The dimension of scores: [batch_size, group_size]
            if reduction:
                loss = loss.mean()
        elif self.loss_type == LossFuncType.FULLSOFTMAX.value:
            pos_scores = torch.gather(scores, 1, labels.reshape(-1,1)).squeeze(-1)
            loss = torch.logsumexp(scores, dim=-1) - pos_scores
            if reduction:
                loss = loss.mean()

        return loss   

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        
        messages = []
        messages.append(super().__str__())
        messages.append('Trainable parameter number: {0}'.format(params))
        
        messages.append('All trainable parameters:')
        for name, param in self.named_parameters():
            if param.requires_grad:
                messages.append('{0} : {1}'.format(name, param.size()))
        
        return '\n'.join(messages)

class BaseRecommender(AbstractRecommender): 
        
    def _init_attributes(self):
        super(BaseRecommender, self)._init_attributes()
        config = self.config
        self.dnn_inner_size = self.embedding_size
        self.time_seq = config.get('time_seq', 0)
    
    def _init_modules(self):
        # predict_layer
        scorer_type = self.config['distance_type']
        if scorer_type == 'mlp':
            self.scorer_layers = modules.MLPScorer(self.embedding_size, self.dnn_inner_size, self.dropout_prob, act_f='tanh')
        elif scorer_type == 'dot':
            self.scorer_layers = modules.InnerProductScorer()
        elif scorer_type == 'cosine':
            self.scorer_layers = modules.CosineScorer(eps=1e-6)
        else:
            raise ValueError('not supported distance_type: {0}'.format(scorer_type))
        
        if self.time_seq:
            self.time_embedding = nn.Embedding(self.time_seq, self.embedding_size, padding_idx=0)

        super(BaseRecommender, self)._init_modules()

    def _define_model_layers(self):
        pass

    def forward_user_emb(self, user_id=None, item_seq=None, item_seq_len=None, item_seq_features=None, time_seq=None):
        user_e = self.user_embedding(user_id)
        return user_e
    
    def forward(self, user_ids=None, item_seq=None, item_ids=None):
        # if min(user_pos.shape) > 0:
        user_emb = self.forward_user_emb(user_id=user_ids, item_seq=item_seq)
        # else:
        #     user_emb = None
        # if min(item_pos.shape) > 0:
        item_emb = self.forward_item_emb(item_ids)
        # else:
        #     item_emb = None
        return user_emb, item_emb
    
    def forward_item_emb(self, items, item_features=None):
        item_emb = self.item_embedding(items) # [batch_size, n_items_inline, embedding_size]
        if self.use_features:
            item_features_emb = self.features_embedding(item_features).sum(-2)
            item_emb = item_emb + item_features_emb
        if self.use_text_emb:
            text_emb = self.text_mlp(self.text_embedding(items))
            item_emb = item_emb + text_emb
        return item_emb

    def _predict_layer(self, user_emb, items_emb, user_id, item_id):  
        scores = self.scorer_layers(user_emb, items_emb)
        
        if self.has_user_bias: 
            user_bias = self.user_bias[user_id] 
            if scores.shape != user_bias.shape:
                user_bias = self.user_bias[user_id].unsqueeze(1) 
                user_bias = torch.repeat_interleave(
                    user_bias, scores.shape[-1], dim=-1
                ) 
            scores = scores + user_bias

        if self.has_item_bias:
            item_bias = self.item_bias[item_id]
            scores = scores + item_bias

        scores = scores / self.tau

        if self.SCORE_CLIP > 0:
            scores = torch.clamp(scores, min=-1.0*self.SCORE_CLIP, max=self.SCORE_CLIP) 
        return scores  


    def predict(self, interaction):
        items_emb = self.forward_item_emb(interaction['item_id'], interaction['item_features'] if self.use_features else None)
        inputs = {k: v for k, v in interaction.items() if k in inspect.signature(self.forward_user_emb).parameters}
        user_emb = self.forward_user_emb(**inputs)
        user_id = interaction['user_id'] if 'user_id' in interaction else None
        item_id = interaction['item_id'] if 'item_id' in interaction else None
        scores = self._predict_layer(user_emb, items_emb, user_id, item_id).detach().cpu().numpy()
        return scores

    def forward_all_item_emb(self, batch_size=None, numpy=True):
        ### get all item's embeddings. when batch_size=None, it will proceed all in one run. 
        ### when numpy=False, it would return a torch.Tensor
        if numpy:
            res = np.zeros((self.n_items, self.embedding_size), dtype=np.float32)
        else:
            res = torch.zeros((self.n_items, self.embedding_size), dtype=torch.float32, device=self.item_embedding.weight.device)
        if batch_size is None:
            batch_size = self.n_items
        
        n_batch = (self.n_items - 1) // batch_size + 1
        for batch in range(n_batch):
            start = batch * batch_size
            end = min(self.n_items, start + batch_size)
            cur_items = torch.arange(start, end, dtype=torch.int32, device=self.item_embedding.weight.device)
            item_features = torch.tensor(self.item2features[start:end], dtype=torch.int32).to(self.item_embedding.weight.device) if self.use_features else None
            cur_items_emb = self.forward_item_emb(cur_items, item_features).detach()
            if numpy:
                cur_items_emb = cur_items_emb.cpu().numpy()
            res[start:end] = cur_items_emb
        return res    

    def get_all_item_bias(self):
        return self.item_bias.detach().cpu().numpy()        
    
    def get_user_bias(self, interaction):
        return self.user_bias[interaction['user_id']].detach().cpu().numpy()
    
    def item_embedding_for_user(self, item_seq, item_seq_features=None, time_seq=None):
        item_emb = self.item_embedding(item_seq)
        if self.use_features:
            item_features_emb = self.features_embedding(item_seq_features).sum(-2)
            item_emb = item_emb + item_features_emb
        if self.time_seq:
            time_embedding = self.time_embedding(time_seq)
            item_emb = item_emb + time_embedding
        if self.use_text_emb:
            text_emb = self.text_mlp(self.text_embedding(item_seq))
            item_emb = item_emb + text_emb
        return item_emb

    def topk(self, interaction:Dict[str, torch.Tensor], k: int, user_hist: torch.Tensor=None, candidates: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Return topk items for a batch of users
        
        Compared with two-stage logic: score and get topk, the method is more simple to use. And the topk operation is done on GPU, which is
        faster then using numpy. A weakness is that user history should be padded to the same length, but it's not required to put on GPU. Now
        the method is used in MoRec Data Sampler, which is used to gather topk items for alignment objective.

        Args:
            interaction (Dict[str, torch.Tensor]): information of the user, including user id, item seq and other required information in `forward_user_emb`.
            k (int): top-k number.
            user_hist (torch.Tensor): padded batchified user history.
            candidates (torch.Tensor): item candidates id. When it's None, regard all items as candidates. The shape of candidates should be [batch_size, #candidates].

        Returns:
            (torch.Tensor, torch.Tensor): (top-k scores, top-k item ids).

        Note: user_hist should be padded to the max length of users' histories in the batch but not the max length of histories of all users, which
        could save memory and improve the efficiency.
        """
        inputs = {k: v for k, v in interaction.items() if k in inspect.signature(self.forward_user_emb).parameters}
        user_emb = self.forward_user_emb(**inputs)
        all_item_emb = self.forward_all_item_emb(numpy=False)
        user_id = interaction['user_id'] if 'user_id' in interaction else None
        if candidates is None:
            candidates = torch.arange(len(all_item_emb), dtype=torch.long, device=all_item_emb.device)
            __all_item = True
        else:
            __all_item = False

        all_scores = self._predict_layer(user_emb, all_item_emb, user_id, candidates)

        if user_hist is not None:
            # Mask items in user history
            if __all_item:
                row_idx = torch.arange(user_emb.size(0), dtype=torch.long).unsqueeze_(-1).expand_as(user_hist)
                all_scores[row_idx, user_hist] = - torch.inf
            else:
                sorted_hist, indices = torch.sort(user_hist, dim=-1)
                _idx = torch.searchsorted(sorted_hist, candidates, side='left')
                _idx[_idx == sorted_hist.size(-1)] = sorted_hist.size(-1) - 1
                _eq = torch.gather(sorted_hist, -1, _idx) == candidates
                all_scores[_eq] = - torch.inf

        topk_scores, _ids = torch.topk(all_scores, k, dim=-1)
        if __all_item:
            topk_ids = _ids
        else:
            topk_ids = torch.gather(candidates, -1, _ids)
        return topk_scores, topk_ids

class SeqRecBase(BaseRecommender):    
    def add_annotation(self):
        super(SeqRecBase, self).add_annotation()
        self.annotations.append('SeqRecBase')