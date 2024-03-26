# coding=utf-8

import torch
import torch.nn.functional as F
import logging
from sklearn.metrics import *
import numpy as np
from models.BaseModel import BaseModel
from utils import utils
from utils.global_p import *


class DeepModel(BaseModel):
    @staticmethod
    def parse_model_args(parser, model_name='DeepModel'):
        parser.add_argument('--f_vector_size', type=int, default=64,
                            help='Size of feature vectors.')
        parser.add_argument('--layers', type=str, default='[64]',
                            help="Size of each layer.")
        return BaseModel.parse_model_args(parser, model_name)

    def __init__(self, feature_dims, f_vector_size, layers, *args, **kwargs):
        self.feature_dims = feature_dims
        self.f_vector_size = f_vector_size
        self.layers = layers if type(layers) == list else eval(layers)
        BaseModel.__init__(self, *args, **kwargs)

    def _init_weights(self):
        self.feature_embeddings = torch.nn.Embedding(self.feature_dims, self.f_vector_size)
        self.l2_embeddings = ['feature_embeddings']
        pre_size = self.f_vector_size * self.feature_num
        for i, layer_size in enumerate(self.layers):
            setattr(self, 'layer_%d' % i, torch.nn.Linear(pre_size, layer_size))
            setattr(self, 'bn_%d' % i, torch.nn.BatchNorm1d(layer_size))
            pre_size = layer_size
        self.prediction = torch.nn.Linear(pre_size, 1)

    def predict(self, feed_dict):
        check_list, embedding_l2 = [], []
        nonzero_embeddings = self.feature_embeddings(feed_dict[X])
        embedding_l2.append(nonzero_embeddings)
        pre_layer = nonzero_embeddings.view([-1, self.feature_num * self.f_vector_size])
        for i in range(0, len(self.layers)):
            pre_layer = getattr(self, 'layer_%d' % i)(pre_layer)
            pre_layer = getattr(self, 'bn_%d' % i)(pre_layer)
            pre_layer = F.relu(pre_layer)
            pre_layer = torch.nn.Dropout(p=feed_dict[DROPOUT])(pre_layer)
        prediction = F.relu(self.prediction(pre_layer)).view([-1])
        out_dict = {PREDICTION: prediction,
                    CHECK: check_list, EMBEDDING_L2: embedding_l2}
        return out_dict
