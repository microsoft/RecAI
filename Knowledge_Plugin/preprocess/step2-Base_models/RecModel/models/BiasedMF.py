# coding=utf-8

import torch
import torch.nn.functional as F
from models.RecModel import RecModel
from utils import utils
from utils.global_p import *

class BiasedMF(RecModel):
    def _init_weights(self):
        self.uid_embeddings = torch.nn.Embedding(self.user_num, self.ui_vector_size)
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.ui_vector_size)
        self.user_bias = torch.nn.Embedding(self.user_num, 1)
        self.item_bias = torch.nn.Embedding(self.item_num, 1)
        self.global_bias = torch.nn.Parameter(torch.tensor(0.1))
        self.l2_embeddings = ['uid_embeddings', 'iid_embeddings', 'user_bias', 'item_bias']

    def predict(self, feed_dict):
        check_list, embedding_l2 = [], []
        u_ids = feed_dict[UID]
        i_ids = feed_dict[IID]

        # bias
        u_bias = self.user_bias(u_ids).view([-1]) # [batch_size]
        i_bias = self.item_bias(i_ids).view([-1]) # [batch_size]
        embedding_l2.extend([u_bias, i_bias])

        cf_u_vectors = self.uid_embeddings(u_ids) # [batch_size, embed_dim]
        cf_i_vectors = self.iid_embeddings(i_ids) # [batch_size, embed_dim]
        embedding_l2.extend([cf_u_vectors, cf_i_vectors])

        prediction = (cf_u_vectors * cf_i_vectors).sum(dim=1).view([-1]) # recommend score
        prediction = prediction + u_bias + i_bias + self.global_bias
        prediction = torch.sigmoid(prediction) # [batch_size]
        # check_list.append(('prediction', prediction))

        out_dict = {PREDICTION: prediction,
                    CHECK: check_list, EMBEDDING_L2: embedding_l2}
        return out_dict