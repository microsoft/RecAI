# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.RecModel import RecModel
from utils import utils
from utils.global_p import *

class AttributeMF(RecModel):
    include_item_features = True
    include_user_features = True

    def __init__(self, user_num, item_num, feature_dims, u_vector_size, i_vector_size,
                 *args, **kwargs):
        self.u_vector_size, self.i_vector_size = u_vector_size, i_vector_size
        assert self.u_vector_size == self.i_vector_size
        self.ui_vector_size = self.u_vector_size
        self.user_num = user_num
        self.item_num = item_num

        self.user_feat_num = 13 + 2 + 20 + 1
        # self.item_feat_num = 17 + (feature_dims - (13 + 2 + 20) - 17) // 2 + 1
        self.item_feat_num = (feature_dims - (13 + 2 + 20)) // 2 + 1
        # self.zero_feat = torch.tensor([-17, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37]) + 17
        self.zero_feat = torch.tensor([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37])

        RecModel.__init__(self, user_num, item_num, u_vector_size, i_vector_size, *args, **kwargs)

    def _init_weights(self):
        self.uid_embeddings = torch.nn.Embedding(self.user_num, self.ui_vector_size)
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.ui_vector_size)
        self.user_feat_embeddings = torch.nn.Embedding(self.user_feat_num, self.ui_vector_size)
        self.item_feat_embeddings = torch.nn.Embedding(self.item_feat_num, self.ui_vector_size)
        self.user_concat_layer = nn.Sequential(
            nn.Linear(self.ui_vector_size * 2, self.ui_vector_size),
            nn.ReLU()
        )
        self.item_concat_layer = nn.Sequential(
            nn.Linear(self.ui_vector_size * 2, self.ui_vector_size),
            nn.ReLU()
        )
        self.l2_embeddings = ['uid_embeddings', 'iid_embeddings', 'user_feat_embeddings', 'item_feat_embeddings']

    def predict(self, feed_dict):
        check_list, embedding_l2 = [], []
        u_ids = feed_dict[UID]
        i_ids = feed_dict[IID]
        feat_ids = feed_dict[X] # [batch_size, feat_len]
        user_feat_ids, item_feat_ids = feat_ids[:, :3], feat_ids[:, 3:] - 35

        cf_u_vectors = self.uid_embeddings(u_ids) # [batch_size, embed_dim]
        cf_i_vectors = self.iid_embeddings(i_ids) # [batch_size, embed_dim]

        user_feat_vector = self.user_feat_embeddings(user_feat_ids) # [batch_size, feat_len, embed_dim]
        user_feat_vector = user_feat_vector.mean(dim=1) # [batch_size, embed_dim]

        valid_item_feat = (item_feat_ids != self.zero_feat.cuda()).long() # [batch_size, feat_len]
        item_feat_ids = torch.floor((item_feat_ids + 1) / torch.tensor(2, dtype=torch.long).cuda()).int()
        item_feat_vector = self.item_feat_embeddings(item_feat_ids * valid_item_feat) # [batch_size, feat_len, embed_dim]
        item_feat_vector = item_feat_vector * valid_item_feat.unsqueeze(-1).float() # [batch_size, feat_len, embed_dim]
        item_feat_vector = item_feat_vector.sum(dim=1) / valid_item_feat.sum(dim=1).unsqueeze(-1) # [batch_size, embed_dim]

        embedding_l2.extend([cf_u_vectors, cf_i_vectors, user_feat_vector, item_feat_vector])

        # u_vectors = self.user_concat_layer(torch.cat([cf_u_vectors, user_feat_vector], dim=1)) # [batch_size, embed_dim]
        # i_vectors = self.item_concat_layer(torch.cat([cf_i_vectors, item_feat_vector], dim=1)) # [batch_size, embed_dim]
        u_vectors = cf_u_vectors + user_feat_vector
        i_vectors = cf_i_vectors + item_feat_vector
        prediction = (u_vectors * i_vectors).sum(dim=1).view([-1]) # [batch_size]
        out_dict = {PREDICTION: prediction,
                    CHECK: check_list, EMBEDDING_L2: embedding_l2}
        return out_dict

    def get_ui_vectors(self, feed_dict):
        u_ids = feed_dict[UID]
        i_ids = feed_dict[IID]
        feat_ids = feed_dict[X] # [batch_size, feat_len]
        user_feat_ids, item_feat_ids = feat_ids[:, :3], feat_ids[:, 3:] - 35

        cf_u_vectors = self.uid_embeddings(u_ids) # [batch_size, embed_dim]
        cf_i_vectors = self.iid_embeddings(i_ids) # [batch_size, embed_dim]

        user_feat_vector = self.user_feat_embeddings(user_feat_ids) # [batch_size, feat_len, embed_dim]
        user_feat_vector = user_feat_vector.mean(dim=1) # [batch_size, embed_dim]

        valid_item_feat = (item_feat_ids != self.zero_feat.cuda()).long() # [batch_size, feat_len]
        # item_feat_ids[:, 1:] = torch.floor((item_feat_ids[:, 1:] - 17 + 1) / torch.tensor(2, dtype=torch.long).cuda()).int() + 17
        item_feat_ids = torch.floor((item_feat_ids + 1) / torch.tensor(2, dtype=torch.long).cuda()).int()
        item_feat_vector = self.item_feat_embeddings(item_feat_ids * valid_item_feat) # [batch_size, feat_len, embed_dim]
        item_feat_vector = item_feat_vector * valid_item_feat.unsqueeze(-1).float() # [batch_size, feat_len, embed_dim]
        item_feat_vector = item_feat_vector.sum(dim=1) / valid_item_feat.sum(dim=1).unsqueeze(-1) # [batch_size, embed_dim]

        u_vectors = cf_u_vectors # + user_feat_vector
        i_vectors = cf_i_vectors + item_feat_vector
        return u_vectors, i_vectors