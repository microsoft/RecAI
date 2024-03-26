# coding=utf-8

import os
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from models.RecModel import RecModel
from utils import utils
from utils.global_p import *

class GMF(nn.Module):
    def __init__(self, user_num, item_num, u_vector_size, i_vector_size):
        super().__init__()
        self.mf_user_emb = nn.Embedding(user_num, u_vector_size)
        self.mf_item_emb = nn.Embedding(item_num, i_vector_size)

    def forward(self, user_id, item_id):
        mf_vec = self.mf_user_emb(user_id) * self.mf_item_emb(item_id)
        return mf_vec

class MLP(nn.Module):
    def __init__(self, user_num, item_num, u_vector_size, i_vector_size):
        super().__init__()
        self.mlp_user_emb = nn.Embedding(user_num, u_vector_size)
        self.mlp_item_emb = nn.Embedding(item_num, i_vector_size)
        mlp_layers = [64]
        self.mlp = nn.Sequential(
            nn.Linear(u_vector_size+i_vector_size, mlp_layers[0]),
            nn.ReLU(),
            # nn.Linear(mlp_layers[0], mlp_layers[1]),
            # nn.ReLU()
        )

    def forward(self, user_id, item_id):
        mlp_vec = torch.cat([self.mlp_user_emb(user_id), self.mlp_item_emb(item_id)], dim=-1)
        mlp_vec = self.mlp(mlp_vec)
        return mlp_vec


class NeuMFModel(RecModel):
    include_id = False
    include_user_features = False
    include_item_features = False
    include_context_features = False

    def _init_weights(self):
        mlp_layers = [64]
        # self.gmf = GMF(self.user_num, self.item_num, self.u_vector_size, self.i_vector_size)
        # self.mlp = MLP(self.user_num, self.item_num, self.u_vector_size, self.i_vector_size)
        self.uid_embeddings = nn.Embedding(self.user_num, self.u_vector_size)
        self.iid_embeddings = nn.Embedding(self.item_num, self.i_vector_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.u_vector_size + self.i_vector_size, mlp_layers[0]),
            nn.ReLU(),
            # nn.Linear(mlp_layers[0], mlp_layers[1]),
            # nn.ReLU()
        )
        self.linear = nn.Sequential(
            nn.Linear(self.u_vector_size + mlp_layers[-1], 1),
            nn.Sigmoid()
        )
        self.l2_embeddings = []

    def predict(self, feed_dict):
        check_list, embedding_l2 = [], []
        u_ids = feed_dict[UID]
        i_ids = feed_dict[IID]

        user_embed, item_embed = self.uid_embeddings(u_ids), self.iid_embeddings(i_ids)

        # gmf_vec = self.gmf(u_ids, i_ids) # [batch_size, embed_dim]
        gmf_vec = user_embed * item_embed  # [batch_size, embed_dim]
        # mlp_vec = self.mlp(u_ids, i_ids) # [batch_size, embed_dim]
        mlp_vec = self.mlp(torch.cat([user_embed, item_embed], dim=-1))
        embedding_l2.extend([gmf_vec, mlp_vec])

        prediction = self.linear(torch.cat([gmf_vec, mlp_vec], dim=-1)).view([-1]) # [batch_size]
        out_dict = {PREDICTION: prediction,
                    CHECK: check_list, EMBEDDING_L2: embedding_l2}
        return out_dict
    
    def get_ui_vectors(self):
        return self.uid_embeddings.weight, self.iid_embeddings.weight

    def rank_loss(self, prediction, label, real_batch_size):
        label = torch.cat([label, torch.zeros(len(prediction) - len(label)).to(label.device)])
        if self.loss_sum == 1:
            loss = torch.nn.BCELoss(reduction='sum')(prediction.view(-1, 1), label.view(-1, 1))
        else:
            loss = torch.nn.BCELoss(reduction='mean')(prediction.view(-1, 1), label.view(-1, 1))
        return loss