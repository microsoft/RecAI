# coding=utf-8

import os
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from models.RecModel import RecModel
from utils import utils
from utils.global_p import *


class DMFModel(RecModel):
    def __init__(self, path, dataset, *args, **kwargs):
        self.train_file = os.path.join(os.path.join(path, dataset), dataset + TRAIN_SUFFIX)
        RecModel.__init__(self, *args, **kwargs)

    def _init_weights(self):
        train_df = pd.read_csv(self.train_file, sep="\t")
        train_df = train_df[train_df["label"] > 0].reset_index(drop=True)
        user_item_indices = torch.LongTensor([train_df.uid, train_df.iid])
        rating_data = torch.FloatTensor(train_df.label)
        self.user_item_matrix = torch.sparse_coo_tensor(user_item_indices, rating_data, torch.Size((self.user_num, self.item_num))).to_dense().cuda()
        
        self.linear_user_1 = nn.Linear(in_features=self.item_num, out_features=self.ui_vector_size)
        self.linear_user_1.weight.detach().normal_(0, 0.01)
        self.linear_item_1 = nn.Linear(in_features=self.user_num, out_features=self.ui_vector_size)
        self.linear_item_1.weight.detach().normal_(0, 0.01)

        self.layers = [self.ui_vector_size, self.ui_vector_size]
        self.user_fc_layers = nn.ModuleList()
        for idx in range(1, len(self.layers)):
            self.user_fc_layers.append(nn.Linear(in_features=self.layers[idx - 1], out_features=self.layers[idx]))
            self.user_fc_layers[-1].weight.detach().normal_(0, 0.01)

        self.item_fc_layers = nn.ModuleList()
        for idx in range(1, len(self.layers)):
            self.item_fc_layers.append(nn.Linear(in_features=self.layers[idx - 1], out_features=self.layers[idx]))
            self.item_fc_layers[-1].weight.detach().normal_(0, 0.01)

        self.l2_embeddings = []

    def predict(self, feed_dict):
        check_list, embedding_l2 = [], []
        u_ids = feed_dict[UID]
        i_ids = feed_dict[IID]

        # bias
        user = self.user_item_matrix[u_ids]
        item = self.user_item_matrix[:, i_ids].t()
        
        user = self.linear_user_1(user)
        item = self.linear_item_1(item)

        for idx in range(len(self.layers) - 1):
            user = F.relu(user)
            user = self.user_fc_layers[idx](user)

        for idx in range(len(self.layers) - 1):
            item = F.relu(item)
            item = self.item_fc_layers[idx](item)

        prediction = torch.cosine_similarity(user, item).view([-1])
        prediction = torch.clamp(prediction, min=1e-6, max=1)

        # check_list.append(('prediction', prediction))

        out_dict = {PREDICTION: prediction,
                    CHECK: check_list, EMBEDDING_L2: embedding_l2}
        return out_dict

    def get_ui_vectors(self):
        u_ids = torch.arange(self.user_num).cuda()
        i_ids = torch.arange(self.item_num).cuda()

        user = self.user_item_matrix[u_ids]
        item = self.user_item_matrix[:, i_ids].t()
        
        user = self.linear_user_1(user)
        item = self.linear_item_1(item)

        for idx in range(len(self.layers) - 1):
            user = F.relu(user)
            user = self.user_fc_layers[idx](user)

        for idx in range(len(self.layers) - 1):
            item = F.relu(item)
            item = self.item_fc_layers[idx](item)
        
        return user, item

    def rank_loss(self, prediction, label, real_batch_size):
        label = torch.cat([label, torch.zeros(len(prediction) - real_batch_size).to(label.device)])
        if self.loss_sum == 1:
            loss = torch.nn.BCELoss(reduction='sum')(prediction.view(-1, 1), label.view(-1,1))
        else:
            loss = torch.nn.BCELoss(reduction='mean')(prediction.view(-1, 1), label.view(-1,1))
        return loss
