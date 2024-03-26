# coding=utf-8

import os
import torch
import pickle
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import *
import torch.nn.functional as F
from collections import defaultdict
from utils import utils
from utils.global_p import *

class BaseModel(torch.nn.Module):
    """
    Base model, functions that children mdoels need to reimplement incluldes:
    parse_model_args,
    __init__,
    _init_weights,
    predict,
    forward,

    The format_data_dict() function of DataProcessor will use the following four parameters
    Usually all features are converted into multi-hot vectors
    e.g.: uid(0-2),iid(0-2),u_age(0-2),i_xx(0-1),
    So uid=0,iid=1,u_age=1,i_xx=0 will be converted to the sparse representation of 100 010 010 10, i.e., 0,4,7,9
    If include_id=False, then multi-hot will not include uid,iid, i.e., u_age=1,i_xx=0 will be converted to the sparse representation of 010 10, which are 1,3
    Same for include_user_features and include_item_features
    append id means if or not to append uid,iid before the input 'X', e.g., when append_id=True, include_id=False:
    uid=0,iid=1,u_age=1,i_xx=0 will be converted to 0,1,1,3
    """

    include_id = True
    include_user_features = True
    include_item_features = True
    include_context_features = True
    data_loader = 'DataLoader'  # Default data_loader
    data_processor = 'DataProcessor'  # Default data_processor
    runner = 'BaseRunner'  # Default runner

    @staticmethod
    def evaluate_method(p, data, metrics, error_skip=False):
        """
        Compute the evaluation measures of the model
        :param p: predicted value, np.array，usually produced by runner.predict
        :param data: data dict，usually produced by DataProcessor
        :param metrics: list of evaluate measures, usually being the runner.metrics, e.g., ['rmse', 'auc']
        :return:
        """
        # data['score'] = p
        # with open("evaluation.pkl", "wb") as fw:
        #     pickle.dump(data, fw)
        l = data[Y]
        evaluations = []
        sample_evaluation = []
        rank = False
        for metric in metrics:
            if '@' in metric:
                rank = True
                break
        
        split_l, split_p, split_l_sum = None, None, None
        if rank:
            uids, times = data[UID].reshape([-1]), data[TIME].reshape([-1])
            if TIME in data:
                sorted_idx = np.lexsort((-l, -p, times, uids))
                sorted_uid, sorted_time = uids[sorted_idx], times[sorted_idx]
                sorted_key, sorted_spl = np.unique([sorted_uid, sorted_time], axis=1, return_index=True)
            else:
                sorted_idx = np.lexsort((-l, -p, uids))
                sorted_uid = uids[sorted_idx]
                sorted_key, sorted_spl = np.unique(sorted_uid, return_index=True)
            sorted_l, sorted_p = l[sorted_idx], p[sorted_idx]
            split_l, split_p = np.split(sorted_l, sorted_spl[1:]), np.split(sorted_p, sorted_spl[1:])
            split_l_sum = [np.sum((d > 0).astype(float)) for d in split_l]
        
        for metric in metrics:
            try:
                if metric == 'rmse':
                    evaluations.append(np.sqrt(mean_squared_error(l, p)))
                elif metric == 'mae':
                    evaluations.append(mean_absolute_error(l, p))
                elif metric == 'auc':
                    evaluations.append(roc_auc_score(l, p))
                elif metric == 'f1':
                    evaluations.append(f1_score(l, np.around(p)))
                elif metric == 'accuracy':
                    evaluations.append(accuracy_score(l, np.around(p)))
                elif metric == 'precision':
                    evaluations.append(precision_score(l, np.around(p)))
                elif metric == 'recall':
                    evaluations.append(recall_score(l, np.around(p)))
                else:
                    k = int(metric.split('@')[-1])
                    if metric.startswith('ndcg@'):
                        max_k = max([len(d) for d in split_l])
                        k_data = np.array([(list(d) + [0] * max_k)[:max_k] for d in split_l])
                        best_rank = -np.sort(-k_data, axis=1)
                        best_dcg = np.sum(best_rank[:, :k] / np.log2(np.arange(2, k + 2)), axis=1)
                        best_dcg[best_dcg == 0] = 1
                        dcg = np.sum(k_data[:, :k] / np.log2(np.arange(2, k + 2)), axis=1)
                        ndcgs = dcg / best_dcg
                        sample_evaluation.append(ndcgs)
                        evaluations.append(np.average(ndcgs))
                    elif metric.startswith('hit@'):
                        k_data = np.array([(list(d) + [0] * k)[:k] for d in split_l])
                        hits = (np.sum((k_data > 0).astype(float), axis=1) > 0).astype(float)
                        sample_evaluation.append(hits)
                        evaluations.append(np.average(hits))
                    elif metric.startswith('precision@'):
                        k_data = [d[:k] for d in split_l]
                        k_data_dict = defaultdict(list)
                        for d in k_data:
                            k_data_dict[len(d)].append(d)
                        precisions = [np.average((np.array(d) > 0).astype(float), axis=1) for d in k_data_dict.values()]
                        sample_evaluation.append(precisions)
                        evaluations.append(np.average(np.concatenate(precisions)))
                    elif metric.startswith('recall@'):
                        k_data = np.array([(list(d) + [0] * k)[:k] for d in split_l])
                        recalls = np.sum((k_data > 0).astype(float), axis=1) / split_l_sum
                        sample_evaluation.append(recalls)
                        evaluations.append(np.average(recalls))
            except Exception as e:
                if error_skip:
                    evaluations.append(-1)
                else:
                    raise e
        return evaluations, sample_evaluation

    def __init__(self, label_min, label_max, feature_num, loss_sum, l2_bias, random_seed, model_path):
        super(BaseModel, self).__init__()
        self.label_min = label_min
        self.label_max = label_max
        self.feature_num = feature_num
        self.loss_sum = loss_sum
        self.l2_bias = l2_bias
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        self.model_path = model_path

        self._init_weights()
        logging.debug(list(self.parameters()))

        self.total_parameters = self.count_variables()
        logging.info('# of params: %d' % self.total_parameters)

        self.optimizer = None
    
    def _init_weights(self):
        """
        Weighted needed by initializatioin (with weight layer)
        :return:
        """
        self.x_bn = torch.nn.BatchNorm1d(self.feature_num)
        self.prediction = torch.nn.Linear(self.feature_num, 1)
        self.l2_embeddings = []

    def count_variables(self):
        """
        Total number of parameters of the model
        :return:
        """
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_parameters

    def init_paras(self, m):
        """
        Model self-defined initialization function, will be called in main.py
        :param m: parameters or layers with parameters
        :return:
        """
        if 'Linear' in str(type(m)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif 'Embedding' in str(type(m)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def l2(self, out_dict):
        """
        Compute the l2 term of the model, by default it's the square sum of all parameters (except for embedding)
        The l2 norm of embedding only consider those embeddings used in the current batch
        :return:
        """
        l2 = utils.numpy_to_torch(np.array(0.0, dtype=np.float32), gpu=True)
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if self.l2_bias == 0 and 'bias' in name:
                continue
            if name.split('.')[0] in self.l2_embeddings:
                continue
            l2 += (p ** 2).sum()
        b_l2 = utils.numpy_to_torch(np.array(0.0, dtype=np.float32), gpu=True)
        for p in out_dict[EMBEDDING_L2]:
            b_l2 += (p ** 2).sum()
        if self.loss_sum == 0:
            l2_batch = out_dict[TOTAL_BATCH_SIZE] if L2_BATCH not in out_dict else out_dict[L2_BATCH]
            b_l2 = b_l2 / l2_batch
        return l2 + b_l2
    
    def predict(self, feed_dict):
        """
        Only makes prediction, does not compute the loss
        :param feed_dict: model input, it's a dict
        :return: output, it's a dict, prediction is the predicted value, check means needs to check the intermediate results
        """
        check_list = []
        x = self.x_bn(feed_dict[X].float())
        x = torch.nn.Dropout(p=feed_dict[DROPOUT])(x)
        prediction = F.relu(self.prediction(x)).view([-1])
        out_dict = {PREDICTION: prediction,
                    CHECK: check_list}
        return out_dict

    def forward(self, feed_dict):
        """
        Except for making prediction, also computes the loss
        :param feed_dict: model input, it's a dict
        :return: output, it's a dict, prediction is the predicted value, check means needs to check the intermediate results, loss is the loss
        """
        out_dict = self.predict(feed_dict)
        if feed_dict[RANK] == 1:
            # Compute the loss of topn recommendation, the first half of batch are positive example, the second half are negative examples
            loss = self.rank_loss(out_dict[PREDICTION], feed_dict[Y], feed_dict[REAL_BATCH_SIZE])
        else:
            # Compute the loss of rating/clicking prediction, by default use mse
            if self.loss_sum == 1:
                loss = torch.nn.MSELoss(reduction='sum')(out_dict[PREDICTION], feed_dict[Y])
            else:
                loss = torch.nn.MSELoss(reduction='mean')(out_dict[PREDICTION], feed_dict[Y])
        out_dict[LOSS] = loss
        out_dict[LOSS_L2] = self.l2(out_dict)
        return out_dict

    def rank_loss(self, prediction, label, real_batch_size):
        '''
        Compute rank loss，similar to BPR-max, refer to paper:
        @inproceedings{hidasi2018recurrent,
          title={Recurrent neural networks with top-k gains for session-based recommendations},
          author={Hidasi, Bal{\'a}zs and Karatzoglou, Alexandros},
          booktitle={Proceedings of the 27th ACM International Conference on Information and Knowledge Management},
          pages={843--852},
          year={2018},
          organization={ACM}
        }
        :param prediction: predicted value [None]
        :param label: label [None]
        :param real_batch_size: batch size of observation values, excluding sample
        :return:
        '''
        pos_neg_tag = (label - 0.5) * 2
        observed, sample = prediction[:real_batch_size], prediction[real_batch_size:]
        # sample = sample.view([-1, real_batch_size]).mean(dim=0)
        sample = sample.view([-1, real_batch_size])
        sample_softmax = (sample * pos_neg_tag.view([1, real_batch_size])).softmax(dim=0)
        sample = (sample * sample_softmax).sum(dim=0)
        if self.loss_sum == 1:
            loss = -(pos_neg_tag * (observed - sample)).sigmoid().log().sum()
        else:
            loss = -(pos_neg_tag * (observed - sample)).sigmoid().log().mean()
        return loss

    def lrp(self):
        pass

    def save_model(self, model_path=None):
        """
        Save the model, usually using the default directory/path
        :param model_path: Specific the directory/path to save the model
        :return:
        """
        if model_path is None:
            model_path = self.model_path
        dir_path = os.path.dirname(model_path)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        torch.save(self.state_dict(), model_path)
        logging.info('Save model to ' + model_path)

    def load_model(self, model_path=None, cpu=False):
        """
        Load the model, usually using the default directory/path
        :param model_path: Specific the directory/path to load the model
        :return:
        """
        if model_path is None:
            model_path = self.model_path
        if cpu:
            self.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(torch.load(model_path))
        self.eval()
        logging.info('Load model from ' + model_path)