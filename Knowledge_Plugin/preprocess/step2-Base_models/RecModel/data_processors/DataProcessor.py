# coding=utf-8
import copy
import torch
import random
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter
from torch.utils.data import Dataset
from utils import utils
from utils.global_p import *

class DataProcessor(Dataset):
    # The key to store the feature information needed by the model in data dict
    # data_columns = [UID, IID, X]
    # info_columns = [SAMPLE_ID, TIME]
    data_columns = [UID, IID, X, C_HISTORY, C_HISTORY_NEG]
    info_columns = [SAMPLE_ID, TIME, C_HISTORY_LENGTH, C_HISTORY_NEG_LENGTH]

    def __init__(self, args, data_loader, model, sample_type="random"):
        self.args = args
        self.rank = args.rank
        self.sample_un_p = args.sample_un_p
        self.train_sample_n = args.train_sample_n
        self.test_sample_n = args.test_sample_n
        self.unlabel_test = args.unlabel_test
        self.sample_type = sample_type

        if args.use_his == 1:
            self.use_his = args.use_his
            self.max_his = args.max_his
            self.sparse_his = args.sparse_his
            self.sup_his = args.sup_his
            self.drop_first = args.drop_first

        self.data_loader = data_loader
        self.get_pos_frequency()
        self.train_data, self.valid_data, self.test_data = None, None, None
        if self.rank == 1:
            # Generate the dict of user interactoins, convenient for querying when doing negative sampling to guaranttee positive examples are not sampled
            self.train_history_pos = defaultdict(set)
            for uid in data_loader.train_user_pos.keys():
                self.train_history_pos[uid] = set(data_loader.train_user_pos[uid])
            self.validation_history_pos = defaultdict(set)
            for uid in data_loader.validation_user_pos.keys():
                self.validation_history_pos[uid] = set(data_loader.validation_user_pos[uid])
            self.test_history_pos = defaultdict(set)
            for uid in data_loader.test_user_pos.keys():
                self.test_history_pos[uid] = set(data_loader.test_user_pos[uid])

            self.train_history_neg = defaultdict(set)
            for uid in data_loader.train_user_neg.keys():
                self.train_history_neg[uid] = set(data_loader.train_user_neg[uid])
            self.validation_history_neg = defaultdict(set)
            for uid in data_loader.validation_user_neg.keys():
                self.validation_history_neg[uid] = set(data_loader.validation_user_neg[uid])
            self.test_history_neg = defaultdict(set)
            for uid in data_loader.test_user_neg.keys():
                self.test_history_neg[uid] = set(data_loader.test_user_neg[uid])
        self.vt_batches_buffer = {}

    def get_pos_frequency(self):
        self.item_pos_freq = [0] * (self.data_loader.item_num + 1)
        train_df = self.data_loader.train_df
        for idx in range(len(train_df)):
            uid, iid, label = train_df[UID][idx], train_df[IID][idx], train_df[LABEL][idx]
            if label > 0:
                self.item_pos_freq[iid] += 1

    def get_train_data(self, epoch, model):
        """
        Convert the training dataset Dataframe in the dataloader into the needed dict and return, need to shuffle in every round
        This dict will be used to generate batches
        :param epoch: if < 0 then no shuffling
        :param model: Model class
        :return: dict
        """
        if self.train_data is None:
            logging.info('Prepare Train Data...')
            self.train_data = self.format_data_dict(self.data_loader.train_df, model)
            self.train_data[SAMPLE_ID] = np.arange(0, len(self.train_data[Y]))
        if epoch >= 0:
            utils.shuffle_in_unison_scary(self.train_data)
        return self.train_data
    
    def get_valid_data(self, model):
        if self.valid_data is None:
            logging.info('Prepare Validation Data...')
            df = self.data_loader.validation_df
            if self.rank == 1:
                tmp_df = df.rename(columns={self.data_loader.label: Y})
                tmp_df = tmp_df.drop(tmp_df[tmp_df[Y] <= 0].index)
                neg_df = self.generate_neg_df(
                    inter_df=tmp_df, feature_df=df, sample_n=self.test_sample_n, train=False)
                df = pd.concat([df, neg_df], ignore_index=True)
            self.valid_data = self.format_data_dict(df, model)
            self.valid_data[SAMPLE_ID] = np.arange(0, len(self.valid_data[Y]))
        return self.valid_data
    
    def get_test_data(self, model):
        if self.test_data is None:
            logging.info('Prepare Test Data...')
            df = self.data_loader.test_df[:1000]
            if self.rank == 1 and self.unlabel_test == 0:
                tmp_df = df.rename(columns={self.data_loader.label: Y})
                tmp_df = tmp_df.drop(tmp_df[tmp_df[Y] <= 0].index)
                neg_df = self.generate_neg_df(
                    inter_df=tmp_df, feature_df=df, sample_n=self.test_sample_n, train=False) #, test_neg_data=self.data_loader.test_neg_data)
                df = pd.concat([df, neg_df], ignore_index=True)
            self.test_data = self.format_data_dict(df, model)
            self.test_data[SAMPLE_ID] = np.arange(0, len(self.test_data[Y]))
        return self.test_data

    def get_ui_data(self, model):
        ui_pairs = []
        for uid in range(self.data_loader.user_num):
            ui_pairs.append([uid, 1, 1])
        for iid in range(self.data_loader.item_num):
            ui_pairs.append([1, iid, 1])
        ui_df = pd.DataFrame(ui_pairs, columns=[UID, IID, LABEL])
        self.ui_data = self.format_data_dict(ui_df, model)
        self.ui_data[SAMPLE_ID] = np.arange(0, len(self.ui_data[Y]))
        return self.ui_data

    def get_feed_dict(self, data, batch_start, batch_size, train, neg_data=None, special_cols=None):
        """
        topn model will produce a batch, if doing training then need to sample a negative example for each positive example, and garanttee that for each batch the first half are positive examples and the second half are negative examples
        :param data: data dict, produced by self.get_*_data() and self.format_data_dict() functions
        :param batch_start: starting index of the batch
        :param batch_size: batch size
        :param train: training or testing
        :param neg_data: data dict of negative examples, if alreay exist can use directly
        :param special_cols: columns that need special treatment
        :return: feed dict of the batch
        """
        if self.args.use_his:
            special_cols=[C_HISTORY, C_HISTORY_NEG] if special_cols is None else [C_HISTORY, C_HISTORY_NEG] + special_cols
        
        # If testing or validation, no need to sample negative examples for each positive example, negative examples are already sampled using ratio 1 : test_sample_n
        total_data_num = len(data[SAMPLE_ID])
        batch_end = min(len(data[self.data_columns[0]]), batch_start + batch_size)
        real_batch_size = batch_end - batch_start
        total_batch_size = real_batch_size * (self.train_sample_n + 1) if self.rank == 1 and train else real_batch_size
        feed_dict = {TRAIN: train, RANK: self.rank, REAL_BATCH_SIZE: real_batch_size,
                     TOTAL_BATCH_SIZE: total_batch_size}
        if Y in data:
            feed_dict[Y] = utils.numpy_to_torch(data[Y][batch_start:batch_start + real_batch_size], gpu=False)
        for c in self.info_columns + self.data_columns:
            if c not in data or data[c].size <= 0:
                continue
            d = data[c][batch_start: batch_start + real_batch_size]
            if self.rank == 1 and train:
                neg_d = np.concatenate(
                    [neg_data[c][total_data_num * i + batch_start: total_data_num * i + batch_start + real_batch_size]
                     for i in range(self.train_sample_n)])
                d = np.concatenate([d, neg_d])
            feed_dict[c] = d
        for c in self.data_columns:
            if c not in feed_dict:
                continue
            if special_cols is not None and c in special_cols:
                continue
            feed_dict[c] = utils.numpy_to_torch(feed_dict[c], gpu=False)
        
        if self.args.use_his:
            his_cs, his_ls = [C_HISTORY], [C_HISTORY_LENGTH]
            # If there are columns of negative histories
            if C_HISTORY_NEG in feed_dict:
                his_cs.append(C_HISTORY_NEG)
                his_ls.append(C_HISTORY_NEG_LENGTH)

            for i, c in enumerate(his_cs):
                lc, d = his_ls[i], feed_dict[c]
                # If it's sparse representation
                if self.sparse_his == 1:
                    x, y, v = [], [], []
                    for idx, iids in enumerate(d):
                        x.extend([idx] * len(iids))
                        y.extend([abs(iid) for iid in iids])
                        v.extend([1.0 if iid > 0 else -1.0 if iid < 0 else 0 for iid in iids])
                    if len(x) <= 0:
                        i = utils.numpy_to_torch(np.array([[0], [0]]), gpu=False)
                        v = utils.numpy_to_torch(np.array([0.0], dtype=np.float32), gpu=False)
                    else:
                        i = utils.numpy_to_torch(np.array([x, y]), gpu=False)
                        v = utils.numpy_to_torch(np.array(v, dtype=np.float32), gpu=False)
                    history = torch.sparse.FloatTensor(
                        i, v, torch.Size([len(d), self.data_loader.item_num]))
                    # if torch.cuda.device_count() > 0:
                    #     history = history.cuda()
                    feed_dict[c] = history
                    feed_dict[lc] = [len(iids) for iids in d]
                    # feed_dict[lc] = utils.numpy_to_torch(np.array([len(iids) for iids in d]), gpu=False)
                else:
                    lengths = [len(iids) for iids in d]
                    max_length = max(lengths)
                    new_d = np.array([x + [0] * (max_length - len(x)) for x in d])
                    feed_dict[c] = utils.numpy_to_torch(new_d, gpu=False)
                    feed_dict[lc] = lengths
        
        return feed_dict

    def _check_vt_buffer(self, data, batch_size, train, model):
        buffer_key = ''
        if data is self.train_data and not train:
            buffer_key = '_'.join(['train', str(batch_size), str(model)])
        elif data is self.valid_data:
            buffer_key = '_'.join(['validation', str(batch_size), str(model)])
        elif data is self.test_data:
            buffer_key = '_'.join(['test', str(batch_size), str(model)])
        if buffer_key != '' and buffer_key in self.vt_batches_buffer:
            return self.vt_batches_buffer[buffer_key]
        return buffer_key

    def prepare_batches(self, data, batch_size, train, model):
        """
        Convert all data dict to batches
        :param data: dict, generated by self.get_*_data() and self.format_data_dict() functions
        :param batch_size: batch size
        :param train: training or testing
        :param model: Model class
        :return: list of batches
        """

        buffer_key = self._check_vt_buffer(data=data, batch_size=batch_size, train=train, model=model)
        if type(buffer_key) != str:
            return buffer_key
        
        if data is None:
            return None
        num_example = len(data[Y])
        total_batch = int((num_example + batch_size - 1) / batch_size)
        assert num_example > 0
        # if training, then need to sample a negative example for all corresponding positive examples
        neg_data = None
        if train and self.rank == 1:
            neg_data = self.generate_neg_data(
                data, self.data_loader.train_df, sample_n=self.train_sample_n,
                train=True, model=model)
        batches = []
        for batch in tqdm(range(total_batch), leave=False, ncols=100, mininterval=1, desc='Prepare Batches'):
            batches.append(self.get_feed_dict(data=data, batch_start=batch * batch_size, batch_size=batch_size,
                                              train=train, neg_data=neg_data))

        if buffer_key != '':
            self.vt_batches_buffer[buffer_key] = batches
        return batches

    def format_data_dict(self, df, model):
        """
        Convert the training, validation and testing Dataframes of the dataloader to the needed data dict
        :param df: pandas Dataframe, usually includes three columns UID,IID,'label' in recommendation problems
        :param model: Model class
        :return: data dict
        """
        if self.args.use_his:
            assert C_HISTORY in df
            his_cs = [C_HISTORY]
            if C_HISTORY_NEG in df:  # (if there exist columns of negative histories)
                his_cs.append(C_HISTORY_NEG)
            if self.drop_first == 1:
                for c in his_cs:
                    df = df[df[c].apply(lambda x: len(x) > 0)]

        data_loader = self.data_loader
        data = {}
        # record uid, iid
        out_columns = []
        if UID in df:
            out_columns.append(UID)
            data[UID] = df[UID].values
        if IID in df:
            out_columns.append(IID)
            data[IID] = df[IID].values
        if TIME in df:
            data[TIME] = df[TIME].values

        # label is recorded in Y
        if data_loader.label in df.columns:
            data[Y] = np.array(df[data_loader.label], dtype=np.float32)
        else:
            logging.warning('No Labels In Data: ' + data_loader.label)
            data[Y] = np.zeros(len(df), dtype=np.float32)
        
        ui_id = df[out_columns]

        # Concat user features and item features according to uid and iid
        out_df = ui_id
        if data_loader.user_df is not None and model.include_user_features:
            out_df = pd.merge(out_df, data_loader.user_df, on=UID, how='left')
        if data_loader.item_df is not None and model.include_item_features:
            out_df = pd.merge(out_df, data_loader.item_df, on=IID, how='left')

        # If or not to include context feature
        if model.include_context_features and len(data_loader.context_features) > 0:
            context = df[data_loader.context_features]
            out_df = pd.concat([out_df, context], axis=1, ignore_index=True)
        out_df = out_df.fillna(0)

        # If model does not consider uid and iid as normal features, i.e., do not convert to multi-hot vectors as other features
        if not model.include_id:
            out_df = out_df.drop(columns=out_columns)
        
        '''
        Convert all features into multi-hot vectors
        e.g., uid(0-2),iid(0-2),u_age(0-2),i_xx(0-1),
        then uid=0,iid=1,u_age=1,i_xx=0 will be converted to the sparse representation of 100 010 010 10, i.e., 0,4,7,9
        '''
        base = 0
        for feature in out_df.columns:
            out_df[feature] = out_df[feature].apply(lambda x: x + base)
            base += int(data_loader.column_max[feature]) # + 1)

        # If needed by the model, uid, iid will be concatenated at the first two columns of x 
        data[X] = out_df.values.astype(int)
        assert len(data[X]) == len(data[Y])

        if self.args.use_his:
            for c in his_cs:
                his = df[c].apply(lambda x: eval('[' + x + ']'))
                data[c] = his.values

        return data

    def generate_neg_data(self, data, feature_df, sample_n, train, model):
        """
        :param data: data_dict
        :param feature_df:
        :param sample_n:
        :param train:
        :param model:
        :return:
        """
        inter_df = pd.DataFrame()
        for c in [UID, IID, Y, TIME]:
            if c in data:
                inter_df[c] = data[c]
            else:
                assert c == TIME
        neg_df = self.generate_neg_df(
            inter_df=inter_df, feature_df=feature_df,
            sample_n=sample_n, train=train)
        neg_data = self.format_data_dict(neg_df, model)
        neg_data[SAMPLE_ID] = np.arange(0, len(neg_data[UID])) + len(data[SAMPLE_ID])
        return neg_data
    
    def generate_neg_df(self, inter_df, feature_df, sample_n, train, test_neg_data=None):
        """
        Generate negative examples according to uid,iid and the dataframe of traininig, validation or testing
        :param sample_n: number of negative examples
        :param train: negative sampling for training set, validation set or testing set
        :return:
        """
        other_columns = [c for c in inter_df.columns if c not in [UID, Y]]
        neg_df = self._sample_neg_from_uid_list(
            uids=inter_df[UID].tolist(), labels=inter_df[Y].tolist(), sample_n=sample_n, train=train,
            other_infos=inter_df[other_columns].to_dict('list'), test_neg_data=test_neg_data)
        neg_df = pd.merge(neg_df, feature_df, on=[UID] + other_columns, how='left')
        neg_df = neg_df.drop(columns=[IID])
        neg_df = neg_df.rename(columns={'iid_neg': IID})
        neg_df = neg_df[feature_df.columns]
        neg_df[self.data_loader.label] = 0
        return neg_df

    def _sample_neg_from_uid_list(self, uids, labels, sample_n, train, other_infos=None, test_neg_data=None):
        """
        Sample the corresponding negative examples according to the list of uid
        :param uids: uid list
        :param sample_n: number of negative examples to sample for each uid
        :param train: Sample for training set or testing set
        :param other_infos: Information that may need to be copied except for uid,iid,label, e.g., history interactions (first n item)
            used to copy original positive example idd in generate_neg_df
        :return: return the DataFrame, also need to be converted to data dict through self.format_data_dict()
        """
        if other_infos is None:
            other_infos = {}
        iid_list = []

        other_info_list = {}
        for info in other_infos:
            other_info_list[info] = []

        item_num = self.data_loader.item_num
        weights = [self.item_pos_freq[x] for x in range(1, item_num)]
        candidate = list(range(1, item_num))

        # Record the iid that are sampled in the sampling procedure, to avoid repeated samples
        for index, uid in enumerate(uids):
            if labels[index] > 0:
                # Avoid known positive examples
                train_history = self.train_history_pos
                validation_history, test_history = self.validation_history_pos, self.test_history_pos
                known_train = self.train_history_neg
            else:
                assert train
                # Avoid known negative examples
                train_history = self.train_history_neg
                validation_history, test_history = self.validation_history_neg, self.test_history_neg
                known_train = self.train_history_pos
            if train:
                # When sampling for training set, avoid known positive or negative examples in training set
                inter_iids = train_history[uid]
            else:
                # When sampling for testing set, avoid all known positive or negative examples
                inter_iids = train_history[uid] | validation_history[uid] | test_history[uid]

            # Check remaining number of samples
            remain_iids_num = item_num - len(inter_iids)
            # If not enough then report error
            assert remain_iids_num >= sample_n

            # If not too many then list all available items to sample using np.choice
            remain_iids = None
            if 1.0 * remain_iids_num / item_num < 0.2:
                remain_iids = [i for i in range(1, item_num) if i not in inter_iids]
            
            sampled = set()
            if test_neg_data is not None:
                unknown_iid_list = test_neg_data[uid][:sample_n]
            elif remain_iids is None:
                unknown_iid_list = []
                for i in range(sample_n):
                    if train:
                        if self.sample_type == 'random':
                            iid = np.random.randint(1, self.data_loader.item_num)
                        elif self.sample_type == 'pop':
                            # candidate = random.sample([x for x in range(item_num+1)], 200)
                            # weights = [self.item_pos_freq[x] for x in candidate]
                            iid = random.choices(candidate, weights=weights, k=1)[0]
                        
                    else:
                        if self.sample_type == 'random':
                            iid = np.random.randint(1, self.data_loader.item_num)
                        elif self.sample_type == 'pop':
                            iid = random.choices(candidate, weights=weights, k=1)[0]
                    while iid in inter_iids or iid in sampled:
                        if train:
                            if self.sample_type == 'random':
                                iid = np.random.randint(1, self.data_loader.item_num)
                            elif self.sample_type == 'pop':
                                # candidate = random.sample([x for x in range(item_num+1)], 200)
                                # weights = [self.item_pos_freq[x] for x in candidate]
                                iid = random.choices(candidate, weights=weights, k=1)[0]
                        else:
                            if self.sample_type == 'random':
                                iid = np.random.randint(1, self.data_loader.item_num)
                            elif self.sample_type == 'pop':
                                iid = random.choices(candidate, weights=weights, k=1)[0]
                    unknown_iid_list.append(iid)
                    sampled.add(iid)
            else:
                unknown_iid_list = np.random.choice(remain_iids, sample_n, replace=False)
            
            # If when training, it could be possible to sample from known negative or positive samples
            if train and self.sample_un_p < 1:
                known_iid_list = list(np.random.choice(
                    list(known_train[uid]), min(sample_n, len(known_train[uid])), replace=False)) \
                    if len(known_train[uid]) != 0 else []
                known_iid_list = known_iid_list + unknown_iid_list
                tmp_iid_list = []
                sampled = set()
                for i in range(sample_n):
                    p = np.random.rand()
                    if p < self.sample_un_p or len(known_iid_list) == 0:
                        iid = unknown_iid_list.pop(0)
                        while iid in sampled:
                            iid = unknown_iid_list.pop(0)
                    else:
                        iid = known_iid_list.pop(0)
                        while iid in sampled:
                            iid = known_iid_list.pop(0)
                    tmp_iid_list.append(iid)
                    sampled.add(iid)
                iid_list.append(tmp_iid_list)
            else:
                iid_list.append(unknown_iid_list)
            
        all_uid_list, all_iid_list = [], []
        for i in range(sample_n):
            for index, uid in enumerate(uids):
                all_uid_list.append(uid)
                all_iid_list.append(iid_list[index][i])
                # # Copy other information
                for info in other_infos:
                    other_info_list[info].append(other_infos[info][index])
            
        neg_df = pd.DataFrame(data=list(zip(all_uid_list, all_iid_list)), columns=[UID, 'iid_neg'])
        neg_df.to_csv("test_neg_df.csv", sep="\t", index=False)
        for info in other_infos:
            neg_df[info] = other_info_list[info]
        return neg_df