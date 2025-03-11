# coding=utf-8
import os
import json
import logging
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from utils.global_p import *
from utils.dataset import group_user_interactions_df

class DataLoader(object):
    """
    Only responsible for loading the dataset file, and recording some information of the dataset
    """

    def __init__(self, path, dataset, label=LABEL, load_data=True, sep=SEP, seq_sep=SEQ_SEP, sample_type="random"):
        """
        :param path
        :param dataset
        :param label
        :param load_data
        :param sep
        :param seq_sep
        """
        self.dataset = dataset
        self.path = os.path.join(path, dataset)
        self.train_file = os.path.join(self.path, dataset + TRAIN_SUFFIX)
        self.validation_file = os.path.join(self.path, dataset + VALIDATION_SUFFIX)
        self.test_file = os.path.join(self.path, dataset + TEST_SUFFIX)
        self.info_file = os.path.join(self.path, dataset + INFO_SUFFIX)
        self.user_file = os.path.join(self.path, dataset + USER_SUFFIX)
        self.item_file = os.path.join(self.path, dataset + ITEM_SUFFIX)
        self.train_pos_file = os.path.join(self.path, dataset + TRAIN_POS_SUFFIX)
        self.validation_pos_file = os.path.join(self.path, dataset + VALIDATION_POS_SUFFIX)
        self.test_pos_file = os.path.join(self.path, dataset + TEST_POS_SUFFIX)
        self.train_neg_file = os.path.join(self.path, dataset + TRAIN_NEG_SUFFIX)
        self.validation_neg_file = os.path.join(self.path, dataset + VALIDATION_NEG_SUFFIX)
        self.test_neg_file = os.path.join(self.path, dataset + TEST_NEG_SUFFIX)
        self.sep, self.seq_sep = sep, seq_sep
        self.load_data = load_data
        self.label = label

        self.train_df, self.validation_df, self.test_df = None, None, None
        self._load_user_item()
        self._load_data()
        self._load_his()
        self._load_info()
        if not os.path.exists(self.info_file) or self.load_data:
            self._save_info()

    def _load_user_item(self):
        """
        :return:
        
        Load the csv feature file of users and items
        :return:
        """
        self.user_df, self.item_df = None, None
        if os.path.exists(self.user_file) and self.load_data:
            logging.info("load user csv...")
            self.user_df = pd.read_csv(self.user_file, sep='\t')
        if os.path.exists(self.item_file) and self.load_data:
            logging.info("load item csv...")
            self.item_df = pd.read_csv(self.item_file, sep='\t')#.drop(columns=['i_title'])
            if 'i_title' in self.item_df:
                self.item_df = self.item_df.drop(columns=['i_title'])
            if 'i_year' in self.item_df:
                self.item_df = self.item_df.drop(columns=['i_year'])
    
    def _load_data(self):
        """
        :return:
        
        Load the training set, validation set, and testing set csv file
        :return:
        """
        if os.path.exists(self.train_file) and self.load_data:
            logging.info("load train csv...")
            self.train_df = pd.read_csv(self.train_file, sep=self.sep)
            logging.info("size of train: %d" % len(self.train_df))
            if self.label in self.train_df:
                logging.info("train label: " + str(dict(Counter(self.train_df[self.label]).most_common())))
        else:
            logging.info(f"{self.train_file} not exist.")
        if os.path.exists(self.validation_file) and self.load_data:
            logging.info("load validation csv...")
            self.validation_df = pd.read_csv(self.validation_file, sep=self.sep)
            logging.info("size of validation: %d" % len(self.validation_df))
            if self.label in self.validation_df:
                logging.info("validation label: " + str(dict(Counter(self.validation_df[self.label]).most_common())))
        if os.path.exists(self.test_file) and self.load_data:
            logging.info("load test csv...")
            self.test_df = pd.read_csv(self.test_file, sep=self.sep)
            logging.info("size of test: %d" % len(self.test_df))
            if self.label in self.test_df:
                logging.info("test label: " + str(dict(Counter(self.test_df[self.label]).most_common())))
    
    def _save_info(self):
        def json_type(o):
            if isinstance(o, np.int64):
                return int(o)
            # if isinstance(o, np.float32): return int(o)
            raise TypeError

        max_json = json.dumps(self.column_max, default=json_type)
        min_json = json.dumps(self.column_min, default=json_type)
        out_f = open(self.info_file, 'w')
        out_f.write(max_json + os.linesep + min_json)
        logging.info('Save dataset info to ' + self.info_file)
    
    def _load_info(self):
        """
        :return:
        
        Load the dataset information file, if does not exist then create the file
        :return:
        """
        max_dict, min_dict = {}, {}
        if not os.path.exists(self.info_file) or self.load_data:
            for df in [self.train_df, self.validation_df, self.test_df, self.user_df, self.item_df]:
                if df is None:
                    continue
                for c in df.columns:
                    if c not in max_dict:
                        try:
                            max_dict[c] = df[c].max()
                        except:
                            print("str column: ", c, df[c][0])
                    else:
                        max_dict[c] = max(df[c].max(), max_dict[c])
                    if c not in min_dict:
                        try:
                            min_dict[c] = df[c].min()
                        except:
                            print("str column: ", c, df[c][0])
                    else:
                        min_dict[c] = min(df[c].min(), min_dict[c])
        else:
            lines = open(self.info_file, 'r').readlines()
            max_dict = json.loads(lines[0])
            min_dict = json.loads(lines[1])

        self.column_max = max_dict
        self.column_min = min_dict

        # Minimum value and maximum value of the labels
        self.label_max = self.column_max[self.label]
        self.label_min = self.column_min[self.label]
        logging.info("label: %d-%d" % (self.label_min, self.label_max))

        # number of users, number of items
        self.user_num, self.item_num = 0, 0
        if UID in self.column_max:
            self.user_num = self.column_max[UID] + 1
        if IID in self.column_max:
            self.item_num = self.column_max[IID] + 1
        logging.info("# of users: %d" % self.user_num)
        logging.info("# of items: %d" % self.item_num)

        # number of features of the dataset
        self.user_features = [f for f in self.column_max.keys() if f.startswith('u_')]
        logging.info("# of user features: %d" % len(self.user_features))
        self.item_features = [f for f in self.column_max.keys() if f.startswith('i_')]
        logging.info("# of item features: %d" % len(self.item_features))
        self.context_features = [f for f in self.column_max.keys() if f.startswith('c_')]
        logging.info("# of context features: %d" % len(self.context_features))
        self.features = self.context_features + self.user_features + self.item_features
        logging.info("# of features: %d" % len(self.features))
    
    def _load_his(self):
        """
        :return:
        
        Load the history interaction records of the dataset merged according to uid, two columns of 'uid' and 'iids', if non-existing then create
        :return:
        """
        if not self.load_data or UID not in self.train_df or IID not in self.train_df:
            return
        if not os.path.exists(self.train_pos_file):
            logging.info("building train pos history csv...")
            train_pos_df = group_user_interactions_df(self.train_df, pos_neg=1, label=self.label, seq_sep=self.seq_sep)
            train_pos_df.to_csv(self.train_pos_file, index=False, sep=self.sep)
        if not os.path.exists(self.validation_pos_file):
            logging.info("building validation pos history csv...")
            validation_pos_df = group_user_interactions_df(
                self.validation_df, pos_neg=1, label=self.label, seq_sep=self.seq_sep)
            validation_pos_df.to_csv(self.validation_pos_file, index=False, sep=self.sep)
        if not os.path.exists(self.test_pos_file):
            logging.info("building test pos history csv...")
            test_pos_df = group_user_interactions_df(self.test_df, pos_neg=1, label=self.label, seq_sep=self.seq_sep)
            test_pos_df.to_csv(self.test_pos_file, index=False, sep=self.sep)

        if not os.path.exists(self.train_neg_file):
            logging.info("building train neg history csv...")
            train_neg_df = group_user_interactions_df(self.train_df, pos_neg=0, label=self.label, seq_sep=self.seq_sep)
            train_neg_df.to_csv(self.train_neg_file, index=False, sep=self.sep)
        if not os.path.exists(self.validation_neg_file):
            logging.info("building validation neg history csv...")
            validation_neg_df = group_user_interactions_df(
                self.validation_df, pos_neg=0, label=self.label, seq_sep=self.seq_sep)
            validation_neg_df.to_csv(self.validation_neg_file, index=False, sep=self.sep)
        if not os.path.exists(self.test_neg_file):
            logging.info("building test neg history csv...")
            test_neg_df = group_user_interactions_df(self.test_df, pos_neg=0, label=self.label, seq_sep=self.seq_sep)
            test_neg_df.to_csv(self.test_neg_file, index=False, sep=self.sep)

        def build_his(his_df, seqs_sep):
            uids = his_df[UID].tolist()
            iids = his_df[IIDS].astype(str).str.split(seqs_sep).values
            # iids = [i.split(self.seq_sep) for i in his_df['iids'].tolist()]
            iids = [[int(j) for j in i] for i in iids]
            user_his = dict(zip(uids, iids))
            return user_his

        self.train_pos_df, self.train_user_pos = None, None
        self.validation_pos_df, self.validation_user_pos = None, None
        self.test_pos_df, self.test_user_pos = None, None
        self.train_neg_df, self.train_user_neg = None, None
        self.validation_neg_df, self.validation_user_neg = None, None
        self.test_neg_df, self.test_user_neg = None, None
        if self.load_data:
            logging.info("load history csv...")
            self.train_pos_df = pd.read_csv(self.train_pos_file, sep=self.sep)
            self.train_user_pos = build_his(self.train_pos_df, self.seq_sep)
            self.validation_pos_df = pd.read_csv(self.validation_pos_file, sep=self.sep)
            self.validation_user_pos = build_his(self.validation_pos_df, self.seq_sep)
            self.test_pos_df = pd.read_csv(self.test_pos_file, sep=self.sep)
            self.test_user_pos = build_his(self.test_pos_df, self.seq_sep)
            self.train_neg_df = pd.read_csv(self.train_neg_file, sep=self.sep)
            self.train_user_neg = build_his(self.train_neg_df, self.seq_sep)
            self.validation_neg_df = pd.read_csv(self.validation_neg_file, sep=self.sep)
            self.validation_user_neg = build_his(self.validation_neg_df, self.seq_sep)
            self.test_neg_df = pd.read_csv(self.test_neg_file, sep=self.sep)
            self.test_user_neg = build_his(self.test_neg_df, self.seq_sep)

    def feature_info(self, include_id=True, include_item_features=True, include_user_features=True, include_context_features=True):
        features = []
        if include_id:
            if UID in self.column_max:
                features.append(UID)
            if IID in self.column_max:
                features.append(IID)
        if include_user_features:
            features.extend(self.user_features)
        if include_item_features:
            features.extend(self.item_features)
        if include_context_features:
            features.extend(self.context_features)
        feature_dims = 1
        feature_min, feature_max = [], []
        for f in features:
            feature_min.append(feature_dims)
            feature_dims += int(self.column_max[f]) # +1)
            feature_max.append(feature_dims-1)
        logging.info('Model # of features %d %s' % (len(features), features))
        logging.info('Model # of feature dims %d' % feature_dims)
        logging.info(f'Model feature_min: {feature_min}, feature_max: {feature_max}.')
        return features, feature_dims, feature_min, feature_max

    def append_his(self, all_his=1, max_his=10, neg_his=0, neg_column=0):
        assert not (all_his == 1 and self.train_df is None)
        his_dict, neg_dict = {}, {}
        for df in [self.train_df, self.validation_df, self.test_df]:
            if df is None or C_HISTORY in df:
                continue
            history, neg_history = [], []
            if all_his != 1 or df is self.train_df:
                uids, iids, labels = df[UID].tolist(), df[IID].tolist(), df[self.label].tolist()
                for i, uid in enumerate(uids):
                    iid, label = iids[i], labels[i]
                    if uid not in his_dict:
                        his_dict[uid] = []
                    if uid not in neg_dict:
                        neg_dict[uid] = []

                    tmp_his = his_dict[uid] if max_his <= 0 else his_dict[uid][-max_his:]
                    tmp_neg = neg_dict[uid] if max_his <= 0 else neg_dict[uid][-max_his:]
                    history.append(str(tmp_his).replace(' ', '')[1:-1])
                    neg_history.append(str(tmp_neg).replace(' ', '')[1:-1])

                    if label <= 0 and neg_his == 1 and neg_column == 0:
                        his_dict[uid].append(-iid)
                    elif label <= 0 and neg_column == 1:
                        neg_dict[uid].append(iid)
                    elif label > 0:
                        his_dict[uid].append(iid)
            
            if all_his == 1:
                history, neg_history = [], []
                uids, iids, labels = df[UID].tolist(), df[IID].tolist(), df[self.label].tolist()
                for i, uid in enumerate(uids):
                    iid, label = str(iids[i]), labels[i]
                    if uid in his_dict:
                        tmp_his = his_dict[uid][:]
                        if iid in tmp_his:
                            tmp_his.remove(iid)
                        history.append(str(tmp_his).replace(' ', '')[1:-1])
                    else:
                        history.append('')
                    if uid in neg_dict:
                        neg_history.append(str(neg_dict[uid]).replace(' ', '')[1:-1])
                    else:
                        neg_history.append('')

            df[C_HISTORY] = history
            if neg_his == 1 and neg_column == 1:
                df[C_HISTORY_NEG] = neg_history

    def drop_neg(self, train=True, validation=True, test=True):
        """
        If it's top n recommendation, only keep the positive examples, negative examples are sampled during training
        :return:
        """
        logging.info('Drop Neg Samples...')
        if train and self.train_df is not None:
            self.train_df = self.train_df[self.train_df[self.label] > 0].reset_index(drop=True)
        if validation and self.validation_df is not None:
            self.validation_df = self.validation_df[self.validation_df[self.label] > 0].reset_index(drop=True)
        if test and self.test_df is not None:
            self.test_df = self.test_df[self.test_df[self.label] > 0].reset_index(drop=True)
        logging.info("size of train: %d" % len(self.train_df))
        logging.info("size of validation: %d" % len(self.validation_df))
        logging.info("size of test: %d" % len(self.test_df))

    def label_01(self, train=True, validation=True, test=True):
        """
        Converte the label to 01 binary values
        :return:
        """
        logging.info("Transform label to 0-1")
        if train and self.train_df is not None and self.label in self.train_df:
            self.train_df[self.label] = self.train_df[self.label].apply(lambda x: 1 if x > 0 else 0)
            logging.info("train label: " + str(dict(Counter(self.train_df[self.label]).most_common())))
        if validation and self.validation_df is not None and self.label in self.validation_df:
            self.validation_df[self.label] = self.validation_df[self.label].apply(lambda x: 1 if x > 0 else 0)
            logging.info("validation label: " + str(dict(Counter(self.validation_df[self.label]).most_common())))
        if test and self.test_df is not None and self.label in self.test_df:
            self.test_df[self.label] = self.test_df[self.label].apply(lambda x: 1 if x > 0 else 0)
            logging.info("test label: " + str(dict(Counter(self.test_df[self.label]).most_common())))
        self.label_min = 0
        self.label_max = 1