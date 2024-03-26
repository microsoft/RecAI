# coding=utf-8
import os
import numpy as np
import pandas as pd
from shutil import copyfile
from collections import Counter, defaultdict
from utils.global_p import *

def group_user_interactions_csv(in_csv, out_csv, label=LABEL, sep=SEP):
    print('group_user_interactions_csv', out_csv)
    all_data = pd.read_csv(in_csv, sep=sep)
    group_inters = group_user_interactions_df(in_df=all_data, label=label)
    group_inters.to_csv(out_csv, sep=sep, index=False)
    return group_inters

def group_user_interactions_df(in_df, pos_neg, label=LABEL, seq_sep=SEQ_SEP):
    all_data = in_df
    if label in all_data.columns:
        if pos_neg == 1:
            all_data = all_data[all_data[label] > 0]
        elif pos_neg == 0:
            all_data = all_data[all_data[label] <= 0]
    uids, inters = [], []
    for name, group in all_data.groupby(UID):
        uids.append(name)
        inters.append(seq_sep.join(group[IID].astype(str).tolist()))
    group_inters = pd.DataFrame()
    group_inters[UID] = uids
    group_inters[IIDS] = inters # sorted by time
    return group_inters

def leave_out_by_time_df(all_df, leave_n=1, warm_n=5, split_n=1, max_user=-1):
    min_label = all_df[LABEL].min()
    if min_label > 0:
        leave_df = all_df.groupby(UID).head(warm_n)
        all_df = all_df.drop(leave_df.index)
        split_dfs = []
        for i in range(split_n):
            total_uids = all_df[UID].unique()
            if 0 < max_user < len(total_uids):
                total_uids = np.random.choice(total_uids, size=max_user, replace=False).tolist()
                gb_uid = all_df.groupby(UID)
                split_df = []
                for uid in total_uids:
                    split_df.append(gb_uid.get_group(uid).tail(leave_n))
                split_df = pd.concat(split_df).sort_index()
            else:
                split_df = all_df.groupby(UID).tail(leave_n).sort_index()
            all_df = all_df.drop(split_df.index)
            split_dfs.append(split_df)
    else:
        leave_df = []
        for uid, group in all_df.groupby(UID):
            found, found_idx = 0, -1
            for idx in group.index:  # 该group的sample id
                if group.loc[idx, LABEL] > 0:
                    found_idx = idx
                    found += 1
                    if found >= warm_n:
                        break
            if found > 0:
                leave_df.append(group.loc[:found_idx + 1])
        leave_df = pd.concat(leave_df)
        all_df = all_df.drop(leave_df.index)

        split_dfs = []
        for i in range(split_n):
            total_uids = all_df[all_df[LABEL] > 0][UID].unique()
            if 0 < max_user < len(total_uids):
                total_uids = np.random.choice(total_uids, size=max_user, replace=False).tolist()
            gb_uid = all_df.groupby(UID)
            split_df = []
            for uid in total_uids:
                group = gb_uid.get_group(uid)
                found, found_idx = 0, -1
                # Look at the user's history inversely, until a positive example is found
                for idx in reversed(group.index):
                    if group.loc[idx, LABEL] > 0:
                        found_idx = idx
                        found += 1
                        if found >= leave_n:
                            break
                # If a positive example is found, then this example and the negative examples after this example are put into testing set
                if found > 0:
                    split_df.append(group.loc[found_idx:])
            split_df = pd.concat(split_df).sort_index()
            all_df = all_df.drop(split_df.index)
            split_dfs.append(split_df)
    leave_df = pd.concat([leave_df, all_df]).sort_index()
    return leave_df, split_dfs

def leave_out_by_time_csv(all_data_file, dataset_name, leave_n=1, warm_n=5, u_f=None, i_f=None):
    """
    By default, the interaction history in all_data are sorting according to timestamp, according to the interaction time, put the last interactions into validation and testing set
    :param all_data_file: data file after pre-processing *.all.csv, interactions are sorted according to timestamp
    :param dataset_name: create a name for the dataset
    :param leave_n: how many interactions to leave out in validation and testing set
    :param warm_n: guranttee that the testing user has at least warn_n number of interactions in training set, otherwise put all interactions into training set
    :param u_f: user feature vector *.user.csv
    :param i_f: item feature vector *.item.csv
    :return: pandas dataframe training set, validation set, testing set
    """
    dir_name = os.path.join(DATASET_DIR, dataset_name)
    print('leave_out_by_time_csv', dir_name, leave_n, warm_n)
    # If the dataset folder data_name does not exist, then create the folder, dataset_name is the name of the folder
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    all_data = pd.read_csv(all_data_file, sep=SEP)

    train_set, (test_set, validation_set) = leave_out_by_time_df(
        all_data, warm_n=warm_n, leave_n=leave_n, split_n=2, max_user=MAX_VT_USER)
    print('train=%d validation=%d test=%d' % (len(train_set), len(validation_set), len(test_set)))
    if UID in train_set.columns:
        print('train_user=%d validation_user=%d test_user=%d' %
              (len(train_set[UID].unique()), len(validation_set[UID].unique()), len(test_set[UID].unique())))

    train_set.to_csv(os.path.join(dir_name, dataset_name + TRAIN_SUFFIX), index=False, sep=SEP)
    validation_set.to_csv(os.path.join(dir_name, dataset_name + VALIDATION_SUFFIX), index=False, sep=SEP)
    test_set.to_csv(os.path.join(dir_name, dataset_name + TEST_SUFFIX), index=False, sep=SEP)
    
    # Copy the user, item feature file
    if u_f is not None:
        copyfile(u_f, os.path.join(dir_name, dataset_name + USER_SUFFIX))
    if i_f is not None:
        copyfile(i_f, os.path.join(dir_name, dataset_name + ITEM_SUFFIX))
    return train_set, validation_set, test_set
