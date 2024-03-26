# coding=utf-8
import sys
import ast
import json
import copy
import pickle
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '../')
sys.path.insert(0, './')

from utils.dataset import *
from utils.global_p import *
from shutil import copyfile

np.random.seed(DEFAULT_SEED)

dataset = "Movies_and_TV"
filter_users = ".filter"
RAW_DATA = f'../../../raw_data/Amazon_5_core/{dataset}/'
REVIEW_FILE = os.path.join(RAW_DATA, f'reviews_{dataset}_5.json')
ITEM_FILE = os.path.join(RAW_DATA, f'meta_{dataset}.json')

DATA_DIR = f'../../../raw_data/Amazon_5_core/{dataset}/'
USER_FEATURE_FILE = os.path.join(DATA_DIR, f'{dataset}{filter_users}.users.csv')
ITEM_FEATURE_FILE = os.path.join(DATA_DIR, f'{dataset}{filter_users}.items.csv')
ALL_DATA_FILE = os.path.join(DATA_DIR, f'{dataset}01{filter_users}.all.csv')

# http://jmcauley.ucsd.edu/data/amazon/
# format amazon 5-core review dataset
def format_user_feature(out_file, uid_dict):
    print("format_user_feature from raw file ", REVIEW_FILE, ", to target file ", out_file)
    records = []
    for line in open(REVIEW_FILE, 'r'):
        record = json.loads(line)
        if record['reviewerID'] not in uid_dict:
            continue
        records.append(record)
    user_df = pd.DataFrame()
    user_df[UID] = [r['reviewerID'] for r in records]
    user_df = user_df.drop_duplicates([UID])
    user_df[UID] = user_df[UID].apply(lambda x: uid_dict[x])
    user_df = user_df.sort_values(by=UID)
    user_df.reset_index(drop=True, inplace=True)
    user_df.to_csv(out_file, index=False, sep='\t')
    return user_df

def format_item_feature(out_file, iid_dict):
    print("format_item_feature from raw file ", ITEM_FILE, ", to target file ", out_file)
    records = []
    categories = dict()
    for line in open(ITEM_FILE, 'r'):
        record = ast.literal_eval(line.strip())
        if record['asin'] not in iid_dict:
            continue
        cate = record['categories'][0][-1]
        if cate not in categories:
            categories[cate] = len(categories)+1
        records.append(record)
    categories[''] = len(categories)+1

    item_df = pd.DataFrame()
    item_df[IID] = [r['asin'] for r in records]
    item_df['i_title'] = [r['title'].replace('\n', ' ') if 'title' in r else 'null' for r in records]
    item_df['i_description'] = [r['description'].replace('\n', ' ') if 'description' in r else 'null' for r in records]
    item_df['i_price'] = [float(r['price']) if 'price' in r else 0 for r in records]
    item_df['i_category'] = [categories[r['categories'][0][-1]] if 'categories' in r else categories[''] for r in records]
    item_df[IID] = item_df[IID].apply(lambda x: iid_dict[x])
    item_df = item_df.drop_duplicates([IID])
    item_df = item_df.sort_values(by=IID)#.drop_duplicates(subset=IID, keep='first', inplace=True)
    item_df.reset_index(drop=True, inplace=True)
    item_df.to_csv(out_file, index=False, sep='\t')
    return item_df

def format_5core_inter(out_file, label01=True):
    print("format_all_interaction from raw file ", REVIEW_FILE, ", to target file ", out_file)
    # read in the json file
    records = []
    for line in open(REVIEW_FILE, 'r'):
        record = json.loads(line)
        records.append(record)

    # K-core user_core item_core, return False if any user/item < core
    def check_Kcore(user_items, user_core, item_core):
        user_count = defaultdict(int)
        item_count = defaultdict(int)
        for user, items in user_items.items():
            for item in items:
                user_count[user] += 1
                item_count[item] += 1

        for user, num in user_count.items():
            if num < user_core:
                return user_count, item_count, False
        for item, num in item_count.items():
            if num < item_core:
                return user_count, item_count, False
        return user_count, item_count, True 

    def filter_Kcore(user_items, user_core, item_core):
        user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
        while not isKcore:
            for user, num in user_count.items():
                if user_count[user] < user_core:
                    user_items.pop(user)
                else:
                    for item in user_items[user]:
                        if item_count[item] < item_core:
                            user_items[user].remove(item)
            user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
        return user_count, item_count

    if filter_users != "":
        user_items = defaultdict(list)
        for r in records:
            if r['overall'] > 3:
                user_items[r['reviewerID']].append(r['asin'])
        user_count, item_count = filter_Kcore(user_items, user_core=20, item_core=5)
        print("#users: ", len(user_count), ", #items: ", len(item_count))
        records = [r for r in records if r['reviewerID'] in user_count and r['asin'] in item_count]


    # Convert json information to pandas DataFrame
    inter_df = pd.DataFrame()
    inter_df[UID] = [r['reviewerID'] for r in records]
    inter_df[IID] = [r['asin'] for r in records]
    inter_df[LABEL] = [r['overall'] for r in records]
    inter_df[TIME] = [r['unixReviewTime'] for r in records]

    # Sorted as time, uid, iid order
    inter_df = inter_df.sort_values(by=[TIME, UID, IID])
    inter_df = inter_df.drop_duplicates([UID, IID]).reset_index(drop=True)

    # Number the uids, begining from 1
    uids = sorted(inter_df[UID].unique())
    uid_dict = dict(zip(uids, range(1, len(uids) + 1)))
    inter_df[UID] = inter_df[UID].apply(lambda x: uid_dict[x])

    # Number the iids, begining from 1
    iids = sorted(inter_df[IID].unique())
    iid_dict = dict(zip(iids, range(1, len(iids) + 1)))
    inter_df[IID] = inter_df[IID].apply(lambda x: iid_dict[x])

    # # Drop the timestamp
    # out_df = out_df.drop(columns=TIME)

    # If format into two labels 0 (negative) and 1 (positive), rather than ratings, then consider ratings > 3 as positive 1, otherse as negative 0
    print(f"max label: {max(inter_df[LABEL])}, min label: {min(inter_df[LABEL])}.")
    if label01:
        inter_df[LABEL] = inter_df[LABEL].apply(lambda x: 1 if x > 3 else 0)
    print('label:', inter_df[LABEL].min(), inter_df[LABEL].max())
    print(Counter(inter_df[LABEL]))

    inter_df.to_csv(out_file, sep='\t', index=False)
    return inter_df, uid_dict, iid_dict

def main():
    _, uid_dict, iid_dict = format_5core_inter(ALL_DATA_FILE, label01=True)
    format_user_feature(USER_FEATURE_FILE, uid_dict)
    format_item_feature(ITEM_FEATURE_FILE, iid_dict)

    if filter_users == "":
        dataset_name = f'5{dataset}01-1-5'
    else:
        dataset_name = f'5{dataset}_Filtered01-1-5'
    leave_out_by_time_csv(ALL_DATA_FILE, dataset_name, leave_n=1, warm_n=5, 
                          u_f=USER_FEATURE_FILE, i_f=ITEM_FEATURE_FILE)
    return

if __name__ == '__main__':
    main()