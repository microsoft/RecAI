# coding=utf-8
import sys
import socket

sys.path.insert(0, '../')
sys.path.insert(0, './')

from utils.dataset import *
from utils.global_p import *
from shutil import copyfile

np.random.seed(DEFAULT_SEED)
print(socket.gethostname())

RAW_DATA = '../../../raw_data/ml-100k/'
RATINGS_FILE = os.path.join(RAW_DATA, 'u.data')
USERS_FILE = os.path.join(RAW_DATA, 'u.user')
ITEMS_FILE = os.path.join(RAW_DATA, 'u.item')

DATA_DIR = '../../../raw_data/ml-100k/'
USER_FEATURE_FILE = os.path.join(DATA_DIR, 'ml100k.users.csv')
ITEM_FEATURE_FILE = os.path.join(DATA_DIR, 'ml100k.items.csv')
ALL_DATA_FILE = os.path.join(DATA_DIR, 'ml100k01.all.csv')

def format_user_feature(out_file):
    print("format_user_feature from raw file ", USERS_FILE, ", to target file ", out_file)
    user_df = pd.read_csv(USERS_FILE, sep='|', header=None)
    user_df = user_df[[0, 1, 2, 3]]
    user_df.columns = [UID, 'u_age', 'u_gender', 'u_occupation']
    min_age, max_age = 10, 60
    user_df['u_age'] = user_df['u_age'].apply(
        lambda x: 1 if x < min_age else int(x / 5) if x <= max_age else int(max_age / 5) + 1 if x > max_age else 0)
    user_df['u_gender'] = user_df['u_gender'].apply(lambda x: defaultdict(int, {'M': 1, 'F': 2})[x])
    occupation = {'none': 0, 'other': 1}
    for o in user_df['u_occupation'].unique():
        if o not in occupation:
            occupation[o] = len(occupation)
    user_df['u_occupation'] = user_df['u_occupation'].apply(lambda x: defaultdict(int, occupation)[x])
    user_df.to_csv(out_file, index=False, sep='\t')
    return user_df

def format_item_feature(out_file):
    print("format_item_feature from raw file ", ITEMS_FILE, ", to target file ", out_file)
    item_df = pd.read_csv(ITEMS_FILE, sep='|', header=None, encoding="ISO-8859-1")
    item_df = item_df.drop([3, 4], axis=1)
    item_df.columns = [IID, 'i_title', 'i_year',
                       'i_Other', 'i_Action', 'i_Adventure', 'i_Animation', "i_Children's", 'i_Comedy',
                       'i_Crime', 'i_Documentary ', 'i_Drama ', 'i_Fantasy ', 'i_Film-Noir ',
                       'i_Horror ', 'i_Musical ', 'i_Mystery ', 'i_Romance ', 'i_Sci-Fi ',
                       'i_Thriller ', 'i_War ', 'i_Western']
    item_df['i_year'] = item_df['i_year'].apply(lambda x: int(str(x).split('-')[-1]) if pd.notnull(x) else -1)
    seps = [0, 1940, 1950, 1960, 1970, 1980, 1985] + list(range(1990, int(item_df['i_year'].max() + 2)))
    print("total year seps: ", len(seps), seps)
    year_dict = {}
    for i, sep in enumerate(seps[:-1]):
        for j in range(seps[i], seps[i + 1]):
            year_dict[j] = i + 1
    item_df['i_year'] = item_df['i_year'].apply(lambda x: defaultdict(int, year_dict)[x])
    for c in item_df.columns[2:]:
        item_df[c] = item_df[c] + 1
    item_df.to_csv(out_file, index=False, sep='\t')
    return item_df

def format_all_inter(out_file, label01=True):
    print("format_all_interaction from raw file ", RATINGS_FILE, ", to target file ", out_file)
    inter_df = pd.read_csv(RATINGS_FILE, sep='\t', header=None)
    inter_df.columns = [UID, IID, LABEL, TIME]
    inter_df = inter_df.sort_values(by=TIME)
    inter_df = inter_df.drop_duplicates([UID, IID]).reset_index(drop=True)
    if label01: 
        inter_df[LABEL] = inter_df[LABEL].apply(lambda x: 1 if x > 3 else 0)
    print('label:', inter_df[LABEL].min(), inter_df[LABEL].max())
    print(Counter(inter_df[LABEL]))
    inter_df.to_csv(out_file, sep='\t', index=False)
    return inter_df

def main():
    format_user_feature(USER_FEATURE_FILE)
    format_item_feature(ITEM_FEATURE_FILE)
    format_all_inter(ALL_DATA_FILE, label01=True)

    dataset_name = 'ml100k01-1-5'
    leave_out_by_time_csv(ALL_DATA_FILE, dataset_name, leave_n=1, warm_n=5,
                          u_f=USER_FEATURE_FILE, i_f=ITEM_FEATURE_FILE)
    return

if __name__ == '__main__':
    main()