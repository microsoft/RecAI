# coding=utf-8
import sys
import socket
import argparse

sys.path.insert(0, '../')
sys.path.insert(0, './')

from utils.dataset import *
from utils.global_p import *
from shutil import copyfile
from collections import defaultdict

np.random.seed(DEFAULT_SEED)
print(socket.gethostname())

split = "online_retail"
RAW_DATA = f'../../../../data/{split}/'
RATINGS_FILE = os.path.join(RAW_DATA, 'sequential_data.txt')
META_FILE = os.path.join(RAW_DATA, 'metadata.json')
TEST_NEG_FILE = os.path.join(RAW_DATA, 'negative_samples.txt')
TEST_POP_NEG_FILE = os.path.join(RAW_DATA, 'negative_samples_pop.txt')

DATA_DIR = f'../data/{split}/'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
TRAIN_DATA_FILE = os.path.join(DATA_DIR, f'{split}.train.csv')
VALID_DATA_FILE = os.path.join(DATA_DIR, f'{split}.validation.csv')
TEST_DATA_FILE = os.path.join(DATA_DIR, f'{split}.test.csv')
TEST_CAND_FILE = os.path.join(DATA_DIR, f'{split}.test_candidate.txt')
TEST_POP_CAND_FILE = os.path.join(DATA_DIR, f'{split}.test_candidate_pop.txt')

def format_all_inter(out_file, label01=True):
    print("format_all_interaction from raw file ", RATINGS_FILE, ", to target file ", out_file)
    user_items = defaultdict()
    user_dict, item_dict = dict(), dict()
    metadata = []
    for line in open(META_FILE):
        line = eval(line)
        if 'app_name' in line:
            metadata.append(line['app_name'])
        elif 'title' in line:
            metadata.append(line['title'])
        else:
            metadata.append("unknown title")

    with open(RATINGS_FILE, 'r') as f:
        for line in f:
            user, items = line.strip().split(' ', 1)
            items = items.split(' ')
            items = [int(item) for item in items]
            user_items[user] = items
            user_dict[user] = 1
            for item in items:
                item_dict[item] = metadata[item-1]

    user_df = pd.DataFrame(sorted(list(user_dict.keys())), columns=[UID])
    item_df = pd.DataFrame(sorted(list(item_dict.items()), key=lambda x:x[0]), columns=[IID, 'Title'])

    user_df.to_csv(os.path.join(DATA_DIR, f'{split}.user.csv'), sep='\t', index=False)
    item_df.to_csv(os.path.join(DATA_DIR, f'{split}.item.csv'), sep='\t', index=False)
    
    train_data, valid_data, test_data = [], [], []
    for user, items in user_items.items():
        valid_data.append([user, items[-2], 1, 0])
        test_data.append([user, items[-1], 1, 0])
        train_data.extend([[user, item, 1, 0] for item in items[:-2]])
    
    train_df = pd.DataFrame(train_data, columns=[UID, IID, LABEL, TIME])
    valid_df = pd.DataFrame(valid_data, columns=[UID, IID, LABEL, TIME])
    test_df = pd.DataFrame(test_data, columns=[UID, IID, LABEL, TIME])

    train_df.to_csv(TRAIN_DATA_FILE, sep='\t', index=False)
    valid_df.to_csv(VALID_DATA_FILE, sep='\t', index=False)
    test_df.to_csv(TEST_DATA_FILE, sep='\t', index=False)

    with open(TEST_NEG_FILE, 'r') as fr, open(TEST_CAND_FILE, 'w') as fw:
        for line in fr:
            fw.write(line)

    with open(TEST_POP_NEG_FILE, 'r') as fr, open(TEST_POP_CAND_FILE, 'w') as fw:
        for line in fr:
            fw.write(line)
    

def main():
    format_all_inter(TRAIN_DATA_FILE)
    return

if __name__ == '__main__':
    main()