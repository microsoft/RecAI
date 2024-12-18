# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pandas as pd
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="data process")
    parser.add_argument(
        "--in_seq_data", type=str, help=""
    )
    parser.add_argument(
        "--train_file", type=str, help=""
    )
    parser.add_argument(
        "--valid_file", type=str, help=""
    )
    parser.add_argument(
        "--test_file", type=str, default='', help=""
    )
    parser.add_argument(
        "--hist_file", type=str, default='', help=""
    )
    parser.add_argument(
        "--out_train_data", type=str, help=""
    )
    parser.add_argument(
        "--out_test_data", type=str, help=""
    )
    args = parser.parse_args()
    return args

def unirec_split(args):
    with open(args.in_seq_data, 'r') as f, open(args.train_file, 'w') as ftrain, open(args.valid_file, 'w') as fvalid, open(args.test_file, 'w') as ftest, open(args.hist_file, 'w') as fhist:
        ftrain.write('user_id\titem_id\n')
        fvalid.write('user_id\titem_id\n')
        ftest.write('user_id\titem_id\n')
        fhist.write('user_id\titem_seq\n')
        for idx, line in enumerate(f):
            userid, itemids = line.strip().split(' ', 1)
            itemids = itemids.split(' ')
            assert len(itemids) >= 5
            for itemid in itemids[1:-1]:
                ftrain.write('{}\t{}\n'.format(userid, itemid))
            fvalid.write('{}\t{}\n'.format(userid, itemids[-1]))
            ftest.write('{}\t{}\n'.format(userid, itemids[-1]))
            fhist.write('{}\t{}\n'.format(userid, ','.join(itemids)))

def rank_split(args):
    train=[]
    test=[]
    with open(args.in_seq_data, 'r') as f:
        for idx, line in enumerate(f):
            userid, itemids = line.strip().split(' ', 1)
            itemids = itemids.split(' ')
            for itemid in itemids[1:-1]:
                train.append([int(userid), int(itemid)])
            test.append([int(userid), int(itemids[-1])])

    train_df = pd.DataFrame(train, columns=['user_id', 'item_id'])
    test_df = pd.DataFrame(test, columns=['user_id', 'item_id'])
    train_df.to_csv(args.out_train_data, index=False, sep='\t')
    test_df.to_csv(args.out_test_data, index=False, sep='\t')

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(os.path.dirname(args.train_file), exist_ok=True)
    unirec_split(args)
    rank_split(args)