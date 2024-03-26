# coding=utf-8

# default paras
DEFAULT_SEED = 2018
SEP = '\t'
SEQ_SEP = ','
MAX_VT_USER = 100000  # (How many users to take at most when leaving out by time)

# # Path
DATA_DIR = '../data/'  # (Directory of the original data and pre-processed data)
DATASET_DIR = '../../../data/'  # (Directory of the properly splited dataset)
MODEL_DIR = '../model/'  # (Directory to save the model)
LOG_DIR = '../log/'  # (Directory to output logs)
RESULT_DIR = '../result/'  # (Directory to save the prediction results)
# COMMAND_DIR = '../command/'  # (Directory to save the command file used by run.py)
# LOG_CSV_DIR = '../log_csv/'  # (Directory to save the result csv file used by run.py)

# LIBREC_DATA_DIR = '../librec/data/'  # (librec original data file and pre-processed file directory)
# LIBREC_DATASET_DIR = '../librec/dataset/'  # (librec properly splited dataset directory)
# LIBREC_MODEL_DIR = '../librec/model/'  # (librec model saving directory)
# LIBREC_LOG_DIR = '../librec/log/'  # (librec log output directory)
# LIBREC_RESULT_DIR = '../librec/result/'  # (librec prediction result saving directory)
# LIBREC_COMMAND_DIR = '../librec/command/'  # (Directory to save the command file used by run_librec.py)
# LIBREC_LOG_CSV_DIR = '../librec/log_csv/'  # (Directory to save the result csv file used by run_librec.py)

# Preprocess/DataLoader
TRAIN_SUFFIX = '.train.csv'  # (suffix of training dataset)
VALIDATION_SUFFIX = '.validation.csv'  # (suffix of validation dataset)
TEST_SUFFIX = '.test.csv'  # (suffix of testing dataset)
INFO_SUFFIX = '.info.json'  # (suffix of dataset statistics information file)
USER_SUFFIX = '.user.csv'  # (suffix of user feature file)
ITEM_SUFFIX = '.item.csv'  # (suffix of item feature file)
TRAIN_POS_SUFFIX = '.train_pos.csv'  # (suffix of the training user positive interactions merged by uid)
VALIDATION_POS_SUFFIX = '.validation_pos.csv'  # (suffix of the validation user positive interactions merged by uid)
TEST_POS_SUFFIX = '.test_pos.csv'  # (suffix of the testing user positive interactions merged by uid)
TRAIN_NEG_SUFFIX = '.train_neg.csv'  # (suffix of the training user negative interactions merged by uid)
VALIDATION_NEG_SUFFIX = '.validation_neg.csv'  # (suffix of the validation user negative interactions merged by uid)
TEST_NEG_SUFFIX = '.test_neg.csv'  # (suffix of the testing user positive interactions merged by uid)
VARIABLE_SUFFIX = '.variable.csv'  # (suffix of the ProLogic variable file)
DICT_SUFFIX = '.dict.csv'
DICT_POS_SUFFIX = '.dict_pos.csv'

C_HISTORY = 'history'  # (column name of interaction history)
C_HISTORY_LENGTH = 'history_length'  # (column name of interaction history length)
C_HISTORY_NEG = 'history_neg'  # (column name of negative interaction history)
C_HISTORY_NEG_LENGTH = 'history_neg_length'  # (column name of negative interaction history length)
# C_HISTORY_POS_TAG = 'history_pos_tag'  # (tag to record if an interaction list is positive interaction 1 or negative interaction 0)

# # # DataProcessor/feed_dict
X = 'x'
Y = 'y'
LABEL = 'label'
UID = 'uid'
IID = 'iid'
IIDS = 'iids'
TIME = 'time'
RANK = 'rank'
REAL_BATCH_SIZE = 'real_batch_size'
TOTAL_BATCH_SIZE = 'total_batch_size'
TRAIN = 'train'
DROPOUT = 'dropout'
SAMPLE_ID = 'sample_id'  # (In the training/validation/testing set, number each example. This is the name of the column in data dict and feed dict)

# # # out dict
# PRE_VALUE = 'pre_value'
PREDICTION = 'prediction'  # (output the prediction)
CHECK = 'check'  # (check the intermediate results)
LOSS = 'loss'  # (output the loss)
LOSS_L2 = 'loss_l2'  # (output the l2 loss)
EMBEDDING_L2 = 'embedding_l2'  # (the l2 of the embeddings related to the current batch)
L2_BATCH = 'l2_batch'  # (batch size of the currently computed embedding l2)
