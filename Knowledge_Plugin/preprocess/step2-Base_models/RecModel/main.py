# coding=utf-8

import os
import sys
import copy
import json
import torch
import pickle
import logging
import argparse
import datetime
import numpy as np

from models.BaseModel import BaseModel
from models.RecModel import RecModel
from models.BiasedMF import BiasedMF
from models.AttributeMF import AttributeMF
from models.DMFModel import DMFModel
from models.NeuMFModel import NeuMFModel
from models.DeepModel import DeepModel
from data_loaders.DataLoader import DataLoader
from data_processors.DataProcessor import DataProcessor

from runners.BaseRunner import BaseRunner

from utils import utils
from utils.global_p import *

def init_parser():
    # init args
    parser = argparse.ArgumentParser(description='Recommendation Model', add_help=False)
    parser.add_argument('--rank', type=int, default=1,
                        help='1=bpr ranking, 2=bceloss, 0=rating/click')
    parser.add_argument('--data_loader', type=str, default='DataLoader',
                        help='Choose data_loader')
    parser.add_argument('--data_processor', type=str, default='DataProcessor',
                        help='Choose data_processor')
    parser.add_argument('--model_name', type=str, default='RecModel',
                        help='Choose model to run.')
    parser.add_argument('--runner_name', type=str, default='BaseRunner',
                        help='Choose runner')
    return parser

def parse_global_args(parser):
    """
    Global command-line parameters
    :param parser:
    :return:
    """
    parser.add_argument('--gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES')
    parser.add_argument('--verbose', type=int, default=logging.INFO,
                        help='Logging Level, 0, 10, ..., 50')
    parser.add_argument('--log_file', type=str, default=os.path.join(LOG_DIR, 'log.txt'),
                        help='Logging file path')
    parser.add_argument('--result_file', type=str, default=os.path.join(RESULT_DIR, 'result.npy'),
                        help='Result file path')
    parser.add_argument('--random_seed', type=int, default=DEFAULT_SEED,
                        help='Random seed of numpy and tensorflow.')
    parser.add_argument('--train', type=int, default=1,
                        help='To train the model or not.')
    return parser

def parse_dataloader_args(parser):
    """
    Command-line parameters of the data loader related to the dataset
    :param parser:
    :return:
    """
    parser.add_argument('--path', type=str, default="../data/",
                        help='Input data dir.')
    parser.add_argument('--dataset', type=str, default='ml100k01-1-5',
                        help='Choose a dataset.')
    parser.add_argument('--sep', type=str, default=SEP,
                        help='sep of csv file.')
    parser.add_argument('--seq_sep', type=str, default=SEQ_SEP,
                        help='sep of sequences in csv file.')
    parser.add_argument('--label', type=str, default=LABEL,
                        help='name of dataset label column.')
    parser.add_argument('--drop_neg', type=int, default=1,
                        help='whether drop all negative samples when ranking')
    return parser

def parse_dataprocessor_args(parser):
    """
    Command-line parameters to generate batches in data processing
    :param parser:
    :return:
    """
    parser.add_argument('--test_sample_n', type=int, default=100,
                        help='Negative sample num for each instance in test/validation set when ranking.')
    parser.add_argument('--train_sample_n', type=int, default=1,
                        help='Negative sample num for each instance in train set when ranking.')
    parser.add_argument('--sample_un_p', type=float, default=1.0,
                        help='Sample from neg/pos with 1-p or unknown+neg/pos with p.')
    parser.add_argument('--unlabel_test', type=int, default=0,
                        help='If the label of test is unknown, do not sample neg of test set.')
    parser.add_argument('--num_workers', type=int, default=8, 
                        help='Number of workers for preprocessing')
    return parser

def parse_historydp_args(parser):
    parser.add_argument('--use_his', type=int, default=0,
                        help='Whether use history interactions.')
    parser.add_argument('--all_his', type=int, default=0,
                        help='Append all history in the training set')
    parser.add_argument('--max_his', type=int, default=-1,
                        help='Max history length. All his if max_his <= 0')
    parser.add_argument('--neg_his', type=int, default=0,
                        help='Whether keep negative interactions in the history')
    parser.add_argument('--neg_column', type=int, default=0,
                        help='Whether keep negative interactions in the history as a single column')
    parser.add_argument('--sparse_his', type=int, default=0,
                        help='Whether use sparse representation of user history.')
    parser.add_argument('--sup_his', type=int, default=0,
                        help='If sup_his > 0, supplement history list with 0')
    parser.add_argument('--drop_first', type=int, default=1,
                        help='If drop_first > 0, drop the first user interacted item with no previous history')
    return parser

def parse_model_args(parser, model_name='BaseModel'):
    """
    Command-line parameters of the model
    :param parser:
    :param model_name: model name
    :return:
    """
    parser.add_argument('--loss_sum', type=int, default=1,
                        help='Reduction of batch loss 1=sum, 0=mean')
    parser.add_argument('--model_path', type=str,
                        default=os.path.join(MODEL_DIR, '%s/%s.pt' % (model_name, model_name)),
                        help='Model save path.')
    parser.add_argument('--u_vector_size', type=int, default=64,
                            help='Size of user vectors.')
    parser.add_argument('--i_vector_size', type=int, default=64,
                            help='Size of item vectors.')
    
    ## params used in attribute based models
    parser.add_argument('--f_vector_size', type=int, default=64,
                        help='Size of feature vectors.')
    parser.add_argument('--layers', type=str, default='[64]',
                        help="Size of each layer.")  # DeepModel
    parser.add_argument('--dropouts', type=str, default='[0.2, 0.2]')  # NFMModel
    parser.add_argument('--mlp_dims', type=str, default='[64]')  # NFMModel
    parser.add_argument('--attention_size', type=int, default=64,
                        help='Attention size for the AFM model.')

    ## params for GRU4Rec
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Size of hidden vectors in GRU.')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of GRU layers.')
    parser.add_argument('--p_layers', type=str, default='[64]',
                        help="Size of each layer.")
    parser.add_argument('--neg_emb', type=int, default=1,
                        help="Whether use negative interaction embeddings.")
    parser.add_argument('--neg_layer', type=str, default='[]',
                        help="Whether use a neg_layer to transfer negative interaction embeddings. "
                            "[] means using -v. It is ignored when neg_emb=1")

    ## params for Trans4Rec
    parser.add_argument('--n_head', type=int, default=8,
                        help='Number of heads in multi-head attention.')
    return parser

def parse_runner_args(parser):
    """
    Command-line parameters to run the model
    :param parser:
    :return:
    """
    parser.add_argument('--load', type=int, default=0,
                        help='Whether load model and continue to train')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--check_epoch', type=int, default=1,
                        help='Check every epochs.')
    parser.add_argument('--early_stop', type=int, default=0,
                        help='whether to early-stop.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size during training.')
    parser.add_argument('--eval_batch_size', type=int, default=128 * 128,
                        help='Batch size during testing.')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout probability for each deep layer')
    parser.add_argument('--l2_bias', type=int, default=0,
                        help='Whether add l2 regularizer on bias.')
    parser.add_argument('--l2', type=float, default=1e-5,
                        help='Weight of l2_regularize in pytorch optimizer.')
    parser.add_argument('--l2s', type=float, default=0.0,
                        help='Weight of l2_regularize in loss.')
    parser.add_argument('--grad_clip', type=float, default=10,
                        help='clip_grad_value_ para, -1 means, no clip')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='optimizer: GD, Adam, Adagrad')
    parser.add_argument('--metrics', type=str, default="RMSE",
                        help='metrics: RMSE, MAE, AUC, F1, Accuracy, Precision, Recall')
    parser.add_argument('--sample_type', type=str, default='random',
                        help='sample_type: rand, pop')
    return parser

def main():
    # initialize argparse
    print("initialize argparse...")
    parser = init_parser()
    parser = parse_global_args(parser)
    parser = parse_dataloader_args(parser)
    parser = parse_dataprocessor_args(parser)
    parser = parse_historydp_args(parser)
    parser = parse_model_args(parser)
    parser = parse_runner_args(parser)
    args = parser.parse_args()
    print("running with arguments: ", args)

    # choose model, data_loader, data_processor, runner
    model_name = eval(args.model_name)
    data_loader_name = eval(args.data_loader)
    data_processor_name = eval(args.data_processor)
    runner_name = eval(args.runner_name)

    # log, model ckpt, eval result filenames
    print("set log/model ckpt/result files...")
    log_file_name = [str(args.rank) + str(args.drop_neg),
                     args.model_name, args.dataset, str(args.random_seed), args.sample_type]
    log_file_name = '_'.join([l.replace(' ', '-').replace('_', '-') for l in log_file_name])
    args.log_file = os.path.join(LOG_DIR, f"{args.model_name}/{log_file_name}.txt")
    utils.check_dir_and_mkdir(args.log_file)
    args.result_file = os.path.join(RESULT_DIR, f"{args.model_name}/{log_file_name}.npy")
    utils.check_dir_and_mkdir(args.result_file)
    args.model_path = os.path.join(MODEL_DIR, f"{args.model_name}/{log_file_name}.pt")
    utils.check_dir_and_mkdir(args.model_path)

    # logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(vars(args))
    logging.info('DataLoader: ' + args.data_loader)
    logging.info('Model: ' + args.model_name)
    logging.info('Runner: ' + args.runner_name)
    logging.info('DataProcessor: ' + args.data_processor)

    # random seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # default '0'
    logging.info("number of cuda devices: %d" % torch.cuda.device_count())

    # create data_loader
    args.load_data = True
    dl_para_dict = utils.get_init_paras_dict(data_loader_name, vars(args))
    logging.info(args.data_loader + ': ' + str(dl_para_dict))
    data_loader = data_loader_name(**dl_para_dict)

    # Need to use data_loader to append_his
    if 'all_his' in args:
        data_loader.append_his(all_his=args.all_his, max_his=args.max_his, neg_his=args.neg_his, neg_column=args.neg_column)

    # If it's top n recommendation, only keep the positive examples, negative examples are sampled during training, also, convert the label into 0/1 binary values
    if args.rank == 1:
        data_loader.label_01()
        if args.drop_neg == 1:
            data_loader.drop_neg()

    # create model
    # Generate the dataset features according to the need of the model, features are one-hot/multi-hot dimension, the max and min value of each field of the feature
    features, feature_dims, feature_min, feature_max = \
        data_loader.feature_info(include_id=model_name.include_id,
                                 include_item_features=model_name.include_item_features,
                                 include_user_features=model_name.include_user_features)
    args.feature_num, args.feature_dims = len(features), feature_dims
    args.user_feature_num = len([f for f in features if f.startswith('u_')])
    args.item_feature_num = len([f for f in features if f.startswith('i_')])
    args.context_feature_num = len([f for f in features if f.startswith('c_')])
    data_loader_vars = vars(data_loader)
    for key in data_loader_vars:
        if key not in args.__dict__:
            args.__dict__[key] = data_loader_vars[key]

    model_para_dict = utils.get_init_paras_dict(model_name, vars(args))
    logging.info(args.model_name + ': ' + str(model_para_dict))
    model = model_name(**model_para_dict)

    # init model paras
    model.apply(model.init_paras)

    # use gpu
    if torch.cuda.device_count() > 0:
        model = model.cuda()

    # create data_processor
    # args.data_loader = data_loader
    data_processor = data_processor_name(args, data_loader, model)

    # create runner
    logging.info(args.runner_name + ': ' + str({'args': 'args', 'metrics': args.metrics}))
    runner = runner_name(args, args.metrics)

    # start training / testing
    # If load > 0, load the model and continue training
    if args.load > 0:
        model.load_model()
    # If train > 0, it means training is needed, otherwise test directly
    if args.train > 0:
        runner.train(model, data_processor)
    
    # save test results
    train_result = runner.predict(model, data_processor.get_train_data(epoch=-1, model=model), data_processor)
    validation_result = runner.predict(model, data_processor.get_valid_data(model=model), data_processor)
    test_result = runner.predict(model, data_processor.get_test_data(model=model), data_processor)
    np.save(args.result_file.replace('.npy', '__train.npy'), train_result)
    np.save(args.result_file.replace('.npy', '__validation.npy'), validation_result)
    np.save(args.result_file.replace('.npy', '__test.npy'), test_result)
    logging.info('Save Results to ' + args.result_file)

    all_metrics = ['rmse', 'mae', 'auc', 'f1', 'accuracy', 'precision', 'recall']
    if args.rank == 1:
        all_metrics = ['ndcg@1', 'ndcg@5', 'ndcg@10', 'ndcg@20', 'ndcg@50', 'ndcg@100'] \
                      + ['hit@1', 'hit@5', 'hit@10', 'hit@20', 'hit@50', 'hit@100'] \
                      + ['precision@1', 'precision@5', 'precision@10', 'precision@20', 'precision@50', 'precision@100'] \
                      + ['recall@1', 'recall@5', 'recall@10', 'recall@20', 'recall@50', 'recall@100']
    results = [train_result, validation_result, test_result]
    name_map = ['Train', 'Valid', 'Test']
    datasets = [data_processor.get_train_data(epoch=-1, model=model), data_processor.get_valid_data(model=model)]
    if args.unlabel_test != 1:
        datasets.append(data_processor.get_test_data(model=model))
    for i, dataset in enumerate(datasets):
        metrics, sample_result = model.evaluate_method(results[i], datasets[i], metrics=all_metrics, error_skip=True)
        log_info = 'Test After Training on %s: ' % name_map[i]
        log_metrics = ['%s=%s' % (metric, utils.format_metric(metrics[j])) for j, metric in enumerate(all_metrics)]
        log_info += ', '.join(log_metrics)
        logging.info(os.linesep + log_info + os.linesep)
    # with open(args.result_file.replace('.npy', '.sample_result.pkl'), "wb") as fw:
    #     evaluation_dict = {}
    #     for idx, metric in enumerate(all_metrics):
    #         evaluation_dict[metric] = sample_result[idx]
    #     pickle.dump(evaluation_dict, fw)
    
    # extract user and item vectors
    if args.model_name in ["RecModel", "BiasedMF", "MLPMFModel", "GMFModel", "NeuMFModel", "DMFModel", "GRU4Rec"]:
        user_vectors, item_vectors = model.get_ui_vectors()
    elif args.model_name == "AttributeMF":
        user_vectors, item_vectors = runner.get_ui_vectors(model, data_processor.get_ui_data(model=model), data_processor)
    torch.save(user_vectors, os.path.join(MODEL_DIR, f"{args.model_name}/{log_file_name}.user_embedding.pt"))
    torch.save(item_vectors, os.path.join(MODEL_DIR, f"{args.model_name}/{log_file_name}.item_embedding.pt"))
    
    embed = [item_vectors.detach().cpu().numpy(), user_vectors[1:].detach().cpu().numpy()]
    pickle.dump(embed, open(f"../../../data/{args.dataset}/MF_embeddings_{args.sample_type}.pkl", "wb"))


    logging.info('# of params: %d' % model.total_parameters)
    # logging.info(vars(args))
    logging.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    return

if __name__ == '__main__':
    main()