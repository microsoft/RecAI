# coding=utf-8

import os
import gc
import copy
import torch
import logging
import numpy as np
from tqdm import tqdm
from time import time
from torch.utils.data import Dataset, DataLoader
from utils import utils
from utils.global_p import *

class BaseRunner(object):
    def __init__(self, args, metrics='RMSE'):
        self.optimizer_name = args.optimizer
        self.lr = args.lr
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.num_workers = args.num_workers
        self.dropout = args.dropout
        self.no_dropout = 0.0
        self.l2_weight = args.l2
        self.l2s_weight = args.l2s
        self.l2_bias = args.l2_bias
        self.grad_clip = args.grad_clip
        self.rank = args.rank

        # Convert metrics to list of str
        self.metrics = metrics.lower().split(',')
        self.check_epoch = args.check_epoch
        self.early_stop = args.early_stop
        self.time = None

        # Used to record the evaluation measures of training, validation and testing set in each round
        self.train_results, self.valid_results, self.test_results = [], [], []
    
    def _build_optimizer(self, model):
        """
        Create the optimizer
        :param model: model
        :return: optimizer
        """
        weight_p, bias_p = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        if self.l2_bias == 1:  # Trick ?
            optimize_dict = [{'params': weight_p + bias_p, 'weight_decay': self.l2_weight}]
        else:
            optimize_dict = [{'params': weight_p, 'weight_decay': self.l2_weight},
                             {'params': bias_p, 'weight_decay': 0.0}]

        optimizer_name = self.optimizer_name.lower()
        if optimizer_name == 'gd':
            logging.info("Optimizer: GD")
            optimizer = torch.optim.SGD(optimize_dict, lr=self.lr)
        elif optimizer_name == 'adagrad':
            logging.info("Optimizer: Adagrad")
            optimizer = torch.optim.Adagrad(optimize_dict, lr=self.lr)
        elif optimizer_name == 'adam':
            logging.info("Optimizer: Adam")
            optimizer = torch.optim.Adam(optimize_dict, lr=self.lr)
        else:
            logging.error("Unknown Optimizer: " + self.optimizer_name)
            assert self.optimizer_name in ['GD', 'Adagrad', 'Adam']
            optimizer = torch.optim.SGD(optimize_dict, lr=self.lr)
        return optimizer

    def _check_time(self, start=False):
        """
        Record the time, self.time records [starting time, time of last step]
        :param start: if or not to start time counting
        :return: the time to reach current position in the previous step
        """
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def batches_add_control(self, batches, train):
        """
        Add some control information into all batches, such as DROPOUT
        :param batches: list of all batches, produced by DataProcessor
        :param train: if or not this is training stage
        :return: list of all batches
        """
        for batch in batches:
            batch[TRAIN] = train
            batch[DROPOUT] = self.dropout if train else self.no_dropout
        return batches

    def train_step(self, model, batch_data):
        with torch.no_grad():
            for key in batch_data.keys():
                if key in [UID, IID, X, Y, C_HISTORY, C_HISTORY_NEG]:
                    batch_data[key] = batch_data[key].cuda()
        batch_data[RANK] = self.rank
        output_dict = model.forward(batch_data)
        return output_dict

    def fit(self, model, data, data_processor, epoch=-1):
        gc.collect()
        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        batches = data_processor.prepare_batches(data, self.batch_size, train=True, model=model)
        batches = self.batches_add_control(batches, train=True)
        
        model.train()
        batch_size = self.batch_size if self.rank == 0 else self.batch_size * 2
        accumulate_size, prediction_list, sample_id_list, output_dict = 0, [], [], None
        loss_list, loss_l2_list = [], []
        for i, batch_data in \
                tqdm(list(enumerate(batches)), leave=False, desc='Epoch %5d' % (epoch + 1), ncols=100, mininterval=1):
            accumulate_size += len(batch_data[Y])
            model.optimizer.zero_grad()
            output_dict = self.train_step(model, batch_data)
            l2 = output_dict[LOSS_L2]
            loss = output_dict[LOSS] + l2 * self.l2s_weight
            loss.backward()
            loss_list.append(loss.detach().cpu().data.numpy())
            loss_l2_list.append(l2.detach().cpu().data.numpy())
            prediction_list.append(output_dict[PREDICTION].detach().cpu().data.numpy()[:self.batch_size])
            sample_id_list.append(batch_data[SAMPLE_ID][:self.batch_size])
            if self.grad_clip > 0:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
                torch.nn.utils.clip_grad_value_(model.parameters(), self.grad_clip)
            if accumulate_size >= batch_size:
                model.optimizer.step()
                accumulate_size = 0
        if accumulate_size > 0:
            model.optimizer.step()
            accumulate_size = 0

        # for name, parms in model.named_parameters():  
        #     print('-->name:', name, "-->grad_requirs:", parms.requires_grad, "-->grad: ", parms.grad)
        
        model.eval()
        gc.collect()

        predictions = np.concatenate(prediction_list)
        sample_ids = np.concatenate(sample_id_list)
        reorder_dict = dict(zip(sample_ids, predictions))
        predictions = np.array([reorder_dict[i] for i in data[SAMPLE_ID]])
        return predictions, output_dict, np.mean(loss_list), np.mean(loss_l2_list)
 
    def train(self, model, data_processor):
        # Obtain the training, validation and testing data, epoch=-1 no shuffling
        train_data = data_processor.get_train_data(epoch=-1, model=model)
        valid_data = data_processor.get_valid_data(model=model)
        test_data = data_processor.get_test_data(model=model) if data_processor.unlabel_test == 0 else None
        # Record start time
        self._check_time(start=True)  

        init_train = self.evaluate(model, train_data, data_processor) if train_data is not None else [-1.0] * len(self.metrics)
        init_valid = self.evaluate(model, valid_data, data_processor) if valid_data is not None else [-1.0] * len(self.metrics)
        init_test = self.evaluate(model, test_data, data_processor) if test_data is not None else [-1.0] * len(self.metrics)
        logging.info("Init: \t train= %s validation= %s test= %s [%.1f s] " % (
            utils.format_metric(init_train), utils.format_metric(init_valid), utils.format_metric(init_test),
            self._check_time()) + ','.join(self.metrics))
        
        try:
            all_training_times = []
            for epoch in range(self.epoch):
                self._check_time()
                
                epoch_train_data = data_processor.get_train_data(epoch=epoch, model=model)
                train_predictions, last_batch, mean_loss, mean_loss_l2 = \
                    self.fit(model, epoch_train_data, data_processor, epoch=epoch)

                if self.check_epoch > 0 and (epoch == 1 or epoch % self.check_epoch == 0):
                    last_batch['mean_loss'] = mean_loss
                    last_batch['mean_loss_l2'] = mean_loss_l2
                    self.check(model, last_batch)
                training_time = self._check_time()
                all_training_times.append(training_time)

                train_result = [mean_loss] + model.evaluate_method(train_predictions, train_data, metrics=['rmse'])[0]
                valid_result = self.evaluate(model, valid_data, data_processor) \
                    if valid_data is not None else [-1.0] * len(self.metrics)
                test_result = self.evaluate(model, test_data, data_processor) \
                    if test_data is not None and data_processor.unlabel_test == 0 else [-1.0] * len(self.metrics)
                testing_time = self._check_time()

                self.train_results.append(train_result)
                self.valid_results.append(valid_result)
                self.test_results.append(test_result)

                logging.info("Epoch %5d [%.1f s]\t train= %s validation= %s test= %s [%.1f s] "
                             % (epoch + 1, training_time, utils.format_metric(train_result),
                                utils.format_metric(valid_result), utils.format_metric(test_result),
                                testing_time) + ','.join(self.metrics))
                
                if utils.best_result(self.metrics[0], self.valid_results) == self.valid_results[-1]:
                    model.save_model()
                
                if self.eva_termination(model) and self.early_stop == 1:
                    logging.info("Early stop at %d based on validation result." % (epoch + 1))
                    break
            logging.info("Avg Train Time [%.1f s]\t Max Train Time [%.1f s]\t Min Train Time [%.1f s]"
                             % (np.mean(all_training_times), max(all_training_times), min(all_training_times)))

        except KeyboardInterrupt:
            logging.info("Early stop manually")
            save_here = input("Save here? (1/0) (default 0):")
            if str(save_here).lower().startswith('1'):
                model.save_model()

        # Find the best validation result across iterations
        best_valid_score = utils.best_result(self.metrics[0], self.valid_results)
        best_epoch = self.valid_results.index(best_valid_score)
        logging.info("Best Iter(validation)= %5d\t train= %s valid= %s test= %s total time [%.1f s] "
                     % (best_epoch + 1,
                        utils.format_metric(self.train_results[best_epoch]),
                        utils.format_metric(self.valid_results[best_epoch]),
                        utils.format_metric(self.test_results[best_epoch]),
                        self.time[1] - self.time[0]) + ','.join(self.metrics))
        best_test_score = utils.best_result(self.metrics[0], self.test_results)
        best_epoch = self.test_results.index(best_test_score)
        logging.info("Best Iter(test)= %5d\t train= %s valid= %s test= %s total time [%.1f s] "
                     % (best_epoch + 1,
                        utils.format_metric(self.train_results[best_epoch]),
                        utils.format_metric(self.valid_results[best_epoch]),
                        utils.format_metric(self.test_results[best_epoch]),
                        self.time[1] - self.time[0]) + ','.join(self.metrics))
        model.load_model()

    def predict(self, model, data, data_processor):
        """
        Predict, not training
        :param model: model
        :param data: data dict，produced by the self.get_*_data() and self.format_data_dict() function of DataProcessor
        :param data_processor: DataProcessor instance
        :return: prediction the concatenated np.array
        """
        gc.collect()
        batches = data_processor.prepare_batches(data, self.eval_batch_size, train=False, model=model)
        batches = self.batches_add_control(batches, train=False)

        model.eval()
        predictions, sample_ids = [], []
        for batch_data in tqdm(batches, leave=False, ncols=100, mininterval=1, desc='Predict'):
            with torch.no_grad():
                for key in batch_data.keys():
                    if key in [UID, IID, X, Y, C_HISTORY, C_HISTORY_NEG]:
                        batch_data[key] = batch_data[key].cuda()
            prediction = model.predict(batch_data)[PREDICTION]
            predictions.append(prediction.detach().cpu().data.numpy())
            sample_ids.append(batch_data[SAMPLE_ID])
            
        predictions = np.concatenate(predictions)
        sample_ids = np.concatenate(sample_ids)
        reorder_dict = dict(zip(sample_ids, predictions))
        predictions = np.array([reorder_dict[i] for i in data[SAMPLE_ID]])
        gc.collect()

        return predictions

    def evaluate(self, model, data, data_processor, metrics=None):
        """
        evaluate the model performance
        :param model: model
        :param test_data: data dict
        :param metrics: list of str
        :return: list of float, each corresponding to a metric
        """
        if metrics is None:
            metrics = self.metrics
        predictions = self.predict(model, data, data_processor)
        return model.evaluate_method(predictions, data, metrics=metrics)[0]

    def get_ui_vectors(self, model, data, data_processor):
        gc.collect()
        batches = data_processor.prepare_batches(data, self.eval_batch_size, train=False, model=model)
        batches = self.batches_add_control(batches, train=False)

        model.eval()
        user_vectors = torch.zeros((model.user_num, model.u_vector_size))
        item_vectors = torch.zeros((model.item_num, model.i_vector_size))
        for batch_data in tqdm(batches, leave=False, ncols=100, mininterval=1, desc='Predict'):
            with torch.no_grad():
                for key in batch_data.keys():
                    if key in [UID, IID, X, Y, C_HISTORY, C_HISTORY_NEG]:
                        batch_data[key] = batch_data[key].cuda()
            u_vectors, i_vectors = model.get_ui_vectors(batch_data)
            user_vectors[batch_data[UID]] = u_vectors.detach().cpu()
            item_vectors[batch_data[IID]] = i_vectors.detach().cpu()
        gc.collect()

        print("user vectors: ", user_vectors, model.uid_embeddings.weight)

        return user_vectors, item_vectors

    def check(self, model, out_dict):
        check = out_dict
        logging.info(os.linesep)
        for i, t in enumerate(check[CHECK]):
            d = np.array(t[1].detach().cpu())
            logging.info(os.linesep.join([t[0] + '\t' + str(d.shape), np.array2string(d, threshold=20)]) + os.linesep)

        loss, l2 = check['mean_loss'], check['mean_loss_l2']
        logging.info('mean loss = %.4f, l2 = %.4f, %.4f' % (loss, l2 * self.l2_weight, l2 * self.l2s_weight))
    
    def eva_termination(self, model):
        """
        Check if or not to stop training, based on validation set
        :param model: model
        :return: if or not to stop training
        """
        metric = self.metrics[0]
        valid = self.valid_results
        # If has been trained for over 20 rounds, and evaluation measure is the smaller the better, and the measure has been non-desceasing for five rounds
        if len(valid) > 20 and metric in utils.LOWER_METRIC_LIST and utils.strictly_increasing(valid[-5:]):
            return True
        # If has been trained for over 20 rounds, and evaluation measure is the larger the better, and the measure has been non-increasing for five rounds
        elif len(valid) > 20 and metric not in utils.LOWER_METRIC_LIST and utils.strictly_decreasing(valid[-5:]):
            return True
        # It has been more than 20 rounds from the best result
        elif len(valid) - valid.index(utils.best_result(metric, valid)) > 20:
            return True
        return False