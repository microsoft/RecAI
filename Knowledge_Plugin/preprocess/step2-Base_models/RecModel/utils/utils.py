# coding=utf-8
import logging
import numpy as np
import torch
from utils.global_p import *
import os
import inspect

LOWER_METRIC_LIST = ["rmse", 'mae']


def format_metric(metric):
    """
    Convert the evaluation measures into str, keep four decimal places for float
    :param metric:
    :return:
    """
    # print(metric, type(metric))
    if type(metric) is not tuple and type(metric) is not list:
        metric = [metric]
    format_str = []
    if type(metric) is tuple or type(metric) is list:
        for m in metric:
            # print(type(m))
            if type(m) is float or type(m) is np.float or type(m) is np.float32 or type(m) is np.float64:
                format_str.append('%.4f' % m)
            elif type(m) is int or type(m) is np.int or type(m) is np.int32 or type(m) is np.int64:
                format_str.append('%d' % m)
    return ','.join(format_str)


def shuffle_in_unison_scary(data):
    """
    shuffle the contents of the dict of whole dataset
    :param data:
    :return:
    """
    rng_state = np.random.get_state()
    for d in data:
        np.random.set_state(rng_state)
        np.random.shuffle(data[d])
    return data


def best_result(metric, results_list):
    """
    Compute the best result in a list of results
    :param metric:
    :param results_list:
    :return:
    """
    if type(metric) is list or type(metric) is tuple:
        metric = metric[0]
    if metric in LOWER_METRIC_LIST:
        return min(results_list)
    return max(results_list)


def strictly_increasing(l):
    """
    Test if monotonically increasing
    :param l:
    :return:
    """
    return all(x < y for x, y in zip(l, l[1:]))


def strictly_decreasing(l):
    """
    Test if monotonically decreasing
    :param l:
    :return:
    """
    return all(x > y for x, y in zip(l, l[1:]))


def non_increasing(l):
    """
    Test if monotonically non-increasing
    :param l:
    :return:
    """
    return all(x >= y for x, y in zip(l, l[1:]))


def non_decreasing(l):
    """
    Test if monotonically non-decreasing
    :param l:
    :return:
    """
    return all(x <= y for x, y in zip(l, l[1:]))


def monotonic(l):
    """
    Test if monotonic
    :param l:
    :return:
    """
    return non_increasing(l) or non_decreasing(l)


def numpy_to_torch(d, gpu=True, requires_grad=True):
    """
    Convert numpy array to pytorch tensor, if there is gpu then put into gpu
    :param d:
    :param gpu: whether put tensor to gpu
    :param requires_grad: whether the tensor requires grad
    :return:
    """
    t = torch.from_numpy(d)
    if d.dtype is np.float:
        t.requires_grad = requires_grad
    if gpu:
        t = tensor_to_gpu(t)
    return t


def tensor_to_gpu(t):
    if torch.cuda.device_count() > 0:
        t = t.cuda()
    return t


def get_init_paras_dict(class_name, paras_dict):
    base_list = inspect.getmro(class_name)
    paras_list = []
    for base in base_list:
        paras = inspect.getfullargspec(base.__init__)
        paras_list.extend(paras.args)
    paras_list = sorted(list(set(paras_list)))
    out_dict = {}
    for para in paras_list:
        if para == 'self':
            continue
        out_dict[para] = paras_dict[para]
    return out_dict


def check_dir_and_mkdir(path):
    if os.path.basename(path).find('.') == -1 or path.endswith('/'):
        dirname = path
    else:
        dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        print('make dirs:', dirname)
        os.makedirs(dirname)
    return
