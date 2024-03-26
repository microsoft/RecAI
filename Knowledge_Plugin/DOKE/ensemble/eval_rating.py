import re
import json
import argparse
import numpy as np

from metrics4rec import mapping_evaluate_all
from TFIDF_model import TFIDF_model
from sklearn.metrics import auc as sk_auc

def _binary_clf_curve(trues, preds):
    """Calculate true and false positives per binary classification threshold

    Args:
        trues (numpy.ndarray): the true scores' list
        preds (numpy.ndarray): the predict scores' list

    Returns:
        fps (numpy.ndarray): A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]
        preds (numpy.ndarray): An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i].

    Note:
        To improve efficiency, we referred to the source code(which is available at sklearn.metrics.roc_curve)
        in SkLearn and made some optimizations.

    """
    trues = trues == 1

    desc_idxs = np.argsort(preds)[::-1]
    preds = preds[desc_idxs]
    trues = trues[desc_idxs]

    unique_val_idxs = np.where(np.diff(preds))[0]
    threshold_idxs = np.r_[unique_val_idxs, trues.size - 1]

    tps = np.cumsum(trues)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    return fps, tps

def evaluation(preds, trues):        
    fps, tps = _binary_clf_curve(trues, preds)
    if len(fps) > 2:
        optimal_idxs = np.where(
            np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True]
        )[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    if fps[-1] <= 0:
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    result = sk_auc(fpr, tpr)
    return result

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str)
    parser.add_argument('--dataset', type=str)

    return parser.parse_args()

args = parse_args()
total = 0
acc = 0
targets = []
results = []
for line in open(f"outputs/{args.dataset}/{args.method}/answer.jsonl"):
    line = json.loads(line)
    target = line["target"]
    result = line["result"]
    targets.append(target)
    results.append(result)
        
eval_result = evaluation(np.array(results), np.array(targets))
print(f'eval result: auc:{eval_result}')