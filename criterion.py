import math
import torch
import pandas as pd
import numpy as np

from torch import nn
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

# sacred import
from sacred import Ingredient
criterion_ingredient = Ingredient('criterion')


@criterion_ingredient.config
def cfg():
    loss   = 'contrastive' # logbce / focal (default: logbce)

@criterion_ingredient.capture
def load_loss(loss):
    if loss == 'contrastive': return contrastiveLoss
    else: return contrastiveLoss

# ==========================
def contrastiveLoss(output, target, margin=2.0):
    output1, output2 = output
    euclid_dist = F.pairwise_distance(output1, output2).float()
    loss = torch.mean((1-target) * torch.pow(euclid_dist, 2)
                      + (target) * torch.pow(torch.clamp(margin - euclid_dist, min=0.0), 2))
    return loss

# ==========================
def focal_loss(output, target, alpha=0.25, gamma=2):
    x, p, t = output, output.sigmoid(), target
    pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
    w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
    w = w * (1-pt).pow(gamma)
    return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)

# ==========================
@criterion_ingredient.capture
def get_weight(label_count, weight, mu=0.5):
    total = np.sum(list(label_count.values()))
    keys = label_count.keys()
    class_weight_linear = OrderedDict()
    class_weight_log = OrderedDict()
    class_weight_flat = OrderedDict()

    for key in keys:
        score = total / float(label_count[key])
        score_log = math.log(mu * total / float(label_count[key]))
        class_weight_linear[key] = round(score, 2) if score > 1.0 else round(1.0, 2)
        class_weight_log[key]    = round(score_log, 2) if score_log > 1.0 else round(1.0, 2)
        class_weight_flat[key]   = 1

    if weight == 'log':
        class_weight = class_weight_log
    if weight == 'linear':
        class_weight = class_weight_linear
    else:
        class_weight = class_weight_flat

    return torch.FloatTensor(list(class_weight.values())).cuda()

class mean_aggregator():
    def __init__(self, loss_func):
        self.loss_func = loss_func
        self.sum = 0
        self.count = 0

    def update(self, output, target):
        self.sum += self.loss_func(output, target)
        self.count += targer.shape[0]

    def mean(self):
        _mean = self.sum / (self.count + 10e-15)
        if torch.is_tensor(_mean):
            return _mean.item()
        else:
            return _mean

class f1_macro_aggregator():
    def __init__(self, threshold, n_class):
        self.cfs_mats = [np.zeros(4) for i in range(n_class)]
        self.threshold = threshold
        self.n_class = n_class
        self.f1_scores = np.nan

    def update(self, output, target):
        self.cfs_mats, self.f1_scores = self.update_macro_f1(output, target, self.cfs_mats,
                                                             self.threshold, self.n_class)

    def mean(self):
        return np.nanmean(self.f1_scores)

    def update_macro_f1(self, output, target, cfs_mats, threshold, n_classes):
        preds = output.sigmoid().cpu() > threshold
        cfs_mats = [cfs_mats[i] + confusion_matrix(target[:, i], preds[:, i]).ravel()
                    for i in range(n_classes)]
        f1_scores = self.cal_f1_scores(cfs_mats)
        return cfs_mats, f1_scores

    def cal_f1_scores(self, cfs_mats):
        f1_scores = []
        for cfs_mat in cfs_mats:
            (tn, fp, fn, tp) = cfs_mat
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)
            f1_scores.append(f1)
        return f1_scores
