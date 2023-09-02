import os
import time
import pprint
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


def ensure_path(path):
    """The function to make log path.
    Args:
      path: the generated saving path.
    """
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

class Averager():
    """The class to calculate the average."""
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

def count_acc(logits, label):
    """The function to calculate the .
    Args:
      logits: input logits.
      label: ground truth labels.
    Return:
      The output accuracy.
    """
    pred = F.softmax(logits, dim=1).argmax(dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    return (pred == label).type(torch.FloatTensor).mean().item()

class Timer():
    """The class for timer."""
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()

def pprint(x):
    _utils_pp.pprint(x)

def compute_confidence_interval(data):
    """The function to calculate the .
    Args:
      data: input records
      label: ground truth labels.
    Return:
      m: mean value
      pm: confidence interval.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


def Proto_loss(query, proto):
    n = query.size(0)
    m = proto.size(0)
    d = query.size(1)
    assert d == proto.size(1)
    query = query.unsqueeze(1).expand(n, m, d)
    proto = proto.unsqueeze(0).expand(n, m, d)
    return -torch.norm(query - proto, p=2, dim=2)


def euclidean_metric(query, proto):
    n = query.shape[0]
    m = proto.shape[0]
    query = query.unsqueeze(1).expand(n, m, -1)
    proto = proto.unsqueeze(0).expand(n, m, -1)
    logits = -((query - proto) ** 2).sum(dim=2)
    return logits


def cosine_metric(query, proto):
    q = query.shape[0]
    p = proto.shape[0]
    que2 = query.unsqueeze(1).expand(q, p, -1)
    pro2 = proto.unsqueeze(0).expand(q, p, -1)
    logit = torch.cosine_similarity(que2, pro2, dim=2)
    return logit


def get_cos_similar_matrix_tc(v1, v2):
    num = torch.mm(v1, v2.T)
    denom = torch.norm(v1, dim=1).reshape(-1, 1) * torch.norm(v2, dim=1)
    res = num / denom
    return 0.5 + 0.5 * res


def one_hot(label, way):
    num = label.size(0)
    M = torch.zeros(num, way)
    for i in range(num):
        M[i][label[i]] = 1
    return M


def compute_proto_tc(feat, label, way):
    feat_proto = torch.zeros(way, feat.size(1))
    for lb in torch.unique(label):
        ds = torch.where(label == lb)[0]
        feat_ = feat[ds]
        feat_proto[lb] = torch.mean(feat_, dim=0)
    if torch.cuda.is_available():
        feat_proto = feat_proto.type(feat.type())
    return feat_proto


def compute_proto_np(feat, label, way):
    feat_proto = np.zeros((way, feat.shape[1]))
    for lb in np.unique(label):
        ds = np.where(label == lb)[0]
        feat_ = feat[ds]
        feat_proto[lb] = np.mean(feat_, axis=0)
    return feat_proto