import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import WeightNorm
import numpy as np
import torch.nn.functional as F
from models.res12 import Res12
from models.wrn28 import Wrn28
from utils.misc import count_acc, compute_proto_tc, Proto_loss, cosine_metric, get_cos_similar_matrix_tc, euclidean_metric, one_hot
import random
from scipy import sparse
from sklearn.cluster import KMeans
import pulp as lp
from scipy.spatial.distance import pdist, cdist


def tc_pinv(M, pin):
    if pin == 1:
        try:
            invM = torch.pinverse(M)
        except:
            invM = torch.cholesky_inverse(M)
    else:
        invM = torch.inverse(M)
    return invM


def AltOpt(X_in, Y_in, W_in, pinv, ld1, ld2, nums, eps, stopmargin):
    '''
    :param X of size (sample, dim), Y of size (sample, way), W of size (way, dim)
    '''
    I = torch.eye(X_in.size(1)).type(X_in.type())
    X_o = X_in.t()
    W_o = W_in
    Y = Y_in.t()
    for ite in range(20):
        r0 = torch.norm(Y - W_o.mm(X_o))
        W_o = Y.mm(X_o.t()).mm(tc_pinv(ld1 * I + X_o.mm(X_o.t()), pinv))
        X_o = tc_pinv(ld2 * I + (W_o.t()).mm(W_o), pinv).mm((W_o.t()).mm(Y))
        W_o = F.normalize(W_o, dim=1) / eps
        logit = X_o.t().mm(W_o.t())
        Y = logit.t()
        r = torch.norm(Y - W_o.mm(X_o))
        r_c = torch.abs(r - r0) / r0  # r_c = torch.abs(r - r0) / r
        if r_c < stopmargin:
            break
    logits_a = Y.t()
    logits_q3 = logits_a[nums:]
    return logits_q3


def intra_ter_dist(X, Y, way):
    intra = 0.0
    centers = []
    for yk in range(way):
        idk = np.where(Y == yk)[0]
        meanc = X[idk].mean(0)
        # intra = intra + cdist(X[idk], meanc.reshape(1,-1)).mean()
        intra = intra + pdist(X[idk]).mean()
        centers.append(meanc)
    intra = intra / way
    centers = np.stack(centers, axis=0)
    inter = pdist(centers).mean()
    return intra, inter


def intra_dist_class(X):
    # meanc = X.mean(0)
    # intra = cdist(X, meanc.reshape(1,-1)).mean()
    intra = pdist(X).mean()
    return intra


def classes_items(logit, logitop, label_q, X_ini, X_op, way):
    Acclist = []
    Acclist_op = []
    Intralist = []
    Intralist_op = []

    for ki in range(way):
        idd = torch.where(label_q == ki)[0]
        Acclist.append(count_acc(logit[idd], label_q[idd])*100)
        Acclist_op.append(count_acc(logitop[idd], label_q[idd])*100)
        idd = idd.cuda().data.cpu().numpy()
        intra_ini = intra_dist_class(X_ini[idd])
        Intralist.append(intra_ini)
        intra_opt = intra_dist_class(X_op[idd])
        Intralist_op.append(intra_opt)
    return Acclist, Acclist_op, Intralist, Intralist_op


def np_proto(feat, label, way):
    feat_proto = np.zeros((way, feat.shape[1]))
    for lb in np.unique(label):
        ds = np.where(label == lb)[0]
        feat_ = feat[ds]
        feat_proto[lb] = np.mean(feat_, axis=0)
    return feat_proto


def route_plan(Dij):
    K = Dij.shape[0]
    model = lp.LpProblem(name='plan_0_1', sense=lp.LpMinimize)
    x = [[lp.LpVariable("x{}{}".format(i, j), cat="Binary") for j in range(K)] for i in range(K)]
    # objective
    objective = 0
    for i in range(K):
        for j in range(K):
            objective = objective + Dij[i, j] * x[i][j]
    model += objective
    # constraints
    for i in range(K):
        in_degree = 0
        for j in range(K):
            in_degree = in_degree + x[i][j]
        model += in_degree == 1
    for i in range(K):
        out_degree = 0
        for j in range(K):
            out_degree = out_degree + x[j][i]
        model += out_degree == 1
    # solution
    model.solve(lp.apis.PULP_CBC_CMD(msg=False))

    W = np.zeros((K, K))
    i = 0
    j = 0
    for v in model.variables():
        W[i, j] = v.varValue
        j = j + 1
        if j % K == 0:
            i = i + 1
            j = 0
    return W


def recti_proto_lp_sem(clus_center, clus_sem, sem, Xs, ys, way):
    clus_sem = clus_sem.cuda().data.cpu().numpy()
    wordemb = sem[:way].cuda().data.cpu().numpy()
    # LP
    dist = ((wordemb[:, np.newaxis, :] - clus_sem[np.newaxis, :, :]) ** 2).sum(2)
    W = route_plan(dist)
    _, id = np.where(W > 0)
    # update prototypes
    prot = np_proto(Xs, ys, way)
    feat_proto = np.zeros((way, Xs.shape[1]))
    alp = 0.5
    query_center = np.zeros(clus_center.shape)
    sem_center = np.zeros(clus_sem.shape)
    for i in range(way):
        #feat_proto[i] = (prot[i] + clus_center[id[i]]) / 2
        feat_proto[i] = alp * prot[i] + (1-alp) * clus_center[id[i]]
        query_center[i] = clus_center[id[i]]
        sem_center[i] = clus_sem[id[i]]
    return feat_proto, query_center, sem_center


def recti_proto_lp_vis(Xs, ys, cls_center, way):
    proto = np_proto(Xs, ys, way)
    # LP
    dist = ((proto[:, np.newaxis, :] - cls_center[np.newaxis, :, :]) ** 2).sum(2)
    W = route_plan(dist)
    _, id = np.where(W > 0)
    # update prototypes
    feat_proto = np.zeros((way, Xs.shape[1]))
    alp = 0.5
    for i in range(way):
        #feat_proto[i] = (proto[i] + cls_center[id[i]]) / 2
        feat_proto[i] = alp * proto[i] + (1 - alp) * cls_center[id[i]]
    return feat_proto


def recti_proto_vis(Xs, ys, cls_center, way):
    proto = np_proto(Xs, ys, way)
    dist = ((proto[:, np.newaxis, :] - cls_center[np.newaxis, :, :]) ** 2).sum(2)
    id = dist.argmin(1)
    # update prototypes
    feat_proto = np.zeros((way, Xs.shape[1]))
    for i in range(way):
        feat_proto[i] = (cls_center[id[i]] + proto[i]) / 2
    return feat_proto


class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias=False)
        WeightNorm.apply(self.L, 'weight', dim=0)
        if outdim <= 200:
            self.scale_factor = 2
        else:
            self.scale_factor = 10

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        cos_dist = self.L(x_normalized)
        scores = self.scale_factor * (cos_dist)
        return scores


class Model(nn.Module):
    def __init__(self, args, mode='', num_cls=64):
        super().__init__()
        self.args = args
        self.mode = mode
        if self.args.dataset == 'cub':
            self.sdim = 312
        else:
            self.sdim = 300

        if self.args.model_type == 'res12':
            self.encoder = Res12()
            self.z_dim = 640
        elif self.args.model_type == 'wrn28':
            self.encoder = Wrn28()
            self.z_dim = 640

        if self.mode == 'pre':
            self.pre_fc = distLinear(self.z_dim, num_cls)
            self.rot_fc = nn.Linear(self.z_dim, 4)
        elif self.mode == 'cmdd':
            self.mseloss = torch.nn.MSELoss()
            self.Nete = nn.Linear(self.z_dim, self.sdim)
            self.Netd = nn.Linear(self.sdim, self.z_dim)

    def forward(self, inp):
        if self.mode == 'pre':
            return self.forward_pretrain(inp)
        elif self.mode == 'preval':
            datas, dataq = inp
            return self.forward_preval(datas, dataq)
        elif self.mode == 'cmdd':
            feat_s, label_s, sem, feat_q = inp
            return self.forward_cmdd(feat_s, label_s, sem, feat_q)
        else:
            raise ValueError('Please set the correct mode.')


    def forward_pretrain(self, inp):
        embedding = self.encoder(inp)
        logits = self.pre_fc(embedding)
        rot = self.rot_fc(embedding)
        return logits, rot


    def forward_preval(self, data_shot, data_query):
        proto = self.encoder(data_shot)
        proto = proto.reshape(self.args.shot, self.args.way, -1).mean(dim=0)
        query = self.encoder(data_query)
        if self.args.metric == 'ED':
            logitq = euclidean_metric(query, proto)
        elif self.args.metric == 'cos':
            x_mul = torch.matmul(query, proto.T)
            Normv = torch.mul(torch.norm(query, dim=1).unsqueeze(1), torch.norm(proto, dim=1).unsqueeze(0))
            logitq = torch.div(x_mul, Normv)
        return logitq


    def forward_cmdd(self, feat_s, label_s, sem, feat_q):
        nums = feat_s.size(0)
        numq = feat_q.size(0)

        proto0 = compute_proto_tc(feat_s, label_s, self.args.way)
        logits_q0 = euclidean_metric(feat_q, proto0)

        # ----------------clustering----------------
        Xq = feat_q.cuda().data.cpu().numpy()
        Xs = feat_s.cuda().data.cpu().numpy()
        ys = label_s.cuda().data.cpu().numpy()
        p_np = np_proto(Xs, ys, self.args.way)
        if self.args.shot == 1:
            clu_alg = KMeans(n_clusters=self.args.way, max_iter=1000, random_state=100)
        else:
            clu_alg = KMeans(n_clusters=self.args.way, init=p_np, max_iter=1000, random_state=100)
        yq_fit = clu_alg.fit(Xq)
        clus_center = yq_fit.cluster_centers_
        proto1 = recti_proto_lp_vis(Xs, ys, clus_center, self.args.way)
        proto1 = torch.tensor(proto1).type(feat_s.type())
        logits_q1 = cosine_metric(feat_q, proto1)
        
        # -------------- train encoder ----------------
        optimizer1 = torch.optim.Adam([
            {'params': self.Nete.parameters()},
            {'params': self.Netd.parameters()}
        ], lr=self.args.lr, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=50, gamma=0.2)

        self.Nete.train()
        self.Netd.train()
        for i in range(20):
            laten = self.Nete(feat_s)
            losslatent = self.mseloss(laten, sem)
            rec = self.Netd(laten)
            lossrec = self.mseloss(rec, feat_s)
            loss = losslatent + self.args.coef * lossrec
            optimizer1.zero_grad()
            loss.backward(retain_graph=True)
            optimizer1.step()
            lr_scheduler.step()
        self.Nete.eval()
        self.Netd.eval()
        semq = self.Nete(feat_q)

        # ------ encode cluster center to semantic space ------
        clus_center1 = torch.tensor(clus_center).type(feat_s.type())
        clus_sem = self.Nete(clus_center1)
        proto2, _, _ = recti_proto_lp_sem(clus_center, clus_sem, sem, Xs, ys, self.args.way)
        proto2 = torch.tensor(proto2).type(feat_s.type())
        logits_q2 = cosine_metric(feat_q, proto2)

        # ---------- alternative optimization -----------
        cats = torch.cat([feat_s, sem], dim=1)
        catq = torch.cat([feat_q, semq], dim=1)
        W = torch.cat([proto2, sem[:self.args.way]], dim=1)

        X = torch.cat([cats, catq], dim=0) #Y = cosine_metric(X, W)
        Yq = cosine_metric(catq, W)
        Ys = one_hot(label_s, self.args.way).type(Yq.type())
        Y = torch.cat([Ys, Yq], dim=0) + (torch.rand((nums + numq), 1).type(Yq.type()))/10

        logits_q3 = AltOpt(X, Y, W, self.args.pinv, self.args.ld1, self.args.ld2, nums, self.args.eps, self.args.stopmargin)

        return logits_q0, logits_q1, logits_q2, logits_q3