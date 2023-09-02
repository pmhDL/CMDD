import os.path as osp
import os
import tqdm
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader.samplers import CategoriesSampler
from models.model import Model
from utils.misc import Averager, Timer, count_acc, compute_confidence_interval
from tensorboardX import SummaryWriter
from dataloader.dataset_loader import DatasetLoader as Dataset


class CMDD(object):
    def __init__(self, args):
        self.args = args
        self.model = Model(self.args, mode='cmdd')
        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()

    def eval(self):
        # Load test dataset
        test_set = Dataset('test', self.args)
        sampler = CategoriesSampler(test_set.label, 600, self.args.way, self.args.shot + self.args.val_query)
        loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
        test_acc_record = np.zeros((600,))
        # Set accuracy averager
        ave_acc0 = Averager()
        ave_acc1 = Averager()
        ave_acc2 = Averager()
        ave_acc3 = Averager()
        '''---------------- Generate labels for FSL ----------------'''
        # Generate labels
        label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        label = torch.arange(self.args.way).repeat(self.args.val_query)
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
            label_shot = label_shot.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)
            label_shot = label_shot.type(torch.LongTensor)

        for i, batch in enumerate(loader, 1):
            if torch.cuda.is_available():
                data, label_abs, semv = [_.cuda() for _ in batch]
            else:
                data, label_abs, semv = batch[0], batch[1], batch[2]
            k = self.args.way * self.args.shot
            data_shot, data_query = data[:k], data[k:]
            semw = semv[:k].type(data_shot.type())

            data_shot = F.normalize(data_shot, dim=1)
            data_query = F.normalize(data_query, dim=1)

            logit_q0, logit_q1, logit_q2, logit_q3 = self.model((data_shot, label_shot, semw, data_query))

            acc0 = count_acc(logit_q0, label)
            acc1 = count_acc(logit_q1, label)
            acc2 = count_acc(logit_q2, label)
            acc3 = count_acc(logit_q3, label)
            ave_acc0.add(acc0)
            ave_acc1.add(acc1)
            ave_acc2.add(acc2)
            ave_acc3.add(acc3)

            test_acc_record[i - 1] = acc3
            if i % 100 == 0:
                print('batch {}: {:.2f} {:.2f} {:.2f} {:.2f} '
                      .format(i, ave_acc0.item() * 100, ave_acc1.item() * 100, ave_acc2.item() * 100, ave_acc3.item() * 100))
        m, pm = compute_confidence_interval(test_acc_record)
        print('Test Acc {:.4f} + {:.4f}'.format(m*100, pm*100))
