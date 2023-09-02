""" Main function for this repo. """
import argparse
import torch
from utils.misc import pprint
from utils.gpu_tools import set_gpu
from trainer.pre import PRETrainer
from trainer.cmdd import CMDD

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='res12', choices=['res12', 'wrn28'])
    parser.add_argument('--dataset', type=str, default='mini', choices=['mini', 'tiered', 'cub', 'cifar_fs'])
    parser.add_argument('--phase', type=str, default='pre', choices=['pre', 'preval', 'cmdd'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', default='1') # GPU id
    parser.add_argument('--dataset_dir', type=str, default='/data/FSL/mini')

    parser.add_argument('--pre_max_epoch', type=int, default=100)
    parser.add_argument('--pre_batch_size', type=int, default=128)
    parser.add_argument('--pre_lr', type=float, default=0.1)
    parser.add_argument('--pre_gamma', type=float, default=0.2)
    parser.add_argument('--pre_step_size', type=int, default=30)
    parser.add_argument('--pre_momentum', type=float, default=0.9)
    parser.add_argument('--pre_weight_decay', type=float, default=0.0005)
    parser.add_argument('--ret', type=int, default=3)
    parser.add_argument('--opt', type=str, default='Adam', choices=['SGD', 'Adam'])
    parser.add_argument('--metric', type=str, default='ED', choices=['ED', 'cos'])
    parser.add_argument('--nesterov', action='store_true', help='use nesterov for SGD, disable it in default')
    parser.add_argument('--scheduler_milestones', default=[60, 80], nargs='+', type=int,
                        help='milestones if using multistep learning rate scheduler')

    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--train_query', type=int, default=15)
    parser.add_argument('--val_query', type=int, default=15)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--eps', type=float, default=5e-1)
    parser.add_argument('--stopmargin', type=float, default=5e-2)
    parser.add_argument('--ld1', type=float, default=2)
    parser.add_argument('--ld2', type=float, default=0.1)
    parser.add_argument('--q_s_q', type=int, default=1)
    parser.add_argument('--pinv', type=int, default=0)

    parser.add_argument('--cls', type=str, default='lr', choices=['lr', 'knn', 'svm'])
    parser.add_argument('--coef', type=float, default=1.0)
    parser.add_argument('--update_lr', type=float, default=0.1)
    parser.add_argument('--update_step', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    # Set and print the parameters
    args = parser.parse_args()
    pprint(vars(args))
    # Set the GPU id
    set_gpu(args.gpu)

    # Set manual seed for PyTorch
    if args.seed == 0:
        print('Using random seed.')
        torch.backends.cudnn.benchmark = True
    else:
        print('Using manual seed:', args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.phase == 'pre':
        trainer = PRETrainer(args)
        trainer.train()
    elif args.phase == 'preval':
        trainer = PRETrainer(args)
        trainer.eval()
    elif args.phase == 'cmdd':
        trainer = CMDD(args)
        trainer.eval()
    else:
        raise ValueError('Please set correct phase.')
