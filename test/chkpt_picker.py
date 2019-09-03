import sys
sys.path.append('.')

import os
import shutil
import time
import setproctitle
import argparse

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from torch.autograd import Variable

import cleverhans
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import ProjectedGradientDescent
from cleverhans.attacks import FastGradientMethod
from cleverhans.model import CallableModelWrapper
from cleverhans.utils import AccuracyReport
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf


from examples.transfer_mnist import select_model, argparser
from examples.trainer import evaluate_robust
from convex_adversarial import robust_loss, robust_loss_transfer
import examples.problems as pblm
from examples.trainer import AverageMeter



if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', default='MNIST')
    # parser.add_argument('--model')
    # # parser.add_argument('--exact_test_batch_size', type=int)
    # parser.add_argument('--epsilon', type=float)
    # # parser.add_argument('--prefix')
    # parser.add_argument('--cuda_ids', default=None)
    #
    # args = parser.parse_args()
    #
    # if args.cuda_ids is not None:
    #     print('Setting CUDA_VISIBLE_DEVICES to {}'.format(args.cuda_ids))
    #     os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_ids

    args = argparser()
    args.prefix = args.M2_prefix

    model = select_model(args.dataset, args.model)

    # test_batch_size = args.exact_test_batch_size
    #
    # print('test base size = %d' % test_batch_size)

    weight_dict, weight_dict2 = dict(), dict()

    for filename in os.listdir('checkpoints/'):
        if filename.startswith(args.prefix + '_ep_'):
            _lst = filename.split('_')
            _epoch_num = _lst[-2]
            _err_rate = _lst[-1].rsplit('.', 1)[0]
            weight_dict[int(_epoch_num)] = float(_err_rate)
        if filename.startswith(args.prefix + '_mutual_model_ep_'):
            _lst = filename.split('_')
            _epoch_num = _lst[-2]
            _err_rate = _lst[-1].rsplit('.', 1)[0]
            weight_dict2[int(_epoch_num)] = float(_err_rate)

    print(len(weight_dict), len(weight_dict2))
    assert len(weight_dict) > 0, len(weight_dict2) > 0

    min_epoch, min_bound = -1, 1e+9
    min_epoch2, min_bound2 = -1, 1e+9
    for i in weight_dict:
        if weight_dict[i] < min_bound:
            min_epoch, min_bound = i, weight_dict[i]
    for i in weight_dict2:
        if weight_dict2[i] < min_bound2:
            min_epoch2, min_bound2 = i, weight_dict2[i]

    print('best M1: ', min_epoch, min_bound)
    print('best M2: ', min_epoch2, min_bound2)

    if min_bound < min_bound2:
        weight_path = ('checkpoints/' + args.prefix + '_ep_%03d_%.2f.pth') % (min_epoch, min_bound)
    else:
        weight_path = ('checkpoints/' + args.prefix + '_mutual_model_ep_%03d_%.2f.pth') % (min_epoch2, min_bound2)
    print('load path:', weight_path)

    shutil.copy(weight_path, 'final_models/%s_%f_%s_%.2f.pth' % (args.dataset, args.epsilon, args.model, min(min_bound, min_bound2)))

    # model.load_state_dict(torch.load(weight_path)['state_dict'])
    #
    # if args.dataset == 'MNIST':
    #     _, test_loader = pblm.mnist_loaders(test_batch_size)
    # elif args.dataset == 'CIFAR10':
    #     _, test_loader = pblm.cifar_loaders(test_batch_size)
    #
    # evaluate_robust(test_loader, model, args.epsilon, 100, res_log, 20, parallel=True)

# if __name__ == '__main__':
#     cuda_ids = 0
#     print('Setting CUDA_VISIBLE_DEVICES to {}'.format(cuda_ids))
#     os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_ids)
#
#     model = select_model('MNIST', 'small')
#
#     for weight_path in ['final_models/mnist_small_0.1_0202.pth', 'final_models/mnist_small_0.1_mutual_best_0202.pth']:
#         model.load_state_dict(torch.load(weight_path)['state_dict'])
#         _, test_loader = pblm.mnist_loaders(50)
#         res_log = open('mnist_small_0202_bound_test.txt', "a")
#         evaluate_robust(test_loader, model, 0.1, 100, res_log, 20)