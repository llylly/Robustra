import sys
sys.path.append('.')

# import waitGPU
# import setGPU
# waitGPU.wait(utilization=50, available_memory=10000, interval=60)
# waitGPU.wait(gpu_ids=[1,3], utilization=20, available_memory=10000, interval=60)

import itertools

import os

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
# cudnn.benchmark = True

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import setproctitle

import examples.problems as pblm
from examples.trainer import *
import math
import numpy as np


def select_model_mnist(m):
    if m == 'large':
        print('Pick large size model')
        model = pblm.mnist_model_large().cuda()
        _, test_loader = pblm.mnist_loaders(8)
    elif m == 'wide':
        print("Using wide model with model_factor={}".format(args.model_factor))
        _, test_loader = pblm.mnist_loaders(64 // args.model_factor)
        model = pblm.mnist_model_wide(args.model_factor).cuda()
    elif m == 'deep':
        print("Using deep model with model_factor={}".format(args.model_factor))
        _, test_loader = pblm.mnist_loaders(64 // (2 ** args.model_factor))
        model = pblm.mnist_model_deep(args.model_factor).cuda()
    elif m == '500':
        model = pblm.mnist_500().cuda()
    elif m == 'tiny':
        model = pblm.mnist_tiny().cuda()
    else:
        model = pblm.mnist_model().cuda()
    return model


def select_model_cifar10(m):
    if m == 'large':
        # raise ValueError
        model = pblm.cifar_model_large().cuda()
    elif m == 'resnet':
        model = pblm.cifar_model_resnet(N=1, factor=1).cuda()
    else:
        model = pblm.cifar_model().cuda()
    return model


def select_model(dataset, m):
    if dataset == 'MNIST':
        return select_model_mnist(m)
    elif dataset == 'CIFAR10':
        return select_model_cifar10(m)
    else:
        raise Exception('Unknown dataset')


def argparser(epochs=20,
              M1_batch_size=512, M2_batch_size=25, M2P_batch_size=50, M2P_test_batch_size=None,
              M2_test_batch_size=None,
              M2_batch_mem_lim=None,
              exact_test_batch_size=None,
              individual=0,
              max_exact_ratio=None,
              min_valid_ratio=None,
              maximize_exact=0,
              individual_rate=0.,
              favor_l1=0,
              seed=None, M1P_seed=None,
              lr=1e-3,
              M2P_lr_length=10,
              M2_lr_length=3,
              distributed=-1,
              epsilon=0.1, starting_epsilon=0.01,
              norm_train='l1', norm_test='l1', proj=50,
              momentum=0.9, weight_decay=5e-4,
              verbose=100,
              load=1,
              start_epoch=0,
              dataset='MNIST'
              ):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default=dataset)

    # optimizer settings
    parser.add_argument('--opt', default='adam')
    parser.add_argument('--momentum', type=float, default=momentum)
    parser.add_argument('--weight_decay', type=float, default=weight_decay)
    parser.add_argument('--M1_batch_size', type=int, default=M1_batch_size)
    parser.add_argument('--M2_batch_size', type=int, default=M2_batch_size)
    parser.add_argument('--M2P_batch_size', type=int, default=M2P_batch_size)
    parser.add_argument('--M2P_test_batch_size', type=int, default=-1)
    parser.add_argument('--M2_batch_mem_lim', type=int, default=M2_batch_mem_lim)
    if M2_test_batch_size is None:
        M2_test_batch_size = -1
    parser.add_argument('--M2_test_batch_size', type=int, default=M2_test_batch_size)
    if exact_test_batch_size is None:
        exact_test_batch_size = -1
    parser.add_argument('--exact_test_batch_size', type=int, default=exact_test_batch_size)
    parser.add_argument('--epochs', type=int, default=epochs)
    parser.add_argument('--lr', type=float, default=lr)

    # extra settings for transfer training
    parser.add_argument('--individual', type=int, default=0)
    parser.add_argument('--max_exact_ratio', type=float, default=1.0)
    parser.add_argument('--min_valid_ratio', type=float, default=0.0)
    parser.add_argument('--maximize_exact', type=int, default=0)
    parser.add_argument('--individual_rate', type=float, default=0.0)
    parser.add_argument('--favor_l1', type=int, default=favor_l1)

    # epsilon settings
    parser.add_argument("--epsilon", type=float, default=epsilon)
    parser.add_argument("--starting_epsilon", type=float, default=starting_epsilon)
    parser.add_argument('--schedule_length', type=int, default=20)
    parser.add_argument('--M2P_lr_length', type=int, default=M2P_lr_length)
    parser.add_argument('--M2_lr_length', type=int, default=M2_lr_length)

    # distributed
    parser.add_argument('--distributed', type=int, default=distributed)

    # projection settings
    parser.add_argument('--norm_train', default=norm_train)
    parser.add_argument('--norm_test', default=norm_test)
    parser.add_argument('--proj', default=proj)

    # model arguments
    parser.add_argument('--model', default=None)

    # other arguments
    parser.add_argument('--prefix')
    parser.add_argument('--M1_prefix')
    parser.add_argument('--M1P_prefix')
    parser.add_argument('--M2_prefix')
    parser.add_argument('--M2P_prefix')
    parser.add_argument('--real_time', action='store_true')

    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--M1P_seed', type=int, default=M1P_seed)

    parser.add_argument('--load', type=int, default=load)
    parser.add_argument('--start_epoch', type=int, default=start_epoch)

    parser.add_argument('--verbose', type=int, default=verbose)
    parser.add_argument('--cuda_ids', default=None)

    args = parser.parse_args()

    if args.M2_test_batch_size == -1:
        args.M2_test_batch_size = args.M2_batch_size
    if args.M2P_test_batch_size == -1:
        args.M2P_test_batch_size = args.M2P_batch_size

    args.load = (args.load != 0)
    args.individual = (args.individual != 0)
    assert 0.0 <= args.individual_rate <= 1.0
    if args.max_exact_ratio == 1.0:
        args.max_exact_ratio = None
    if args.min_valid_ratio == 0.0:
        args.min_valid_ratio = None
    args.maximize_exact = (args.maximize_exact != 0)

    if args.starting_epsilon is None:
        args.starting_epsilon = args.epsilon
    if args.prefix is None:
        args.prefix = 'temporary'
    if args.model is not None:
        args.prefix += '_' + args.model
    args.M1_prefix = args.prefix + '_M1'
    args.M1P_prefix = args.prefix + '_M1P'
    args.M2_prefix = args.prefix + '_M2'
    args.M2P_prefix = args.prefix + '_M2P'
    args.M2PM_prefix = args.prefix + '_M2PM'

    banned = ['opt', 'momentum', 'weight_decay', 'M1_batch_size', 'lr', 'starting_epsilon', 'schedule_length',
              'proj', 'prefix', 'M1_prefix', 'M1P_prefix', 'M2_prefix', 'M2P_prefix', 'M2PM_prefix',
              'real_time', 'verbose', 'cuda_ids', 'load', 'individual',
              'M2P_test_batch_size', 'M2_test_batch_size', 'dataset', 'M2_batch_mem_lim', 'exact_test_batch_size',
              'start_epoch', 'distributed']
    if args.maximize_exact is False:
        banned += ['maximize_exact']
    if args.favor_l1 == 0:
        banned += ['favor_l1']
    M1_banned = banned + ['M2P_batch_size', 'M2P_lr_length', 'M2_lr_length', 'M2_batch_size', 'epsilon', 'max_exact_ratio', 'min_valid_ratio', 'maximize_exact', 'individual_rate', 'norm_train', 'norm_test', 'favor_l1']
    M2P_banned = banned + ['M2_batch_size', 'max_exact_ratio', 'min_valid_ratio', 'M2_lr_length', 'M2_mutual', 'maximize_exact', 'individual', 'individual_rate', 'favor_l1']
    M2_banned = banned + ['M2P_batch_size', 'M2P_lr_length']
    for arg in sorted(vars(args)):
        if getattr(args, arg) is not None:
            addstr = '_' + arg + '_' + str(getattr(args, arg))
            if arg not in banned:
                args.prefix += addstr
            if arg not in M1_banned:
                args.M1_prefix += addstr
                args.M1P_prefix += addstr
            if arg not in M2_banned:
                args.M2_prefix += addstr
            if arg not in M2P_banned:
                args.M2P_prefix += addstr
                args.M2PM_prefix += addstr

    if args.schedule_length > args.epochs:
        raise ValueError('Schedule length for epsilon ({}) is greater then '
                         'number of epochs ({})'.format(args.schedule_length, args.epochs))

    if args.cuda_ids is not None:
        print('Setting CUDA_VISIBLE_DEVICES to {}'.format(args.cuda_ids))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_ids

    return args

if __name__ == "__main__":
    args = argparser()
    assert args.dataset in ['MNIST', 'CIFAR10']
    kwargs = {'parallel': (args.cuda_ids is not None)}

    print("saving prefix: {}".format(args.prefix))

    # if args.distributed == -1:
    #     ## ----- Train M1P -----
    #     print('----- Train M1P -----')
    #     setproctitle.setproctitle(args.M1P_prefix)
    #     model_M1P = select_model(args.dataset, args.model)
    #     if args.M1P_seed is not None:
    #         torch.manual_seed(args.M1P_seed)
    #         torch.cuda.manual_seed(args.M1P_seed)
    #     if args.load and os.path.exists(args.M1P_prefix + '_best.pth'):
    #         checkpoint = torch.load(args.M1P_prefix + '_best.pth')
    #         model_M1P.load_state_dict(checkpoint['state_dict'])
    #
    #         tmp_log = open('tmp.log', 'w')
    #         if args.dataset == 'MNIST':
    #             _, test_loader = pblm.mnist_loaders(args.M1_batch_size)
    #         elif args.dataset == 'CIFAR10':
    #             _, test_loader = pblm.cifar_loaders(args.M1_batch_size)
    #         evaluate_baseline(test_loader, model_M1P, args.epochs, tmp_log, args.verbose)
    #         tmp_log.close()
    #     else:
    #         if args.dataset == 'MNIST':
    #             train_loader, _ = pblm.mnist_loaders(args.M1_batch_size)
    #             _, test_loader = pblm.mnist_loaders(args.M1_batch_size)
    #         elif args.dataset == 'CIFAR10':
    #             train_loader, _ = pblm.cifar_loaders(args.M1_batch_size)
    #             _, test_loader = pblm.cifar_loaders(args.M1_batch_size)
    #
    #         if args.opt == 'adam':
    #             opt = optim.Adam(model_M1P.parameters(), lr=args.lr)
    #         elif args.opt == 'sgd':
    #             opt = optim.SGD(model_M1P.parameters(), lr=args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    #         lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=args.M2P_lr_length, gamma=0.5)
    #
    #         train_log = open(args.M1P_prefix + "_train.log", 'w')
    #         test_log = open(args.M1P_prefix + "_test.log", 'w')
    #
    #         best_err = 1.0
    #         for t in range(args.epochs):
    #             lr_scheduler.step(epoch=t)
    #             print('Epoch %d' % t)
    #             train_baseline(train_loader, model_M1P, opt, t, train_log, args.verbose)
    #             err = evaluate_baseline(test_loader, model_M1P, t, test_log, args.verbose)
    #             if err < best_err:
    #                 best_err = err
    #                 torch.save({
    #                     'state_dict': model_M1P.state_dict(),
    #                     'err': best_err,
    #                     'epoch': t
    #                 }, args.M1P_prefix + "_best.pth")
    #             torch.save({
    #                 'state_dict': model_M1P.state_dict(),
    #                 'err': best_err,
    #                 'epoch': t
    #             }, args.M1P_prefix + "_checkpoint.pth")
    #
    #         train_log.close()
    #         test_log.close()

        # # ----- Train M1 -----
        # print('----- Train M1 -----')
        # setproctitle.setproctitle(args.M1_prefix)
        # model_M1 = select_model(args.dataset, args.model)
        # if args.seed is not None:
        #     torch.manual_seed(args.seed)
        #     torch.cuda.manual_seed(args.seed)
        # if args.load and os.path.exists(args.M1_prefix + '_best.pth'):
        #     checkpoint = torch.load(args.M1_prefix + '_best.pth')
        #     model_M1.load_state_dict(checkpoint['state_dict'])
        #
        #     tmp_log = open('tmp.log', 'w')
        #     if args.dataset == 'MNIST':
        #         _, test_loader = pblm.mnist_loaders(args.M1_batch_size)
        #     elif args.dataset == 'CIFAR10':
        #         _, test_loader = pblm.cifar_loaders(args.M1_batch_size)
        #     evaluate_baseline(test_loader, model_M1, args.epochs, tmp_log, args.verbose)
        #     tmp_log.close()
        # else:
        #     if args.dataset == 'MNIST':
        #         train_loader, _ = pblm.mnist_loaders(args.M1_batch_size)
        #         _, test_loader = pblm.mnist_loaders(args.M1_batch_size)
        #     elif args.dataset == 'CIFAR10':
        #         train_loader, _ = pblm.cifar_loaders(args.M1_batch_size)
        #         _, test_loader = pblm.cifar_loaders(args.M1_batch_size)
        #
        #     if args.opt == 'adam':
        #         opt = optim.Adam(model_M1.parameters(), lr=args.lr)
        #     elif args.opt == 'sgd':
        #         opt = optim.SGD(model_M1.parameters(), lr=args.lr,
        #                         momentum=args.momentum,
        #                         weight_decay=args.weight_decay)
        #     lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=args.M2P_lr_length, gamma=0.5)
        #
        #     train_log = open(args.M1_prefix + "_train.log", 'w')
        #     test_log = open(args.M1_prefix + "_test.log", 'w')
        #
        #     best_err = 1.0
        #     for t in range(args.epochs):
        #         lr_scheduler.step(epoch=t)
        #         print('Epoch %d' % t)
        #         train_baseline(train_loader, model_M1, opt, t, train_log, args.verbose)
        #         err = evaluate_baseline(test_loader, model_M1, t, test_log, args.verbose)
        #         if err < best_err:
        #             best_err = err
        #             torch.save({
        #                 'state_dict': model_M1.state_dict(),
        #                 'err': best_err,
        #                 'epoch': t
        #             }, args.M1_prefix + "_best.pth")
        #         torch.save({
        #             'state_dict': model_M1.state_dict(),
        #             'err': best_err,
        #             'epoch': t
        #         }, args.M1_prefix + "_checkpoint.pth")
        #
        #     train_log.close()
        #     test_log.close()
        #
        # # ----- Train M2P & M2PM -----
        # for m_type in ['', 'M']:
        #     print('----- Train M2P%s -----' % m_type)
        #     now_prefix = args.M2P_prefix if m_type == '' else args.M2PM_prefix
        #     setproctitle.setproctitle(now_prefix)
        #     model_M2P = select_model(args.dataset, args.model)
        #     if args.load and os.path.exists(now_prefix + '_best.pth'):
        #         checkpoint = torch.load(now_prefix + '_best.pth')
        #         model_M2P.load_state_dict(checkpoint['state_dict'])
        #
        #         tmp_log = open('tmp.log', 'w')
        #         if args.dataset == 'MNIST':
        #             _, test_loader = pblm.mnist_loaders(args.M2P_test_batch_size)
        #         elif args.dataset == 'CIFAR10':
        #             _, test_loader = pblm.cifar_loaders(args.M2P_test_batch_size)
        #         evaluate_robust(test_loader, model_M2P, args.epsilon,
        #                         args.epochs, tmp_log, args.verbose, args.real_time,
        #                         norm_type=args.norm_test, proj=args.proj, bounded_input=False, **kwargs)
        #     else:
        #         if args.dataset == 'MNIST':
        #             train_loader, _ = pblm.mnist_loaders(args.M2P_batch_size)
        #             _, test_loader = pblm.mnist_loaders(args.M2P_test_batch_size)
        #         elif args.dataset == 'CIFAR10':
        #             train_loader, _ = pblm.cifar_loaders(args.M2P_batch_size)
        #             _, test_loader = pblm.cifar_loaders(args.M2P_test_batch_size)
        #
        #         if args.opt == 'adam':
        #             opt = optim.Adam(model_M2P.parameters(), lr=args.lr)
        #         elif args.opt == 'sgd':
        #             opt = optim.SGD(model_M2P.parameters(), lr=args.lr,
        #                             momentum=args.momentum,
        #                             weight_decay=args.weight_decay)
        #         lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=args.M2P_lr_length, gamma=0.5)
        #         eps_schedule = np.linspace(args.starting_epsilon,
        #                                    args.epsilon,
        #                                    args.schedule_length)
        #
        #         train_log = open(now_prefix + "_train.log", 'w+')
        #         test_log = open(now_prefix + "_test.log", 'w+')
        #
        #         best_err = 1.0
        #         for t in range(args.epochs):
        #             lr_scheduler.step(epoch=max(t-len(eps_schedule), 0))
        #             if t < len(eps_schedule) and args.starting_epsilon is not None:
        #                 epsilon = float(eps_schedule[t])
        #             else:
        #                 epsilon = args.epsilon
        #
        #             print('Epoch %d, epsilon=%f' % (t, epsilon))
        #
        #             train_robust(train_loader, model_M2P, opt, epsilon, t,
        #                          train_log, args.verbose, args.real_time,
        #                          norm_type=args.norm_train, proj=args.proj, bounded_input=False,
        #                          clip_grad=1 if args.dataset == 'CIFAR10' else None,
        #                          **kwargs)
        #             err = evaluate_robust(test_loader, model_M2P, args.epsilon,
        #                                   t, test_log, args.verbose, args.real_time,
        #                                   norm_type=args.norm_test, proj=args.proj, bounded_input=False, **kwargs)
        #
        #             if err < best_err:
        #                 best_err = err
        #                 torch.save({
        #                     'state_dict': model_M2P.state_dict(),
        #                     'err': best_err,
        #                     'epoch': t
        #                 }, now_prefix + "_best.pth")
        #
        #             torch.save({
        #                 'state_dict': model_M2P.state_dict(),
        #                 'err': err,
        #                 'epoch': t
        #             }, now_prefix + "_checkpoint.pth")
        #
        #         train_log.close()
        #         test_log.close()

    # # ----- Train M2 & M2M -----
    m2_time = time.time()

    print('----- Train M2 & M2M -----')
    setproctitle.setproctitle(args.M2_prefix)
    model_M2 = select_model(args.dataset, args.model)
    model_M2M = select_model(args.dataset, args.model)

    if args.dataset == 'MNIST':
        train_loader, _ = pblm.mnist_loaders(args.M2_batch_size)
        _, test_loader = pblm.mnist_loaders(args.M2_test_batch_size)
    elif args.dataset == 'CIFAR10':
        train_loader, _ = pblm.cifar_loaders(args.M2_batch_size)
        _, test_loader = pblm.cifar_loaders(args.M2_test_batch_size)

    eps_schedule = np.linspace(args.starting_epsilon,
                               args.epsilon,
                               args.schedule_length)

    if args.opt == 'adam':
        opt = optim.Adam(model_M2.parameters(), lr=args.lr)
    elif args.opt == 'sgd':
        opt = optim.SGD(model_M2.parameters(), lr=args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=args.M2_lr_length, gamma=0.5)

    if args.opt == 'adam':
        t_opt = optim.Adam(model_M2M.parameters(), lr=args.lr)
    elif args.opt == 'sgd':
        t_opt = optim.SGD(model_M2M.parameters(), lr=args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay)
    t_lr_scheduler = optim.lr_scheduler.StepLR(t_opt, step_size=args.M2_lr_length, gamma=0.5)

    if args.distributed in [-1, 0]:
        train_log = open(args.M2_prefix + "_train.log", 'a')
    if args.distributed in [-1, 1]:
        train_t_log = open(args.M2_prefix + "_t_train.log", 'a')
    test_log = open(args.M2_prefix + "_test.log", 'a')

    best_err1, best_err2 = 1.0, 1.0
    for t in range(args.epochs):
        lr_scheduler.step(epoch=max(t - len(eps_schedule), 0))
        t_lr_scheduler.step(epoch=max(t - len(eps_schedule), 0))
        if t < len(eps_schedule) and args.starting_epsilon is not None:
            epsilon = float(eps_schedule[t])
        else:
            epsilon = args.epsilon

        print('Epoch %d, epsilon=%f' % (t, epsilon))

        if t < args.start_epoch:
            if t == args.start_epoch - 1:
                prefix = args.M2_prefix + ("_ep_%03d_" % t)
                prefix_t = args.M2_prefix + ("_mutual_model_ep_%03d_" % t)
                print('loading %s' % prefix)
                loaded = 0
                for file in os.listdir('checkpoints/'):
                    if file.startswith(prefix):
                        ckp = torch.load('checkpoints/' + file)
                        model_M2.load_state_dict(ckp['state_dict'])
                        loaded += 1
                    if file.startswith(prefix_t):
                        ckp = torch.load('checkpoints/' + file)
                        model_M2M.load_state_dict(ckp['state_dict'])
                        loaded += 1
                assert loaded == 2
                print('Weight loaded')
            continue

        if args.distributed in [0, 1] and t > args.start_epoch:
            another_model_path = None
            another_model_err = 999.9
            for file in os.listdir('checkpoints/'):
                if not file.endswith('.pth'):
                    continue
                if args.distributed == 0 and file.startswith(args.M2_prefix + "_mutual_model_ep_%03d_" % (t-1)):
                    nowacc = float(file.rsplit('_', 1)[-1].split('.', 1)[0])
                    if nowacc < another_model_err:
                        another_model_path = file
                if args.distributed == 1 and file.startswith(args.M2_prefix + "_ep_%03d_" % (t-1)):
                    nowacc = float(file.rsplit('_', 1)[-1].split('.', 1)[0])
                    if nowacc < another_model_err:
                        another_model_path = file
            assert another_model_path is not None
            ckp = torch.load('checkpoints/' + another_model_path)
            if args.distributed == 0:
                model_M2M.load_state_dict(ckp['state_dict'])
            elif args.distributed == 1:
                model_M2.load_state_dict(ckp['state_dict'])
            print('Another weight loaded')

        if args.distributed in [-1, 0]:

            print("train M2")
            model_M2M.eval()
            model_M2.train()
            for param in model_M2.parameters():
                param.requires_grad = True
            for param in model_M2M.parameters():
                param.requires_grad = False
            robust_transfer(train_loader, model_M2, model_M2M, opt, epsilon, t,
                            train_log, args.verbose, args.real_time,
                            individual=args.individual,
                            max_exact_ratio=args.max_exact_ratio,
                            min_valid_ratio=args.min_valid_ratio,
                            maximize_exact=args.maximize_exact,
                            individual_rate=args.individual_rate,
                            norm_type=args.norm_train, proj=args.proj,
                            clip_grad=1 if args.dataset == 'CIFAR10' else None,
                            M2_batch_mem_lim=args.M2_batch_mem_lim,
                            bounded_input=False, **kwargs)

        if args.distributed in [-1, 1]:
            print("train M2M")
            model_M2M.train()
            model_M2.eval()
            for param in model_M2.parameters():
                param.requires_grad = False
            for param in model_M2M.parameters():
                param.requires_grad = True
            # alternative bound
            robust_transfer(train_loader, model_M2M, model_M2, t_opt, epsilon, t,
                            train_t_log, args.verbose, args.real_time,
                            individual=args.individual,
                            max_exact_ratio=args.max_exact_ratio,
                            min_valid_ratio=args.min_valid_ratio,
                            maximize_exact=args.maximize_exact,
                            individual_rate=args.individual_rate,
                            norm_type=args.norm_train, proj=args.proj,
                            clip_grad=1 if args.dataset == 'CIFAR10' else None,
                            M2_batch_mem_lim=args.M2_batch_mem_lim,
                            bounded_input=False, **kwargs)

        torch.cuda.empty_cache()

        # test
        model_M2M.eval()
        model_M2.eval()
        for param in model_M2M.parameters():
            param.requires_grad = False
        for param in model_M2.parameters():
            param.requires_grad = False

        # if args.distributed in [-1, 0]:
        #     print("evaluate M2")
        #     err1 = robust_transfer(test_loader, model_M2, model_M2M, None, args.epsilon, t,
        #                            test_log, args.verbose, args.real_time,
        #                            evaluate=True,
        #                            individual=False,
        #                            max_exact_ratio=None,
        #                            min_valid_ratio=None,
        #                            maximize_exact=False,
        #                            norm_type=args.norm_train, proj=args.proj,
        #                            M2_batch_mem_lim=args.M2_batch_mem_lim,
        #                            bounded_input=False, **kwargs)
        #     if err1 < best_err1:
        #         best_err1 = err1
        #         torch.save({
        #             'state_dict': model_M2.state_dict(),
        #             'err': best_err1,
        #             'epoch': t
        #         }, args.M2_prefix + "_best.pth")
        #     torch.save({
        #         'state_dict': model_M2.state_dict(),
        #         'err': err1,
        #         'epoch': t
        #     }, args.M2_prefix + "_checkpoint.pth")
        #     # save things for each epoch
        #     torch.save({
        #         'state_dict': model_M2.state_dict(),
        #         'err': err1,
        #         'epoch': t
        #     }, 'checkpoints/' + args.M2_prefix + ("_ep_%03d_%.2f.pth" % (t, err1 * 100.)))
        #
        # if args.distributed in [-1, 1]:
        #     print("evaluate M2M")
        #     err2 = robust_transfer(test_loader, model_M2M, model_M2, None, args.epsilon, t,
        #                            test_log, args.verbose, args.real_time,
        #                            evaluate=True,
        #                            individual=False,
        #                            max_exact_ratio=None,
        #                            min_valid_ratio=None,
        #                            maximize_exact=False,
        #                            norm_type=args.norm_train, proj=args.proj,
        #                            M2_batch_mem_lim=args.M2_batch_mem_lim,
        #                            bounded_input=False, **kwargs)
        #     if err2 < best_err2:
        #         torch.save({
        #             'state_dict': model_M2M.state_dict(),
        #             'err': best_err2,
        #             'epoch': t
        #         }, args.M2_prefix + "_mutual_model_best.pth")
        #     torch.save({
        #         'state_dict': model_M2M.state_dict(),
        #         'err': err2,
        #         'epoch': t
        #     }, args.M2_prefix + "_mutual_model_checkpoint.pth")
        #     # save things for each epoch
        #     torch.save({
        #         'state_dict': model_M2M.state_dict(),
        #         'err': err2,
        #         'epoch': t
        #     }, 'checkpoints/' + args.M2_prefix + ("_mutual_model_ep_%03d_%.2f.pth" % (t, err2 * 100.)))
        # torch.cuda.empty_cache()
        #
        # if args.distributed in [0, 1]:
        #     print('Get another model...')
        #     while True:
        #         find = False
        #         for file in os.listdir('checkpoints/'):
        #             if not file.endswith('.pth'):
        #                 continue
        #             if args.distributed == 0 and file.startswith(args.M2_prefix + "_mutual_model_ep_%03d_" % t):
        #                 find = True
        #                 break
        #             if args.distributed == 1 and file.startswith(args.M2_prefix + "_ep_%03d_" % t):
        #                 find = True
        #                 break
        #         if find:
        #             break
        #         time.sleep(10)
        #     print('Get it...')
        #     time.sleep(20)
        #     print('Continue to next epoch')

    print(f'Finish, elapsed time = {time.time() - m2_time}')


