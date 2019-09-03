import sys
sys.path.append('.')

import argparse
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
plt.rcParams['hatch.linewidth'] = 0.4

import os
import time
import setproctitle

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from torch.autograd import Variable


from examples.transfer_mnist import select_model
from convex_adversarial import robust_loss, robust_loss_transfer
import examples.problems as pblm
from examples.trainer import AverageMeter


def draw_cosine_approach(natural_test='motivation/MNIST_0.1_base_natural_test.log',
                         transfer_test='motivation/run4_MNIST_0.1_base_transfer_test.log',
                         out='stat/fig_cosine_approach.pdf'):
    series = dict()
    n_test = open(natural_test, 'r')
    lines = n_test.readlines()
    numbers = [line.split(' ') for line in lines]
    numbers = [[int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])] for x in numbers]
    series['NaturalClean'] = [x[3] for x in numbers]
    series['NaturalPGD'] = [x[4] for x in numbers]
    n_test.close()
    t_test = open(transfer_test, 'r')
    lines = t_test.readlines()
    numbers = [line.split(' ') for line in lines]
    numbers = [[int(x[0])] + [float(y) for y in x[1:]] for x in numbers]
    series['TransClean'] = [x[5] for x in numbers]
    series['TransPGD'] = [x[8] for x in numbers]
    series['TransNaturalToIt'] = [x[6] for x in numbers]
    series['TransItToNatural'] = [x[7] for x in numbers]
    t_test.close()

    print('Series: %s' % str(series.keys()))
    epochs = np.arange(1, 21, 1)

    fig = plt.figure()
    fig.suptitle("Error Rate Curves Under PGD Attack on MNIST, $\epsilon = 0.1$", y=1.0)

    plt.grid(True)
    plt.xlim(0, 20)
    plt.ylim(0.0, 1.0)
    plt.xticks(np.arange(0, 20, 2))
    plt.xlabel('# Epochs', fontsize=14)
    plt.ylabel('Error Rate', fontsize=14)

    for tp in ['NaturalPGD', 'TransPGD', 'TransNaturalToIt', 'TransItToNatural']:
        style = {
            'NaturalPGD': ('bo-', "Natural Model $g$"),
            'TransPGD': ('ro-', "Overlap-Reducing Model $f$"),
            'TransNaturalToIt': ('c.--', "Trans. from $f$ to $g$"),
            'TransItToNatural': ('m.-.', "Trans. from $g$ to $f$")
        }
        plt.plot(epochs, series[tp], style[tp][0], label=style[tp][1])

    fig.tight_layout()
    plt.legend(prop={'size': 14})

    plt.savefig(out, dpi=300, bbox_inches='tight')


def draw_training_curve(train_log, train_t_log, test_log, title, out):
    data = list()
    for fhd in [train_log, train_t_log]:
        hd = open(fhd, 'r')
        lines = hd.readlines()
        lines = [l.split(' ') for l in lines]
        lines = [[int(l[0]), int(l[1])] + [float(y) for y in l[2:]] for l in lines]
        epoch_cluster = []
        for l in lines:
            if l[0] == len(epoch_cluster) - 1:
                assert epoch_cluster[-1][1] + 1 == l[1]
                epoch_cluster[-1] = [l[0], l[1]] + [epoch_cluster[-1][i] + l[i] for i in range(2, len(l))]
            elif l[0] == len(epoch_cluster):
                epoch_cluster.append(l)
            else:
                raise Exception("Wrong format.")
        for i in range(len(epoch_cluster)):
            ncnt = epoch_cluster[i][1] + 1
            for j in range(2, len(epoch_cluster[i])):
                epoch_cluster[i][j] /= ncnt
        hd.close()

        series = dict()
        series['TransRobustLoss'] = [x[2] for x in epoch_cluster]
        series['TransRobustErr'] = [x[3] for x in epoch_cluster]
        series['RobustLoss'] = [x[4] for x in epoch_cluster]
        series['RobustErr'] = [x[5] for x in epoch_cluster]
        series['CleanLoss'] = [x[6] for x in epoch_cluster]
        series['CleanErr'] = [x[7] for x in epoch_cluster]
        series['t'] = [-x[11] for x in epoch_cluster]
        data.append(series)

    hd = open(test_log, 'r')
    lines = hd.readlines()
    lines = [l.split(' ') for l in lines]
    lines = [[int(l[0]), int(l[1])] + [float(y) for y in l[2:]] for l in lines]
    epoch_cluster = [[], []]
    tick = 1
    for l in lines:
        if l[1] == 0:
            tick = 1 - tick
            assert l[0] == len(epoch_cluster[tick])
            epoch_cluster[tick].append(l)
        else:
            assert l[0] == len(epoch_cluster[tick]) - 1
            assert epoch_cluster[tick][-1][1] + 1 == l[1]
            epoch_cluster[tick][-1] = [l[0], l[1]] + [epoch_cluster[tick][-1][i] + l[i] for i in range(2, len(l))]
    for i in range(len(epoch_cluster)):
        for j in range(len(epoch_cluster[i])):
            ncnt = epoch_cluster[i][j][1] + 1
            for k in range(2, len(epoch_cluster[i][j])):
                epoch_cluster[i][j][k] /= ncnt
    for i in range(len(epoch_cluster)):
        now = epoch_cluster[i]
        data[i]['TestTransRobustLoss'] = [x[2] for x in now]
        data[i]['TestTransRobustErr'] = [x[3] for x in now]
        data[i]['TestRobustLoss'] = [x[4] for x in now]
        data[i]['TestRobustErr'] = [x[5] for x in now]
        data[i]['TestCleanLoss'] = [x[6] for x in now]
        data[i]['TestCleanErr'] = [x[7] for x in now]
        data[i]['TestExactRatio'] = [x[8] for x in now]
        data[i]['TestRelaxRatio'] = [x[9] for x in now]
        data[i]['TestInvalidRatio'] = [x[10] for x in now]
    hd.close()

    print('Series: %s' % (data[0].keys()))
    epochs = np.arange(1, 101, 1)

    # draw four figures
    for i in range(4):
        fig = plt.figure()
        # fig.suptitle(title[i], y=1.0)

        if i in [0, 1]:
            plt.figtext(.5, 0.92, title[i], fontsize=20, ha='center')
        else:
            plt.figtext(.5, 0.93, title[i], fontsize=28, ha='center')

        # plt.grid(True)
        plt.xticks(np.arange(0, 100, 10))
        plt.xlim(0, 100)
        if i in [0, 1]:
            plt.xlabel('Epochs', fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
        else:
            plt.xlabel('Epochs', fontsize=24)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
        if i in [1, 3]:
            plt.ylim(0.0, 1.0)
        if i in [0]:
            plt.yscale("log")

        style = {
            ('TransRobustLoss', 0): ('b-o', 'Train, Trans Robust Loss ($f$)'),
            ('TransRobustLoss', 1): ('b-x', 'Train, Trans Robust Loss ($g$)'),
            ('RobustLoss', 0): ('r-o', 'Train, Robust Loss ($f$)'),
            ('RobustLoss', 1): ('r-x', 'Train, Robust Loss ($g$)'),
            ('TestRobustLoss', 0): ('g-o', 'Robust Loss ($f$)'),
            ('TestRobustLoss', 1): ('g-x', 'Robust Loss ($g$)'),
            ('TestCleanLoss', 0): ('c-o', 'Clean Loss ($f$)'),
            ('TestCleanLoss', 1): ('c-x', 'Clean Loss ($g$)'),

            ('TransRobustErr', 0): ('b-o', 'Train, Trans Provable Error ($f$)'),
            ('TransRobustErr', 1): ('b-x', 'Train, Trans Provable Error ($g$)'),
            ('RobustErr', 0): ('r-o', 'Train, Provable Error ($f$)'),
            ('RobustErr', 1): ('r-x', 'Train, Provable Error ($g$)'),
            ('TestRobustErr', 0): ('r-o', 'Provable Error ($f$)'),
            ('TestRobustErr', 1): ('g-x', 'Provable Error ($g$)'),
            ('TestCleanErr', 0): ('r--o', 'Clean Error ($f$)'),
            ('TestCleanErr', 1): ('g--x', 'Clean Error ($g$)'),

            ('t', 0): ('b-o', 'Selected t ($f$)'),
            ('t', 1): ('b--x', 'Selected t ($g$)'),

            ('TestRelaxRatio',): ((175./255., 48./255., 117./255., 0.5), 'Loose', None, '////'),
            ('TestExactRatio',): ((130./255., 79./255., 149./255., 0.5), 'Tight', None, '-'),
            ('TestInvalidRatio',): ((92./255., 66./255., 164./255., 0.5), 'Infeasible', None, '\\\\')
        }
        if i in [0, 1, 2]:
            if i == 0:
                # things = ['TransRobustLoss', 'RobustLoss', 'TestRobustLoss', 'TestCleanLoss']
                things = ['TestRobustLoss', 'TestCleanLoss']
            elif i == 1:
                # things = ['TransRobustErr', 'RobustErr', 'TestRobustErr', 'TestCleanErr']
                things = ['TestRobustErr', 'TestCleanErr']
            elif i == 2:
                things = ['t']
            for tp in things:
                for s in [0, 1]:
                    plt.plot(epochs, data[s][tp], style[(tp, s)][0], label=style[(tp, s)][1], markersize=2, markevery=3, alpha=0.4)
        if i == 3:
            now_bottom0 = [0. for _ in epochs]
            now_upper0 = [0. for _ in epochs]
            now_bottom1 = [0. for _ in epochs]
            now_upper1 = [0. for _ in epochs]
            for tp in ['TestRelaxRatio', 'TestExactRatio', 'TestInvalidRatio']:
                now_upper0 = [x + y for x, y in zip(now_upper0, data[0][tp])]
                now_upper1 = [x + y for x, y in zip(now_upper1, data[1][tp])]
                plt.bar(epochs, [(x + y) / 2. for x, y in zip(data[0][tp], data[1][tp])],
                        bottom=[(x + y) / 2. for x, y in zip(now_bottom0, now_bottom1)], width=0.9, label=style[(tp,)][1],
                        edgecolor=style[(tp,)][2], hatch=style[(tp,)][3])
                now_bottom0 = now_upper0
                now_bottom1 = now_upper1

        fig.tight_layout()
        if i in [0, 1]:
            plt.legend(prop = {'size': 20})
            plt.subplots_adjust(left=0.10, right=0.95, bottom=0.15, top=0.9)
        else:
            plt.legend(prop = {'size': 30})
            plt.subplots_adjust(left=0.12, right=0.99, bottom=0.17, top=0.88)

        plt.savefig(out[i], dpi=300)


if __name__ == '__main__':
    # draw_cosine_approach()
    draw_training_curve('logs/mnistB_small_batch_size_50_epsilon_0.3_train.log',
                        'logs/mnistB_small_batch_size_50_epsilon_0.3_t_train.log',
                        'logs/mnistB_small_batch_size_50_epsilon_0.3_test.log',
                        title=('Loss Curve on MNIST, $\epsilon = 0.3$',
                               'Err. Curve on MNIST, $\epsilon = 0.3$',
                               '$t$ Curve',
                               'Status of Adv. Constraints if $t = 0$'
                               ),
                        out=('stat/fig_loss_curve_mnist.pdf', 'stat/fig_err_curve_mnist.pdf',
                             'stat/fig_t_curve_mnist.pdf', 'stat/fig_tightness_curve_mnist.pdf'))

