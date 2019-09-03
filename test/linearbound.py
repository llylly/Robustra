import sys
sys.path.append('.')

import argparse
import os
import torch
from torch import nn

import examples.problems as pblm
from examples.trainer import evaluate_robust

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--epsilon', type=str)
    parser.add_argument('--model')

    parser.add_argument('--cuda_ids', default=None)

    args = parser.parse_args()
    if args.cuda_ids is not None:
        print('Setting CUDA_VISIBLE_DEVICES to {}'.format(args.cuda_ids))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_ids
    return args


def select_model(dataset, model):
    assert dataset in ['MNIST', 'CIFAR10']
    if dataset == 'MNIST':
        if model == 'small':
            return pblm.mnist_model()
        elif model == 'large':
            return pblm.mnist_model_large()
        elif model == 'base':
            net = nn.Sequential(
                nn.Conv2d(1, 32, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=1),
                nn.ReLU(),
                Flatten(),
                nn.Linear(64 * 7 * 7, 1024),
                nn.ReLU(),
                nn.Linear(1024, 10)
            )
            return net
    elif dataset == 'CIFAR10':
        if model == 'small':
            return pblm.cifar_model()
        elif model == 'large':
            return pblm.cifar_model_large()
        elif model == 'resnet':
            return pblm.cifar_model_resnet(1, 1)


if __name__ == '__main__':
    args = parse_args()

    model = select_model(args.dataset, args.model)

    for fname in os.listdir('final_models'):
        if fname.endswith('.pth') and fname.startswith('%s_%s_%s_' % (args.dataset, args.epsilon, args.model)):
            print('Find weights in path: final_models/%s' % fname)
            model.load_state_dict(torch.load('final_models/' + fname)['state_dict'])
            model = model.cuda()
            if args.dataset == 'MNIST':
                _, test_loader = pblm.mnist_loaders(1)
            else:
                _, test_loader = pblm.cifar_loaders(1)
            res_log = open('linear_bound_tmp.txt', 'a')
            evaluate_robust(test_loader, model, float(args.epsilon), 100, res_log, 20)

