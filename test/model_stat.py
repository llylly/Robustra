
import torch

from examples.transfer_mnist import select_model


if __name__ == '__main__':
    for type in ['MNIST', 'CIFAR10']:
        for scale in ['small', 'large', 'resnet']:
            if type == 'MNIST' and scale == 'resnet':
                continue
            m = select_model(type, scale)
            print(m)
            pytorch_total_params = sum(p.numel() for p in m.parameters())
            print(type, scale, pytorch_total_params)
