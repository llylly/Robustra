
import os
import torch
import numpy as np
import scipy.io as sio

best_ones = dict()

def fname_parse(fname):
    dataset, epsilon, model, prebound = fname.split('_')
    epsilon = float(epsilon)
    prebound = float(prebound[:-4])
    return dataset, epsilon, model, prebound

if __name__ == '__main__':
    for fname in os.listdir('final_models'):
        if fname.endswith('.pth'):
            dataset, epsilon, model, prebound = fname_parse(fname)
            if (dataset, epsilon, model) not in best_ones or prebound < best_ones[(dataset, epsilon, model)]:
                best_ones[(dataset, epsilon, model)] = prebound


    for name in os.listdir('final_models'):
        if name.endswith('.pth'):
            dataset, epsilon, model, prebound = fname_parse(name)
            if prebound > best_ones[(dataset, epsilon, model)] + 1e-6:
                continue
            # only parse the largest models
            print(dataset, epsilon, model, prebound)
            if (dataset, epsilon, model) in [('MNIST', 0.3, 'small'), ('MNIST', 0.3, 'large'), ('MNIST', 0.1, 'small'), ('MNIST', 0.1, 'large'), ('CIFAR10', 0.1394, 'small'), ('CIFAR10', 0.0347, 'large'),
                                             ('CIFAR10', 0.0347, 'resnet'), ('CIFAR10', 0.0347, 'small'), ('CIFAR10', 0.1394, 'resnet'), ('MNIST', 0.1, 'base'), ('CIFAR10', 0.1394, 'large')]:
                raw = torch.load('final_models/' + name, map_location='cpu')['state_dict']
                trans = dict()
                for k in raw:
                    v = raw[k].numpy()
                    if np.array(v).ndim == 4:
                        # conv layer kernel, permute to correspond MILP configs.
                        v = np.transpose(v, (2, 3, 1, 0))
                    elif np.array(v).ndim == 2:
                        # FC layer weight, permute to correspond MILP configs
                        v = np.transpose(v, (1, 0))
                    trans[str(k).replace('.', '/')] = v
                    print(str(k).replace('.', '/'), v.shape)
                sio.savemat('final_models/' + name[:-4] + '.mat', mdict=trans)