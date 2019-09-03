import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

from .utils import Dense, DenseSequential
from .dual_inputs import select_input
from .dual_layers import select_layer

import warnings

# need to be positive to avoid grad jumping at p = 0
P_EPS = 1e-7

class DualNetwork(nn.Module):   
    def __init__(self, net, X, epsilon, 
                 proj=None, norm_type='l1', bounded_input=False, 
                 data_parallel=True):
        """  
        This class creates the dual network. 

        net : ReLU network
        X : minibatch of examples
        epsilon : size of l1 norm ball to be robust against adversarial examples
        alpha_grad : flag to propagate gradient through alpha
        scatter_grad : flag to propagate gradient through scatter operation
        l1 : size of l1 projection
        l1_eps : the bound is correct up to a 1/(1-l1_eps) factor
        m : number of probabilistic bounds to take the max over
        """
        super(DualNetwork, self).__init__()
        # need to change that if no batchnorm, can pass just a single example
        if not isinstance(net, (nn.Sequential, DenseSequential)): 
            raise ValueError("Network must be a nn.Sequential or DenseSequential module")
        with torch.no_grad(): 
            if any('BatchNorm2d' in str(l.__class__.__name__) for l in net): 
                zs = [X]
            else:
                zs = [X[:1]]
            nf = [zs[0].size()]
            for l in net: 
                if isinstance(l, Dense): 
                    zs.append(l(*zs))
                else:
                    zs.append(l(zs[-1]))
                nf.append(zs[-1].size())


        # Use the bounded boxes
        dual_net = [select_input(X, epsilon, proj, norm_type, bounded_input)]

        for i,(in_f,out_f,layer) in enumerate(zip(nf[:-1], nf[1:], net)): 
            dual_layer = select_layer(layer, dual_net, X, proj, norm_type,
                                      in_f, out_f, zs[i])

            # skip last layer
            # ###: because in equation (7), the last layer is not in objective
            if i < len(net)-1: 
                for l in dual_net: 
                    l.apply(dual_layer)
                dual_net.append(dual_layer)
            else: 
                self.last_layer = dual_layer

        self.dual_net = dual_net
        return 

    def forward(self, c):
        """ For the constructed given dual network, compute the objective for
        some given vector c """
        nu = [-c]
        nu.append(self.last_layer.T(*nu))
        for l in reversed(self.dual_net[1:]): 
            nu.append(l.T(*nu))
        dual_net = self.dual_net + [self.last_layer]
        
        return sum(l.objective(*nu[:min(len(dual_net)-i+1, len(dual_net))]) for
           i,l in enumerate(dual_net))


class DualNetBounds(DualNetwork): 
    def __init__(self, *args, **kwargs):
        warnings.warn("DualNetBounds is deprecated. Use the proper "
                      "PyTorch module DualNetwork instead. ")
        super(DualNetBounds, self).__init__(*args, **kwargs)

    def g(self, c):
        return self(c)


class DualNetworkDouble(nn.Module):
    def __init__(self, net, t_net, X, epsilon,
                 proj=None, norm_type='l1', bounded_input=False, favor_l1=False,
                 data_parallel=True):
        """
        This class creates the dual network.

        net : ReLU network
        t_net: Another network for transferability robust training
        X : minibatch of examples
        epsilon : size of l1 norm ball to be robust against adversarial examples
        alpha_grad : flag to propagate gradient through alpha
        scatter_grad : flag to propagate gradient through scatter operation
        l1 : size of l1 projection
        l1_eps : the bound is correct up to a 1/(1-l1_eps) factor
        m : number of probabilistic bounds to take the max over
        """
        super(DualNetworkDouble, self).__init__()

        # need to change that if no batchnorm, can pass just a single example
        if not isinstance(net, (nn.Sequential, DenseSequential)):
            raise ValueError("Network must be a nn.Sequential or DenseSequential module")
        if not isinstance(t_net, (nn.Sequential, DenseSequential)):
            raise ValueError("Transfer network must be a nn.Sequential or DenseSequential module")

        with torch.no_grad():
            if any('BatchNorm2d' in str(l.__class__.__name__) for l in net):
                zs = [X]
            else:
                zs = [X[:1]]
            if any('BatchNorm2d' in str(l.__class__.__name__) for l in t_net):
                t_zs = [X]
            else:
                t_zs = [X[:1]]

            nf = [zs[0].size()]
            t_nf = [t_zs[0].size()]

            for l in net:
                if isinstance(l, Dense):
                    zs.append(l(*zs))
                else:
                    zs.append(l(zs[-1]))
                nf.append(zs[-1].size())

            for l in t_net:
                if isinstance(l, Dense):
                    t_zs.append(l(*t_zs))
                else:
                    t_zs.append(l(t_zs[-1]))
                t_nf.append(t_zs[-1].size())

        # Use the bounded boxes
        dual_net = [select_input(X, epsilon, proj, norm_type, bounded_input, favor_l1)]
        t_dual_net = [select_input(X, epsilon, proj, norm_type, bounded_input, favor_l1)]

        for i, (in_f, out_f, layer) in enumerate(zip(nf[:-1], nf[1:], net)):
            dual_layer = select_layer(layer, dual_net, X, proj, norm_type,
                                      in_f, out_f, zs[i], transfer_training=True)
            # skip last layer
            # ###: because in equation (7), the last layer is not in objective
            if i < len(net) - 1:
                for l in dual_net:
                    l.apply(dual_layer)
                dual_net.append(dual_layer)
            else:
                self.last_layer = dual_layer

        for i, (in_f, out_f, layer) in enumerate(zip(t_nf[:-1], t_nf[1:], t_net)):
            dual_layer = select_layer(layer, t_dual_net, X, proj, norm_type,
                                      in_f, out_f, t_zs[i], transfer_training=True)
            # skip last layer
            # ###: because in equation (7), the last layer is not in objective
            if i < len(t_net) - 1:
                for l in t_dual_net:
                    l.apply(dual_layer)
                t_dual_net.append(dual_layer)
            else:
                self.t_last_layer = dual_layer

        self.dual_net = dual_net
        self.t_dual_net = t_dual_net

        return

    def forward(self, p, c, eta=None):
        """ For the constructed given dual network, compute the objective for
        some given vector c and slack variable p """
        nu = [-c]

        # t_c = p.view(p.size()[0], p.size()[1], 1) * c
        # q = [-t_c]
        q = [-c]

        nu.append(self.last_layer.T(*nu))
        q.append(self.t_last_layer.T(*q))
        for l in reversed(self.dual_net[1:]):
            nu.append(l.T(*nu))
        for l in reversed(self.t_dual_net[1:]):
            q.append(l.T(*q))
        dual_net = self.dual_net + [self.last_layer]
        t_dual_net = self.t_dual_net + [self.t_last_layer]

        if eta is not None:
            appen = torch.tensor(eta).cuda() * p
        else:
            appen = 0.

        # return sum(l.objective(*nu[:min(len(dual_net) - (i + 1) + 1, len(dual_net))]) for
        #            i, l in enumerate(dual_net[1:])) + \
        #        sum(l.objective(*q[:min(len(t_dual_net) - (i + 1) + 1, len(t_dual_net))]) for
        #                i, l in enumerate(t_dual_net[1:])) + \
        #        dual_net[0].objective(nu[-1] + q[-1]) + appen

        return sum(l.objective(*nu[:min(len(dual_net) - (i + 1) + 1, len(dual_net))]) for
                   i, l in enumerate(dual_net[1:])) + \
               p * sum(l.objective(*q[:min(len(t_dual_net) - (i + 1) + 1, len(t_dual_net))]) for
                       i, l in enumerate(t_dual_net[1:])) + \
               dual_net[0].objective(nu[-1].view(nu[-1].size(0), nu[-1].size(1), -1) + p.view(p.size(0), p.size(1), -1) * q[-1].view(q[-1].size(0), q[-1].size(1), -1)) + appen

    def forward_orig(self, c):
        """ For the constructed given dual network, compute the objective for
                some given vector c """
        nu = [-c]
        nu.append(self.last_layer.T(*nu))
        for l in reversed(self.dual_net[1:]):
            nu.append(l.T(*nu))
        dual_net = self.dual_net + [self.last_layer]

        return sum(l.objective(*nu[:min(len(dual_net) - i + 1, len(dual_net))]) for
                   i, l in enumerate(dual_net))

    def nu1(self, c):
        nu = [-c]
        nu.append(self.last_layer.T(*nu))
        for l in reversed(self.dual_net[1:]):
            nu.append(l.T(*nu))
        return nu[-1]

    def q1(self, c):
        q = [-c]
        q.append(self.t_last_layer.T(*q))
        for l in reversed(self.t_dual_net[1:]):
            q.append(l.T(*q))
        return q[-1]
    #
    # def fix_item(self, c):
    #     # loss items whose derivatives are independent of the value of p
    #     nu = [-c]
    #     q = [-c]
    #     nu.append(self.last_layer.T(*nu))
    #     q.append(self.t_last_layer.T(*q))
    #     for l in reversed(self.dual_net[1:]):
    #         nu.append(l.T(*nu))
    #     for l in reversed(self.t_dual_net[1:]):
    #         q.append(l.T(*q))
    #     dual_net = self.dual_net + [self.last_layer]
    #     t_dual_net = self.t_dual_net + [self.t_last_layer]
    #     return sum(l.objective(*nu[:min(len(dual_net) - (i + 1) + 1, len(dual_net))]) for
    #                i, l in enumerate(dual_net[1:])) +\
    #            sum(l.objective(*q[:min(len(t_dual_net) - (i + 1) + 1, len(t_dual_net))]) for
    #                i, l in enumerate(t_dual_net[1:]))

    def grad_p(self, p, c, retain_graph=False):

        p = Variable(torch.tensor(p).cuda(), requires_grad=True)
        loss = self(p, c)

        ans = torch.autograd.grad(loss.sum(), p, retain_graph=retain_graph, only_inputs=True)[0]
        return ans.cpu()


class RobustBounds(nn.Module): 
    def __init__(self, net, epsilon, **kwargs): 
        super(RobustBounds, self).__init__()
        self.net = net
        self.epsilon = epsilon
        self.kwargs = kwargs

    def forward(self, X,y): 
        num_classes = self.net[-1].out_features
        dual = DualNetwork(self.net, X, self.epsilon, **self.kwargs)
        c = Variable(torch.eye(num_classes).type_as(X)[y].unsqueeze(1) - torch.eye(num_classes).type_as(X).unsqueeze(0))
        if X.is_cuda:
            c = c.cuda()
        f = -dual(c)
        return f


def robust_loss(net, epsilon, X, y, 
                size_average=True, device_ids=None, parallel=False, **kwargs):
    if parallel:
        f = nn.DataParallel(RobustBounds(net, epsilon, **kwargs))(X,y)
    else: 
        f = RobustBounds(net, epsilon, **kwargs)(X,y)
    err = (f.max(1)[1] != y)
    if size_average: 
        err = err.sum().item()/X.size(0)
    ce_loss = nn.CrossEntropyLoss(reduce=size_average)(f, y)
    return ce_loss, err


def robust_loss_transfer(net, t_net, epsilon, X, y,
                         size_average=True, device_ids=None, parallel=False, evaluate=False,
                         individual=False,
                         max_exact_ratio=None,
                         min_valid_ratio=None,
                         maximize_exact=None,
                         individual_rate=0.0,
                         **kwargs):
    '''
    :param net:
    :param t_net:
    :param epsilon:
    :param X:
    :param y:
    :param size_average:
    :param device_ids:
    :param parallel:
    :param evaluate:
    :param max_exact_ratio: constrain maximum exact bound ratio
    :param min_valid_ratio: constrain minimum valid bound ratio, which includes relax bound and exact bound
    :param kwargs:
    :return:
    '''
    num_classes = net[-1].out_features
    num_samples = X.size()[0]

    # np.set_printoptions(threshold=np.nan)

    if not evaluate:
        if not individual:
            # same eta use for the whole batch

            # --- calculate possible value interval of slack variable p ---
            # lower bound: p_lbound, upper bound: p_ubound

            dual = DualNetworkDouble(net, t_net, X, epsilon, **kwargs)
            c = Variable(torch.eye(num_classes).type_as(X)[y].unsqueeze(1) - torch.eye(num_classes).type_as(X).unsqueeze(0))
            # c_value = c.cpu().numpy()

            nu1, q1 = dual.nu1(c), dual.q1(c)
            nu1 = nu1.view(num_samples, num_classes, -1).cpu().detach().numpy()
            q1 = q1.view(num_samples, num_classes, -1).cpu().detach().numpy()

            p_lbound = np.zeros([num_samples, num_classes], dtype=np.float32) + P_EPS
            p_ubound = np.zeros([num_samples, num_classes], dtype=np.float32)
            for i in range(num_samples):
                for j in range(num_classes):
                    _cel = (np.abs(q1[i][j]) > 1e-7)
                    if np.sum(_cel) > 0:
                        p_ubound[i][j] = max(P_EPS, np.amax(- nu1[i][j][_cel] / q1[i][j][_cel]))
                    else:
                        p_ubound[i][j] = P_EPS

            # findout gradient range w.r.t. p for the interval endpoint
            grad_l = (dual.grad_p(p_lbound, c, retain_graph=True).detach()).numpy()
            grad_r = (dual.grad_p(p_ubound, c, retain_graph=True).detach()).numpy()

            # grad_range = [[(grad_r[i][j], grad_l[i][j]) for j in range(num_classes)] for i in range(num_samples)]
            # print(grad_range)

            # determine eta by firstly assumes eta = 0 then adjust eta to satisfy the ratio constraints if they exist
            exact_n, valid_n = 0, 0
            for i in range(num_samples):
                for j in range(num_classes):
                    if j != y[i]:
                        if grad_r[i][j] <= 0.:
                            valid_n += 1
                            if grad_l[i][j] > 0.:
                                exact_n += 1
            exact_ratio = float(exact_n) / float((num_classes - 1) * num_samples)
            valid_ratio = float(valid_n) / float((num_classes - 1) * num_samples)

            eta = 0.
            if max_exact_ratio is not None and exact_ratio > max_exact_ratio:
                # constrain exact point ratio to not be too large
                _fls = list()
                for i in range(num_samples):
                    for j in range(num_classes):
                        if j != y[i]:
                            if grad_l[i][j] <= 0.0:
                                _fls.append((grad_l[i][j], 1))
                            if grad_r[i][j] <= 0.0:
                                _fls.append((grad_r[i][j], -1))
                _fls = sorted(_fls, key=lambda x: x[0], reverse=True)
                _n = exact_n
                _th = max_exact_ratio * ((num_classes - 1) * num_samples)
                assert 0 <= _th <= _n
                i = 0
                while _n > _th:
                    _n += _fls[i][1]
                    eta = -_fls[i][0] + P_EPS
                    # eta > 0
                    i += 1
            elif maximize_exact:
                # just maximize the exact point ratio
                _fls = list()
                for i in range(num_samples):
                    for j in range(num_classes):
                        if j != y[i]:
                            _fls.append((grad_l[i][j], -1))
                            _fls.append((grad_r[i][j], 1))
                _fls = sorted(_fls, key=lambda x: x[0])
                _n = 0
                maxn = 0
                for i in range(len(_fls)):
                    _n += _fls[i][1]
                    # past: constrain eta > 0
                    # if _n > maxn and (-_fls[i][0] + P_EPS > 0.0):

                    # now: eta could < 0
                    if _n > maxn:
                        maxn = _n
                        eta = -_fls[i][0] + P_EPS
            elif min_valid_ratio is not None and valid_ratio < min_valid_ratio:
                # find the smallest -eta, which satisfies the min valid point ratio requirement
                _fls = list()
                for i in range(num_samples):
                    for j in range(num_classes):
                        if j != y[i]:
                            if grad_r[i][j] > 0.:
                                _fls.append((grad_r[i][j], 1))
                _fls = sorted(_fls, key=lambda x: x[0])
                _n = valid_n
                _th = min_valid_ratio * ((num_classes - 1) * num_samples)
                assert 0 <= _n <= _th
                i = 0
                while _n <= _th:
                    _n += _fls[i][1]
                    eta = -_fls[i][0] - P_EPS
                    # eta < 0
                    i += 1

            # -- calculate need cells and needed ratios --
            need_cells = np.zeros([num_samples, num_classes], dtype=np.int)
            relax_n, exact_n, invalid_n = 0, 0, 0
            for i in range(num_samples):
                for j in range(num_classes):
                    if j != y[i]:
                        if grad_r[i][j] <= -eta:
                            if grad_l[i][j] > -eta:
                                need_cells[i][j] = 1
                                exact_n += 1
                            else:
                                relax_n += 1
                        else:
                            invalid_n += 1
            newN = np.sum(np.sum(need_cells, axis=1, dtype=np.int) > 0)
            newM = np.max(np.sum(need_cells, axis=1, dtype=np.int))

            relax_ratio, exact_ratio, invalid_ratio = \
                float(relax_n) / float((num_classes - 1) * num_samples), \
                float(exact_n) / float((num_classes - 1) * num_samples), \
                float(invalid_n) / float((num_classes - 1) * num_samples),

            # -- construct tiny network to binary search P which maximizes the target function --
            if newN == 0:
                targetp = np.zeros([num_samples, num_classes], dtype=np.float32)
            else:
                # compress, only contains useful P
                l = np.zeros([newN, newM], dtype=np.float32)
                r = np.ones([newN, newM], dtype=np.float32)
                newX = np.zeros([newN] + list(X.size()[1:]), dtype=np.float32)
                newp_lbound = np.zeros_like(l, dtype=np.float32) + P_EPS
                # newp_ubound will be filled later
                newp_ubound = np.zeros_like(l, dtype=np.float32)
                newc = np.zeros([newN, newM] + list(c.size()[2:]), dtype=np.float32)

                ii = 0
                for i in range(num_samples):
                    if np.sum(need_cells[i]) == 0:
                        continue
                    newX[ii] = X[i]

                    jj = 0
                    for j in range(num_classes):
                        if not need_cells[i][j]:
                            continue
                        newp_ubound[ii][jj] = p_ubound[i][j]
                        newc[ii][jj] = c[i][j]
                        l[ii][jj], r[ii][jj] = 0., 1.
                        jj += 1
                    # zero-padding
                    while jj < newM:
                        newp_ubound[ii][jj] = P_EPS
                        newc[ii][jj] = c[i][y[i]]
                        l[ii][jj], r[ii][jj] = 0., 0.
                        jj += 1
                    ii += 1

                tiny_dual = DualNetworkDouble(net, t_net, torch.tensor(newX).cuda(), epsilon, **kwargs)
                newc = Variable(torch.tensor(newc)).cuda()

                mid = (l + r) / 2.0
                t_cnt = 0
                # terminate condition: sum of all p difference <= 1e-6
                while np.sum((r - l) * (newp_ubound - newp_lbound)) > 1e-6:
                    t_cnt += 1
                    if t_cnt >= 30: # empirical
                        break
                    grad_mid_p = tiny_dual.grad_p(newp_lbound + (newp_ubound - newp_lbound) * mid, newc, retain_graph=True).numpy()
                    for i in range(newN):
                        for j in range(newM):
                            if grad_mid_p[i][j] > -eta:
                                l[i][j] = mid[i][j]
                            else:
                                r[i][j] = mid[i][j]
                    mid = (l + r) / 2.0
                # just to release the graph
                tiny_dual.grad_p(newp_lbound + (newp_ubound - newp_lbound) * mid, newc)

                # restore the original p matrix
                targetp = np.zeros_like(p_ubound)
                ii = 0
                for i in range(num_samples):
                    if np.sum(need_cells[i]) == 0:
                        continue
                    jj = 0
                    for j in range(num_classes):
                        if not need_cells[i][j]:
                            continue
                        targetp[i][j] = newp_lbound[ii][jj] + (newp_ubound[ii][jj] - newp_lbound[ii][jj]) * mid[ii][jj]
                        jj += 1
                    ii += 1

                # print(targetp)

                del tiny_dual
                torch.cuda.empty_cache()

            targetp = torch.tensor(targetp).cuda()
        else:
            # different eta use for each sample point

            dual = DualNetworkDouble(net, t_net, X, epsilon, **kwargs)
            c = Variable(torch.eye(num_classes).type_as(X)[y].unsqueeze(1) - torch.eye(num_classes).type_as(X).unsqueeze(0))
            # c_value = c.cpu().numpy()

            nu1, q1 = dual.nu1(c), dual.q1(c)
            nu1 = nu1.view(num_samples, num_classes, -1).cpu().detach().numpy()
            q1 = q1.view(num_samples, num_classes, -1).cpu().detach().numpy()

            targetp = np.zeros([num_samples, num_classes], dtype=np.float32)

            num_features = nu1.shape[2]

            for i in range(num_samples):
                for j in range(num_classes):
                    _cel = (np.abs(q1[i][j]) > 1e-7)
                    _p_points = - nu1[i][j][_cel] / q1[i][j][_cel]
                    _p_points = _p_points[_p_points > P_EPS]
                    _p_points = [P_EPS] + list(_p_points)
                    _p_points = sorted(_p_points)
                    _ind = min(max(int(individual_rate * len(_p_points)), 0), len(_p_points) - 1)
                    targetp[i][j] = _p_points[_ind]
            grad_p = dual.grad_p(targetp, c, retain_graph=True).numpy()
            eta = -grad_p

            targetp = torch.tensor(targetp).cuda()
            relax_ratio, exact_ratio, invalid_ratio = 0., 1., 0.

        # --- given target P & eta, now start normal calculation ---

        # the criteria is -eta

        f = -dual(targetp, c, eta=eta)
        f_orig = -dual.forward_orig(c)

        err, err_orig = (f.max(1)[1] != y), (f_orig.max(1)[1] != y)
        if size_average:
            err, err_orig = \
                err.sum().item() / X.size(0), err_orig.sum().item() / X.size(0)

        ce_loss, ce_loss_orig = \
            nn.CrossEntropyLoss(reduce=size_average)(f, y), \
            nn.CrossEntropyLoss(reduce=size_average)(f_orig, y)

        if not isinstance(eta, float):
            eta = np.mean(eta)
        return ce_loss, err, ce_loss_orig, err_orig, exact_ratio, relax_ratio, invalid_ratio, -eta

    else:
        # just evaluate

        torch.set_grad_enabled(False)
        dual = DualNetwork(net, X, epsilon, **kwargs)
        c = Variable(torch.eye(num_classes).type_as(X)[y].unsqueeze(1) - torch.eye(num_classes).type_as(X).unsqueeze(0))
        if X.is_cuda:
            c = c.cuda()
        f = -dual(c)

        err_orig = (f.max(1)[1] != y)
        if size_average:
            err_orig = err_orig.sum().item() / X.size(0)

        ce_loss_orig = nn.CrossEntropyLoss(reduce=size_average)(f, y)

        torch.cuda.empty_cache()
        torch.set_grad_enabled(True)

        eta = 0
        return ce_loss_orig, err_orig, ce_loss_orig, err_orig, 0.0, 0.0, 0.0, -eta


class InputSequential(nn.Sequential): 
    def __init__(self, *args, **kwargs): 
        self.i = 0
        super(InputSequential, self).__init__(*args, **kwargs)

    def set_start(self, i): 
        self.i = i

    def forward(self, input): 
        """ Helper class to apply a sequential model starting at the ith layer """
        xs = [input]
        for j,module in enumerate(self._modules.values()): 
            if j >= self.i: 
                if 'Dense' in type(module).__name__:
                    xs.append(module(*xs))
                else:
                    xs.append(module(xs[-1]))
        return xs[-1]


# Data parallel versions of the loss calculation
def robust_loss_parallel(net, epsilon, X, y, proj=None, 
                 norm_type='l1', bounded_input=False, size_average=True): 
    if any('BatchNorm2d' in str(l.__class__.__name__) for l in net): 
        raise NotImplementedError
    if bounded_input: 
        raise NotImplementedError('parallel loss for bounded input spaces not implemented')
    if X.size(0) != 1: 
        raise ValueError('Only use this function for a single example. This is '
            'intended for the use case when a single example does not fit in '
            'memory.')
    zs = [X[:1]]
    nf = [zs[0].size()]
    for l in net: 
        if isinstance(l, Dense): 
            zs.append(l(*zs))
        else:
            zs.append(l(zs[-1]))
        nf.append(zs[-1].size())

    dual_net = [select_input(X, epsilon, proj, norm_type, bounded_input)]

    for i,(in_f,out_f,layer) in enumerate(zip(nf[:-1], nf[1:], net)): 
        if isinstance(layer, nn.ReLU): 
            # compute bounds
            D = (InputSequential(*dual_net[1:]))
            Dp = nn.DataParallel(D)
            zl,zu = 0,0
            for j,dual_layer in enumerate(dual_net): 
                D.set_start(j)
                out = dual_layer.bounds(network=Dp)
                zl += out[0]
                zu += out[1]

            dual_layer = select_layer(layer, dual_net, X, proj, norm_type,
                in_f, out_f, zs[i], zl=zl, zu=zu)
        else:
            dual_layer = select_layer(layer, dual_net, X, proj, norm_type,
                in_f, out_f, zs[i])
        
        dual_net.append(dual_layer)

    num_classes = net[-1].out_features
    c = Variable(torch.eye(num_classes).type_as(X)[y].unsqueeze(1) - torch.eye(num_classes).type_as(X).unsqueeze(0))
    if X.is_cuda:
        c = c.cuda()

    # same as f = -dual.g(c)
    nu = [-c]
    for l in reversed(dual_net[1:]): 
        nu.append(l.T(*nu))
    
    f = -sum(l.objective(*nu[:min(len(dual_net)-i+1, len(dual_net))]) 
             for i,l in enumerate(dual_net))

    err = (f.max(1)[1] != y)

    if size_average: 
        err = err.sum().item()/X.size(0)
    ce_loss = nn.CrossEntropyLoss(reduce=size_average)(f, y)
    return ce_loss, err