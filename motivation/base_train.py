import sys
sys.path.append('.')
# sys.path.append('..')
# sys.path.append('../test')

import time
import os
import argparse
import examples.problems as pblm
from examples.trainer import AverageMeter

import setproctitle

import torch
from torch import optim
from torch.autograd import Variable, grad
from torch import nn
import torch.nn.functional as F

import numpy as np
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

import cleverhans
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import ProjectedGradientDescent
from cleverhans.attacks import FastGradientMethod
from cleverhans.model import CallableModelWrapper
from cleverhans.utils import AccuracyReport
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf

DEBUG = False

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str)
    parser.add_argument('--dataset')
    parser.add_argument('--epsilon', type=str)
    parser.add_argument('--model')

    parser.add_argument('--batch_size', type=int)

    parser.add_argument('--epochs', type=int, default=20)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_length', type=int, default=5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_dacay', type=float, default=5e-4)
    parser.add_argument('--opt', type=str, default='adam')

    parser.add_argument('--ref_model', type=str, default=None)
    parser.add_argument('--lbda', type=float, default=0.0)

    parser.add_argument('--cuda_ids', default=None)

    parser.add_argument('--prefix', default=None)

    args = parser.parse_args()

    assert args.method in ['natural', 'adaptive', 'transfer', 'mutual']
    if args.method == 'transfer':
        assert os.path.exists('motivation/' + args.ref_model)

    # if args.method == 'natural':
    #     addons = '%s_%s_%s' % (args.dataset, args.model, args.method)
    # else:
    addons = '%s_%s_%s_%s' % (args.dataset, args.epsilon, args.model, args.method)
    if args.prefix is None:
        args.prefix = addons
    else:
        args.prefix += '_' + addons
    args.epsilon = float(args.epsilon)

    if args.cuda_ids is not None:
        print('Setting CUDA_VISIBLE_DEVICES to {}'.format(args.cuda_ids))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_ids
    return args


def train(loader, model, opt, epoch, verbose):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()

    model.train()

    end = time.time()
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda()
        data_time.update(time.time() - end)

        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)

        opt.zero_grad()
        ce.backward()
        opt.step()

        batch_time.update(time.time()-end)
        end = time.time()
        losses.update(ce.item(), X.size(0))
        errors.update(err.item(), X.size(0))

        if verbose and i % verbose == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {errors.val:.3f} ({errors.avg:.3f})'.format(
                   epoch, i, len(loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, errors=errors))

        if DEBUG and i == 10:
            break

    return losses.avg, errors.avg

def adv_train(loader, model, opt, epoch, epsilon,
              clip_min=0., clip_max=1., eps_iter=0.005, nb_iter=100, rand_init=False, verbose=20):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()

    model.train()

    params = {'eps': epsilon,
              'clip_min': clip_min,
              'clip_max': clip_max,
              'eps_iter': eps_iter,
              'nb_iter': nb_iter,
              'rand_init': rand_init}

    sess = tf.Session(config=config)
    x_op = tf.placeholder(tf.float32, shape=(None, 1, 28, 28,))

    tf_model = convert_pytorch_model_to_tf(model)
    cleverhans_model = CallableModelWrapper(tf_model, output_layer='logits')
    attk = ProjectedGradientDescent(cleverhans_model, sess=sess)
    adv_x_op = attk.generate(x_op, **params)

    end = time.time()
    for i, (X, y) in enumerate(loader):
        X_adv = sess.run((adv_x_op), feed_dict={x_op: X})

        X, y = Variable(torch.tensor(X_adv)).cuda(), y.cuda()
        data_time.update(time.time() - end)

        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum() / X.size(0)

        opt.zero_grad()
        ce.backward()
        opt.step()

        batch_time.update(time.time() - end)
        end = time.time()
        losses.update(ce.item(), X.size(0))
        errors.update(err.item(), X.size(0))

        if verbose and i % verbose == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {errors.val:.3f} ({errors.avg:.3f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, errors=errors))

        if DEBUG and i == 10:
            break

    return losses.avg, errors.avg

def trans_train(loader, model, opt, epoch, epsilon, ref_model,
                clip_min=0., clip_max=1., eps_iter=0.005, nb_iter=100, rand_init=False, verbose=20):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()

    model.train()

    params = {'eps': epsilon,
              'clip_min': clip_min,
              'clip_max': clip_max,
              'eps_iter': eps_iter,
              'nb_iter': nb_iter,
              'rand_init': rand_init}

    sess = tf.Session(config=config)
    x_op = tf.placeholder(tf.float32, shape=(None, 1, 28, 28,))

    tf_model = convert_pytorch_model_to_tf(ref_model)
    cleverhans_model = CallableModelWrapper(tf_model, output_layer='logits')
    attk = ProjectedGradientDescent(cleverhans_model, sess=sess)
    adv_x_op = attk.generate(x_op, **params)

    end = time.time()
    for i, (X, y) in enumerate(loader):
        X_adv = sess.run((adv_x_op), feed_dict={x_op: X})

        X, y = Variable(torch.tensor(X_adv)).cuda(), y.cuda()
        data_time.update(time.time() - end)

        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum() / X.size(0)

        opt.zero_grad()
        ce.backward()
        opt.step()

        batch_time.update(time.time() - end)
        end = time.time()
        losses.update(ce.item(), X.size(0))
        errors.update(err.item(), X.size(0))

        if verbose and i % verbose == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {errors.val:.3f} ({errors.avg:.3f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, errors=errors))

        if DEBUG and i == 10:
            break

    return losses.avg, errors.avg

def trans_reg_train(loader, model, opt, epoch, epsilon, ref_model, lbda, verbose=20):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    dps = AverageMeter()

    model.train()
    ref_model.train()

    end = time.time()
    for i, (X, y) in enumerate(loader):

        X, y = Variable(X.cuda(), requires_grad=True), Variable(y).cuda()
        data_time.update(time.time() - end)

        pred2 = ref_model(X)
        loss2 = F.cross_entropy(pred2, y)
        (grad2,) = grad(loss2, X, create_graph=True)
        grad2 = grad2.view(len(X), -1)
        grad2 = F.normalize(grad2)
        grad2 = torch.tensor(grad2.detach().cpu().numpy()).cuda()

        pred1 = model(X)
        loss1 = F.cross_entropy(pred1, y)
        (grad1,) = grad(loss1, X, create_graph=True)
        grad1 = grad1.view(len(X), -1)
        grad1 = F.normalize(grad1)

        X.requires_grad_(False)

        dp = torch.sum(torch.mul(grad1, grad2), dim=1)
        dp = torch.mean(dp)

        loss = loss1 + dp * lbda
        opt.zero_grad()
        loss.backward()
        opt.step()

        err = (pred1.data.max(1)[1] != y).float().sum() / X.size(0)

        batch_time.update(time.time() - end)
        end = time.time()
        losses.update(loss.item(), X.size(0))
        errors.update(err.item(), X.size(0))
        dps.update(dp, X.size(0))

        if verbose and i % verbose == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {errors.val:.3f} ({errors.avg:.3f})\t'
                  'Dp {dps.val:.3f} ({dps.avg:.3f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, errors=errors,
                dps=dps))

        if DEBUG and i == 10:
            break

    return losses.avg, errors.avg, dps.avg

def evaluate_clean(loader, model, epoch, verbose=20):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()

    model.eval()

    end = time.time()
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda()
        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum() / X.size(0)

        # measure accuracy and record loss
        losses.update(ce.item(), X.size(0))
        errors.update(err.item(), X.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        endline = '\n' if i % verbose == 0 else '\r'
        print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Error {error.val:.3f} ({error.avg:.3f})'.format(
                  i, len(loader), batch_time=batch_time, loss=losses,
                  error=errors), end=endline)

        if DEBUG and i == 10:
            break

    print('\n * Error {error.avg:.3f}'
          .format(error=errors))
    return losses.avg, errors.avg

def evaluate_adv(loader, dataset, model, epoch, epsilon,
                 clip_min=0., clip_max=1., eps_iter=0.005, nb_iter=100, rand_init=False, verbose=20):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()

    params = {'eps': epsilon,
              'clip_min': clip_min,
              'clip_max': clip_max,
              'eps_iter': eps_iter,
              'nb_iter': nb_iter,
              'rand_init': rand_init}

    sess = tf.Session(config=config)
    x_op = tf.placeholder(tf.float32, shape=(None, 1, 28, 28,))

    model.eval()
    tf_model = convert_pytorch_model_to_tf(model)
    cleverhans_model = CallableModelWrapper(tf_model, output_layer='logits')
    attk = ProjectedGradientDescent(cleverhans_model, sess=sess)
    adv_x_op = attk.generate(x_op, **params)
    adv_preds_op = tf_model(adv_x_op)

    end = time.time()
    for i, (X,y) in enumerate(loader):
        adv_preds = sess.run((adv_preds_op), feed_dict={x_op: X})

        y_arr = y.numpy()
        err = float((np.argmax(adv_preds, axis=1) != y_arr).sum()) / X.size(0)
        ce = nn.CrossEntropyLoss()(torch.Tensor(adv_preds), y)

        # measure accuracy and record loss
        losses.update(ce.item(), X.size(0))
        errors.update(err, X.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        endline = '\n' if i % verbose == 0 else '\r'
        print('Adv test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Error {error.val:.3f} ({error.avg:.3f})'.format(
                  i, len(loader), batch_time=batch_time, loss=losses,
                  error=errors), end=endline)

        if DEBUG and i == 10:
            break

    print('\n * Error {error.avg:.3f}'
          .format(error=errors))
    return losses.avg, errors.avg

def evaluate_trans(loader, dataset, model, epoch, epsilon, ref_model,
                 clip_min=0., clip_max=1., eps_iter=0.005, nb_iter=100, rand_init=False, verbose=20):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()

    params = {'eps': epsilon,
              'clip_min': clip_min,
              'clip_max': clip_max,
              'eps_iter': eps_iter,
              'nb_iter': nb_iter,
              'rand_init': rand_init}

    sess = tf.Session(config=config)
    x_op = tf.placeholder(tf.float32, shape=(None, 1, 28, 28,))

    model.eval()
    ref_model.eval()
    tf_model = convert_pytorch_model_to_tf(ref_model)
    cleverhans_model = CallableModelWrapper(tf_model, output_layer='logits')
    attk = ProjectedGradientDescent(cleverhans_model, sess=sess)
    adv_x_op = attk.generate(x_op, **params)

    end = time.time()
    for i, (X,y) in enumerate(loader):

        X_adv = sess.run((adv_x_op), feed_dict={x_op: X})
        X, y = Variable(torch.tensor(X_adv)).cuda(), y.cuda()
        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum() / X.size(0)

        # measure accuracy and record loss
        losses.update(ce.item(), X.size(0))
        errors.update(err.item(), X.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        endline = '\n' if i % verbose == 0 else '\r'
        print('Adv test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Error {error.val:.3f} ({error.avg:.3f})'.format(
                  i, len(loader), batch_time=batch_time, loss=losses,
                  error=errors), end=endline)

        if DEBUG and i == 10:
            break

    print('\n * Error {error.avg:.3f}'
          .format(error=errors))
    return losses.avg, errors.avg

if __name__ == '__main__':
    args = parse_args()

    setproctitle.setproctitle(args.prefix)

    model = select_model(args.dataset, args.model)
    model = model.cuda()

    if args.method == 'transfer' or args.method == 'mutual':
        ref_model = select_model(args.dataset, args.model)
        ref_model = ref_model.cuda()
        if args.method == 'transfer':
            ref_model.load_state_dict(torch.load('motivation/' + args.ref_model)['state_dict'])

    if args.dataset == 'MNIST':
        train_loader, _ = pblm.mnist_loaders(args.batch_size)
        _, test_loader = pblm.mnist_loaders(args.batch_size)
    elif args.dataset == 'CIFAR10':
        train_loader, _ = pblm.cifar_loaders(args.batch_size)
        _, test_loader = pblm.cifar_loaders(args.batch_size)
    assert train_loader is not None

    train_log = open('motivation/' + args.prefix + '_train.log', 'a')
    test_log = open('motivation/' + args.prefix + '_test.log', 'a')

    if args.opt == 'adam':
        opt = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'sgd':
        opt = optim.SGD(model.parameters(), lr=args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=args.lr_length, gamma=0.5)

    if args.method == 'mutual':
        if args.opt == 'adam':
            t_opt = optim.Adam(ref_model.parameters(), lr=args.lr)
        elif args.opt == 'sgd':
            t_opt = optim.SGD(ref_model.parameters(), lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
        t_lr_scheduler = optim.lr_scheduler.StepLR(t_opt, step_size=args.lr_length, gamma=0.5)

    best_err = 1.0
    for t in range(args.epochs):
        lr_scheduler.step(epoch=t)
        if args.method == 'mutual':
            t_lr_scheduler.step(epoch=t)
        print('Epoch %d' % t)

        if args.method == 'natural':
            loss, err = train(train_loader, model, opt, t, 20)
        elif args.method == 'adaptive':
            loss, err = adv_train(train_loader, model, opt, t, args.epsilon, verbose=20)
        elif args.method == 'transfer':
            if args.lbda > 1e-6:
                loss, err, dp = trans_reg_train(train_loader, model, opt, t, args.epsilon, ref_model, lbda=args.lbda, verbose=20)
            else:
                loss, err = trans_train(train_loader, model, opt, t, args.epsilon, ref_model, verbose=20)
        elif args.method == 'mutual':
            loss, err = trans_train(train_loader, model, opt, t, args.epsilon, ref_model, verbose=20)
            loss_t, err_t = trans_train(train_loader, ref_model, t_opt, t, args.epsilon, model, verbose=20)

        if args.method == 'transfer' and args.lbda > 1e-6:
            print(t, loss, err, dp, file=train_log)
        else:
            print(t, loss, err, file=train_log)
        if args.method == 'mutual':
            print(t, loss_t, err_t, file=train_log)

        loss, err = evaluate_clean(test_loader, model, t, verbose=20)
        adv_loss, adv_err = evaluate_adv(test_loader, args.dataset, model, t, args.epsilon, verbose=20)
        if args.method == 'transfer':
            adv_trans_loss, adv_trans_err = evaluate_trans(test_loader, args.dataset, model, t, args.epsilon, ref_model, verbose=20)
            adv_trans2_loss, adv_trans2_err = evaluate_trans(test_loader, args.dataset, ref_model, t, args.epsilon, model, verbose=20)

        if adv_err < best_err:
            best_err = err
            torch.save({
                'state_dict': model.state_dict(),
                'err': best_err,
                'epoch': t
            }, 'motivation/weights/' + args.prefix + "_best.pth")
        if args.method == 'mutual':
            loss_t, err_t = evaluate_clean(test_loader, ref_model, t, verbose=20)
            adv_loss_t, adv_err_t = evaluate_adv(test_loader, args.dataset, ref_model, t, args.epsilon, verbose=20)
            if adv_err_t < best_err:
                best_err = adv_err_t
                torch.save({
                    'state_dict': ref_model.state_dict(),
                    'err': best_err,
                    'epoch': t
                }, 'motivation/weights/' + args.prefix + "_best.pth")

        torch.save({
            'state_dict': model.state_dict(),
            'err': adv_err,
            'epoch': t
        }, 'motivation/weights/checkpoints/' + args.prefix + "_ep_%02d_%.2f_%.2f.pth" % (t, adv_err * 100., err * 100.))
        if args.method == 'mutual':
            torch.save({
                'state_dict': ref_model.state_dict(),
                'err': adv_err_t,
                'epoch': t
            }, 'motivation/weights/checkpoints/' + args.prefix + "_t_" + "_ep_%02d_%.2f_%.2f.pth" % (t, adv_err_t * 100., err_t * 100.))

        if args.method == 'transfer':
            print(t, adv_loss, adv_trans_loss, adv_trans2_loss, loss, err, adv_trans_err, adv_trans2_err, adv_err, file=test_log)
        elif args.method == 'mutual':
            print(t, adv_loss, loss, err, adv_err, adv_loss_t, loss_t, err_t, adv_err_t, file=test_log)
        else:
            print(t, adv_loss, loss, err, adv_err, file=test_log)

    train_log.close()
    test_log.close()