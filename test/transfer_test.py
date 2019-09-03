import sys
sys.path.append('.')

import os
import time
import setproctitle

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
from convex_adversarial import robust_loss, robust_loss_transfer
import examples.problems as pblm
from examples.trainer import AverageMeter


DEBUG = False
params = {
    'cuda_ids': 1,
    'batch_size': 128,
    'robust_batch_size': 50,
    'robust_transfer_batch_size': 30,
    'actual_attack_batch_size': 400
}

CW_LEARNING_RATE = .2
CW_ATTACK_ITERATIONS = 100

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

## -------



def evaluate(loader, model, apply_func, log, verbose):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()

    end = time.time()

    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda().long()
        if apply_func is not None:
            X,y = apply_func(model, X, y)
        if y.dim() == 2:
            y = y.squeeze(1)

        with torch.no_grad():
            out = model(Variable(X))
            ce = nn.CrossEntropyLoss()(out, Variable(y))
            err = (out.max(1)[1] != y).float().sum()  / X.size(0)

        # measure accuracy and record loss
        losses.update(ce.item(), X.size(0))
        errors.update(err.item(), X.size(0))

        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()

        # print(i, ce.item(), err.item(), file=log)

        if verbose:
            endline = '\n' if i % verbose == 0 else '\r'
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {errors.val:.3f} ({errors.avg:.3f})'.format(
                   i, len(loader), batch_time=batch_time,
                   loss=losses, errors=errors), end=endline)
        log.flush()

        del X, y, out, ce, err
        if DEBUG and i == 10:
            break
    return losses.avg, errors.avg


def evaluate_robust(loader, model, epsilon, log, verbose):
    batch_time = AverageMeter()
    robust_losses = AverageMeter()
    robust_errors = AverageMeter()

    end = time.time()

    torch.set_grad_enabled(False)
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda().long()
        if y.dim() == 2:
            y = y.squeeze(1)

        robust_ce, robust_err = robust_loss(model, epsilon, X, y)

        # measure accuracy and record loss
        robust_losses.update(robust_ce.item(), X.size(0))
        robust_errors.update(robust_err, X.size(0))

        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()

        # print(i, robust_ce.item(), robust_err, file=log)

        if verbose:
            endline = '\n' if i % verbose == 0 else '\r'
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Robust loss {rloss.val:.3f} ({rloss.avg:.3f})\t'
                  'Robust error {rerrors.val:.3f} ({rerrors.avg:.3f})'.format(
                      i, len(loader), batch_time=batch_time,
                      rloss = robust_losses, rerrors = robust_errors), end=endline)
        log.flush()

        del X, y, robust_ce
        if DEBUG and i == 10:
            break
    torch.set_grad_enabled(True)
    torch.cuda.empty_cache()
    return robust_losses.avg, robust_errors.avg


def evaluate_transfer_robust(loader, t_model, model, epsilon, adaptive_vp_rate, log, verbose,
                             real_time=False, evaluate=False, clip_grad=None, **kwargs):

    batch_time = AverageMeter()
    robust_losses = AverageMeter()
    robust_errors = AverageMeter()
    vp_rates = AverageMeter()
    invp_rates = AverageMeter()

    end = time.time()
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda().long()
        if y.dim() == 2:
            y = y.squeeze(1)

        robust_ce, robust_err, _, _, v_point_rate, eta, inv_point_rate = \
            robust_loss_transfer(model, t_model, epsilon,
                                 Variable(X), Variable(y), **kwargs)

        # measure accuracy and record loss
        robust_losses.update(robust_ce.detach().item(), X.size(0))
        robust_errors.update(robust_err, X.size(0))
        vp_rates.update(v_point_rate, X.size(0))
        invp_rates.update(inv_point_rate, X.size(0))

        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()

        # print(i, robust_ce.detach().item(), robust_err, v_point_rate, inv_point_rate, file=log)

        if verbose:
            endline = '\n' if i % verbose == 0 else '\r'
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'T Robust loss {rloss.val:.4f} ({rloss.avg:.4f})\t'
                  'T Robust error {rerrors.val:.3f} ({rerrors.avg:.3f})\t'
                  'VP Rate {vp_rate.val: .3f} ({vp_rate.avg:.3f})\t'
                  'INVP Rate {invp_rate.val: .3f} ({invp_rate.avg:.3f})\t'.format(
                i, len(loader), batch_time=batch_time,
                vp_rate=vp_rates, invp_rate=invp_rates,
                rloss=robust_losses, rerrors=robust_errors), end=endline)

        log.flush()

        del X, y, robust_ce, robust_err
        if DEBUG and i == 10:
            break
    torch.cuda.empty_cache()
    return robust_losses.avg, robust_errors.avg, vp_rates.avg, invp_rates.avg


def test_transferability(loader, attack_method, epsilon, torch_model1, torch_model2, verbose, batch_size):
    batch_time = AverageMeter()
    err11s = AverageMeter()
    err12s = AverageMeter()
    err21s = AverageMeter()
    err22s = AverageMeter()

    end = time.time()

    sess = tf.Session(config=config)
    x_op = tf.placeholder(tf.float32, shape=(None, 1, 28, 28,))

    # Convert pytorch model to a tf_model and wrap it in cleverhans
    tf_model_fn1 = convert_pytorch_model_to_tf(torch_model1)
    if torch_model2 is not None:
        tf_model_fn2 = convert_pytorch_model_to_tf(torch_model2)

    # Attack Parameters
    if attack_method == 'CW':
        params = {'binary_search_steps': 1,
                  # 'y': None,
                  'max_iterations': CW_ATTACK_ITERATIONS,
                  'learning_rate': CW_LEARNING_RATE,
                  'batch_size': batch_size,
                  'initial_const': 10}
    elif attack_method == 'PGD':
        params = {'eps': epsilon,
                  'clip_min': 0.,
                  'clip_max': 1.,
                  'eps_iter': 0.005,
                  'nb_iter': 100,
                  'rand_init': False}
    elif attack_method == 'FGSM':
        params = {'eps': epsilon,
                  'clip_min': 0.,
                  'clip_max': 1.}
    else:
        raise Exception('Unknown attack method %s'.format(attack_method))

    # Model1 --> Model2
    cleverhans_model1 = CallableModelWrapper(tf_model_fn1, output_layer='logits')
    if torch_model2 is not None:
        cleverhans_model2 = CallableModelWrapper(tf_model_fn2, output_layer='logits')

    # Create an attack
    if attack_method == 'CW':
        attk1 = CarliniWagnerL2(cleverhans_model1, sess=sess)
    if attack_method == 'PGD':
        attk1 = ProjectedGradientDescent(cleverhans_model1, sess=sess)
    if attack_method == 'FGSM':
        attk1 = FastGradientMethod(cleverhans_model1, sess=sess)
    if torch_model2 is not None:
        if attack_method == 'CW':
            attk2 = CarliniWagnerL2(cleverhans_model2, sess=sess)
        if attack_method == 'PGD':
            attk2 = ProjectedGradientDescent(cleverhans_model2, sess=sess)
        if attack_method == 'FGSM':
            attk2 = FastGradientMethod(cleverhans_model2, sess=sess)

    adv_x_op1 = attk1.generate(x_op, **params)
    if torch_model2 is not None:
        adv_x_op2 = attk2.generate(x_op, **params)

    # Test on model1 and model2
    adv_preds_op11 = tf_model_fn1(adv_x_op1)
    if torch_model2 is not None:
        adv_preds_op12 = tf_model_fn2(adv_x_op1)
        adv_preds_op21 = tf_model_fn1(adv_x_op2)
        adv_preds_op22 = tf_model_fn2(adv_x_op2)

    for i, (xs, ys) in enumerate(loader):
        if torch_model2 is not None:
            (adv_preds11, adv_preds12) = sess.run((adv_preds_op11, adv_preds_op12), feed_dict={x_op: xs})
            (adv_preds21, adv_preds22) = sess.run((adv_preds_op21, adv_preds_op22), feed_dict={x_op: xs})
            err11 = float((np.argmax(adv_preds11, axis=1) != ys).sum()) / xs.size(0)
            err12 = float((np.argmax(adv_preds12, axis=1) != ys).sum()) / xs.size(0)
            err21 = float((np.argmax(adv_preds21, axis=1) != ys).sum()) / xs.size(0)
            err22 = float((np.argmax(adv_preds22, axis=1) != ys).sum()) / xs.size(0)
            err11s.update(err11, xs.size(0))
            err12s.update(err12, xs.size(0))
            err21s.update(err21, xs.size(0))
            err22s.update(err22, xs.size(0))
        else:
            adv_preds11 = sess.run((adv_preds_op11), feed_dict={x_op: xs})
            err11 = float((np.argmax(adv_preds11, axis=1) != ys).sum()) / xs.size(0)
            err11s.update(err11, xs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if verbose:
            endline = '\n' if i % verbose == 0 else '\r'
            if torch_model2 is not None:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'error 1->1 {err11.val:.3f} ({err11.avg:.3f})\t'
                      'error 1->2 {err12.val:.3f} ({err12.avg:.3f})\t'
                      'error 2->1 {err21.val:.3f} ({err21.avg:.3f})\t'
                      'error 2->2 {err22.val:.3f} ({err22.avg:.3f})\t'.format(
                    i, len(loader), batch_time=batch_time,
                    err11=err11s, err12=err12s, err21=err21s, err22=err22s), end=endline)
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'error {err11.val:.3f} ({err11.avg:.3f})\t'.format(
                    i, len(loader), batch_time=batch_time,
                    err11=err11s), end=endline)

    sess.close()
    if torch_model2 is not None:
        return err11s.avg, err12s.avg, err21s.avg, err22s.avg
    else:
        return err11s.avg


def test_transferability_subset(loader, attack_method, epsilon, torch_model1, torch_model2, verbose, batch_size):
    batch_time = AverageMeter()
    err12s = AverageMeter()
    err21s = AverageMeter()

    end = time.time()

    sess = tf.Session(config=config)
    x_op = tf.placeholder(tf.float32, shape=(None, 1, 28, 28,))

    # Convert pytorch model to a tf_model and wrap it in cleverhans
    tf_model_fn1 = convert_pytorch_model_to_tf(torch_model1)
    tf_model_fn2 = convert_pytorch_model_to_tf(torch_model2)

    # Attack Parameters
    if attack_method == 'CW':
        params = {'binary_search_steps': 1,
                  # 'y': None,
                  'max_iterations': CW_ATTACK_ITERATIONS,
                  'learning_rate': CW_LEARNING_RATE,
                  'batch_size': batch_size,
                  'initial_const': 10}
    elif attack_method == 'PGD':
        params = {'eps': epsilon,
                  'clip_min': 0.,
                  'clip_max': 1.,
                  'eps_iter': 0.005,
                  'nb_iter': 100,
                  'rand_init': False}
    elif attack_method == 'FGSM':
        params = {'eps': epsilon,
                  'clip_min': 0.,
                  'clip_max': 1.}
    else:
        raise Exception('Unknown attack method %s'.format(attack_method))

    # Model1 --> Model2
    cleverhans_model1 = CallableModelWrapper(tf_model_fn1, output_layer='logits')
    cleverhans_model2 = CallableModelWrapper(tf_model_fn2, output_layer='logits')

    # Create an attack
    if attack_method == 'CW':
        attk1 = CarliniWagnerL2(cleverhans_model1, sess=sess)
    if attack_method == 'PGD':
        attk1 = ProjectedGradientDescent(cleverhans_model1, sess=sess)
    if attack_method == 'FGSM':
        attk1 = FastGradientMethod(cleverhans_model1, sess=sess)

    if attack_method == 'CW':
        attk2 = CarliniWagnerL2(cleverhans_model2, sess=sess)
    if attack_method == 'PGD':
        attk2 = ProjectedGradientDescent(cleverhans_model2, sess=sess)
    if attack_method == 'FGSM':
        attk2 = FastGradientMethod(cleverhans_model2, sess=sess)

    adv_x_op1 = attk1.generate(x_op, **params)
    adv_x_op2 = attk2.generate(x_op, **params)

    # Test on model1 and model2
    adv_preds_op11 = tf_model_fn1(adv_x_op1)
    adv_preds_op12 = tf_model_fn2(adv_x_op1)
    adv_preds_op21 = tf_model_fn1(adv_x_op2)
    adv_preds_op22 = tf_model_fn2(adv_x_op2)

    for i, (xs, ys) in enumerate(loader):
        (adv_preds11, adv_preds12) = sess.run((adv_preds_op11, adv_preds_op12), feed_dict={x_op: xs})
        (adv_preds21, adv_preds22) = sess.run((adv_preds_op21, adv_preds_op22), feed_dict={x_op: xs})
        cnt11 = int((np.argmax(adv_preds11, axis=1) != ys).sum())
        cnt22 = int((np.argmax(adv_preds22, axis=1) != ys).sum())
        if cnt11 > 0:
            err12 = float(((np.argmax(adv_preds12, axis=1) != ys) * (np.argmax(adv_preds11, axis=1) != ys)).sum()) / float(cnt11)
            err12s.update(err12, cnt11)
        if cnt22 > 0:
            err21 = float(((np.argmax(adv_preds22, axis=1) != ys) * (np.argmax(adv_preds21, axis=1) != ys)).sum()) / float(cnt22)
            err21s.update(err21, cnt22)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose:
            endline = '\n' if i % verbose == 0 else '\r'
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'error 1->2 {err12.val:.3f} ({err12.avg:.3f})\t'
                  'error 2->1 {err21.val:.3f} ({err21.avg:.3f})\t'.format(
                i, len(loader), batch_time=batch_time,
                err12=err12s, err21=err21s), end=endline)

    sess.close()
    return err12s.avg, err21s.avg


os.environ['CUDA_VISIBLE_DEVICES'] = str(params['cuda_ids'])
if __name__ == '__main__':

    args = argparser()
    model = dict()
    for m_type in ['M1', 'M1P', 'M2P', 'M2PM', 'M2', 'M2M']:
        weight_path = None
        if m_type == 'M1':
            _psbl_path = args.M1_prefix + '_best.pth'
        elif m_type == 'M1P':
            _psbl_path = args.M1P_prefix + '_best.pth'
        elif m_type == 'M2P':
            _psbl_path = args.M2P_prefix + '_best.pth'
        elif m_type == 'M2PM':
            _psbl_path = args.M2PM_prefix + '_best.pth'
        elif m_type == 'M2':
            _psbl_path = args.M2_prefix + '_best.pth'
        elif m_type == 'M2M':
            _psbl_path = args.M2_prefix + '_mutual_model_best.pth'
        if os.path.exists(_psbl_path):
            model[m_type] = select_model(args.model)
            model[m_type].load_state_dict(torch.load(_psbl_path)['state_dict'])

    _, test_loader = pblm.mnist_loaders(params['batch_size'])
    _, robust_test_loader = pblm.mnist_loaders(params['robust_batch_size'])
    _, robust_transfer_test_loader = pblm.mnist_loaders(params['robust_transfer_batch_size'])
    _, actual_test_loader = pblm.mnist_loaders(params['actual_attack_batch_size'])

    res_log = open(args.prefix + '_test.txt', "w")
    setproctitle.setproctitle('test_clean_accuracy')
    print('Clean accuracy')
    print('Clean accuracy', file=res_log)
    for k in model:
        losses, errors = evaluate(test_loader, model[k], None, res_log, 10)
        print('\n')
        print(k, 'loss', losses, 'error', errors)
        print(k, 'loss', losses, 'error', errors, file=res_log)

    setproctitle.setproctitle('test_robust_bound')
    print('Robust bound')
    print('Robust bound', file=res_log)
    for k in model:
        losses, errors = evaluate_robust(robust_test_loader, model[k], args.epsilon, res_log, 10)
        print('\n')
        print(k, 'loss', losses, 'error', errors)
        print(k, 'loss', losses, 'error', errors, file=res_log)
    res_log.flush()

    # res_trans_log = open('test.trans.log', "w")
    # setproctitle.setproctitle('test_transfer_bound')
    # print('Transfer bound')
    # print('Transfer bound', file=res_trans_log)
    # for e in [0.1, 0.3]:
    #     # robust_bound
    #     m1 = model_set[('N', e)]
    #     for t in ['NP', 'T', 'TB']:
    #         m2 = model_set[(t, e)]
    #         losses, errors, vprates, invprates = evaluate_transfer_robust(robust_transfer_test_loader, m1, m2, e, None, res_trans_log, 10)
    #         print(t, e, losses, errors, vprates, invprates, file=res_trans_log)
    # res_trans_log.flush()

    res_attack_log = open(args.prefix + '_test.attack.txt', "w")
    setproctitle.setproctitle('actual_white_attack')
    print('actual_white_attack', file=res_attack_log)
    print('actual_white_attack')
    for attack_method in ['FGSM', 'PGD']:
        print(attack_method)
        print(attack_method, file=res_attack_log)
        for k in model:
            m = model[k]
            err = test_transferability(actual_test_loader, attack_method, args.epsilon, m, None, 10,
                                       params['actual_attack_batch_size'])
            print(k, attack_method, 'err', err)
            print(k, attack_method, 'err', err, file=res_attack_log)
    setproctitle.setproctitle('actual_transfer_attack')
    print('actual_transfer_attack', file=res_attack_log)
    print('actual_transfer_attack')
    for attack_method in ['FGSM', 'PGD']:
        print(attack_method)
        print(attack_method, file=res_attack_log)
        appeared = set()
        for s, t in [('M1', 'M1P'), ('M2P', 'M2PM'), ('M2', 'M2M')]:
        # for s in model:
        #     for t in model:
            if s == t:
                break
            # if (s, t) in appeared and (t, s) in appeared:
            #     break
            m1 = model[s]
            m2 = model[t]
            _, err12, err21, _ = test_transferability(actual_test_loader, attack_method, args.epsilon, m1, m2, 10,
                                                      params['actual_attack_batch_size'])
            print(attack_method, s, '->', t, err12, t, '->', s, err21)
            print(attack_method, s, '->', t, err12, t, '->', s, err21, file=res_attack_log)
            err12, err21 = test_transferability_subset(actual_test_loader, attack_method, args.epsilon, m1, m2, 10,
                                                       params['actual_attack_batch_size'])

            print(attack_method, 'subset', s, '->', t, err12, t, '->', s, err21)
            print(attack_method, 'subset', s, '->', t, err12, t, '->', s, err21, file=res_attack_log)
            appeared.add((s, t))
            appeared.add((t, s))
    res_attack_log.flush()

    # res_trans_bound_log = open('test.trans.rev.log', "w")
    # setproctitle.setproctitle('test_transfer_bound_rev')
    # print('Transfer bound Rev')
    # print('Transfer bound Rev', file=res_trans_bound_log)
    # for e in [0.1, 0.3]:
    #     # robust_bound
    #     m1 = model_set[('N', e)]
    #     for t in ['NP', 'T', 'TB']:
    #         m2 = model_set[(t, e)]
    #         losses, errors, vprates, invprates = evaluate_transfer_robust(robust_transfer_test_loader, m2, m1, e, None, res_trans_bound_log, 10)
    #         print(t, 'N', e, losses, errors, vprates, invprates, file=res_trans_bound_log)
    # res_trans_bound_log.flush()

