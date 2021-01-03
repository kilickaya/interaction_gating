# coding=utf-8

import os
import sys
import time
import argparse
import numpy as np
from collections import OrderedDict

import torch
from torch import nn
from torch import onnx
from sklearn import metrics as sk_metrics
from torchviz import make_dot, make_dot_from_trace
from thop import profile

from modules import layers_pytorch as pl
from core import utils, data_utils, configs, const, metrics
from core.utils import Path as Pth

# region Constants

RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# endregion

# region Model Training

def train_model(model, optimizer, loss_fn, metric_fn, xs_tr, y_tr, xs_te, y_te, n_epochs, batch_size_tr, batch_size_te, is_shuffle=True, callbacks=None):
    """
    Train using given input features.
    """

    # convert input to enumerables
    xs_tr = xs_tr if utils.is_enumerable(xs_tr) else [xs_tr]
    xs_te = xs_te if utils.is_enumerable(xs_te) else [xs_te]

    n_tr = len(y_tr)
    n_te = len(y_te)

    n_batch_tr = utils.calc_num_batches(n_tr, batch_size_tr)
    n_batch_te = utils.calc_num_batches(n_te, batch_size_te)

    print('... [tr]: n, n_batch, batch_size: %d, %d, %d' % (n_tr, n_batch_tr, batch_size_tr))
    print('... [te]: n, n_batch, batch_size: %d, %d, %d' % (n_te, n_batch_te, batch_size_te))

    acc_max_tr = 0.0
    acc_max_te = 0.0

    # loop on epochs
    sys.stdout.write('\n')
    for idx_epoch in range(n_epochs):

        # shuffle data
        if is_shuffle:
            idx_tr = np.arange(len(xs_tr[0]))
            np.random.shuffle(idx_tr)
        else:
            idx_tr = None

        epoch_num = idx_epoch + 1
        loss_tr = 0.0
        loss_te = 0.0
        acc_tr = 0.0
        acc_te = 0.0
        tt1 = time.time()

        # switch to training mode
        model.train()

        # loop on batches for train
        for idx_batch in range(n_batch_tr):
            batch_num = idx_batch + 1
            idx_start = idx_batch * batch_size_tr
            idx_end = (idx_batch + 1) * batch_size_tr

            # get data of batch
            if is_shuffle:
                xs_tr_b = [x_tr[idx_tr[idx_start:idx_end]] for x_tr in xs_tr]
                y_tr_b = y_tr[idx_tr[idx_start:idx_end]]
            else:
                xs_tr_b = [x_tr[idx_start:idx_end] for x_tr in xs_tr]
                y_tr_b = y_tr[idx_start:idx_end]

            # convert data to tensors, and copy to gpu
            t_xs_tr_b = [torch.from_numpy(x_tr_b).float().cuda() for x_tr_b in xs_tr_b]
            t_y_tr_b = torch.from_numpy(y_tr_b).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + loss
            t_y_tr_pred_b = model(*t_xs_tr_b)
            t_loss_b = loss_fn(t_y_tr_pred_b, t_y_tr_b)

            # backward + optimize
            t_loss_b.backward()
            optimizer.step()
            loss_b = t_loss_b.item()

            # loss + accuracy
            acc_b = metric_fn(t_y_tr_pred_b, t_y_tr_b)
            loss_tr += loss_b
            acc_tr += acc_b
            loss_b = loss_b / float(batch_num)
            acc_b = 100 * acc_b / float(batch_num)

            del t_loss_b
            del t_xs_tr_b
            del t_y_tr_pred_b

            # calling the callbacks
            if callbacks is not None:
                for cb in callbacks:
                    cb.on_batch_ends(batch_num, is_training=True)

            tt2 = time.time()
            duration = tt2 - tt1
            sys.stdout.write('\r%04ds | epoch %02d/%02d | batch [tr] %02d/%02d | loss_tr %.02f | acc_tr %02.02f' % (duration, epoch_num, n_epochs, batch_num, n_batch_tr, loss_b, acc_b))

        # switch to eval mode
        model.eval()

        # loop on batches for test
        for idx_batch in range(n_batch_te):
            batch_num = idx_batch + 1
            idx_start = idx_batch * batch_size_te
            idx_end = (idx_batch + 1) * batch_size_te

            # get data of batch
            xs_tre_b = [x_te[idx_start:idx_end] for x_te in xs_te]
            y_te_b = y_te[idx_start:idx_end]

            # convert data to tensors, and copy to gpu
            t_xs_te_b = [torch.from_numpy(x_te_b).float().cuda() for x_te_b in xs_tre_b]
            t_y_te_b = torch.from_numpy(y_te_b).cuda()

            # nograd + forward
            t_y_te_pred_b = model(*t_xs_te_b)

            # loss + accuracy
            t_loss_b = loss_fn(t_y_te_pred_b, t_y_te_b)
            loss_b = t_loss_b.item()
            acc_b = metric_fn(t_y_te_pred_b, t_y_te_b)
            loss_te += loss_b
            acc_te += acc_b
            loss_b = loss_b / float(batch_num)
            acc_b = 100 * acc_b / float(batch_num)

            del t_loss_b
            del t_xs_te_b
            del t_y_te_pred_b

            # calling the callbacks
            if callbacks is not None:
                for cb in callbacks:
                    cb.on_batch_ends(batch_num, is_training=False)

            tt2 = time.time()
            duration = tt2 - tt1
            sys.stdout.write('\r%04ds | epoch %02d/%02d | batch [te] %02d/%02d | loss_te %.02f | acc_te %02.02f' % (duration, epoch_num, n_epochs, batch_num, n_batch_te, loss_b, acc_b))

        tt2 = time.time()
        duration = tt2 - tt1

        loss_tr /= float(n_batch_tr)
        loss_te /= float(n_batch_te)
        acc_tr = 100 * acc_tr / float(n_batch_tr)
        acc_te = 100 * acc_te / float(n_batch_te)
        acc_max_tr = max(acc_max_tr, acc_tr)
        acc_max_te = max(acc_max_te, acc_te)

        sys.stdout.write('\r%04ds | epoch %02d/%02d | loss_tr %.02f | loss_te %.02f | acc_tr %02.02f | acc_te %02.02f | acc_max_tr %02.02f | acc_max_te %02.02f\n' % (duration, epoch_num, n_epochs, loss_tr, loss_te, acc_tr, acc_te, acc_max_tr, acc_max_te))

        # calling the callbacks
        if callbacks is not None:
            for cb in callbacks:
                cb.on_epoch_ends(epoch_num)


def train_model_custom_metric(model, optimizer, loss_fn, metric_fn, xs_tr, y_tr, xs_te, y_te, n_epochs, batch_size_tr, batch_size_te, is_shuffle=True, callbacks=None):
    """
    Train using given input features.
    """

    # convert input to enumerables
    xs_tr = xs_tr if utils.is_enumerable(xs_tr) else [xs_tr]
    xs_te = xs_te if utils.is_enumerable(xs_te) else [xs_te]

    n_tr = len(y_tr)
    n_te = len(y_te)

    n_batch_tr = utils.calc_num_batches(n_tr, batch_size_tr)
    n_batch_te = utils.calc_num_batches(n_te, batch_size_te)

    print('... [tr]: n, n_batch, batch_size: %d, %d, %d' % (n_tr, n_batch_tr, batch_size_tr))
    print('... [te]: n, n_batch, batch_size: %d, %d, %d' % (n_te, n_batch_te, batch_size_te))

    acc_max_tr = 0.0
    acc_max_te = 0.0

    # loop on epochs
    sys.stdout.write('\n')
    for idx_epoch in range(n_epochs):

        # shuffle data
        if is_shuffle:
            idx_tr = np.arange(len(xs_tr[0]))
            np.random.shuffle(idx_tr)
        else:
            idx_tr = None

        epoch_num = idx_epoch + 1
        loss_tr = 0.0
        loss_te = 0.0
        acc_tr = 0.0
        acc_te = 0.0
        y_pred_te = None
        tt1 = time.time()

        # switch to training mode
        model.train()

        # loop on batches for train
        for idx_batch in range(n_batch_tr):
            batch_num = idx_batch + 1
            idx_start = idx_batch * batch_size_tr
            idx_end = (idx_batch + 1) * batch_size_tr

            # get data of batch
            if is_shuffle:
                xs_tr_b = [x_tr[idx_tr[idx_start:idx_end]] for x_tr in xs_tr]
                y_tr_b = y_tr[idx_tr[idx_start:idx_end]]
            else:
                xs_tr_b = [x_tr[idx_start:idx_end] for x_tr in xs_tr]
                y_tr_b = y_tr[idx_start:idx_end]

            # convert data to tensors, and copy to gpu
            t_xs_tr_b = [torch.from_numpy(x_tr_b).float().cuda() for x_tr_b in xs_tr_b]
            t_y_tr_b = torch.from_numpy(y_tr_b).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + loss
            t_y_tr_pred_b = model(*t_xs_tr_b)
            t_loss_b = loss_fn(t_y_tr_pred_b, t_y_tr_b)

            # backward + optimize
            t_loss_b.backward()
            optimizer.step()
            loss_b = t_loss_b.item()

            # loss + accuracy
            acc_b = 0.0
            loss_tr += loss_b
            acc_tr += acc_b
            loss_b = loss_b / float(batch_num)
            acc_b = 100 * acc_b / float(batch_num)

            del t_loss_b
            del t_xs_tr_b
            del t_y_tr_pred_b

            # calling the callbacks
            if callbacks is not None:
                for cb in callbacks:
                    cb.on_batch_ends(batch_num, is_training=True)

            tt2 = time.time()
            duration = tt2 - tt1
            sys.stdout.write('\r%04ds | epoch %02d/%02d | batch [tr] %02d/%02d | loss_tr %.02f | acc_tr %02.02f' % (duration, epoch_num, n_epochs, batch_num, n_batch_tr, loss_b, acc_b))

        # switch to eval mode
        model.eval()

        # loop on batches for test
        for idx_batch in range(n_batch_te):
            batch_num = idx_batch + 1
            idx_start = idx_batch * batch_size_te
            idx_end = (idx_batch + 1) * batch_size_te

            # get data of batch
            xs_tre_b = [x_te[idx_start:idx_end] for x_te in xs_te]
            y_te_b = y_te[idx_start:idx_end]

            # convert data to tensors, and copy to gpu
            t_xs_te_b = [torch.from_numpy(x_te_b).float().cuda() for x_te_b in xs_tre_b]
            t_y_te_b = torch.from_numpy(y_te_b).cuda()

            # nograd + forward
            t_y_te_pred_b = model(*t_xs_te_b)

            # append predictions
            y_te_pred_b = np.array(t_y_te_pred_b.tolist())
            y_pred_te = y_te_pred_b if y_pred_te is None else np.vstack((y_pred_te, y_te_pred_b))

            # loss + accuracy
            t_loss_b = loss_fn(t_y_te_pred_b, t_y_te_b)
            loss_b = t_loss_b.item()
            acc_b = 0.0
            loss_te += loss_b
            acc_te += acc_b
            loss_b = loss_b / float(batch_num)
            acc_b = 100 * acc_b / float(batch_num)

            del t_loss_b
            del t_xs_te_b
            del t_y_te_pred_b

            # calling the callbacks
            if callbacks is not None:
                for cb in callbacks:
                    cb.on_batch_ends(batch_num, is_training=False)

            tt2 = time.time()
            duration = tt2 - tt1
            sys.stdout.write('\r%04ds | epoch %02d/%02d | batch [te] %02d/%02d | loss_te %.02f | acc_te %02.02f' % (duration, epoch_num, n_epochs, batch_num, n_batch_te, loss_b, acc_b))

        tt2 = time.time()
        duration = tt2 - tt1

        acc_te = metric_fn(y_pred_te, y_te)
        loss_tr /= float(n_batch_tr)
        loss_te /= float(n_batch_te)
        acc_tr = 100 * acc_tr
        acc_te = 100 * acc_te
        acc_max_tr = max(acc_max_tr, acc_tr)
        acc_max_te = max(acc_max_te, acc_te)

        sys.stdout.write('\r%04ds | epoch %02d/%02d | loss_tr %.02f | loss_te %.02f | acc_tr %02.02f | acc_te %02.02f | acc_max_tr %02.02f | acc_max_te %02.02f\n' % (duration, epoch_num, n_epochs, loss_tr, loss_te, acc_tr, acc_te, acc_max_tr, acc_max_te))

        # calling the callbacks
        if callbacks is not None:
            for cb in callbacks:
                cb.on_epoch_ends(epoch_num)



def train_model_custom_metric_mask(model, optimizer, loss_fn, metric_fn, xs_tr, y_tr, y_tr_mask, xs_te, y_te, y_te_mask, n_epochs, batch_size_tr, batch_size_te, is_shuffle=True, callbacks=None):
    """
    Train using given input features.
    """

    # convert input to enumerables
    xs_tr = xs_tr if utils.is_enumerable(xs_tr) else [xs_tr]
    xs_te = xs_te if utils.is_enumerable(xs_te) else [xs_te]

    n_tr = len(y_tr)
    n_te = len(y_te)

    n_batch_tr = utils.calc_num_batches(n_tr, batch_size_tr)
    n_batch_te = utils.calc_num_batches(n_te, batch_size_te)

    print('... [tr]: n, n_batch, batch_size: %d, %d, %d' % (n_tr, n_batch_tr, batch_size_tr))
    print('... [te]: n, n_batch, batch_size: %d, %d, %d' % (n_te, n_batch_te, batch_size_te))

    acc_max_tr = 0.0
    acc_max_te = 0.0

    # loop on epochs
    sys.stdout.write('\n')
    for idx_epoch in range(n_epochs):

        # shuffle data
        if is_shuffle:
            idx_tr = np.arange(len(xs_tr[0]))
            np.random.shuffle(idx_tr)
        else:
            idx_tr = None

        epoch_num = idx_epoch + 1
        loss_tr = 0.0
        loss_te = 0.0
        acc_tr = 0.0
        acc_te = 0.0
        y_pred_te = None
        tt1 = time.time()

        # switch to training mode
        model.train()

        # loop on batches for train
        for idx_batch in range(n_batch_tr):
            batch_num = idx_batch + 1
            idx_start = idx_batch * batch_size_tr
            idx_end = (idx_batch + 1) * batch_size_tr

            # get data of batch
            if is_shuffle:
                xs_tr_b = [x_tr[idx_tr[idx_start:idx_end]] for x_tr in xs_tr]
                y_tr_b = y_tr[idx_tr[idx_start:idx_end]]
                y_tr_mask_b = y_tr_mask[idx_tr[idx_start:idx_end]]

            else:
                xs_tr_b = [x_tr[idx_start:idx_end] for x_tr in xs_tr]
                y_tr_b = y_tr[idx_start:idx_end]
                y_tr_mask_b = y_tr_mask[idx_start:idx_end]

            # convert data to tensors, and copy to gpu
            t_xs_tr_b = [torch.from_numpy(x_tr_b).float().cuda() for x_tr_b in xs_tr_b]
            t_y_tr_b = torch.from_numpy(y_tr_b).cuda()
            t_y_tr_mask_b = torch.from_numpy(y_tr_mask_b).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + loss
            t_y_tr_pred_b = model(*t_xs_tr_b)
            # mask out ambiguous labels
            t_y_tr_pred_b = t_y_tr_pred_b  * t_y_tr_mask_b
            t_loss_b = loss_fn(t_y_tr_pred_b, t_y_tr_b)

            # backward + optimize
            t_loss_b.backward()
            optimizer.step()
            loss_b = t_loss_b.item()

            # loss + accuracy
            acc_b = 0.0
            loss_tr += loss_b
            acc_tr += acc_b
            loss_b = loss_b / float(batch_num)
            acc_b = 100 * acc_b / float(batch_num)

            del t_loss_b
            del t_xs_tr_b
            del t_y_tr_pred_b

            # calling the callbacks
            if callbacks is not None:
                for cb in callbacks:
                    cb.on_batch_ends(batch_num, is_training=True)

            tt2 = time.time()
            duration = tt2 - tt1
            sys.stdout.write('\r%04ds | epoch %02d/%02d | batch [tr] %02d/%02d | loss_tr %.02f | acc_tr %02.02f' % (duration, epoch_num, n_epochs, batch_num, n_batch_tr, loss_b, acc_b))

        # switch to eval mode
        model.eval()

        # loop on batches for test
        for idx_batch in range(n_batch_te):
            batch_num = idx_batch + 1
            idx_start = idx_batch * batch_size_te
            idx_end = (idx_batch + 1) * batch_size_te

            # get data of batch
            xs_tre_b = [x_te[idx_start:idx_end] for x_te in xs_te]
            y_te_b = y_te[idx_start:idx_end]
            y_te_mask_b = y_te_mask[idx_start:idx_end]

            # convert data to tensors, and copy to gpu
            t_xs_te_b = [torch.from_numpy(x_te_b).float().cuda() for x_te_b in xs_tre_b]
            t_y_te_b = torch.from_numpy(y_te_b).cuda()
            t_y_te_mask_b = torch.from_numpy(y_te_mask_b).cuda()

            # nograd + forward
            t_y_te_pred_b = model(*t_xs_te_b)

            # mask out ambiguous predictions
            t_y_te_pred_b = t_y_te_pred_b * t_y_te_mask_b

            # append predictions
            y_te_pred_b = np.array(t_y_te_pred_b.tolist())
            y_pred_te = y_te_pred_b if y_pred_te is None else np.vstack((y_pred_te, y_te_pred_b))

            # loss + accuracy
            t_loss_b = loss_fn(t_y_te_pred_b, t_y_te_b)
            loss_b = t_loss_b.item()
            acc_b = 0.0
            loss_te += loss_b
            acc_te += acc_b
            loss_b = loss_b / float(batch_num)
            acc_b = 100 * acc_b / float(batch_num)

            del t_loss_b
            del t_xs_te_b
            del t_y_te_pred_b

            # calling the callbacks
            if callbacks is not None:
                for cb in callbacks:
                    cb.on_batch_ends(batch_num, is_training=False)

            tt2 = time.time()
            duration = tt2 - tt1
            sys.stdout.write('\r%04ds | epoch %02d/%02d | batch [te] %02d/%02d | loss_te %.02f | acc_te %02.02f' % (duration, epoch_num, n_epochs, batch_num, n_batch_te, loss_b, acc_b))

        tt2 = time.time()
        duration = tt2 - tt1

        acc_te = metric_fn(y_pred_te, y_te)
        loss_tr /= float(n_batch_tr)
        loss_te /= float(n_batch_te)
        acc_tr = 100 * acc_tr
        acc_te = 100 * acc_te
        acc_max_tr = max(acc_max_tr, acc_tr)
        acc_max_te = max(acc_max_te, acc_te)

        sys.stdout.write('\r%04ds | epoch %02d/%02d | loss_tr %.02f | loss_te %.02f | acc_tr %02.02f | acc_te %02.02f | acc_max_tr %02.02f | acc_max_te %02.02f\n' % (duration, epoch_num, n_epochs, loss_tr, loss_te, acc_tr, acc_te, acc_max_tr, acc_max_te))

        # calling the callbacks
        if callbacks is not None:
            for cb in callbacks:
                cb.on_epoch_ends(epoch_num)


def train_model_custom_metric_ldecay(model, optimizer, lrate, lrate_epoch, loss_fn, metric_fn, xs_tr, y_tr, xs_te, y_te, n_epochs, batch_size_tr, batch_size_te, is_shuffle=True, callbacks=None):
    """
    Train using given input features.
    """

    initial_lrate = lrate 

    # convert input to enumerables
    xs_tr = xs_tr if utils.is_enumerable(xs_tr) else [xs_tr]
    xs_te = xs_te if utils.is_enumerable(xs_te) else [xs_te]

    n_tr = len(y_tr)
    n_te = len(y_te)

    n_batch_tr = utils.calc_num_batches(n_tr, batch_size_tr)
    n_batch_te = utils.calc_num_batches(n_te, batch_size_te)

    print('... [tr]: n, n_batch, batch_size: %d, %d, %d' % (n_tr, n_batch_tr, batch_size_tr))
    print('... [te]: n, n_batch, batch_size: %d, %d, %d' % (n_te, n_batch_te, batch_size_te))

    acc_max_tr = 0.0
    acc_max_te = 0.0

    # loop on epochs
    sys.stdout.write('\n')
    for idx_epoch in range(n_epochs):

        # shuffle data
        if is_shuffle:
            idx_tr = np.arange(len(xs_tr[0]))
            np.random.shuffle(idx_tr)
        else:
            idx_tr = None

        epoch_num = idx_epoch + 1
        loss_tr = 0.0
        loss_te = 0.0
        acc_tr = 0.0
        acc_te = 0.0
        y_pred_te = None
        tt1 = time.time()

        # switch to training mode
        model.train()

        # loop on batches for train
        for idx_batch in range(n_batch_tr):
            batch_num = idx_batch + 1
            idx_start = idx_batch * batch_size_tr
            idx_end = (idx_batch + 1) * batch_size_tr

            # get data of batch
            if is_shuffle:
                xs_tr_b = [x_tr[idx_tr[idx_start:idx_end]] for x_tr in xs_tr]
                y_tr_b = y_tr[idx_tr[idx_start:idx_end]]
            else:
                xs_tr_b = [x_tr[idx_start:idx_end] for x_tr in xs_tr]
                y_tr_b = y_tr[idx_start:idx_end]

            # convert data to tensors, and copy to gpu
            t_xs_tr_b = [torch.from_numpy(x_tr_b).float().cuda() for x_tr_b in xs_tr_b]
            t_y_tr_b = torch.from_numpy(y_tr_b).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + loss
            t_y_tr_pred_b = model(*t_xs_tr_b)
            t_loss_b = loss_fn(t_y_tr_pred_b, t_y_tr_b)

            # backward + optimize
            t_loss_b.backward()
            optimizer.step()
            loss_b = t_loss_b.item()

            # loss + accuracy
            acc_b = 0.0
            loss_tr += loss_b
            acc_tr += acc_b
            loss_b = loss_b / float(batch_num)
            acc_b = 100 * acc_b / float(batch_num)

            del t_loss_b
            del t_xs_tr_b
            del t_y_tr_pred_b

            # calling the callbacks
            if callbacks is not None:
                for cb in callbacks:
                    cb.on_batch_ends(batch_num = batch_num, is_training=True)

            tt2 = time.time()
            duration = tt2 - tt1
            sys.stdout.write('\r%04ds | epoch %02d/%02d | batch [tr] %02d/%02d | loss_tr %.02f | acc_tr %02.02f' % (duration, epoch_num, n_epochs, batch_num, n_batch_tr, loss_b, acc_b))

        # Decay learning rate here
        if (epoch_num+1) % lrate_epoch == 0:
            initial_lrate = initial_lrate * 0.1
            print('\nLearning rate decayed: %0.4f\n' %(initial_lrate))
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lrate

        # switch to eval mode
        model.eval()

        # loop on batches for test
        for idx_batch in range(n_batch_te):
            batch_num = idx_batch + 1
            idx_start = idx_batch * batch_size_te
            idx_end = (idx_batch + 1) * batch_size_te

            # get data of batch
            xs_tre_b = [x_te[idx_start:idx_end] for x_te in xs_te]
            y_te_b = y_te[idx_start:idx_end]

            # convert data to tensors, and copy to gpu
            t_xs_te_b = [torch.from_numpy(x_te_b).float().cuda() for x_te_b in xs_tre_b]
            t_y_te_b = torch.from_numpy(y_te_b).cuda()

            # nograd + forward
            t_y_te_pred_b = model(*t_xs_te_b)

            # append predictions
            y_te_pred_b = np.array(t_y_te_pred_b.tolist())
            y_pred_te = y_te_pred_b if y_pred_te is None else np.vstack((y_pred_te, y_te_pred_b))

            # loss + accuracy
            t_loss_b = loss_fn(t_y_te_pred_b, t_y_te_b)
            loss_b = t_loss_b.item()
            acc_b = 0.0
            loss_te += loss_b
            acc_te += acc_b
            loss_b = loss_b / float(batch_num)
            acc_b = 100 * acc_b / float(batch_num)

            del t_loss_b
            del t_xs_te_b
            del t_y_te_pred_b

            # calling the callbacks
            if callbacks is not None:
                for cb in callbacks:
                    cb.on_batch_ends(batch_num, is_training=False)

            tt2 = time.time()
            duration = tt2 - tt1
            sys.stdout.write('\r%04ds | epoch %02d/%02d | batch [te] %02d/%02d | loss_te %.02f | acc_te %02.02f' % (duration, epoch_num, n_epochs, batch_num, n_batch_te, loss_b, acc_b))

        tt2 = time.time()
        duration = tt2 - tt1

        acc_te = metric_fn(y_pred_te, y_te)
        loss_tr /= float(n_batch_tr)
        loss_te /= float(n_batch_te)
        acc_tr = 100 * acc_tr
        acc_te = 100 * acc_te
        acc_max_tr = max(acc_max_tr, acc_tr)
        acc_max_te = max(acc_max_te, acc_te)

        sys.stdout.write('\r%04ds | epoch %02d/%02d | loss_tr %.02f | loss_te %.02f | acc_tr %02.02f | acc_te %02.02f | acc_max_tr %02.02f | acc_max_te %02.02f\n' % (duration, epoch_num, n_epochs, loss_tr, loss_te, acc_tr, acc_te, acc_max_tr, acc_max_te))

        # calling the callbacks
        if callbacks is not None:
            for cb in callbacks:
                cb.on_epoch_ends(epoch_num)


def train_model_using_sampler(model, optimizer, loss_fn, metric_fn, sampler, n_epochs, batch_size_tr, batch_size_te, callbacks=None):
    """
    Train using input sampler. Each epoch, we sample inputs from the sampler
    """

    # sample input
    x_tr, y_tr = sampler.sample_train()
    x_te, y_te = sampler.sample_test()

    n_tr = len(y_tr)
    n_te = len(y_te)

    n_batch_tr = utils.calc_num_batches(n_tr, batch_size_tr)
    n_batch_te = utils.calc_num_batches(n_te, batch_size_te)

    print('... [tr]: n, n_batch, batch_size: %d, %d, %d' % (n_tr, n_batch_tr, batch_size_tr))
    print('... [te]: n, n_batch, batch_size: %d, %d, %d' % (n_te, n_batch_te, batch_size_te))

    acc_max_tr = 0.0
    acc_max_te = 0.0

    # loop on epochs
    sys.stdout.write('\n')
    for idx_epoch in range(n_epochs):

        epoch_num = idx_epoch + 1
        loss_tr = 0.0
        loss_te = 0.0
        acc_tr = 0.0
        acc_te = 0.0
        tt1 = time.time()

        # switch to training mode
        model.train()

        # loop on batches for train
        for idx_batch in range(n_batch_tr):
            batch_num = idx_batch + 1
            idx_start = idx_batch * batch_size_tr
            idx_end = (idx_batch + 1) * batch_size_tr

            # get data of batch
            x_tr_b = x_tr[idx_start:idx_end]
            y_tr_b = y_tr[idx_start:idx_end]

            # convert data to tensors, and copy to gpu
            t_x_tr_b = torch.from_numpy(x_tr_b).cuda()
            t_y_tr_b = torch.from_numpy(y_tr_b).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + loss
            t_y_tr_pred_b = model(t_x_tr_b)
            t_loss_b = loss_fn(t_y_tr_pred_b, t_y_tr_b)

            # backward + optimize
            t_loss_b.backward()
            optimizer.step()
            loss_b = t_loss_b.item()

            # loss + accuracy
            acc_b = metric_fn(t_y_tr_pred_b, t_y_tr_b)
            loss_tr += loss_b
            acc_tr += acc_b
            loss_b = loss_b / float(batch_num)
            acc_b = 100 * acc_b / float(batch_num)

            del t_loss_b
            del t_x_tr_b
            del t_y_tr_pred_b

            # calling the callbacks
            if callbacks is not None:
                for cb in callbacks:
                    cb.on_batch_ends(batch_num, is_training=True)

            tt2 = time.time()
            duration = tt2 - tt1
            sys.stdout.write('\r%04ds | epoch %02d/%02d | batch [tr] %02d/%02d | loss_tr %.02f | acc_tr %02.02f' % (duration, epoch_num, n_epochs, batch_num, n_batch_tr, loss_b, acc_b))

        # switch to eval mode
        model.eval()

        # loop on batches for test
        for idx_batch in range(n_batch_te):
            batch_num = idx_batch + 1
            idx_start = idx_batch * batch_size_te
            idx_end = (idx_batch + 1) * batch_size_te

            # get data of batch
            x_te_b = x_te[idx_start:idx_end]
            y_te_b = y_te[idx_start:idx_end]

            # convert data to tensors, and copy to gpu
            t_x_te_b = torch.from_numpy(x_te_b).cuda()
            t_y_te_b = torch.from_numpy(y_te_b).cuda()

            # nograd + forward
            t_y_te_pred_b = model(t_x_te_b)

            # loss + accuracy
            t_loss_b = loss_fn(t_y_te_pred_b, t_y_te_b)
            loss_b = t_loss_b.item()
            acc_b = metric_fn(t_y_te_pred_b, t_y_te_b)
            loss_te += loss_b
            acc_te += acc_b
            loss_b = loss_b / float(batch_num)
            acc_b = 100 * acc_b / float(batch_num)

            del t_loss_b
            del t_x_te_b
            del t_y_te_pred_b

            # calling the callbacks
            if callbacks is not None:
                for cb in callbacks:
                    cb.on_batch_ends(batch_num, is_training=False)

            tt2 = time.time()
            duration = tt2 - tt1
            sys.stdout.write('\r%04ds | epoch %02d/%02d | batch [te] %02d/%02d | loss_te %.02f | acc_te %02.02f' % (duration, epoch_num, n_epochs, batch_num, n_batch_te, loss_b, acc_b))

        tt2 = time.time()
        duration = tt2 - tt1

        loss_tr /= float(n_batch_tr)
        loss_te /= float(n_batch_te)
        acc_tr = 100 * acc_tr / float(n_batch_tr)
        acc_te = 100 * acc_te / float(n_batch_te)
        acc_max_tr = max(acc_max_tr, acc_tr)
        acc_max_te = max(acc_max_te, acc_te)

        sys.stdout.write('\r%04ds | epoch %02d/%02d | loss_tr %.02f | loss_te %.02f | acc_tr %02.02f | acc_te %02.02f | acc_max_tr %02.02f | acc_max_te %02.02f\n' % (duration, epoch_num, n_epochs, loss_tr, loss_te, acc_tr, acc_te, acc_max_tr, acc_max_te))

        # sample inputs again
        x_tr, y_tr = sampler.sample_train()
        x_te, y_te = sampler.sample_test()

        # calling the callbacks
        if callbacks is not None:
            for cb in callbacks:
                cb.on_epoch_ends(epoch_num)

def train_model_using_sampler_custom_metric(model, optimizer, loss_fn, metric_fn, sampler, n_epochs, batch_size_tr, batch_size_te, callbacks=None):
    """
    Train using input sampler. Each epoch, we sample inputs from the sampler
    """

    # sample input
    x_tr, y_tr = sampler.sample_train()
    x_te, y_te = sampler.sample_test()

    n_tr = len(y_tr)
    n_te = len(y_te)

    n_batch_tr = utils.calc_num_batches(n_tr, batch_size_tr)
    n_batch_te = utils.calc_num_batches(n_te, batch_size_te)

    print('... [tr]: n, n_batch, batch_size: %d, %d, %d' % (n_tr, n_batch_tr, batch_size_tr))
    print('... [te]: n, n_batch, batch_size: %d, %d, %d' % (n_te, n_batch_te, batch_size_te))

    acc_max_tr = 0.0
    acc_max_te = 0.0

    # loop on epochs
    sys.stdout.write('\n')
    for idx_epoch in range(n_epochs):

        epoch_num = idx_epoch + 1
        loss_tr = 0.0
        loss_te = 0.0
        acc_tr = 0.0
        acc_te = 0.0
        y_pred_te = None
        tt1 = time.time()

        # switch to training mode
        model.train()

        # loop on batches for train
        for idx_batch in range(n_batch_tr):
            batch_num = idx_batch + 1
            idx_start = idx_batch * batch_size_tr
            idx_end = (idx_batch + 1) * batch_size_tr

            # get data of batch
            x_tr_b = x_tr[idx_start:idx_end]
            y_tr_b = y_tr[idx_start:idx_end]

            # convert data to tensors, and copy to gpu
            t_x_tr_b = torch.from_numpy(x_tr_b).cuda()
            t_y_tr_b = torch.from_numpy(y_tr_b).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + loss
            t_y_tr_pred_b = model(t_x_tr_b)
            t_loss_b = loss_fn(t_y_tr_pred_b, t_y_tr_b)

            # backward + optimize
            t_loss_b.backward()
            optimizer.step()
            loss_b = t_loss_b.item()

            # loss + accuracy
            acc_b = 0.0
            loss_tr += loss_b
            acc_tr += acc_b
            loss_b = loss_b / float(batch_num)
            acc_b = 100 * acc_b / float(batch_num)

            del t_loss_b
            del t_x_tr_b
            del t_y_tr_pred_b

            # calling the callbacks
            if callbacks is not None:
                for cb in callbacks:
                    cb.on_batch_ends(batch_num, is_training=True)

            tt2 = time.time()
            duration = tt2 - tt1
            sys.stdout.write('\r%04ds | epoch %02d/%02d | batch [tr] %02d/%02d | loss_tr %.02f | acc_tr %02.02f' % (duration, epoch_num, n_epochs, batch_num, n_batch_tr, loss_b, acc_b))

        # switch to eval mode
        model.eval()

        # loop on batches for test
        for idx_batch in range(n_batch_te):
            batch_num = idx_batch + 1
            idx_start = idx_batch * batch_size_te
            idx_end = (idx_batch + 1) * batch_size_te

            # get data of batch
            x_te_b = x_te[idx_start:idx_end]
            y_te_b = y_te[idx_start:idx_end]

            # convert data to tensors, and copy to gpu
            t_x_te_b = torch.from_numpy(x_te_b).cuda()
            t_y_te_b = torch.from_numpy(y_te_b).cuda()

            # nograd + forward
            t_y_te_pred_b = model(t_x_te_b)

            # append predictions
            y_te_pred_b = np.array(t_y_te_pred_b.tolist())
            y_pred_te = y_te_pred_b if y_pred_te is None else np.vstack((y_pred_te, y_te_pred_b))

            # loss + accuracy
            t_loss_b = loss_fn(t_y_te_pred_b, t_y_te_b)
            loss_b = t_loss_b.item()
            acc_b = 0.0
            loss_te += loss_b
            acc_te += acc_b
            loss_b = loss_b / float(batch_num)
            acc_b = 100 * acc_b / float(batch_num)

            del t_loss_b
            del t_x_te_b
            del t_y_te_pred_b

            # calling the callbacks
            if callbacks is not None:
                for cb in callbacks:
                    cb.on_batch_ends(batch_num, is_training=False)

            tt2 = time.time()
            duration = tt2 - tt1
            sys.stdout.write('\r%04ds | epoch %02d/%02d | batch [te] %02d/%02d | loss_te %.02f | acc_te %02.02f' % (duration, epoch_num, n_epochs, batch_num, n_batch_te, loss_b, acc_b))

        tt2 = time.time()
        duration = tt2 - tt1

        acc_te = metric_fn(y_pred_te, y_te)
        loss_tr /= float(n_batch_tr)
        loss_te /= float(n_batch_te)
        acc_tr = 100 * acc_tr
        acc_te = 100 * acc_te
        acc_max_tr = max(acc_max_tr, acc_tr)
        acc_max_te = max(acc_max_te, acc_te)

        sys.stdout.write('\r%04ds | epoch %02d/%02d | loss_tr %.02f | loss_te %.02f | acc_tr %02.02f | acc_te %02.02f | acc_max_tr %02.02f | acc_max_te %02.02f\n' % (duration, epoch_num, n_epochs, loss_tr, loss_te, acc_tr, acc_te, acc_max_tr, acc_max_te))

        # sample inputs again
        x_tr, y_tr = sampler.sample_train()
        x_te, y_te = sampler.sample_test()

        # calling the callbacks
        if callbacks is not None:
            for cb in callbacks:
                cb.on_epoch_ends(epoch_num)

def train_model_using_sampler_multi(model, optimizer, loss_fn, metric_fn, sampler, n_epochs, batch_size_tr, batch_size_te, callbacks=None):
    """
    Train using input sampler. Each epoch, we sample inputs from the sampler
    """

    # sample input
    xs_tr, y_tr = sampler.sample_train()
    xs_te, y_te = sampler.sample_test()

    n_tr = len(y_tr)
    n_te = len(y_te)

    n_batch_tr = utils.calc_num_batches(n_tr, batch_size_tr)
    n_batch_te = utils.calc_num_batches(n_te, batch_size_te)

    print('... [tr]: n, n_batch, batch_size: %d, %d, %d' % (n_tr, n_batch_tr, batch_size_tr))
    print('... [te]: n, n_batch, batch_size: %d, %d, %d' % (n_te, n_batch_te, batch_size_te))

    acc_max_tr = 0.0
    acc_max_te = 0.0

    # loop on epochs
    sys.stdout.write('\n')
    for idx_epoch in range(n_epochs):

        epoch_num = idx_epoch + 1
        loss_tr = 0.0
        loss_te = 0.0
        acc_tr = 0.0
        acc_te = 0.0
        tt1 = time.time()

        # switch to training mode
        model.train()

        # loop on batches for train
        for idx_batch in range(n_batch_tr):
            batch_num = idx_batch + 1
            idx_start = idx_batch * batch_size_tr
            idx_end = (idx_batch + 1) * batch_size_tr

            # get data of batch
            xs_tr_b = [x_tr[idx_start:idx_end] for x_tr in xs_tr]
            y_tr_b = y_tr[idx_start:idx_end]

            # convert data to tensors, and copy to gpu
            t_xs_tr_b = [torch.from_numpy(x_tr_b).cuda() for x_tr_b in xs_tr_b]
            t_y_tr_b = torch.from_numpy(y_tr_b).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + loss
            t_y_tr_pred_b = model(*t_xs_tr_b)
            t_loss_b = loss_fn(t_y_tr_pred_b, t_y_tr_b)

            # backward + optimize
            t_loss_b.backward()
            optimizer.step()
            loss_b = t_loss_b.item()

            # loss + accuracy
            acc_b = metric_fn(t_y_tr_pred_b, t_y_tr_b)
            loss_tr += loss_b
            acc_tr += acc_b
            loss_b = loss_b / float(batch_num)
            acc_b = 100 * acc_b / float(batch_num)

            del t_loss_b
            del t_xs_tr_b
            del t_y_tr_pred_b

            # calling the callbacks
            if callbacks is not None:
                for cb in callbacks:
                    cb.on_batch_ends(batch_num, is_training=True)

            tt2 = time.time()
            duration = tt2 - tt1
            sys.stdout.write('\r%04ds | epoch %02d/%02d | batch [tr] %02d/%02d | loss_tr %.02f | acc_tr %02.02f' % (duration, epoch_num, n_epochs, batch_num, n_batch_tr, loss_b, acc_b))

        # switch to eval mode
        model.eval()

        # loop on batches for test
        for idx_batch in range(n_batch_te):
            batch_num = idx_batch + 1
            idx_start = idx_batch * batch_size_te
            idx_end = (idx_batch + 1) * batch_size_te

            # get data of batch
            xs_te_b = [x_te[idx_start:idx_end] for x_te in xs_te]
            y_te_b = y_te[idx_start:idx_end]

            # convert data to tensors, and copy to gpu
            t_xs_te_b = [torch.from_numpy(x_te_b).cuda() for x_te_b in xs_te_b]
            t_y_te_b = torch.from_numpy(y_te_b).cuda()

            # nograd + forward
            t_y_te_pred_b = model(*t_xs_te_b)

            # loss + accuracy
            t_loss_b = loss_fn(t_y_te_pred_b, t_y_te_b)
            loss_b = t_loss_b.item()
            acc_b = metric_fn(t_y_te_pred_b, t_y_te_b)
            loss_te += loss_b
            acc_te += acc_b
            loss_b = loss_b / float(batch_num)
            acc_b = 100 * acc_b / float(batch_num)

            del t_loss_b
            del t_xs_te_b
            del t_y_te_pred_b

            # calling the callbacks
            if callbacks is not None:
                for cb in callbacks:
                    cb.on_batch_ends(batch_num, is_training=False)

            tt2 = time.time()
            duration = tt2 - tt1
            sys.stdout.write('\r%04ds | epoch %02d/%02d | batch [te] %02d/%02d | loss_te %.02f | acc_te %02.02f' % (duration, epoch_num, n_epochs, batch_num, n_batch_te, loss_b, acc_b))

        tt2 = time.time()
        duration = tt2 - tt1

        loss_tr /= float(n_batch_tr)
        loss_te /= float(n_batch_te)
        acc_tr = 100 * acc_tr / float(n_batch_tr)
        acc_te = 100 * acc_te / float(n_batch_te)
        acc_max_tr = max(acc_max_tr, acc_tr)
        acc_max_te = max(acc_max_te, acc_te)

        sys.stdout.write('\r%04ds | epoch %02d/%02d | loss_tr %.02f | loss_te %.02f | acc_tr %02.02f | acc_te %02.02f | acc_max_tr %02.02f | acc_max_te %02.02f\n' % (duration, epoch_num, n_epochs, loss_tr, loss_te, acc_tr, acc_te, acc_max_tr, acc_max_te))

        # sample inputs again
        xs_tr, y_tr = sampler.sample_train()
        xs_te, y_te = sampler.sample_test()

        # calling the callbacks
        if callbacks is not None:
            for cb in callbacks:
                cb.on_epoch_ends(epoch_num)

def train_model_using_sampler_multi_custom_metric(model, optimizer, loss_fn, metric_fn, sampler, n_epochs, batch_size_tr, batch_size_te, callbacks=None):
    """
    Train using input sampler. Each epoch, we sample inputs from the sampler
    """

    # sample input
    xs_tr, y_tr = sampler.sample_train()
    xs_te, y_te = sampler.sample_test()

    n_tr = len(y_tr)
    n_te = len(y_te)

    n_batch_tr = utils.calc_num_batches(n_tr, batch_size_tr)
    n_batch_te = utils.calc_num_batches(n_te, batch_size_te)

    print('... [tr]: n, n_batch, batch_size: %d, %d, %d' % (n_tr, n_batch_tr, batch_size_tr))
    print('... [te]: n, n_batch, batch_size: %d, %d, %d' % (n_te, n_batch_te, batch_size_te))

    acc_max_tr = 0.0
    acc_max_te = 0.0

    # loop on epochs
    sys.stdout.write('\n')
    for idx_epoch in range(n_epochs):

        epoch_num = idx_epoch + 1
        loss_tr = 0.0
        loss_te = 0.0
        acc_tr = 0.0
        acc_te = 0.0
        y_pred_te = None
        alpha_values = None
        tt1 = time.time()

        # switch to training mode
        model.train()

        # loop on batches for train
        for idx_batch in range(n_batch_tr):
            batch_num = idx_batch + 1
            idx_start = idx_batch * batch_size_tr
            idx_end = (idx_batch + 1) * batch_size_tr

            # get data of batch
            xs_tr_b = [x_tr[idx_start:idx_end] for x_tr in xs_tr]
            y_tr_b = y_tr[idx_start:idx_end]

            # convert data to tensors, and copy to gpu
            t_xs_tr_b = [torch.from_numpy(x_tr_b).cuda() for x_tr_b in xs_tr_b]
            t_y_tr_b = torch.from_numpy(y_tr_b).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + loss
            t_y_tr_pred_b = model(*t_xs_tr_b)
            t_loss_b = loss_fn(t_y_tr_pred_b, t_y_tr_b)

            # backward + optimize
            t_loss_b.backward()
            optimizer.step()
            loss_b = t_loss_b.item()

            # loss + accuracy
            acc_b = 0.0
            loss_tr += loss_b
            acc_tr += acc_b
            loss_b = loss_b / float(batch_num)
            acc_b = 100 * acc_b / float(batch_num)

            del t_loss_b
            del t_xs_tr_b
            del t_y_tr_pred_b

            # calling the callbacks
            if callbacks is not None:
                for cb in callbacks:
                    cb.on_batch_ends(batch_num, is_training=True)

            tt2 = time.time()
            duration = tt2 - tt1
            sys.stdout.write('\r%04ds | epoch %02d/%02d | batch [tr] %02d/%02d | loss_tr %.02f | acc_tr %02.02f' % (duration, epoch_num, n_epochs, batch_num, n_batch_tr, loss_b, acc_b))

        # switch to eval mode
        model.eval()

        # loop on batches for test
        for idx_batch in range(n_batch_te):
            batch_num = idx_batch + 1
            idx_start = idx_batch * batch_size_te
            idx_end = (idx_batch + 1) * batch_size_te

            # get data of batch
            xs_te_b = [x_te[idx_start:idx_end] for x_te in xs_te]
            y_te_b = y_te[idx_start:idx_end]

            # convert data to tensors, and copy to gpu
            t_xs_te_b = [torch.from_numpy(x_te_b).cuda() for x_te_b in xs_te_b]
            t_y_te_b = torch.from_numpy(y_te_b).cuda()

            # nograd + forward
            t_y_te_pred_b = model(*t_xs_te_b)

            # loss + accuracy
            t_loss_b = loss_fn(t_y_te_pred_b, t_y_te_b)
            loss_b = t_loss_b.item()
            acc_b = 0.0
            loss_te += loss_b
            acc_te += acc_b
            loss_b = loss_b / float(batch_num)
            acc_b = 100 * acc_b / float(batch_num)

            # append predictions
            y_te_pred_b = np.array(t_y_te_pred_b.tolist())
            y_pred_te = y_te_pred_b if y_pred_te is None else np.vstack((y_pred_te, y_te_pred_b))

            # get tensor value and append it to the list
            alpha_values_b = model_get_tensor_value(model, ('temporal_selection', 'attention_values_after'))  # (None, T)
            alpha_values = alpha_values_b if alpha_values is None else np.vstack((alpha_values, alpha_values_b))  # (None, T)

            del t_loss_b
            del t_xs_te_b
            del t_y_te_pred_b

            # calling the callbacks
            if callbacks is not None:
                for cb in callbacks:
                    cb.on_batch_ends(batch_num, is_training=False)

            tt2 = time.time()
            duration = tt2 - tt1
            sys.stdout.write('\r%04ds | epoch %02d/%02d | batch [te] %02d/%02d | loss_te %.02f | acc_te %02.02f' % (duration, epoch_num, n_epochs, batch_num, n_batch_te, loss_b, acc_b))

        tt2 = time.time()
        duration = tt2 - tt1

        acc_te = metric_fn(y_pred_te, y_te)
        loss_tr /= float(n_batch_tr)
        loss_te /= float(n_batch_te)
        acc_tr = 100 * acc_tr
        acc_te = 100 * acc_te
        acc_max_tr = max(acc_max_tr, acc_tr)
        acc_max_te = max(acc_max_te, acc_te)

        sys.stdout.write('\r%04ds | epoch %02d/%02d | loss_tr %.02f | loss_te %.02f | acc_tr %02.02f | acc_te %02.02f | acc_max_tr %02.02f | acc_max_te %02.02f\n' % (duration, epoch_num, n_epochs, loss_tr, loss_te, acc_tr, acc_te, acc_max_tr, acc_max_te))

        # sample inputs again
        xs_tr, y_tr = sampler.sample_train()
        xs_te, y_te = sampler.sample_test()

        # calling the callbacks
        if callbacks is not None:
            for cb in callbacks:
                cb.on_epoch_ends(epoch_num)

def train_model_using_async_reader(model, optimizer, loss_fn, metric_fn, reader, sampler, n_epochs, batch_size_tr, batch_size_te, callbacks=None):
    # sample image pathes
    img_pathes_tr, y_tr = sampler.sample_train()
    img_pathes_te, y_te = sampler.sample_test()

    n_tr = len(y_tr)
    n_te = len(y_te)

    n_batch_tr = utils.calc_num_batches(n_tr, batch_size_tr)
    n_batch_te = utils.calc_num_batches(n_te, batch_size_te)

    print('... [tr]: n, n_batch, batch_size: %d, %d, %d' % (n_tr, n_batch_tr, batch_size_tr))
    print('... [te]: n, n_batch, batch_size: %d, %d, %d' % (n_te, n_batch_te, batch_size_te))

    acc_max_tr = 0.0
    acc_max_te = 0.0

    # loop on epochs
    sys.stdout.write('\n')
    for idx_epoch in range(n_epochs):

        epoch_num = idx_epoch + 1
        waiting_duration = 0.0
        loss_tr = 0.0
        loss_te = 0.0
        acc_tr = 0.0
        acc_te = 0.0
        tt1 = time.time()

        # start loading images of the first training batch
        img_pathes_batch = img_pathes_tr[:batch_size_tr]  # (B, T,)
        reader.load_batch(img_pathes_batch)

        # switch to training mode
        model.train()

        # loop on batches for train
        for idx_batch in range(n_batch_tr):
            batch_num = idx_batch + 1
            idx_start = idx_batch * batch_size_tr
            idx_end = (idx_batch + 1) * batch_size_tr

            # wait if data is not ready yet
            t1 = time.time()
            while reader.is_busy():
                time.sleep(0.1)
            t2 = time.time()
            waiting_duration += (t2 - t1)

            # get data of batch, convert data to tensors, and copy to gpu
            x_tr_b = reader.get_batch()  # (B, 3, T, 224, 224)
            y_tr_b = y_tr[idx_start:idx_end]
            t_x_tr_b = torch.from_numpy(x_tr_b).float().cuda()
            t_y_tr_b = torch.from_numpy(y_tr_b).cuda()

            # start getting images for next batch
            if batch_num < n_batch_tr:
                next_idx_start = (batch_num) * batch_size_tr
                next_idx_end = (batch_num + 1) * batch_size_tr
                next_img_pathes_batch = img_pathes_tr[next_idx_start:next_idx_end]  # (B, T,)
                reader.load_batch(next_img_pathes_batch)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + loss
            t_y_tr_pred_b = model(t_x_tr_b)
            t_loss_b = loss_fn(t_y_tr_pred_b, t_y_tr_b)

            # backward + optimize
            t_loss_b.backward()
            optimizer.step()
            loss_b = t_loss_b.item()

            # loss + accuracy
            acc_b = metric_fn(t_y_tr_pred_b, t_y_tr_b)
            loss_tr += loss_b
            acc_tr += acc_b
            loss_b = loss_b / float(batch_num)
            acc_b = 100 * acc_b / float(batch_num)

            del t_x_tr_b
            del t_y_tr_b
            del t_y_tr_pred_b
            del t_loss_b

            # calling the callbacks
            if callbacks is not None:
                for cb in callbacks:
                    cb.on_batch_ends(batch_num, is_training=True)

            tt2 = time.time()
            duration = tt2 - tt1
            sys.stdout.write('\r%04ds | epoch %02d/%02d | batch [tr] %02d/%02d | loss_tr %.02f | acc_tr %02.02f' % (duration, epoch_num, n_epochs, batch_num, n_batch_tr, loss_b, acc_b))

        # start loading images of the first test batch
        img_pathes_batch = img_pathes_te[:batch_size_te]  # (B, T,)
        reader.load_batch(img_pathes_batch)

        # switch to eval mode
        model.eval()

        # loop on batches for test
        for idx_batch in range(n_batch_te):
            batch_num = idx_batch + 1
            idx_start = idx_batch * batch_size_te
            idx_end = (idx_batch + 1) * batch_size_te

            # wait if data is not ready yet
            t1 = time.time()
            while reader.is_busy():
                time.sleep(0.1)
            t2 = time.time()
            waiting_duration += (t2 - t1)

            # get data of batch, convert data to tensors, and copy to gpu
            x_te_b = reader.get_batch()  # (B, 3, T, 224, 224)
            y_te_b = y_te[idx_start:idx_end]
            t_x_te_b = torch.from_numpy(x_te_b).float().cuda()
            t_y_te_b = torch.from_numpy(y_te_b).cuda()

            # start getting images for next batch
            if batch_num < n_batch_te:
                next_idx_start = (batch_num) * batch_size_te
                next_idx_end = (batch_num + 1) * batch_size_te
                next_img_pathes_batch = img_pathes_te[next_idx_start:next_idx_end]  # (B, T,)
                reader.load_batch(next_img_pathes_batch)

            # nograd + forward
            t_y_te_pred_b = model(t_x_te_b)

            # loss + accuracy
            t_loss_b = loss_fn(t_y_te_pred_b, t_y_te_b)
            loss_b = t_loss_b.item()
            acc_b = metric_fn(t_y_te_pred_b, t_y_te_b)
            loss_te += loss_b
            acc_te += acc_b
            loss_b = loss_b / float(batch_num)
            acc_b = 100 * acc_b / float(batch_num)

            del t_x_te_b
            del t_y_te_b
            del t_y_te_pred_b
            del t_loss_b

            # calling the callbacks
            if callbacks is not None:
                for cb in callbacks:
                    cb.on_batch_ends(batch_num, is_training=False)

            tt2 = time.time()
            duration = tt2 - tt1
            sys.stdout.write('\r%04ds | epoch %02d/%02d | batch [te] %02d/%02d | loss_te %.02f | acc_te %02.02f' % (duration, epoch_num, n_epochs, batch_num, n_batch_te, loss_b, acc_b))

        # callbacks
        for cb in callbacks:
            cb.on_epoch_ends(epoch_num)

        tt2 = time.time()
        duration = tt2 - tt1

        loss_tr /= float(n_batch_tr)
        loss_te /= float(n_batch_te)
        acc_tr = 100 * acc_tr / float(n_batch_tr)
        acc_te = 100 * acc_te / float(n_batch_te)
        acc_max_tr = max(acc_max_tr, acc_tr)
        acc_max_te = max(acc_max_te, acc_te)

        sys.stdout.write('\r%04ds | epoch %02d/%02d | loss_tr %.02f | loss_te %.02f | acc_tr %02.02f | acc_te %02.02f | acc_max_tr %02.02f | acc_max_te %02.02f | waited %d\n' % (duration, epoch_num, n_epochs, loss_tr, loss_te, acc_tr, acc_te, acc_max_tr, acc_max_te, waiting_duration))

        # sample image pathes
        img_pathes_tr, y_tr = sampler.sample_train()
        img_pathes_te, y_te = sampler.sample_test()

def train_model_using_async_reader_custom_metric(model, optimizer, loss_fn, metric_fn, reader, sampler, n_epochs, batch_size_tr, batch_size_te, callbacks=None):
    # sample image pathes
    img_pathes_tr, y_tr = sampler.sample_train()
    img_pathes_te, y_te = sampler.sample_test()

    n_tr = len(y_tr)
    n_te = len(y_te)

    n_batch_tr = utils.calc_num_batches(n_tr, batch_size_tr)
    n_batch_te = utils.calc_num_batches(n_te, batch_size_te)

    print('... [tr]: n, n_batch, batch_size: %d, %d, %d' % (n_tr, n_batch_tr, batch_size_tr))
    print('... [te]: n, n_batch, batch_size: %d, %d, %d' % (n_te, n_batch_te, batch_size_te))

    acc_max_tr = 0.0
    acc_max_te = 0.0

    # loop on epochs
    sys.stdout.write('\n')
    for idx_epoch in range(n_epochs):

        epoch_num = idx_epoch + 1
        waiting_duration = 0.0
        loss_tr = 0.0
        loss_te = 0.0
        acc_tr = 0.0
        acc_te = 0.0
        y_pred_te = None
        tt1 = time.time()

        # start loading images of the first training batch
        img_pathes_batch = img_pathes_tr[:batch_size_tr]  # (B, T,)
        reader.load_batch(img_pathes_batch)

        # switch to training mode
        model.train()

        # loop on batches for train
        for idx_batch in range(n_batch_tr):
            batch_num = idx_batch + 1
            idx_start = idx_batch * batch_size_tr
            idx_end = (idx_batch + 1) * batch_size_tr

            # wait if data is not ready yet
            t1 = time.time()
            while reader.is_busy():
                time.sleep(0.1)
            t2 = time.time()
            waiting_duration += (t2 - t1)

            # get data of batch, convert data to tensors, and copy to gpu
            x_tr_b = reader.get_batch()  # (B, 3, T, 224, 224)
            y_tr_b = y_tr[idx_start:idx_end]
            t_x_tr_b = torch.from_numpy(x_tr_b).float().cuda()
            t_y_tr_b = torch.from_numpy(y_tr_b).cuda()

            # start getting images for next batch
            if batch_num < n_batch_tr:
                next_idx_start = (batch_num) * batch_size_tr
                next_idx_end = (batch_num + 1) * batch_size_tr
                next_img_pathes_batch = img_pathes_tr[next_idx_start:next_idx_end]  # (B, T,)
                reader.load_batch(next_img_pathes_batch)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + loss
            t_y_tr_pred_b = model(t_x_tr_b)
            t_loss_b = loss_fn(t_y_tr_pred_b, t_y_tr_b)

            # backward + optimize
            t_loss_b.backward()
            optimizer.step()
            loss_b = t_loss_b.item()

            # loss + accuracy
            acc_b = 0.0
            loss_tr += loss_b
            acc_tr += acc_b
            loss_b = loss_b / float(batch_num)
            acc_b = 100 * acc_b / float(batch_num)

            del t_x_tr_b
            del t_y_tr_b
            del t_y_tr_pred_b
            del t_loss_b

            # calling the callbacks
            if callbacks is not None:
                for cb in callbacks:
                    cb.on_batch_ends(batch_num, is_training=True)

            tt2 = time.time()
            duration = tt2 - tt1
            sys.stdout.write('\r%04ds | epoch %02d/%02d | batch [tr] %02d/%02d | loss_tr %.02f | acc_tr %02.02f' % (duration, epoch_num, n_epochs, batch_num, n_batch_tr, loss_b, acc_b))

        # start loading images of the first test batch
        img_pathes_batch = img_pathes_te[:batch_size_te]  # (B, T,)
        reader.load_batch(img_pathes_batch)

        # switch to eval mode
        model.eval()

        # loop on batches for test
        for idx_batch in range(n_batch_te):
            batch_num = idx_batch + 1
            idx_start = idx_batch * batch_size_te
            idx_end = (idx_batch + 1) * batch_size_te

            # wait if data is not ready yet
            t1 = time.time()
            while reader.is_busy():
                time.sleep(0.1)
            t2 = time.time()
            waiting_duration += (t2 - t1)

            # get data of batch, convert data to tensors, and copy to gpu
            x_te_b = reader.get_batch()  # (B, 3, T, 224, 224)
            y_te_b = y_te[idx_start:idx_end]
            t_x_te_b = torch.from_numpy(x_te_b).float().cuda()
            t_y_te_b = torch.from_numpy(y_te_b).cuda()

            # start getting images for next batch
            if batch_num < n_batch_te:
                next_idx_start = (batch_num) * batch_size_te
                next_idx_end = (batch_num + 1) * batch_size_te
                next_img_pathes_batch = img_pathes_te[next_idx_start:next_idx_end]  # (B, T,)
                reader.load_batch(next_img_pathes_batch)

            # nograd + forward
            t_y_te_pred_b = model(t_x_te_b)

            # loss + accuracy
            t_loss_b = loss_fn(t_y_te_pred_b, t_y_te_b)
            loss_b = t_loss_b.item()
            acc_b = 0.0
            loss_te += loss_b
            acc_te += acc_b
            loss_b = loss_b / float(batch_num)
            acc_b = 100 * acc_b / float(batch_num)

            # append predictions
            y_te_pred_b = np.array(t_y_te_pred_b.tolist())
            y_pred_te = y_te_pred_b if y_pred_te is None else np.vstack((y_pred_te, y_te_pred_b))

            del t_x_te_b
            del t_y_te_b
            del t_y_te_pred_b
            del t_loss_b

            # calling the callbacks
            if callbacks is not None:
                for cb in callbacks:
                    cb.on_batch_ends(batch_num, is_training=False)

            tt2 = time.time()
            duration = tt2 - tt1
            sys.stdout.write('\r%04ds | epoch %02d/%02d | batch [te] %02d/%02d | loss_te %.02f | acc_te %02.02f' % (duration, epoch_num, n_epochs, batch_num, n_batch_te, loss_b, acc_b))

        # callbacks
        for cb in callbacks:
            cb.on_epoch_ends(epoch_num)

        tt2 = time.time()
        duration = tt2 - tt1

        acc_te = metric_fn(y_pred_te, y_te)
        loss_tr /= float(n_batch_tr)
        loss_te /= float(n_batch_te)
        acc_tr = 100 * acc_tr
        acc_te = 100 * acc_te
        acc_max_tr = max(acc_max_tr, acc_tr)
        acc_max_te = max(acc_max_te, acc_te)

        sys.stdout.write('\r%04ds | epoch %02d/%02d | loss_tr %.02f | loss_te %.02f | acc_tr %02.02f | acc_te %02.02f | acc_max_tr %02.02f | acc_max_te %02.02f | waited %d\n' % (duration, epoch_num, n_epochs, loss_tr, loss_te, acc_tr, acc_te, acc_max_tr, acc_max_te, waiting_duration))

        # sample image pathes
        img_pathes_tr, y_tr = sampler.sample_train()
        img_pathes_te, y_te = sampler.sample_test()

# endregion

# region Model Testing

def test_model(model, loss_fn, metric_fn, x_te, y_te, batch_size_te):
    # convert input to enumerables

    n_te = len(y_te)
    n_batch_te = utils.calc_num_batches(n_te, batch_size_te)
    print('... [te]: n, n_batch, batch_size: %d, %d, %d' % (n_te, n_batch_te, batch_size_te))

    loss_te = 0.0
    acc_te = 0.0
    tt1 = time.time()

    # switch to eval mode
    model.eval()
    model.training = False

    # loop on batches for test
    for idx_batch in range(n_batch_te):
        idx_start = idx_batch * batch_size_te
        idx_end = (idx_batch + 1) * batch_size_te

        # get data of batch
        x_te_b = x_te[idx_start:idx_end]
        y_te_b = y_te[idx_start:idx_end]

        # convert data to tensors, and copy to gpu
        t_xs_te_b = torch.from_numpy(x_te_b).cuda()
        t_y_te_b = torch.from_numpy(y_te_b).cuda()

        # nograd + forward
        t_y_te_pred_b = model(t_xs_te_b)

        # loss + accuracy
        t_loss_b = loss_fn(t_y_te_pred_b, t_y_te_b)
        loss_b = t_loss_b.item()
        acc_b = metric_fn(t_y_te_pred_b, t_y_te_b)
        loss_te += loss_b
        acc_te += acc_b

    tt2 = time.time()
    duration = tt2 - tt1

    loss_te /= float(n_batch_te)
    acc_te = 100 * acc_te / float(n_batch_te)

    print('%04ds | loss_te %.02f | acc_te %02.02f' % (duration, loss_te, acc_te))

def test_model_using_sampler(model, loss_fn, metric_fn, sampler, batch_size_te):
    """
    Train using input sampler. Each epoch, we sample inputs from the sampler
    """

    # sample input

    x_te, y_te = sampler.sample_test()

    n_te = len(y_te)
    n_batch_te = utils.calc_num_batches(n_te, batch_size_te)
    print('... [te]: n, n_batch, batch_size: %d, %d, %d' % (n_te, n_batch_te, batch_size_te))

    # loop on epochs
    sys.stdout.write('\n')
    loss_te = 0.0
    acc_te = 0.0
    tt1 = time.time()

    # switch to eval mode
    model.eval()
    model.training = False

    # loop on batches for test
    for idx_batch in range(n_batch_te):
        batch_num = idx_batch + 1
        idx_start = idx_batch * batch_size_te
        idx_end = (idx_batch + 1) * batch_size_te

        # get data of batch
        x_te_b = x_te[idx_start:idx_end]
        y_te_b = y_te[idx_start:idx_end]

        # convert data to tensors, and copy to gpu
        t_x_te_b = torch.from_numpy(x_te_b).cuda()
        t_y_te_b = torch.from_numpy(y_te_b).cuda()

        # nograd + forward
        t_y_te_pred_b = model(t_x_te_b)

        # loss + accuracy
        t_loss_b = loss_fn(t_y_te_pred_b, t_y_te_b)
        loss_b = t_loss_b.item()
        acc_b = metric_fn(t_y_te_pred_b, t_y_te_b)
        loss_te += loss_b
        acc_te += acc_b
        loss_b = loss_b / float(batch_num)
        acc_b = 100 * acc_b / float(batch_num)

        tt2 = time.time()
        duration = tt2 - tt1
        sys.stdout.write('\r%04ds | batch [te] %02d/%02d | loss_te %.02f | acc_te %02.02f' % (duration, batch_num, n_batch_te, loss_b, acc_b))

    tt2 = time.time()
    duration = tt2 - tt1

    loss_te /= float(n_batch_te)
    acc_te = 100 * acc_te / float(n_batch_te)

    sys.stdout.write('\r%04ds | loss_te %.02f | acc_te %02.02f                            \n' % (duration, loss_te, acc_te))

def test_model_using_sampler_n_epochs(model, loss_fn, metric_fn, sampler, n_epochs, batch_size_te):
    """
    Train using input sampler. Each epoch, we sample inputs from the sampler
    """

    # sample input
    x_te, y_te = sampler.sample_test()
    n_te = len(y_te)
    n_batch_te = utils.calc_num_batches(n_te, batch_size_te)
    print('... [te]: n, n_batch, batch_size: %d, %d, %d' % (n_te, n_batch_te, batch_size_te))

    # loop on epochs
    sys.stdout.write('\n')
    losses_te = []
    accs_te = []

    # switch to eval mode
    model.eval()
    model.training = False

    for idx_epoch in range(n_epochs):
        epoch_num = idx_epoch
        loss_te = 0.0
        acc_te = 0.0
        tt1 = time.time()

        # loop on batches for test
        for idx_batch in range(n_batch_te):
            batch_num = idx_batch + 1
            idx_start = idx_batch * batch_size_te
            idx_end = (idx_batch + 1) * batch_size_te

            # get data of batch
            x_te_b = x_te[idx_start:idx_end]
            y_te_b = y_te[idx_start:idx_end]

            # convert data to tensors, and copy to gpu
            t_x_te_b = torch.from_numpy(x_te_b).cuda()
            t_y_te_b = torch.from_numpy(y_te_b).cuda()

            # nograd + forward
            t_y_te_pred_b = model(t_x_te_b)

            # loss + accuracy
            t_loss_b = loss_fn(t_y_te_pred_b, t_y_te_b)
            loss_b = t_loss_b.item()
            acc_b = metric_fn(t_y_te_pred_b, t_y_te_b)
            loss_te += loss_b
            acc_te += acc_b
            loss_b = loss_b / float(batch_num)
            acc_b = 100 * acc_b / float(batch_num)

            tt2 = time.time()
            duration = tt2 - tt1
            sys.stdout.write('\r%04ds | epoch %02d/%02d | batch [te] %02d/%02d | loss_te %.02f | acc_te %02.02f' % (duration, epoch_num, n_epochs, batch_num, n_batch_te, loss_b, acc_b))

        tt2 = time.time()
        duration = tt2 - tt1

        loss_te /= float(n_batch_te)
        acc_te = 100 * acc_te / float(n_batch_te)

        # save current loss and acc for epoch
        losses_te.append(loss_te)
        accs_te.append(acc_te)

        # sample input
        x_te, y_te = sampler.sample_test()

        sys.stdout.write('\r%04ds | epoch %02d/%02d | loss_te %.02f | acc_te %02.02f   \n' % (duration, epoch_num, n_epochs, loss_te, acc_te))

    # average losses
    loss_te = np.mean(losses_te)
    acc_te = np.mean(accs_te)

    sys.stdout.write('\raverage_loss_te %.02f | average_acc_te %02.02f   \n' % (loss_te, acc_te))

def test_model_using_sampler_n_epochs_custom_metric(model, loss_fn, metric_fn, sampler, n_epochs, batch_size_te):
    """
    Train using input sampler. Each epoch, we sample inputs from the sampler
    """

    # sample input
    x_te, y_te = sampler.sample_test()
    n_te = len(y_te)
    n_batch_te = utils.calc_num_batches(n_te, batch_size_te)
    print('... [te]: n, n_batch, batch_size: %d, %d, %d' % (n_te, n_batch_te, batch_size_te))

    # loop on epochs
    sys.stdout.write('\n')
    losses_te = []
    accs_te = []

    # switch to eval mode
    model.eval()
    model.training = False

    for idx_epoch in range(n_epochs):
        epoch_num = idx_epoch
        loss_te = 0.0
        acc_te = 0.0
        y_pred_te = None
        tt1 = time.time()

        # loop on batches for test
        for idx_batch in range(n_batch_te):
            batch_num = idx_batch + 1
            idx_start = idx_batch * batch_size_te
            idx_end = (idx_batch + 1) * batch_size_te

            # get data of batch
            x_te_b = x_te[idx_start:idx_end]
            y_te_b = y_te[idx_start:idx_end]

            # convert data to tensors, and copy to gpu
            t_x_te_b = torch.from_numpy(x_te_b).cuda()
            t_y_te_b = torch.from_numpy(y_te_b).cuda()

            # nograd + forward
            t_y_te_pred_b = model(t_x_te_b)
            y_te_pred_b = np.array(t_y_te_pred_b.tolist())

            y_pred_te = y_te_pred_b if y_pred_te is None else np.vstack((y_pred_te, y_te_pred_b))

            # loss + accuracy
            t_loss_b = loss_fn(t_y_te_pred_b, t_y_te_b)
            loss_b = t_loss_b.item()
            acc_b = 0.0
            loss_te += loss_b
            acc_te += acc_b
            loss_b = loss_b / float(batch_num)
            acc_b = 100 * acc_b / float(batch_num)

            tt2 = time.time()
            duration = tt2 - tt1
            sys.stdout.write('\r%04ds | epoch %02d/%02d | batch [te] %02d/%02d | loss_te %.02f | acc_te %02.02f' % (duration, epoch_num, n_epochs, batch_num, n_batch_te, loss_b, acc_b))

        tt2 = time.time()
        duration = tt2 - tt1

        acc_te = metric_fn(y_pred_te, y_te)
        loss_te /= float(n_batch_te)
        acc_te = 100 * acc_te

        # save current loss and acc for epoch
        losses_te.append(loss_te)
        accs_te.append(acc_te)

        # sample input
        x_te, y_te = sampler.sample_test()

        sys.stdout.write('\r%04ds | epoch %02d/%02d | loss_te %.02f | acc_te %02.02f   \n' % (duration, epoch_num, n_epochs, loss_te, acc_te))

    # average losses
    loss_te = np.mean(losses_te)
    acc_te = np.mean(accs_te)

    sys.stdout.write('\raverage_loss_te %.02f | average_acc_te %02.02f   \n' % (loss_te, acc_te))

def test_model_using_sampler_multi(model, loss_fn, metric_fn, sampler, batch_size_te, callbacks=None):
    """
    Train using input sampler. Each epoch, we sample inputs from the sampler
    """

    # sample input
    xs_te, y_te = sampler.sample_test()

    n_te = len(y_te)
    n_batch_te = utils.calc_num_batches(n_te, batch_size_te)
    print('... [te]: n, n_batch, batch_size: %d, %d, %d' % (n_te, n_batch_te, batch_size_te))

    # switch to eval mode
    model.eval()
    model.training = False

    # loop on epochs
    sys.stdout.write('\n')

    loss_te = 0.0
    acc_te = 0.0
    epoch_num = 1
    tt1 = time.time()

    # loop on batches for test
    for idx_batch in range(n_batch_te):
        batch_num = idx_batch + 1
        idx_start = idx_batch * batch_size_te
        idx_end = (idx_batch + 1) * batch_size_te

        # get data of batch
        xs_te_b = [x_te[idx_start:idx_end] for x_te in xs_te]
        y_te_b = y_te[idx_start:idx_end]

        # convert data to tensors, and copy to gpu
        t_xs_te_b = [torch.from_numpy(x_te_b).cuda() for x_te_b in xs_te_b]
        t_y_te_b = torch.from_numpy(y_te_b).cuda()

        # nograd + forward
        t_y_te_pred_b = model(*t_xs_te_b)

        # loss + accuracy
        t_loss_b = loss_fn(t_y_te_pred_b, t_y_te_b)
        loss_b = t_loss_b.item()
        acc_b = metric_fn(t_y_te_pred_b, t_y_te_b)
        loss_te += loss_b
        acc_te += acc_b

        del t_loss_b
        del t_xs_te_b
        del t_y_te_pred_b

        # calling the callbacks
        if callbacks is not None:
            for cb in callbacks:
                cb.on_batch_ends(batch_num, is_training=False)

    tt2 = time.time()
    duration = tt2 - tt1

    loss_te /= float(n_batch_te)
    acc_te = 100 * acc_te / float(n_batch_te)

    sys.stdout.write('\r%04ds loss_te %.02f | acc_te %02.02f ' % (duration, loss_te, acc_te))

    # calling the callbacks
    if callbacks is not None:
        for cb in callbacks:
            cb.on_epoch_ends(epoch_num)

def test_model_using_sampler_multi_n_epochs(model, loss_fn, metric_fn, sampler, n_epochs, batch_size_te, callbacks=None):
    """
    Train using input sampler. Each epoch, we sample inputs from the sampler
    """

    # sample input
    xs_te, y_te = sampler.sample_test()

    n_te = len(y_te)
    n_batch_te = utils.calc_num_batches(n_te, batch_size_te)
    print('... [te]: n, n_batch, batch_size: %d, %d, %d' % (n_te, n_batch_te, batch_size_te))

    # switch to eval mode
    model.eval()
    model.training = False

    losses_te = []
    accs_te = []

    # loop on epochs
    sys.stdout.write('\n')
    for idx_epoch in range(n_epochs):

        epoch_num = idx_epoch + 1
        loss_te = 0.0
        acc_te = 0.0
        tt1 = time.time()

        # loop on batches for test
        for idx_batch in range(n_batch_te):
            batch_num = idx_batch + 1
            idx_start = idx_batch * batch_size_te
            idx_end = (idx_batch + 1) * batch_size_te

            # get data of batch
            xs_te_b = [x_te[idx_start:idx_end] for x_te in xs_te]
            y_te_b = y_te[idx_start:idx_end]

            # convert data to tensors, and copy to gpu
            t_xs_te_b = [torch.from_numpy(x_te_b).cuda() for x_te_b in xs_te_b]
            t_y_te_b = torch.from_numpy(y_te_b).cuda()

            # nograd + forward
            t_y_te_pred_b = model(*t_xs_te_b)

            # loss + accuracy
            t_loss_b = loss_fn(t_y_te_pred_b, t_y_te_b)
            loss_b = t_loss_b.item()
            acc_b = metric_fn(t_y_te_pred_b, t_y_te_b)
            loss_te += loss_b
            acc_te += acc_b
            loss_b = loss_b / float(batch_num)
            acc_b = 100 * acc_b / float(batch_num)

            del t_loss_b
            del t_xs_te_b
            del t_y_te_pred_b

            # calling the callbacks
            if callbacks is not None:
                for cb in callbacks:
                    cb.on_batch_ends(batch_num, is_training=False)

            tt2 = time.time()
            duration = tt2 - tt1
            sys.stdout.write('\r%04ds | epoch %02d/%02d | batch [te] %02d/%02d | loss_te %.02f | acc_te %02.02f' % (duration, epoch_num, n_epochs, batch_num, n_batch_te, loss_b, acc_b))

        tt2 = time.time()
        duration = tt2 - tt1

        loss_te /= float(n_batch_te)
        acc_te = 100 * acc_te / float(n_batch_te)

        # save current loss and acc for epoch
        losses_te.append(loss_te)
        accs_te.append(acc_te)

        sys.stdout.write('\r%04ds | epoch %02d/%02d | loss_te %.02f | acc_te %02.02f' % (duration, epoch_num, n_epochs, loss_te, acc_te))

        # calling the callbacks
        if callbacks is not None:
            for cb in callbacks:
                cb.on_epoch_ends(epoch_num)

        # sample inputs again
        xs_te, y_te = sampler.sample_test()

    # average losses
    loss_te = np.mean(losses_te)
    acc_te = np.mean(accs_te)

    sys.stdout.write('\raverage_loss_te %.02f | average_acc_te %02.02f' % (loss_te, acc_te))

def test_model_using_sampler_multi_n_epochs_custom_metric(model, loss_fn, metric_fn, sampler, n_epochs, batch_size_te, callbacks=None):
    """
    Train using input sampler. Each epoch, we sample inputs from the sampler
    """

    # sample input
    xs_te, y_te = sampler.sample_test()

    n_te = len(y_te)
    n_batch_te = utils.calc_num_batches(n_te, batch_size_te)
    print('... [te]: n, n_batch, batch_size: %d, %d, %d' % (n_te, n_batch_te, batch_size_te))

    # switch to eval mode
    model.eval()
    model.training = False

    losses_te = []
    accs_te = []

    # loop on epochs
    sys.stdout.write('\n')
    for idx_epoch in range(n_epochs):

        epoch_num = idx_epoch + 1
        loss_te = 0.0
        acc_te = 0.0
        y_pred_te = None
        tt1 = time.time()

        # loop on batches for test
        for idx_batch in range(n_batch_te):
            batch_num = idx_batch + 1
            idx_start = idx_batch * batch_size_te
            idx_end = (idx_batch + 1) * batch_size_te

            # get data of batch
            xs_te_b = [x_te[idx_start:idx_end] for x_te in xs_te]
            y_te_b = y_te[idx_start:idx_end]

            # convert data to tensors, and copy to gpu
            t_xs_te_b = [torch.from_numpy(x_te_b).cuda() for x_te_b in xs_te_b]
            t_y_te_b = torch.from_numpy(y_te_b).cuda()

            # nograd + forward
            t_y_te_pred_b = model(*t_xs_te_b)

            y_pred_te_b = np.array(t_y_te_pred_b.tolist())
            y_pred_te = y_pred_te_b if y_pred_te is None else np.vstack((y_pred_te, y_pred_te_b))

            # loss + accuracy
            t_loss_b = loss_fn(t_y_te_pred_b, t_y_te_b)
            loss_b = t_loss_b.item()
            acc_b = 0.0
            loss_te += loss_b
            acc_te += acc_b
            loss_b = loss_b / float(batch_num)
            acc_b = 100 * acc_b / float(batch_num)

            del t_loss_b
            del t_xs_te_b
            del t_y_te_pred_b

            # calling the callbacks
            if callbacks is not None:
                for cb in callbacks:
                    cb.on_batch_ends(batch_num, is_training=False)

            tt2 = time.time()
            duration = tt2 - tt1
            sys.stdout.write('\r%04ds | epoch %02d/%02d | batch [te] %02d/%02d | loss_te %.02f | acc_te %02.02f' % (duration, epoch_num, n_epochs, batch_num, n_batch_te, loss_b, acc_b))

        tt2 = time.time()
        duration = tt2 - tt1

        acc_te = metric_fn(y_pred_te, y_te)
        loss_te /= float(n_batch_te)
        acc_te = 100 * acc_te

        # save current loss and acc for epoch
        losses_te.append(loss_te)
        accs_te.append(acc_te)

        sys.stdout.write('\r%04ds | epoch %02d/%02d | loss_te %.02f | acc_te %02.02f' % (duration, epoch_num, n_epochs, loss_te, acc_te))

        # calling the callbacks
        if callbacks is not None:
            for cb in callbacks:
                cb.on_epoch_ends(epoch_num)

        # sample inputs again
        xs_te, y_te = sampler.sample_test()

    # average losses
    loss_te = np.mean(losses_te)
    acc_te = np.mean(accs_te)

    sys.stdout.write('\raverage_loss_te %.02f | average_acc_te %02.02f' % (loss_te, acc_te))

def test_model_using_async_reader_custom_metric(model, loss_fn, metric_fn, reader, sampler, batch_size_te, callbacks=None):
    # sample image pathes
    img_pathes_te, y_te = sampler.sample_test()
    n_te = len(y_te)
    n_batch_te = utils.calc_num_batches(n_te, batch_size_te)
    print('... [te]: n, n_batch, batch_size: %d, %d, %d' % (n_te, n_batch_te, batch_size_te))

    # loop on epochs
    sys.stdout.write('\n')
    waiting_duration = 0.0
    loss_te = 0.0
    acc_te = 0.0
    y_pred_te = None
    tt1 = time.time()

    # start loading images of the first test batch
    img_pathes_batch = img_pathes_te[:batch_size_te]  # (B, T,)
    reader.load_batch(img_pathes_batch)

    # switch to eval mode
    model.eval()

    # loop on batches for test
    for idx_batch in range(n_batch_te):
        batch_num = idx_batch + 1
        idx_start = idx_batch * batch_size_te
        idx_end = (idx_batch + 1) * batch_size_te

        # wait if data is not ready yet
        t1 = time.time()
        while reader.is_busy():
            time.sleep(0.1)
        t2 = time.time()
        waiting_duration += (t2 - t1)

        # get data of batch, convert data to tensors, and copy to gpu
        x_te_b = reader.get_batch()  # (B, 3, T, 224, 224)
        y_te_b = y_te[idx_start:idx_end]
        t_x_te_b = torch.from_numpy(x_te_b).float().cuda()
        t_y_te_b = torch.from_numpy(y_te_b).cuda()

        # start getting images for next batch
        if batch_num < n_batch_te:
            next_idx_start = (batch_num) * batch_size_te
            next_idx_end = (batch_num + 1) * batch_size_te
            next_img_pathes_batch = img_pathes_te[next_idx_start:next_idx_end]  # (B, T,)
            reader.load_batch(next_img_pathes_batch)

        # nograd + forward
        t_y_te_pred_b = model(t_x_te_b)

        # loss + accuracy
        t_loss_b = loss_fn(t_y_te_pred_b, t_y_te_b)
        loss_b = t_loss_b.item()
        acc_b = 0.0
        loss_te += loss_b
        acc_te += acc_b
        loss_b = loss_b / float(batch_num)
        acc_b = 100 * acc_b / float(batch_num)

        # append predictions
        y_te_pred_b = np.array(t_y_te_pred_b.tolist())
        y_pred_te = y_te_pred_b if y_pred_te is None else np.vstack((y_pred_te, y_te_pred_b))

        del t_x_te_b
        del t_y_te_b
        del t_y_te_pred_b
        del t_loss_b

        # calling the callbacks
        if callbacks is not None:
            for cb in callbacks:
                cb.on_batch_ends(batch_num, is_training=False)

        tt2 = time.time()
        duration = tt2 - tt1
        sys.stdout.write('\r%04ds | batch [te] %02d/%02d | loss_te %.02f | acc_te %02.02f' % (duration, batch_num, n_batch_te, loss_b, acc_b))

    # callbacks
    for cb in callbacks:
        cb.on_epoch_ends(1)

    tt2 = time.time()
    duration = tt2 - tt1

    # calc amp
    acc_te = metric_fn(y_pred_te, y_te)
    loss_te /= float(n_batch_te)
    acc_te = 100 * acc_te

    sys.stdout.write('\r%04ds | loss_te %.02f | acc_te %02.02f | waited %d\n' % (duration, loss_te, acc_te, waiting_duration))

def test_model_prediction(model, x_tr, x_te, batch_size_tr, batch_size_te):
    n_tr = len(x_tr)
    n_te = len(x_te)

    n_batch_tr = utils.calc_num_batches(n_tr, batch_size_tr)
    n_batch_te = utils.calc_num_batches(n_te, batch_size_te)

    print('... [tr]: n, n_batch, batch_size: %d, %d, %d' % (n_tr, n_batch_tr, batch_size_tr))
    print('... [te]: n, n_batch, batch_size: %d, %d, %d' % (n_te, n_batch_te, batch_size_te))

    y_tr_pred = None
    y_te_pred = None

    # switch to eval mode
    model.eval()

    # loop on batches for train
    for idx_batch in range(n_batch_tr):
        idx_start = idx_batch * batch_size_tr
        idx_end = (idx_batch + 1) * batch_size_tr

        # get data of batch
        x_tr_b = x_tr[idx_start:idx_end]

        # convert data to tensors, and copy to gpu
        t_x_tr_b = torch.from_numpy(x_tr_b).float().cuda()

        # nograd + forward
        t_y_tr_pred_b = model(t_x_tr_b)
        y_tr_pred_b = np.array(t_y_tr_pred_b.tolist())
        y_tr_pred = y_tr_pred_b if y_tr_pred is None else np.vstack((y_tr_pred, y_tr_pred_b))

        del t_x_tr_b
        del t_y_tr_pred_b

    # loop on batches for test
    for idx_batch in range(n_batch_te):
        idx_start = idx_batch * batch_size_te
        idx_end = (idx_batch + 1) * batch_size_te

        # get data of batch
        x_te_b = x_te[idx_start:idx_end]

        # convert data to tensors, and copy to gpu
        t_x_te_b = torch.from_numpy(x_te_b).float().cuda()

        # nograd + forward
        t_y_te_pred_b = model(t_x_te_b)
        y_te_pred_b = np.array(t_y_te_pred_b.tolist())

        y_te_pred = y_te_pred_b if y_te_pred is None else np.vstack((y_te_pred, y_te_pred_b))

        del t_x_te_b
        del t_y_te_pred_b

    return y_tr_pred, y_te_pred

# endregion

# region Model Feedforward

def model_get_tensor_using_features(model, inputs, batch_size, tensor_name_stack):
    """
    Feedforward data all the way to the model, then get a value of a tensor.
    :param model:
    :param x:
    :param y:
    :return:
    """

    # convert input to enumerable
    xs = inputs if utils.is_enumerable(inputs) else [inputs]

    n_samples = len(xs[0])
    n_batch = utils.calc_num_batches(n_samples, batch_size)

    # switch to eval mode
    model.eval()

    value = None

    # loop on batches for test
    for idx_batch in range(n_batch):
        batch_num = idx_batch + 1
        idx_start = idx_batch * batch_size
        idx_end = (idx_batch + 1) * batch_size

        # get data of batch
        xs_b = [x[idx_start:idx_end] for x in xs]

        # convert data to tensors, and copy to gpu
        t_xs_b = [torch.from_numpy(x_b).float().cuda() for x_b in xs_b]

        # nograd + forward
        model(*t_xs_b)

        # get value and accumulate
        v_b = model_get_tensor_value(model, tensor_name_stack)
        value = v_b if value is None else np.vstack((value, v_b))

    return value

def model_get_tensor_async_video_reader(model, x, batch_size, tensor_name_stacks, async_reader, n_timesteps):
    """
    Feedforward data all the way to the model, then get a value of a tensor.
    :param model:
    :param x:
    :param y:
    :return:
    """

    n_samples = len(x)
    n_batch = utils.calc_num_batches(n_samples, batch_size)

    # switch to eval mode
    model.eval()

    value1 = None
    value2 = None

    # start loading images of the first test batch
    x_b = x[:batch_size]
    async_reader.load_batch(x_b)

    # loop on batches for test
    for idx_batch in range(n_batch):
        batch_num = idx_batch + 1
        idx_start = idx_batch * batch_size
        idx_end = (idx_batch + 1) * batch_size

        # wait if loading is not done
        while async_reader.is_busy():
            time.sleep(0.1)

        # convert data to tensors, and copy to gpu
        x_b = async_reader.get_batch()  # (B*T, 224, 224, 3)
        x_b = torch.from_numpy(x_b).float().cuda()

        # start getting images for next batch
        if batch_num < n_batch:
            next_idx_start = (batch_num) * batch_size
            next_idx_end = (batch_num + 1) * batch_size
            next_x_b = x[next_idx_start:next_idx_end]  # (B, T,)
            async_reader.load_batch(next_x_b)

        # nograd + forward
        model(x_b)

        v1_b = model_get_tensor_value(model, tensor_name_stacks[0])
        v2_b = model_get_tensor_value(model, tensor_name_stacks[1])

        value1 = v1_b if value1 is None else np.vstack((value1, v1_b))
        value2 = v2_b if value2 is None else np.vstack((value2, v2_b))

        # delete all tensors
        del x_b

    return value1, value2

def model_get_tensor_async_image_reader(model, x, batch_size, tensor_name_stacks, async_reader, n_timesteps):
    """
    Feedforward data all the way to the model, then get a value of a tensor.
    :param model:
    :param x:
    :param y:
    :return:
    """

    n_samples = len(x)
    n_batch = utils.calc_num_batches(n_samples, batch_size)

    # switch to eval mode
    model.eval()

    value1 = None
    value2 = None

    # start loading images of the first test batch
    x_b = x[0:batch_size]
    x_b = np.reshape(x_b, (-1,))  # (B*T,)
    async_reader.load_batch(x_b)

    # loop on batches for test
    for idx_batch in range(n_batch):
        batch_num = idx_batch + 1
        idx_start = idx_batch * batch_size
        idx_end = (idx_batch + 1) * batch_size

        # wait if loading is not done
        while async_reader.is_busy():
            time.sleep(0.1)

        # convert data to tensors, and copy to gpu
        x_b = async_reader.get_batch()  # (B*T, 224, 224, 3)
        x_b = np.reshape(x_b, (-1, n_timesteps, 224, 224, 3))  # (B, T, 224, 224, 3)
        x_b = np.transpose(x_b, (0, 4, 1, 2, 3))  # (B, 3, T, 224, 224)
        x_b = torch.from_numpy(x_b).float().cuda()

        # start getting images for next batch
        if batch_num < n_batch:
            next_idx_start = (batch_num) * batch_size
            next_idx_end = (batch_num + 1) * batch_size
            next_x_b = x[next_idx_start:next_idx_end]  # (B, T,)
            next_x_b = np.reshape(next_x_b, (-1,))  # (B*T,)
            async_reader.load_batch(next_x_b)

        # nograd + forward
        model(x_b)

        v1_b = model_get_tensor_value(model, tensor_name_stacks[0])
        v2_b = model_get_tensor_value(model, tensor_name_stacks[1])

        value1 = v1_b if value1 is None else np.vstack((value1, v1_b))
        value2 = v2_b if value2 is None else np.vstack((value2, v2_b))

        # delete all tensors
        del x_b

    return value1, value2

def model_get_tensor_async_feature_reader(model, x, batch_size, tensor_name_stacks, async_reader, n_timesteps):
    """
    Feedforward data all the way to the model, then get a value of a tensor.
    :param model:
    :param x:
    :param y:
    :return:
    """

    n_samples = len(x)
    n_batch = utils.calc_num_batches(n_samples, batch_size)

    # switch to eval mode
    model.eval()

    value1 = None
    value2 = None

    # start loading images of the first test batch
    x_b = x[0:batch_size]
    async_reader.load_batch(x_b)

    # loop on batches for test
    for idx_batch in range(n_batch):
        batch_num = idx_batch + 1
        idx_start = idx_batch * batch_size
        idx_end = (idx_batch + 1) * batch_size

        # wait if loading is not done
        while async_reader.is_busy():
            time.sleep(0.1)

        # convert data to tensors, and copy to gpu
        x_b = async_reader.get_features()  # (B, T, C, H, W)
        x_b = np.transpose(x_b, (0, 2, 1, 3, 4))  # (B, C, T, H, W)
        x_b = torch.from_numpy(x_b).float().cuda()

        # start getting images for next batch
        if batch_num < n_batch:
            next_idx_start = (batch_num) * batch_size
            next_idx_end = (batch_num + 1) * batch_size
            next_x_b = x[next_idx_start:next_idx_end]  # (B, T,)
            async_reader.load_batch(next_x_b)

        # nograd + forward
        model(x_b)

        v1_b = model_get_tensor_value(model, tensor_name_stacks[0])
        v2_b = model_get_tensor_value(model, tensor_name_stacks[1])

        value1 = v1_b if value1 is None else np.vstack((value1, v1_b))
        value2 = v2_b if value2 is None else np.vstack((value2, v2_b))

        # delete all tensors
        del x_b

    return value1, value2

def model_get_tensor_value(model, tensor_name_stack):
    # get value and accumulate
    class_attr = None
    class_instance = model
    for n in tensor_name_stack:
        class_attr = getattr(class_instance, n)
        class_instance = class_attr
    v_b = class_attr

    return v_b

def __test_waiting_time(reader, sampler, n_epochs, batch_size_tr, batch_size_te):
    # sample image pathes
    img_pathes_tr, y_tr = sampler.sample_train()
    img_pathes_te, y_te = sampler.sample_test()

    n_tr = len(y_tr)
    n_te = len(y_te)

    n_batch_tr = utils.calc_num_batches(n_tr, batch_size_tr)
    n_batch_te = utils.calc_num_batches(n_te, batch_size_te)

    # loop on epochs
    sys.stdout.write('\n')
    for idx_epoch in range(n_epochs):

        epoch_num = idx_epoch + 1
        waiting_duration = 0.0
        tt1 = time.time()

        # start loading images of the first training batch
        img_pathes_batch = img_pathes_tr[:batch_size_tr]  # (B, T,)
        reader.load_batch(img_pathes_batch)

        # loop on batches for train
        for idx_batch in range(n_batch_tr):
            batch_num = idx_batch + 1

            # wait if data is not ready yet
            t1 = time.time()
            while reader.is_busy():
                time.sleep(0.1)
            t2 = time.time()
            waiting_duration += (t2 - t1)

            # get data of batch, convert data to tensors, and copy to gpu
            x_tr_b = reader.get_batch()  # (B, 3, T, 224, 224)

            # start getting images for next batch
            if batch_num < n_batch_tr:
                next_idx_start = (batch_num) * batch_size_tr
                next_idx_end = (batch_num + 1) * batch_size_tr
                next_img_pathes_batch = img_pathes_tr[next_idx_start:next_idx_end]  # (B, T,)
                reader.load_batch(next_img_pathes_batch)

            tt2 = time.time()
            duration = tt2 - tt1
            sys.stdout.write('\r%04ds | epoch %02d/%02d | batch [tr] %02d/%02d' % (duration, epoch_num, n_epochs, batch_num, n_batch_tr))

        # start loading images of the first test batch
        img_pathes_batch = img_pathes_te[:batch_size_te]  # (B, T,)
        reader.load_batch(img_pathes_batch)

        # loop on batches for test
        for idx_batch in range(n_batch_te):
            batch_num = idx_batch + 1

            # wait if data is not ready yet
            t1 = time.time()
            while reader.is_busy():
                time.sleep(0.1)
            t2 = time.time()
            waiting_duration += (t2 - t1)

            # get data of batch, convert data to tensors, and copy to gpu
            x_te_b = reader.get_batch()  # (B, 3, T, 224, 224)

            # start getting images for next batch
            if batch_num < n_batch_te:
                next_idx_start = (batch_num) * batch_size_te
                next_idx_end = (batch_num + 1) * batch_size_te
                next_img_pathes_batch = img_pathes_te[next_idx_start:next_idx_end]  # (B, T,)
                reader.load_batch(next_img_pathes_batch)

            tt2 = time.time()
            duration = tt2 - tt1
            sys.stdout.write('\r%04ds | epoch %02d/%02d | batch [te] %02d/%02d' % (duration, epoch_num, n_epochs, batch_num, n_batch_te))

        tt2 = time.time()
        duration = tt2 - tt1

        sys.stdout.write('\r%04ds | epoch %02d/%02d | waited %d\n' % (duration, epoch_num, n_epochs, waiting_duration))

        # sample image pathes
        img_pathes_tr, y_tr = sampler.sample_train()
        img_pathes_te, y_te = sampler.sample_test()

# endregion

# region Model Evaluating

def batched_feedforward(model, x, batch_size, func_name=None, output_type=np.float32):
    n = len(x)
    n_batch = utils.calc_num_batches(n, batch_size)

    forward_func = model if func_name is None else getattr(model, func_name)
    y = None

    # loop on batches
    for idx_batch in range(n_batch):

        batch_num = idx_batch + 1
        idx_start = idx_batch * batch_size
        idx_end = (idx_batch + 1) * batch_size

        # get data of batch
        x_b = x[idx_start:idx_end]

        # convert data to tensors, and copy to gpu
        x_b = torch.from_numpy(x_b).cuda()


        # forward
        y_b = forward_func(x_b)
        y_b = y_b.detach().cpu().numpy()

        # define output if not defined yet
        if y is None:
            output_shape = y_b.shape[1:]
            output_shape = [n] + list(output_shape)
            y = np.zeros(output_shape, dtype=output_type)

        # append output
        y[idx_start:idx_end] = y_b

    return y


def batched_feedforward_multi(model, xs, batch_size, func_name=None, output_type=np.float32):
    # convert input to enumerables
    xs = xs if utils.is_enumerable(xs) else [xs]
    n = len(xs[0])
    n_batch = utils.calc_num_batches(n, batch_size)

    forward_func = model if func_name is None else getattr(model, func_name)
    y = None

    # loop on batches
    for idx_batch in range(n_batch):

        batch_num = idx_batch + 1
        idx_start = idx_batch * batch_size
        idx_end = (idx_batch + 1) * batch_size

        # get data of batch
        xs_b = [x[idx_start:idx_end] for x in xs]

        xs_b = [torch.from_numpy(x_tr_b).float().cuda() for x_tr_b in xs_b]

        # forward
        y_b = forward_func(*xs_b)
        y_b = y_b.detach().cpu().numpy()

        # define output if not defined yet
        if y is None:
            output_shape = y_b.shape[1:]
            output_shape = [n] + list(output_shape)
            y = np.zeros(output_shape, dtype=output_type)

        # append output
        y[idx_start:idx_end] = y_b

    return y


# endregion

# region Model Save/Load

def save_model(model, path):
    torch.save(model, path)

def load_model(path):
    model = torch.load(path)
    return model

def save_model_dict(model, path):
    model_dict = model.state_dict()
    torch.save(model_dict, path)

def load_model_dict(model, path, strict=True):
    model_dict = torch.load(path)
    model.load_state_dict(model_dict, strict=strict)
    return model

def load_model_checkpoint(model, path, strict=True):
    checkpoint = torch.load(path)
    model_dict = checkpoint['state_dict']
    model.load_state_dict(model_dict, strict=strict)
    return model

# endregion

# region Model Summary/Visualization

def model_summary(model, input_size, batch_size=-1, device="cuda"):
    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("-------------------------------------------------------------------------------------------------------")
    line_new = "{:>50}  {:>30} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("=======================================================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>50}  {:>30} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("=======================================================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("-------------------------------------------------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("-------------------------------------------------------------------------------------------------------")
    # return summary

def model_summary_multi_input(model, input_sizes, batch_size=-1, device="cuda"):
    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_sizes]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("-------------------------------------------------------------------------------------------------------")
    line_new = "{:>50}  {:>30} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("=======================================================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>50}  {:>30} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = sum([abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.)) for input_size in input_sizes])
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("=======================================================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("-------------------------------------------------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("-------------------------------------------------------------------------------------------------------")
    # return summary

def visualize_model(model, input_shape):
    # add batch size
    input_shape = [2] + list(input_shape)

    # create input tensor
    input_tensor = torch.randn(input_shape)
    input_tensor = torch.autograd.Variable(input_tensor).cuda()

    # feedforward to get output tensor
    output_tensor = model(input_tensor)

    # get model params
    model_params = dict(model.named_parameters())

    # plot graph
    g = make_dot(output_tensor.mean(), params=model_params)

    g.view()

def export_model_definition(model, input_shape, model_path='model.onnx'):
    # add batch size
    input_shape = [2] + list(input_shape)

    # create input tensor
    input_tensor = torch.randn(input_shape)
    input_tensor = torch.autograd.Variable(input_tensor).cuda()

    torch.onnx.export(model, input_tensor, model_path)

def calc_model_params(model, mode='MB'):
    KB = 1000.0
    MB = 1000 * KB
    GB = 1000 * MB
    parms = sum(p.numel() for p in model.parameters())

    if mode == 'GB':
        parms /= GB
    elif mode == 'MB':
        parms /= MB
    elif mode == 'KB':
        parms /= KB
    else:
        raise Exception('Sorry, unsupported mode: %s' % (mode))

    return parms

def calc_model_flops(model, batch_shape, mode='M'):
    input = torch.randn(*batch_shape).cuda()
    flops, _ = profile(model, (input,))
    if mode is None:
        pass
    elif mode == 'K':
        flops /= (1024.0)
    elif mode == 'M':
        flops /= (1024.0 * 1024.0)
    elif mode == 'G':
        flops /= (1024.0 * 1024.0 * 1024.0)
    return flops

# import torchviz
# import torchsummary
# from core import i3d_factory
#
# model_path = '%s/Charades/baseline_models/i3d/rgb_charades.pt' % (c.data_root_path)
# model = i3d_factory.load_model_i3d_charades_rgb_for_testing(model_path)
# input_shape = (3, 8, 224, 224)
# torchsummary.summary(model, input_size=input_shape)

# import torch
# import torch.autograd
# import networkx as nx
# import matplotlib.pyplot as plt
# from core import torch_utils
#
# # prepare input variable
# with torch.no_grad():
#     # extract features
#     # input_var = torch.from_numpy(video_frames).cuda()
#     # output_var = model(input_var)
#     # output_var = output_var.cpu()
#     # features = output_var.data.numpy()  # (1, 1024, 20, 7, 7)
#
#
#     model_input_shape = (10, 3, 8, 224, 224)
#     # x = np.zeros(model_input_shape, dtype=np.float32)
#     # x = torch.from_numpy(x).cuda()
#     x = torch.randn(10, 3, 8, 224, 224)
#     x = torch.autograd.Variable(x).cuda()
#     y = model(x)
#     # y.mean()
#     #
#
#     model_params = dict(model.named_parameters())
#
#     # graph = torch_utils.visualize_model(model, x)
#     graph = torchviz.make_dot(y, params=model_params)
#     graph.view()
#     # graph.format = 'svg'
#     # graph.render('/home/nour/Downloads/torch_model_svg', view=True)
#
# # nx.draw(graph, cmap = plt.get_cmap('jet'))
# # plt.show()

# endregion

# region Misc

def freeze_layers(layers):
    # unfreeze given layers
    for layer in layers:
        for param in layer.parameters():
            param.requires_grad = False

def freeze_layers_recursive(layers):
    # freeze given layers
    for layer in layers:
        for param in layer.parameters():
            param.requires_grad = False
        if isinstance(layer, nn.Sequential):
            for sub_layer in layer:
                if isinstance(sub_layer, (pl.Conv1dPaded, pl.Conv2dPaded, pl.Conv3dPaded)):
                    sub_layer = sub_layer.conv
                    freeze_layers([sub_layer])
                else:
                    freeze_layers_recursive([sub_layer])

def freeze_model_layers(model, layer_names):
    # freeze given layers
    layers = [getattr(model, l_name) for l_name in layer_names]
    freeze_layers(layers)

def freeze_model_layers_recursive(model, layer_names):
    # freeze given layers
    layers = [getattr(model, l_name) for l_name in layer_names]
    freeze_layers_recursive(layers)

def unfreeze_layers(layers):
    # unfreeze given layers
    for layer in layers:
        for param in layer.parameters():
            param.requires_grad = True

def unfreeze_layers_recursive(layers):
    for layer in layers:
        for param in layer.parameters():
            param.requires_grad = True
        if isinstance(layer, nn.Sequential):
            for sub_layer in layer:
                if isinstance(sub_layer, (pl.Conv1dPaded, pl.Conv2dPaded, pl.Conv3dPaded)):
                    sub_layer = sub_layer.conv
                else:
                    pass
                unfreeze_layers([sub_layer])

def unfreeze_model_layers(model, layer_names):
    # unfreeze given layers
    layers = [getattr(model, l_name) for l_name in layer_names]
    unfreeze_layers(layers)

def unfreeze_model_layers_recursive(model, layer_names):
    # unfreeze given layers
    layers = [getattr(model, l_name) for l_name in layer_names]
    unfreeze_layers_recursive(layers)

def calc_padding_1d(input_size, kernel_size, stride=1, dilation=1):
    """
    Calculate the padding.
    """

    # i = input
    # o = output
    # p = padding
    # k = kernel_size
    # s = stride
    # d = dilation
    # the equation is
    # o = [i + 2 * p - k - (k - 1) * (d - 1)] / s + 1
    # give that we want i = o, then we solve the equation for p gives us

    i = input_size
    s = stride
    k = kernel_size
    d = dilation

    padding = 0.5 * (k - i + s * (i - 1) + (k - 1) * (d - 1))
    padding = int(padding)

    return padding

def configure_specific_gpu():
    _is_local_machine = configs.is_local_machine()

    if _is_local_machine:
        gpu_core_id = 0
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--gpu_core_id', default='-1', type=int)
        args = parser.parse_args()
        gpu_core_id = args.gpu_core_id

        if gpu_core_id < 0 or gpu_core_id > 3:
            msg = 'Please specify a correct GPU core!!!'
            raise Exception(msg)

    torch.cuda.set_device(gpu_core_id)

    # set which device to be used
    const.GPU_CORE_ID = gpu_core_id

def get_shape(tensor):
    t_shape = list(tensor.shape)
    return t_shape

def print_shape(tensor):
    print(get_shape(tensor))

# endregion

# region Metric Functions

def metric_fn_accuracy(y_pred, y_true):
    n_y = y_true.size(0)
    idx_predicted = torch.argmax(y_pred, 1)
    acc = (idx_predicted == y_true).sum().item()
    acc = acc / float(n_y)
    return acc

def metric_fn_binary_accuracy(y_pred, y_true):
    n_y = y_true.size(0)
    y_pred = torch.squeeze(y_pred)
    y_pred_thresholded = torch.zeros_like(y_pred)
    y_pred_thresholded[y_pred >= 0.5] = 1

    acc = (y_pred_thresholded == y_true)
    acc = torch.sum(acc).item()
    acc = acc / float(n_y)
    return acc

def metric_fn_map_charades(y_pred, y_true):
    y_pred = np.array(y_pred.tolist())
    y_true = np.array(y_true.tolist())
    map = metrics.mean_avg_precision_charades(y_true, y_pred)
    map = torch.tensor(map)
    return map

def metric_fun_ap_hico_old(y_pred, y_true):
    y_pred = np.array(y_pred.tolist())
    y_true = np.array(y_true.tolist())

    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    ap = sk_metrics.average_precision_score(y_true, y_pred)
    return ap

def metric_fun_ap_hico_all(y_pred, y_true):
    y_pred = np.array(y_pred.tolist())
    y_true = np.array(y_true.tolist())

    ap = np.array([sk_metrics.average_precision_score(yi_true, yi_pred) for (yi_true, yi_pred) in zip(y_true, y_pred)])

    return ap

def metric_fun_ap_hico(y_pred, y_true):
    y_pred = np.array(y_pred.tolist())
    y_true = np.array(y_true.tolist())

    ap = np.array([sk_metrics.average_precision_score(yi_true, yi_pred) for (yi_true, yi_pred) in zip(y_true, y_pred)])
    ap = np.mean(ap)

    return ap

class METRIC_FUNCTIONS:
    accuracy = metric_fn_accuracy
    binary_accuracy = metric_fn_binary_accuracy
    map_charades = metric_fn_map_charades
    ap_hico = metric_fun_ap_hico
    ap_hico_all = metric_fun_ap_hico_all

# endregion

# region Classes: Callbacks

class ModelSaveCallback():
    def __init__(self, model, model_root_path):
        self.__is_local_machine = configs.is_local_machine()
        self.model = model

        if not os.path.exists(model_root_path):
            os.mkdir(model_root_path)

        self.model_root_path = model_root_path

    def on_batch_ends(self, batch_num, is_training):
        pass

    def on_epoch_ends(self, epoch_num):
        model = self.model
        model_root_path = self.model_root_path
        model_dict_path = '%s/%03d.pt' % (model_root_path, epoch_num)

        # save both model and model dict
        # save_model(model, model_path)
        save_model_dict(model, model_dict_path)

# endregion

# region Classes: Misc

class Logger():
    def __init__(self, model_name):
        self.model_name = model_name
        self.log_file_path = Pth('Breakfast/logs/%s.txt' % (self.model_name,))

        super(Logger, self).__init__()

    def write(self, logs):
        log_file_path = self.log_file_path
        with open(log_file_path, 'a') as f:
            for log_line in logs:
                f.write(log_line)

# endregion
