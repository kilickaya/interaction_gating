#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
PyTorch Implementation of Experiment 3
This experiment is for Breakfast dataset
 Exp_03 is used for activity classification
 Exp_04 is used for unit_action classification
"""

import sys
import time
import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics as sk_metrics

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torch.nn.modules
import torch.nn.functional as F
import torch.optim as optim
import torchsummary
import torchviz

from torch.autograd import Variable


from modules import node_attention, self_attention, context_fusion
from modules import layers_pytorch as pl
from modules import functions_pytorch as pf

from core import const as c
from core import utils, image_utils, plot_utils, configs, data_utils, pytorch_utils
from core.utils import Obj, Path as Pth

#from datasets import ds_breakfast
from nets import resnet_torch

# region Const

N_CLASSES = 600  # how many classes in our problem



def train_part_states():

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    # Features of the image: f_scene
    feats_c1_path = Pth('Hico/features/features_local_part_states.h5')
    x_cs_shape = (1032, 1, 1, 1)

    # Annotation of the image
    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())

    print('--- start time')
    print(datetime.datetime.now())

    print('... loading data')
    t1 = time.time()

    (img_names_tr, y_tr,y_tr_mask, img_names_te, y_te, y_te_mask) = utils.pkl_load(annot_path)
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)

    y_tr_mask = y_tr_mask.astype(np.float32)
    y_te_mask = y_te_mask.astype(np.float32)
    print('... context features')
    (x_tr_c1, x_te_c1) = utils.h5_load_multi(feats_c1_path, ['x_tr', 'x_te'])

    x_tr_c1 = x_tr_c1.reshape(-1, 1032, 1,1,1)
    x_te_c1 = x_te_c1.reshape(-1, 1032, 1,1,1)

    print('train_set_shape_context-1: ', x_tr_c1.shape)
    print('test_set_shape_context-1: ',  x_te_c1.shape)

    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    model = ClassifierPartState(n_classes, x_cs_shape)
    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    #pytorch_utils.model_summary(model, input_size=x_cs_shape, batch_size=-1, device='cuda')
    print('... model built, duration (sec): %d' % (duration))

    # callbacks

    model_name = 'part_states_%s' % (utils.timestamp(),)
    model_root_path = Pth('Hico/models_finetuned/%s', (model_name))

    # callbacks
    model_save_callback =  pytorch_utils.ModelSaveCallback(model, model_root_path)
    print('first_context: %s' %(feats_c1_path))

    print('Training: %s' %(model_name))

    # start training
    pytorch_utils.train_model_custom_metric_mask(model, model._optimizer, model._loss_fn, model._metric_fn, x_tr_c1, y_tr, y_tr_mask, x_te_c1, y_te, y_te_mask, n_epochs, batch_size_tr, batch_size_te, callbacks=[model_save_callback])

    print('--- finish time')
    print(datetime.datetime.now())


class ClassifierPartState(nn.Module):
    def __init__(self, n_classes, x_cs_shape):
        super(ClassifierPartState, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_fusion = 'context_fusion_%d'
        self.__init_layers(n_classes, x_cs_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_cs_shape):
        """
        Define model layers.
        """

        n_units = 512
        n_channels_out = x_cs_shape[0]

        # sparial pooling
        self.spatial_pooling = pl.Max(dim=(3, 4))

        # layers for classification
        classifier_layers = []
        classifier_layers.append(nn.BatchNorm1d(n_channels_out))
        classifier_layers.append(nn.Dropout(0.25))
        classifier_layers.append(nn.Linear(n_channels_out, n_units))
        classifier_layers.append(nn.BatchNorm1d(n_units))
        classifier_layers.append(nn.LeakyReLU(0.2))
        classifier_layers.append(nn.Dropout(0.25))
        classifier_layers.append(nn.Linear(n_units, n_classes))
        self.classifier_layers = nn.Sequential(*classifier_layers)

    def __init_optimizer(self):
        """
        Define loss, metric and optimizer.
        """
        self._loss_fn = F.binary_cross_entropy
        self._metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
        #self._optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-4)
        self._optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9)

    def forward(self, *input):
        """
        input is two features: full image feature and context feature
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x = input[0]

        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)

        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x = x.contiguous().view(B * N, C)  # (B*N, C)

        # feed to classifier
        for l in self.classifier_layers:
            x = l(x)

        # recover the dimension of N
        _, C = pytorch_utils.get_shape(x)
        x = x.view(B, N, C)  # (B, N, C)

        # max over N dimension, then apply activation
        x, _ = torch.max(x, dim=1)  # (B, C)
        x = torch.sigmoid(x)

        return x

    def inference(self, *input):
        """
        input is two features: full image feature and context feature
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x = input[0]

        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)

        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x = x.contiguous().view(B * N, C)  # (B*N, C)

        # feed to classifier
        #for l in self.classifier_layers[:-1]:
        #    x = l(x)

        return x

def train_classifier_coco_stuff():

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    # Features of the image: f_scene
    feats_c1_path = Pth('Hico/features/extra/features_coco_stuff.h5')
    x_cs_shape = [(182, 1, 1, 1), ]

    # Annotation of the image
    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())

    print('--- start time')
    print(datetime.datetime.now())

    print('... loading data')
    t1 = time.time()

    (img_names_tr, y_tr,y_tr_mask, img_names_te, y_te, y_te_mask) = utils.pkl_load(annot_path)
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)

    y_tr_mask = y_tr_mask.astype(np.float32)
    y_te_mask = y_te_mask.astype(np.float32)
    print('... context features')
    (x_tr_c1, x_te_c1) = utils.h5_load_multi(feats_c1_path, ['x_tr', 'x_te'])
    x_tr_c1 = np.swapaxes(x_tr_c1, 1,2)
    x_te_c1 = np.swapaxes(x_te_c1, 1,2)
    print('train_set_shape_context-1: ', x_tr_c1.shape)
    print('test_set_shape_context-1: ',  x_te_c1.shape)

    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    model = ClassifierCocoStuff(n_classes, x_cs_shape)
    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    #pytorch_utils.model_summary(model, input_size=x_cs_shape, batch_size=-1, device='cuda')
    print('... model built, duration (sec): %d' % (duration))

    # callbacks

    model_name = 'coco_stuff_%s' % (utils.timestamp(),)
    model_root_path = Pth('Hico/models_finetuned/%s', (model_name))

    # callbacks
    model_save_callback =  pytorch_utils.ModelSaveCallback(model, model_root_path)
    print('first_context: %s' %(feats_c1_path))

    # start training
    pytorch_utils.train_model_custom_metric_mask(model, model._optimizer, model._loss_fn, model._metric_fn, x_tr_c1, y_tr, y_tr_mask, x_te_c1, y_te, y_te_mask, n_epochs, batch_size_tr, batch_size_te, callbacks=[model_save_callback])

    print('--- finish time')
    print(datetime.datetime.now())


class ClassifierCocoStuff(nn.Module):
    def __init__(self, n_classes, x_cs_shape):
        super(ClassifierCocoStuff, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_fusion = 'context_fusion_%d'
        self.__init_layers(n_classes, x_cs_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_cs_shape):
        """
        Define model layers.
        """

        n_units = 512
        n_channels_out = x_cs_shape[0]

        # sparial pooling
        self.spatial_pooling = pl.Max(dim=(3, 4))

        # layers for classification
        classifier_layers = []
        classifier_layers.append(nn.BatchNorm1d(n_channels_out))
        classifier_layers.append(nn.Dropout(0.25))
        classifier_layers.append(nn.Linear(n_channels_out, n_units))
        classifier_layers.append(nn.BatchNorm1d(n_units))
        classifier_layers.append(nn.LeakyReLU(0.2))
        classifier_layers.append(nn.Dropout(0.25))
        classifier_layers.append(nn.Linear(n_units, n_classes))
        self.classifier_layers = nn.Sequential(*classifier_layers)

    def __init_optimizer(self):
        """
        Define loss, metric and optimizer.
        """
        self._loss_fn = F.binary_cross_entropy
        self._metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
        #self._optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-4)
        self._optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9)

    def forward(self, *input):
        """
        input is two features: full image feature and context feature
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x = input[0]

        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)

        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x = x.contiguous().view(B * N, C)  # (B*N, C)

        # feed to classifier
        for l in self.classifier_layers:
            x = l(x)

        # recover the dimension of N
        _, C = pytorch_utils.get_shape(x)
        x = x.view(B, N, C)  # (B, N, C)

        # max over N dimension, then apply activation
        x, _ = torch.max(x, dim=1)  # (B, C)
        x = torch.sigmoid(x)

        return x




def train_stuff_context():

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    # combine coco_stuff with places and scene attribute predictions

    # Features of the image: f_scene
    feats_c1_path = Pth('Hico/features/extra/features_coco_stuff.h5')
    feats_c2_path = Pth('Hico/features/legacy/features_scene_places.h5')
    feats_c3_path = Pth('Hico/features/legacy/features_scene_att.h5')

    x_cs_shape = [(182, 1, 1, 1), (365, 1, 1, 1), (102, 1, 1, 1)]

    # Annotation of the image
    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())

    print('--- start time')
    print(datetime.datetime.now())

    print('... loading data')
    t1 = time.time()

    (img_names_tr, y_tr,y_tr_mask, img_names_te, y_te, y_te_mask) = utils.pkl_load(annot_path)
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)

    y_tr_mask = y_tr_mask.astype(np.float32)
    y_te_mask = y_te_mask.astype(np.float32)
    print('... context features')
    (x_tr_c1, x_te_c1) = utils.h5_load_multi(feats_c1_path, ['x_tr', 'x_te'])
    (x_tr_c2, x_te_c2) = utils.h5_load_multi(feats_c2_path, ['x_tr', 'x_te'])
    (x_tr_c3, x_te_c3) = utils.h5_load_multi(feats_c3_path, ['x_tr', 'x_te'])

    x_tr_c1 = np.swapaxes(x_tr_c1, 1,2)
    x_te_c1 = np.swapaxes(x_te_c1, 1,2)

    print('train_set_shape_context-1: ', x_tr_c1.shape)
    print('test_set_shape_context-1: ',  x_te_c1.shape)

    print('train_set_shape_context-2: ', x_tr_c2.shape)
    print('test_set_shape_context-2: ',  x_te_c2.shape)

    print('train_set_shape_context-3: ', x_tr_c3.shape)
    print('test_set_shape_context-3: ',  x_te_c3.shape)

    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    model = ClassifierCombination(n_classes, x_cs_shape)
    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    #pytorch_utils.model_summary(model, input_size=x_cs_shape, batch_size=-1, device='cuda')
    print('... model built, duration (sec): %d' % (duration))

    # callbacks

    model_name = 'stuff_%s' % (utils.timestamp(),)
    model_root_path = Pth('Hico/models_finetuned/%s', (model_name))

    # callbacks
    model_save_callback =  pytorch_utils.ModelSaveCallback(model, model_root_path)
    print('first_context: %s' %(feats_c1_path))

    print('Training: %s' %(model_name))

    # start training
    pytorch_utils.train_model_custom_metric_mask(model, model._optimizer, model._loss_fn, model._metric_fn, [x_tr_c1, x_tr_c2, x_tr_c3], y_tr, y_tr_mask, [x_te_c1, x_te_c2, x_te_c3], y_te, y_te_mask, n_epochs, batch_size_tr, batch_size_te, callbacks=[model_save_callback])

    print('--- finish time')
    print(datetime.datetime.now())


class ClassifierCombination(nn.Module):
    def __init__(self, n_classes, x_cs_shape):
        super(ClassifierCombination, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_fusion = 'context_fusion_%d'
        self.__init_layers(n_classes, x_cs_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_cs_shape):
        """
        Define model layers.
        """

        n_units = 512
        n_channels_out = np.sum([i[0] for i in x_cs_shape])

        # sparial pooling
        self.spatial_pooling = pl.Max(dim=(3, 4))

        # layers for classification
        classifier_layers = []
        classifier_layers.append(nn.BatchNorm1d(n_channels_out))
        classifier_layers.append(nn.Dropout(0.25))
        classifier_layers.append(nn.Linear(n_channels_out, n_units))
        classifier_layers.append(nn.BatchNorm1d(n_units))
        classifier_layers.append(nn.LeakyReLU(0.2))
        classifier_layers.append(nn.Dropout(0.25))
        classifier_layers.append(nn.Linear(n_units, n_classes))
        self.classifier_layers = nn.Sequential(*classifier_layers)

    def __init_optimizer(self):
        """
        Define loss, metric and optimizer.
        """
        self._loss_fn = F.binary_cross_entropy
        self._metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
        #self._optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-4)
        self._optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9)

    def forward(self, *input):
        """
        input is two features: full image feature and context feature
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x = input[:]
        x = torch.cat(x, dim=1)

        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)

        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x = x.contiguous().view(B * N, C)  # (B*N, C)

        # feed to classifier
        for l in self.classifier_layers:
            x = l(x)

        # recover the dimension of N
        _, C = pytorch_utils.get_shape(x)
        x = x.view(B, N, C)  # (B, N, C)

        # max over N dimension, then apply activation
        x, _ = torch.max(x, dim=1)  # (B, C)
        x = torch.sigmoid(x)

        return x

    def inference(self, *input):
        """
        input is two features: full image feature and context feature
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x = input[:]
        x = torch.cat(x, dim=1)

        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)

        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x = x.contiguous().view(B * N, C)  # (B*N, C)

        # feed to classifier
        #for l in self.classifier_layers[:-1]:
        #    x = l(x)

        return x
def expand_feats(feat):

    feat = np.expand_dims(feat, 3)
    feat = np.expand_dims(feat, 4)

    return feat

def train_classifier_local_pose_pooling():

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    # Features of the image: f_scene
    feats_c1_path = Pth('Hico/features/legacy/features_pairattn_pose.h5')
    x_cs_shape = (4096, 3, 1, 1)

    # Annotation of the image
    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())

    print('--- start time')
    print(datetime.datetime.now())

    print('... loading data')
    t1 = time.time()

    (img_names_tr, y_tr,y_tr_mask, img_names_te, y_te, y_te_mask) = utils.pkl_load(annot_path)
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)

    y_tr_mask = y_tr_mask.astype(np.float32)
    y_te_mask = y_te_mask.astype(np.float32)
    print('... context features')
    (x_tr_c1, x_te_c1) = utils.h5_load_multi(feats_c1_path, ['x_tr', 'x_te'])
    x_tr_c1 = np.swapaxes(x_tr_c1, 1,2)
    x_te_c1 = np.swapaxes(x_te_c1, 1,2)

    x_tr_c1 = expand_feats(x_tr_c1)
    x_te_c1 = expand_feats(x_te_c1)

    print('train_set_shape_context-1: ', x_tr_c1.shape)
    print('test_set_shape_context-1: ',  x_te_c1.shape)

    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    model = ClassifierLocalContextPooling(n_classes, x_cs_shape)
    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    #pytorch_utils.model_summary(model, input_size=x_cs_shape, batch_size=-1, device='cuda')
    print('... model built, duration (sec): %d' % (duration))

    # callbacks

    model_name = 'local_pose_%s' % (utils.timestamp(),)
    model_root_path = Pth('Hico/models_finetuned/%s', (model_name))

    # callbacks
    model_save_callback =  pytorch_utils.ModelSaveCallback(model, model_root_path)
    print('first_context: %s' %(feats_c1_path))

    print('Training: %s' %(model_name))

    # start training
    pytorch_utils.train_model_custom_metric_mask(model, model._optimizer, model._loss_fn, model._metric_fn, x_tr_c1, y_tr, y_tr_mask, x_te_c1, y_te, y_te_mask, n_epochs, batch_size_tr, batch_size_te, callbacks=[model_save_callback])

    print('--- finish time')
    print(datetime.datetime.now())

def train_classifier_local_segment_pooling():

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    # Features of the image: f_scene
    feats_c1_path = Pth('Hico/features/extra/features_local_scene.h5')
    x_cs_shape = (2048, 6, 1, 1)

    # Annotation of the image
    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())

    print('--- start time')
    print(datetime.datetime.now())

    print('... loading data')
    t1 = time.time()

    (img_names_tr, y_tr,y_tr_mask, img_names_te, y_te, y_te_mask) = utils.pkl_load(annot_path)
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)

    y_tr_mask = y_tr_mask.astype(np.float32)
    y_te_mask = y_te_mask.astype(np.float32)
    print('... context features')
    (x_tr_c1, x_te_c1) = utils.h5_load_multi(feats_c1_path, ['x_tr', 'x_te'])
    x_tr_c1 = np.swapaxes(x_tr_c1, 1,2)
    x_te_c1 = np.swapaxes(x_te_c1, 1,2)

    print('train_set_shape_context-1: ', x_tr_c1.shape)
    print('test_set_shape_context-1: ',  x_te_c1.shape)

    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    model = ClassifierLocalContextPooling(n_classes, x_cs_shape)
    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    #pytorch_utils.model_summary(model, input_size=x_cs_shape, batch_size=-1, device='cuda')
    print('... model built, duration (sec): %d' % (duration))

    # callbacks

    model_name = 'local_scene_%s' % (utils.timestamp(),)
    model_root_path = Pth('Hico/models_finetuned/%s', (model_name))

    # callbacks
    model_save_callback =  pytorch_utils.ModelSaveCallback(model, model_root_path)
    print('first_context: %s' %(feats_c1_path))

    print('Training: %s' %(model_name))

    # start training
    pytorch_utils.train_model_custom_metric_mask(model, model._optimizer, model._loss_fn, model._metric_fn, x_tr_c1, y_tr, y_tr_mask, x_te_c1, y_te, y_te_mask, n_epochs, batch_size_tr, batch_size_te, callbacks=[model_save_callback])

    print('--- finish time')
    print(datetime.datetime.now())

class ClassifierLocalContextMax(nn.Module):
    def __init__(self, n_classes, x_cs_shape):
        super(ClassifierLocalContextMax, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_fusion = 'context_fusion_%d'
        self.__init_layers(n_classes, x_cs_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_cs_shape):
        """
        Define model layers.
        """

        n_units = 512
        n_channels_out = x_cs_shape[0]

        # sparial pooling
        self.spatial_pooling = pl.Max(dim=(3, 4))

        # layers for classification
        classifier_layers = []
        classifier_layers.append(nn.BatchNorm1d(n_channels_out))
        classifier_layers.append(nn.Dropout(0.25))
        classifier_layers.append(nn.Linear(n_channels_out, n_units))
        classifier_layers.append(nn.BatchNorm1d(n_units))
        classifier_layers.append(nn.LeakyReLU(0.2))
        classifier_layers.append(nn.Dropout(0.25))
        classifier_layers.append(nn.Linear(n_units, n_classes))
        self.classifier_layers = nn.Sequential(*classifier_layers)

    def __init_optimizer(self):
        """
        Define loss, metric and optimizer.
        """
        self._loss_fn = F.binary_cross_entropy
        self._metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
        #self._optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-4)
        self._optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9)

    def forward(self, *input):
        """
        input is two features: full image feature and context feature
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x = input[0]

        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)

        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x = x.contiguous().view(B * N, C)  # (B*N, C)

        # feed to classifier
        for l in self.classifier_layers:
            x = l(x)

        # recover the dimension of N
        _, C = pytorch_utils.get_shape(x)
        x = x.view(B, N, C)  # (B, N, C)

        # max over N dimension, then apply activation
        x, _ = torch.max(x, dim=1)  # (B, C)
        x = torch.sigmoid(x)

        return x

    def forward_for_extraction(self, *input):
        """
        input is two features: full image feature and context feature
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x = input[0]

        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)

        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x = x.contiguous().view(B * N, C)  # (B*N, C)

        # feed to classifier
        for l in self.classifier_layers:
            x = l(x)

        # recover the dimension of N
        _, C = pytorch_utils.get_shape(x)
        x = x.view(B, N, C)  # (B, N, C)

        return x



class ClassifierLocalContextPooling(nn.Module):
    def __init__(self, n_classes, x_cs_shape):
        super(ClassifierLocalContextPooling, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_fusion = 'context_fusion_%d'
        self.__init_layers(n_classes, x_cs_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_cs_shape):
        """
        Define model layers.
        """

        n_units = 512
        n_channels_out = x_cs_shape[0]

        # sparial pooling
        self.spatial_pooling = pl.Max(dim=(3, 4))
        self.softmax = nn.Softmax(dim = 1)

        # layers for pooling
        pool_layers = []
        pool_layers.append(nn.BatchNorm1d(n_channels_out))
        pool_layers.append(nn.Dropout(0.25))
        pool_layers.append(nn.Linear(n_channels_out, 1))
        self.pool_layers = nn.Sequential(*pool_layers)

        # layers for classification

        classifier_layers = []
        classifier_layers.append(nn.BatchNorm1d(n_channels_out))
        classifier_layers.append(nn.Dropout(0.25))
        classifier_layers.append(nn.Linear(n_channels_out, n_units))
        classifier_layers.append(nn.BatchNorm1d(n_units))
        classifier_layers.append(nn.LeakyReLU(0.2))
        classifier_layers.append(nn.Dropout(0.25))
        classifier_layers.append(nn.Linear(n_units, n_classes))
        self.classifier_layers = nn.Sequential(*classifier_layers)


    def __init_optimizer(self):
        """
        Define loss, metric and optimizer.
        """
        self._loss_fn = F.binary_cross_entropy
        self._metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
        self._optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-4)

    def forward(self, *input):
        """
        input is two features: full image feature and context feature
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x = input[0]

        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)

        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x = x.contiguous().view(B * N, C)  # (B*N, C)

        # get attention values per region
        alpha = x
        for l in self.pool_layers:
            alpha = l(alpha) # (B*N, 1)

        #alpha = torch.sigmoid(alpha)

        alpha = alpha.view(B, N)
        alpha = self.softmax(alpha)
        alpha = alpha.view(B*N,1)

        # reweight features
        x = torch.mul(x, alpha) # (B*N, C)
        x = x.view(B, N, C)  # (B, N, C)
        #x = torch.mean(x, 1)
        x = torch.sum(x, 1)

        # feed to classifier
        for l in self.classifier_layers:
            x = l(x)

        # max over N dimension, then apply activation
        x = torch.sigmoid(x)

        return x

    def inference(self, *input):
        """
        input is two features: full image feature and context feature
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x = input[0]

        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)

        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x = x.contiguous().view(B * N, C)  # (B*N, C)

        # get attention values per region
        alpha = x
        for l in self.pool_layers:
            alpha = l(alpha) # (B*N, 1)

        #alpha = torch.sigmoid(alpha)

        alpha = alpha.view(B, N)
        alpha = self.softmax(alpha)
        alpha = alpha.view(B*N,1)

        # reweight features
        x = torch.mul(x, alpha) # (B*N, C)
        x = x.view(B, N, C)  # (B, N, C)
        #x = torch.mean(x, 1)
        x = torch.sum(x, 1)

        '''
        for l in self.classifier_layers[:-1]:
            x = l(x)
        '''

        return x


class ClassifierLocalContextPooling_v2(nn.Module):
    def __init__(self, n_classes, x_cs_shape):
        super(ClassifierLocalContextPooling_v2, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_fusion = 'context_fusion_%d'
        self.__init_layers(n_classes, x_cs_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_cs_shape):
        """
        Define model layers.
        """

        n_units = 512
        n_channels_out = x_cs_shape[0]

        # sparial pooling
        self.spatial_pooling = pl.Max(dim=(3, 4))

        # layers for pooling
        pool_layers = []
        pool_layers.append(nn.BatchNorm1d(n_units))
        pool_layers.append(nn.Dropout(0.25))
        pool_layers.append(nn.Linear(n_units, 1))
        self.pool_layers = nn.Sequential(*pool_layers)

        # layers for classification
        classifier_layers = []
        classifier_layers.append(nn.BatchNorm1d(n_channels_out))
        classifier_layers.append(nn.Dropout(0.25))
        classifier_layers.append(nn.Linear(n_channels_out, n_units))
        classifier_layers.append(nn.BatchNorm1d(n_units))
        classifier_layers.append(nn.LeakyReLU(0.2))
        classifier_layers.append(nn.Dropout(0.25))
        
        self.classifier_layers_1 = nn.Sequential(*classifier_layers)
        classifier_layers = []
        classifier_layers.append(nn.Linear(n_units, n_classes))

        self.classifier_layers_2 = nn.Sequential(*classifier_layers)

    def __init_optimizer(self):
        """
        Define loss, metric and optimizer.
        """
        self._loss_fn = F.binary_cross_entropy
        self._metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
        self._optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-4)

    def forward(self, *input):
        """
        input is two features: full image feature and context feature
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x = input[0]

        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)

        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x = x.contiguous().view(B * N, C)  # (B*N, C)

        # feed to classifier
        for l in self.classifier_layers_1:
            x = l(x)

        # get attention values per region
        alpha = x
        for l in self.pool_layers:
            alpha = l(alpha) # (B*N, 1)

        alpha = torch.sigmoid(alpha)

        # reweight features
        x = torch.mul(x, alpha) # (B*N, C)
        _, C = pytorch_utils.get_shape(x)

        x = x.view(B, N, C)  # (B, N, C)
        x = torch.mean(x, 1)

        # feed to classifier
        for l in self.classifier_layers_2:
            x = l(x)

        # max over N dimension, then apply activation
        x = torch.sigmoid(x)

        return x

def train_classifier_deformation():

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    # Features of the image: f_scene
    feats_c1_path = Pth('Hico/features/extra/features_deformation.h5')
    x_cs_shape = (80, 1, 32, 32)

    # Annotation of the image
    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())

    print('--- start time')
    print(datetime.datetime.now())

    print('... loading data')
    t1 = time.time()

    (img_names_tr, y_tr, y_tr_mask, img_names_te, y_te, y_te_mask) = utils.pkl_load(annot_path)
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)

    y_tr_mask = y_tr_mask.astype(np.float32)
    y_te_mask = y_te_mask.astype(np.float32)
    print('... context features')
    (x_tr_c1, x_te_c1) = utils.h5_load_multi(feats_c1_path, ['x_tr', 'x_te'])

    print('train_set_shape_context-1: ', x_tr_c1.shape)
    print('test_set_shape_context-1: ',  x_te_c1.shape)

    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    model = ClassifierDeformation(n_classes, x_cs_shape)
    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    #pytorch_utils.model_summary(model, input_size=x_cs_shape, batch_size=-1, device='cuda')
    print('... model built, duration (sec): %d' % (duration))

    model_name = 'deformation_%s' % (utils.timestamp(),)
    model_root_path = Pth('Hico/models_finetuned/%s', (model_name))

    # callbacks
    model_save_callback =  pytorch_utils.ModelSaveCallback(model, model_root_path)

    print('first_context: %s' %(feats_c1_path))

    print('Training: %s' %(model_name))

    # start training
    pytorch_utils.train_model_custom_metric_mask(model, model._optimizer, model._loss_fn, model._metric_fn, x_tr_c1, y_tr, y_tr_mask, x_te_c1, y_te, y_te_mask, n_epochs, batch_size_tr, batch_size_te, callbacks=[model_save_callback])

    print('--- finish time')
    print(datetime.datetime.now())

class ClassifierDeformation(nn.Module):
    def __init__(self, n_classes, x_cs_shape):
        super(ClassifierDeformation, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_fusion = 'context_fusion_%d'
        self.__init_layers(n_classes, x_cs_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_cs_shape):
        """
        Define model layers.
        """

        n_inter_unit_1 = 128
        n_inter_unit_2 = 256

        n_out_unit = 512

        n_channels_in = x_cs_shape[0]

        self.squeeze = pl.Squeeze(dim=1)

        # layers for classification: (3,3,80,128) -> (3,3,128,512)

        classifier_layers = []
        classifier_layers.append(nn.BatchNorm2d(n_channels_in))
        classifier_layers.append(nn.Dropout(0.25))

        classifier_layers.append(nn.Conv2d(n_channels_in, n_inter_unit_1, 3, 2))
        classifier_layers.append(nn.BatchNorm2d(n_inter_unit_1))
        classifier_layers.append(nn.LeakyReLU(0.2))
        classifier_layers.append(nn.Dropout(0.25))

        classifier_layers.append(nn.Conv2d(n_inter_unit_1, n_out_unit, 3, 2))
        classifier_layers.append(nn.BatchNorm2d(n_out_unit))
        classifier_layers.append(nn.LeakyReLU(0.2))
        classifier_layers.append(nn.Dropout(0.25))

        classifier_layers.append(nn.AvgPool2d(7))

        self.classifier_layers = nn.Sequential(*classifier_layers)

        last_layer = []
        last_layer.append(nn.Linear(n_out_unit, n_classes))
        self.last_layer = nn.Sequential(*last_layer)

    def __init_optimizer(self):
        """
        Define loss, metric and optimizer.
        """
        self._loss_fn = F.binary_cross_entropy
        self._metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
        self._optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-4)

    def forward(self, *input):
        """
        input is two features: full image feature and context feature
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x = input[0] # (B, C, N, H, W)

        # remove singleton dimension
        x = self.squeeze(x)  # (B, C, H,W)

        # feed to classifier
        for l in self.classifier_layers:
            x = l(x)

        x = torch.squeeze(x)

        for l in self.last_layer:
            x = l(x)

        x = torch.sigmoid(x) # (B, C)

        return x

    def feature_extractor(self, *input):
        """
        input is two features: full image feature and context feature
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x = input[0] # (B, C, N, H, W)

        # remove singleton dimension
        x = self.squeeze(x)  # (B, C, H,W)

        # feed to classifier
        for l in self.classifier_layers:
            x = l(x)

        x = torch.squeeze(x)


        return x

    def inference(self, *input):
        """
        input is two features: full image feature and context feature
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x = input[0] # (B, C, N, H, W)

        # remove singleton dimension
        x = self.squeeze(x)  # (B, C, H,W)

        # feed to classifier
        for l in self.classifier_layers:
            x = l(x)

        x = torch.squeeze(x)

        for l in self.last_layer:
            x = l(x)

        x = torch.sigmoid(x) # (B, C)

        return x

def train_classifier_openpose():

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    # Features of the image: f_scene
    feats_c1_path = Pth('Hico/features/extra/features_open_pose.h5')
    x_cs_shape = (81, 1, 32, 32)

    # Annotation of the image
    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())

    print('--- start time')
    print(datetime.datetime.now())

    print('... loading data')
    t1 = time.time()

    (img_names_tr, y_tr, y_tr_mask, img_names_te, y_te, y_te_mask) = utils.pkl_load(annot_path)
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)

    y_tr_mask = y_tr_mask.astype(np.float32)
    y_te_mask = y_te_mask.astype(np.float32)
    print('... context features')
    (x_tr_c1, x_te_c1) = utils.h5_load_multi(feats_c1_path, ['x_tr', 'x_te'])

    print('train_set_shape_context-1: ', x_tr_c1.shape)
    print('test_set_shape_context-1: ',  x_te_c1.shape)

    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    model = ClassifierOpenPose(n_classes, x_cs_shape)
    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    #pytorch_utils.model_summary(model, input_size=x_cs_shape, batch_size=-1, device='cuda')
    print('... model built, duration (sec): %d' % (duration))

    model_name = 'deformation_%s' % (utils.timestamp(),)
    model_root_path = Pth('Hico/models_finetuned/%s', (model_name))

    # callbacks
    callbacks = []
    print('first_context: %s' %(feats_c1_path))

    # start training
    pytorch_utils.train_model_custom_metric_mask(model, model._optimizer, model._loss_fn, model._metric_fn, x_tr_c1, y_tr, y_tr_mask, x_te_c1, y_te, y_te_mask, n_epochs, batch_size_tr, batch_size_te, callbacks=[])

    print('--- finish time')
    print(datetime.datetime.now())


class ClassifierOpenPose(nn.Module):
    def __init__(self, n_classes, x_cs_shape):
        super(ClassifierOpenPose, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_fusion = 'context_fusion_%d'
        self.__init_layers(n_classes, x_cs_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_cs_shape):
        """
        Define model layers.
        """

        n_inter_unit_1 = 128
        n_inter_unit_2 = 256

        n_out_unit = 512

        n_channels_in = x_cs_shape[0]

        self.squeeze = pl.Squeeze(dim=1)

        # layers for classification: (3,3,80,128) -> (3,3,128,512)

        classifier_layers = []
        classifier_layers.append(nn.BatchNorm2d(n_channels_in))
        classifier_layers.append(nn.Dropout(0.25))

        classifier_layers.append(nn.Conv2d(n_channels_in, n_inter_unit_1, 3, 2))
        classifier_layers.append(nn.BatchNorm2d(n_inter_unit_1))
        classifier_layers.append(nn.LeakyReLU(0.2))
        classifier_layers.append(nn.Dropout(0.25))

        classifier_layers.append(nn.Conv2d(n_inter_unit_1, n_out_unit, 3, 2))
        classifier_layers.append(nn.BatchNorm2d(n_out_unit))
        classifier_layers.append(nn.LeakyReLU(0.2))
        classifier_layers.append(nn.Dropout(0.25))

        classifier_layers.append(nn.AvgPool2d(7))

        self.classifier_layers = nn.Sequential(*classifier_layers)

        last_layer = []
        last_layer.append(nn.Linear(n_out_unit, n_classes))
        self.last_layer = nn.Sequential(*last_layer)

    def __init_optimizer(self):
        """
        Define loss, metric and optimizer.
        """
        self._loss_fn = F.binary_cross_entropy
        self._metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
        self._optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-4)
        #self._optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9)

    def forward(self, *input):
        """
        input is two features: full image feature and context feature
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x = input[0] # (B, C, N, H, W)

        # remove singleton dimension
        x = self.squeeze(x)  # (B, C, H,W)

        # feed to classifier
        for l in self.classifier_layers:
            x = l(x)

        x = torch.squeeze(x)

        for l in self.last_layer:
            x = l(x)

        x = torch.sigmoid(x) # (B, C)

        return x


def train_classifier_scene():

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    # Features of the image: f_scene
    feats_c1_path = Pth('Hico/features/extra/features_scene.h5')
    x_cs_shape = (512, 1, 14, 14)

    # Annotation of the image
    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())

    print('--- start time')
    print(datetime.datetime.now())

    print('... loading data')
    t1 = time.time()

    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)

    print('... context features')
    (x_tr_c1, x_te_c1) = utils.h5_load_multi(feats_c1_path, ['x_tr', 'x_te'])

    print('train_set_shape_context-1: ', x_tr_c1.shape)
    print('test_set_shape_context-1: ',  x_te_c1.shape)

    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    model = ClassifierScene(n_classes, x_cs_shape)
    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    #pytorch_utils.model_summary(model, input_size=x_cs_shape, batch_size=-1, device='cuda')
    print('... model built, duration (sec): %d' % (duration))

    model_name = 'global_scene_%s' % (utils.timestamp(),)
    model_root_path = Pth('Hico/models_finetuned/%s', (model_name))

    # callbacks
    model_save_callback =  pytorch_utils.ModelSaveCallback(model, model_root_path)

    print('first_context: %s' %(feats_c1_path))

    # start training
    pytorch_utils.train_model_custom_metric(model, model._optimizer, model._loss_fn, model._metric_fn, x_tr_c1, y_tr, x_te_c1, y_te, n_epochs, batch_size_tr, batch_size_te, callbacks=[model_save_callback])

    print('--- finish time')
    print(datetime.datetime.now())


class ClassifierScene(nn.Module):
    def __init__(self, n_classes, x_cs_shape):
        super(ClassifierScene, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_fusion = 'context_fusion_%d'
        self.__init_layers(n_classes, x_cs_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_cs_shape):
        """
        Define model layers.
        """

        n_inter_unit = 128
        n_out_unit = 512

        n_channels_in = x_cs_shape[0]

        self.squeeze = pl.Squeeze(dim=1)

        # layers for classification: (3,3,80,128) -> (3,3,128,512)
        classifier_layers = []
        classifier_layers.append(nn.BatchNorm2d(n_channels_in))
        classifier_layers.append(nn.Dropout(0.25))

        classifier_layers.append(nn.Conv2d(n_channels_in, n_inter_unit, 3, 2, 1))
        classifier_layers.append(nn.BatchNorm2d(n_inter_unit))
        classifier_layers.append(nn.LeakyReLU(0.2))
        classifier_layers.append(nn.Dropout(0.25))

        classifier_layers.append(nn.Conv2d(n_inter_unit, n_out_unit, 3, 1, 1))
        classifier_layers.append(nn.BatchNorm2d(n_out_unit))
        classifier_layers.append(nn.LeakyReLU(0.2))
        classifier_layers.append(nn.Dropout(0.25))

        classifier_layers.append(nn.AvgPool2d(7))

        self.classifier_layers = nn.Sequential(*classifier_layers)

        last_layer = []
        last_layer.append(nn.Linear(n_out_unit, n_classes))
        self.last_layer = nn.Sequential(*last_layer)

    def __init_optimizer(self):
        """
        Define loss, metric and optimizer.
        """
        self._loss_fn = F.binary_cross_entropy
        self._metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
        self._optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-4)

    def forward(self, *input):
        """
        input is two features: full image feature and context feature
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x = input[0] # (B, C, N, H, W)

        # remove singleton dimension
        x = self.squeeze(x)  # (B, C, H,W)

        # feed to classifier
        for l in self.classifier_layers:
            x = l(x)

        x = torch.squeeze(x)

        for l in self.last_layer:
            x = l(x)

        x = torch.sigmoid(x) # (B, C)

        return x

    def forward_for_extraction(self, *input):
        """
        input is two features: full image feature and context feature
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x = input[0] # (B, C, N, H, W)

        # remove singleton dimension
        x = self.squeeze(x)  # (B, C, H,W)

        # feed to classifier
        for l in self.classifier_layers:
            x = l(x)

        x = torch.squeeze(x)

        return x


def train_classifier_comb_scene_deformation():

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    # Features of the image: f_scene
    feats_c1_path = Pth('Hico/features/extra/features_semantic_seg.h5')
    feats_c2_path = Pth('Hico/features/extra/features_deformation.h5')

    x_cs_shape = (150, 1, 14, 14)

    # Annotation of the image
    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())

    print('--- start time')
    print(datetime.datetime.now())

    print('... loading data')
    t1 = time.time()

    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)

    print('... context features')
    (x_tr_c1, x_te_c1) = utils.h5_load_multi(feats_c1_path, ['x_tr', 'x_te'])
    (x_tr_c2, x_te_c2) = utils.h5_load_multi(feats_c2_path, ['x_tr', 'x_te'])

    print('train_set_shape_context-1: ', x_tr_c1.shape)
    print('test_set_shape_context-1: ',  x_te_c1.shape)

    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    model = ClassifierComb(n_classes, x_cs_shape)
    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    #pytorch_utils.model_summary(model, input_size=x_cs_shape, batch_size=-1, device='cuda')
    print('... model built, duration (sec): %d' % (duration))

    # callbacks
    callbacks = []

    # start training
    pytorch_utils.train_model_custom_metric(model, model._optimizer, model._loss_fn, model._metric_fn, [x_tr_c1, x_tr_c2], y_tr, [x_te_c1, x_te_c2], y_te, n_epochs, batch_size_tr, batch_size_te, callbacks=[])

    print('--- finish time')
    print(datetime.datetime.now())

class ClassifierComb(nn.Module):
    def __init__(self, n_classes, x_cs_shape):
        super(ClassifierComb, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_fusion = 'context_fusion_%d'
        self.__init_layers(n_classes, x_cs_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_cs_shape):
        """
        Define model layers.
        """

        n_inter_unit = 128
        n_out_unit = 512

        n_channels_scene = 150
        n_channels_in_object = 80

        self.squeeze = pl.Squeeze(dim=1)

        ## Scene layers to process places (14,14,512)
        scene_layers = []
        scene_layers.append(nn.BatchNorm2d(n_channels_scene))
        scene_layers.append(nn.Dropout(0.25))
        scene_layers.append(nn.Conv2d(n_channels_scene, n_inter_unit, 3, 2, 1))
        scene_layers.append(nn.BatchNorm2d(n_inter_unit))
        scene_layers.append(nn.LeakyReLU(0.2))
        scene_layers.append(nn.Dropout(0.25))

        scene_layers.append(nn.Conv2d(n_inter_unit, n_out_unit, 3, 1, 1))
        scene_layers.append(nn.BatchNorm2d(n_out_unit))
        scene_layers.append(nn.LeakyReLU(0.2))
        scene_layers.append(nn.Dropout(0.25))

        scene_layers.append(nn.AvgPool2d(7))

        self.scene_layers = nn.Sequential(*scene_layers)

        deform_layers = []
        deform_layers.append(nn.BatchNorm2d(n_channels_in_object))
        deform_layers.append(nn.Dropout(0.25))

        deform_layers.append(nn.Conv2d(n_channels_in_object, n_inter_unit, 3, 2))
        deform_layers.append(nn.BatchNorm2d(n_inter_unit))
        deform_layers.append(nn.LeakyReLU(0.2))
        deform_layers.append(nn.Dropout(0.25))

        deform_layers.append(nn.Conv2d(n_inter_unit, n_out_unit, 3, 2))
        deform_layers.append(nn.BatchNorm2d(n_out_unit))
        deform_layers.append(nn.LeakyReLU(0.2))
        deform_layers.append(nn.Dropout(0.25))

        deform_layers.append(nn.AvgPool2d(7))

        self.deform_layers = nn.Sequential(*deform_layers)

        # Joint classifier layer
        last_layer = []
        last_layer.append(nn.Linear(n_out_unit*2, n_classes))
        self.last_layer = nn.Sequential(*last_layer)

    def __init_optimizer(self):
        """
        Define loss, metric and optimizer.
        """
        self._loss_fn = F.binary_cross_entropy
        self._metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
        self._optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-4)

    def forward(self, *input):
        """
        input is two features: full image feature and context feature
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x = input[0] # scene feature
        # remove singleton dimension
        x = self.squeeze(x)  # (B, C, H,W)
        # feed to classifier
        for l in self.scene_layers:
            x = l(x)

        x_scene = x
        x_scene = torch.squeeze(x_scene)

        x = input[1] # scene feature
        # remove singleton dimension
        x = self.squeeze(x)  # (B, C, H,W)
        # feed to classifier
        for l in self.deform_layers:
            x = l(x)

        x_deform = x
        x_deform = torch.squeeze(x_deform)

        x = torch.cat((x_scene, x_deform), 1)

        x = torch.squeeze(x)

        for l in self.last_layer:
            x = l(x)

        x = torch.sigmoid(x) # (B, C)

        return x

def train_classifier_comb():

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    # Features of the image: f_scene
    feats_c1_path = Pth('Hico/features/extra/features_scene.h5')
    feats_c2_path = Pth('Hico/features/extra/features_semantic_seg.h5')
    feats_c3_path = Pth('Hico/features/extra/features_local_scene.h5')

    x_cs_shape = (2048, 6, 1, 1)

    # Annotation of the image
    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())

    print('--- start time')
    print(datetime.datetime.now())

    print('... loading data')
    t1 = time.time()

    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)

    print('... context features')
    (x_tr_c1, x_te_c1) = utils.h5_load_multi(feats_c1_path, ['x_tr', 'x_te'])
    (x_tr_c2, x_te_c2) = utils.h5_load_multi(feats_c2_path, ['x_tr', 'x_te'])
    (x_tr_c3, x_te_c3) = utils.h5_load_multi(feats_c3_path, ['x_tr', 'x_te'])
    x_tr_c3 = np.swapaxes(x_tr_c3, 1,2)
    x_te_c3 = np.swapaxes(x_te_c3, 1,2)

    print(feats_c1_path, feats_c2_path, feats_c3_path)

    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    model = ClassifierCombAll(n_classes, x_cs_shape)
    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    #pytorch_utils.model_summary(model, input_size=x_cs_shape, batch_size=-1, device='cuda')
    print('... model built, duration (sec): %d' % (duration))

    # callbacks
    callbacks = []

    # start training
    pytorch_utils.train_model_custom_metric(model, model._optimizer, model._loss_fn, model._metric_fn, [x_tr_c1, x_tr_c2, x_tr_c3], y_tr, [x_te_c1, x_te_c2, x_te_c3], y_te, n_epochs, batch_size_tr, batch_size_te, callbacks=[])

    print('--- finish time')
    print(datetime.datetime.now())

class ClassifierCombAll(nn.Module):
    def __init__(self, n_classes, x_cs_shape):
        super(ClassifierCombAll, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_fusion = 'context_fusion_%d'
        self.__init_layers(n_classes, x_cs_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_cs_shape):
        """
        Define model layers.
        """

        n_inter_unit = 128
        n_out_unit = 512

        n_channels_locals = 2048

        n_channels_scene = 512
        n_channels_segment = 150

        self.squeeze = pl.Squeeze(dim=1)
        self.spatial_pooling = pl.Max(dim=(3, 4))

        # Local layers for classification
        # layers for pooling
        pool_layers = []
        pool_layers.append(nn.BatchNorm1d(n_channels_locals))
        pool_layers.append(nn.Dropout(0.25))
        pool_layers.append(nn.Linear(n_channels_locals, 1))
        self.pool_layers = nn.Sequential(*pool_layers)

        # layers for classification
        local_layers = []
        local_layers.append(nn.BatchNorm1d(n_channels_locals))
        local_layers.append(nn.Dropout(0.25))
        local_layers.append(nn.Linear(n_channels_locals, n_out_unit))
        local_layers.append(nn.BatchNorm1d(n_out_unit))
        local_layers.append(nn.LeakyReLU(0.2))
        local_layers.append(nn.Dropout(0.25))
        self.local_layers = nn.Sequential(*local_layers)

        ## Scene layers to process places (14,14,512)
        scene_layers = []
        scene_layers.append(nn.BatchNorm2d(n_channels_scene))
        scene_layers.append(nn.Dropout(0.25))
        scene_layers.append(nn.Conv2d(n_channels_scene, n_inter_unit, 3, 2, 1))
        scene_layers.append(nn.BatchNorm2d(n_inter_unit))
        scene_layers.append(nn.LeakyReLU(0.2))
        scene_layers.append(nn.Dropout(0.25))

        scene_layers.append(nn.Conv2d(n_inter_unit, n_out_unit, 3, 1, 1))
        scene_layers.append(nn.BatchNorm2d(n_out_unit))
        scene_layers.append(nn.LeakyReLU(0.2))
        scene_layers.append(nn.Dropout(0.25))

        scene_layers.append(nn.AvgPool2d(7))

        self.scene_layers = nn.Sequential(*scene_layers)

        segment_layers = []
        segment_layers.append(nn.BatchNorm2d(n_channels_segment))
        segment_layers.append(nn.Dropout(0.25))
        segment_layers.append(nn.Conv2d(n_channels_segment, n_inter_unit, 3, 2, 1))
        segment_layers.append(nn.BatchNorm2d(n_inter_unit))
        segment_layers.append(nn.LeakyReLU(0.2))
        segment_layers.append(nn.Dropout(0.25))

        segment_layers.append(nn.Conv2d(n_inter_unit, n_out_unit, 3, 1, 1))
        segment_layers.append(nn.BatchNorm2d(n_out_unit))
        segment_layers.append(nn.LeakyReLU(0.2))
        segment_layers.append(nn.Dropout(0.25))

        segment_layers.append(nn.AvgPool2d(7))

        self.segment_layers = nn.Sequential(*segment_layers)

        # Joint classifier layer
        last_layer = []
        last_layer.append(nn.Linear(n_out_unit*3, n_out_unit))
        last_layer.append(nn.BatchNorm1d(n_out_unit))
        last_layer.append(nn.LeakyReLU(0.2))
        last_layer.append(nn.Dropout(0.25))
        last_layer.append(nn.Linear(n_out_unit, n_classes))

        self.last_layer = nn.Sequential(*last_layer)

    def __init_optimizer(self):
        """
        Define loss, metric and optimizer.
        """
        self._loss_fn = F.binary_cross_entropy
        self._metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
        self._optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-4)

    def forward(self, *input):
        """
        input is two features: full image feature and context feature
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        # spatial pooling
        x = input[2] # local scene
        x = self.spatial_pooling(x)  # (B, C, N)

        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x = x.contiguous().view(B * N, C)  # (B*N, C)

        # get attention values per region
        alpha = x
        for l in self.pool_layers:
            alpha = l(alpha) # (B*N, 1)

        alpha = torch.sigmoid(alpha)

        # reweight features
        x = torch.mul(x, alpha) # (B*N, C)
        x = x.view(B, N, C)  # (B, N, C)
        x = torch.mean(x, 1)

        # feed to classifier
        for l in self.local_layers:
            x = l(x)

        x_local = x

        x = input[0] # scene feature
        # remove singleton dimension
        x = self.squeeze(x)  # (B, C, H,W)
        # feed to classifier
        for l in self.scene_layers:
            x = l(x)

        x_scene = x
        x_scene = torch.squeeze(x_scene)

        x = input[1] # segment
        # remove singleton dimension
        x = self.squeeze(x)  # (B, C, H,W)
        # feed to classifier
        for l in self.segment_layers:
            x = l(x)

        x_segment = x
        x_segment = torch.squeeze(x_segment)

        x = torch.cat((x_scene, x_segment, x_local), 1)

        for l in self.last_layer:
            x = l(x)

        x = torch.sigmoid(x) # (B, C)

        return x

def train_local_context():

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    # Features of the image: f_interaction
    feature_path_interaction = Pth('Hico/features/extra/features_local_locality.h5')
    n_channels, n_regions, channel_side_dim = 4096*2, 12, 1

    # Annotation of the image
    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())
    input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)

    print('--- start time')
    print(datetime.datetime.now())

    print('... loading data')
    t1 = time.time()

    print('... interaction features')
    (img_names_tr, y_tr, y_tr_mask, img_names_te, y_te, y_te_mask) = utils.pkl_load(annot_path)
    (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)

    y_tr_mask = y_tr_mask.astype(np.float32)
    y_te_mask = y_te_mask.astype(np.float32)

    x_tr = np.swapaxes(x_tr, 1,2)
    x_te = np.swapaxes(x_te, 1,2)

    print('train_set_shape_interaction: ', x_tr.shape)
    print('test_set_shape_interaction: ', x_te.shape)

    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    model = ClassifierMultiHumanObjectContextPooling(n_classes, input_shape) 
    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    input_sizes = [input_shape]
    pytorch_utils.model_summary_multi_input(model, input_sizes=input_sizes, batch_size=-1, device='cuda')    
    print('... model built, duration (sec): %d' % (duration))

    # callbacks
    model_name = 'aura_%s' % (utils.timestamp(),)
    model_root_path = Pth('Hico/models_finetuned/%s', (model_name))
    model_save_callback =  pytorch_utils.ModelSaveCallback(model, model_root_path)

    print('Training: %s' %(model_name))

    # start training
    pytorch_utils.train_model_custom_metric_mask(model, model._optimizer, model._loss_fn, model._metric_fn, x_tr, y_tr, y_tr_mask, x_te, y_te, y_te_mask, n_epochs, batch_size_tr, batch_size_te, callbacks=[model_save_callback])

    print('--- finish time')
    print(datetime.datetime.now())

def train_human_object_pooling():

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    feature_path_interaction = Pth('Hico/features/features_base_subject_object.h5')
    n_channels, n_regions, channel_side_dim = 4096, 12, 1

    # Annotation of the image
    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())
    input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)

    print('--- start time')
    print(datetime.datetime.now())

    print('... loading data')
    t1 = time.time()

    print('... interaction features')
    (img_names_tr, y_tr, y_tr_mask, img_names_te, y_te, y_te_mask) = utils.pkl_load(annot_path)
    (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)

    y_tr_mask = y_tr_mask.astype(np.float32)
    y_te_mask = y_te_mask.astype(np.float32)

    x_tr = np.swapaxes(x_tr, 1,2)
    x_te = np.swapaxes(x_te, 1,2)

    print('train_set_shape_interaction: ', x_tr.shape)
    print('test_set_shape_interaction: ', x_te.shape)

    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    model = ClassifierMultiHumanObjectContextPooling(n_classes, input_shape) 
    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    input_sizes = [input_shape]
    #pytorch_utils.model_summary_multi_input(model, input_sizes=input_sizes, batch_size=-1, device='cuda')    
    print('... model built, duration (sec): %d' % (duration))

    # callbacks
    model_name = 'base_human_object_%s' % (utils.timestamp(),)
    model_root_path = Pth('Hico/models_finetuned/%s', (model_name))
    model_save_callback =  pytorch_utils.ModelSaveCallback(model, model_root_path)

    print('Training: %s' %(model_name))

    # start training
    pytorch_utils.train_model_custom_metric_mask(model, model._optimizer, model._loss_fn, model._metric_fn, x_tr, y_tr, y_tr_mask, x_te, y_te, y_te_mask, n_epochs, batch_size_tr, batch_size_te, callbacks=[model_save_callback])

    print('--- finish time')
    print(datetime.datetime.now())

class ClassifierMultiHumanObjectContext(nn.Module):
    def __init__(self, n_classes, x_so_shape):
        super(ClassifierMultiHumanObjectContext, self).__init__()

        self.layer_name_fusion = 'context_fusion_%d'
        self.__init_layers(n_classes, x_so_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_so_shape):
        """
        Define model layers.
        """

        n_units = 512
        n_channels_in_action   = x_so_shape[0] 

        print('n_channels_in_action: ', n_channels_in_action)

        self.N = x_so_shape[1]

        print('Number of human-object: ', self.N)

        # sparial pooling
        self.spatial_pooling = pl.Max(dim=(3, 4))

        # layers for action
        classifier_layers_action = []
        classifier_layers_action.append(nn.BatchNorm1d(n_channels_in_action))
        classifier_layers_action.append(nn.Dropout(0.25))
        classifier_layers_action.append(nn.Linear(n_channels_in_action, n_units))
        classifier_layers_action.append(nn.BatchNorm1d(n_units))
        classifier_layers_action.append(nn.LeakyReLU(0.2))
        classifier_layers_action.append(nn.Dropout(0.25))
        self.classifier_layers_action = nn.Sequential(*classifier_layers_action)

        classifier = []
        classifier.append(nn.Linear(n_units, n_classes))
        self.classifier = nn.Sequential(*classifier) 

    def __init_optimizer(self):
        """
        Define loss, metric and optimizer.
        """

        self._loss_fn = F.binary_cross_entropy
        self._metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
        self._optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-4)
        #self._optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9)

    def forward(self, *input):
        """
        input is two features: full image feature and context feature
        :param x_so: full image feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x = input[0]
        N = self.N

        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)
        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x = x.contiguous().view(B * N, C)  # (B*N, C)

        # TODO: Feed interaction and context features to appropriate MLPs
        # feed to classifier
        for l in self.classifier_layers_action:
            x = l(x)

        # recover the dimension of N
        _, C = pytorch_utils.get_shape(x)
        x = x.view(B, N, C)  # (B, N, C)

        # Feed-to-joint-classifier
        x = self.classifier(x)

        # apply max ops there
        x, _ = torch.max(x, dim=1)  # (B, C)
        x = torch.sigmoid(x)

        return x

    def forward_for_extraction(self, *input):
        """
        input is two features: full image feature and context feature
        :param x_so: full image feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x = input[0]
        N = self.N

        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)
        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x = x.contiguous().view(B * N, C)  # (B*N, C)

        # feed to classifier
        for l in self.classifier_layers_action:
            x = l(x)

        # recover the dimension of N
        _, C = pytorch_utils.get_shape(x)
        x = x.view(B, N, C)  # (B, N, C)

        return x



class ClassifierMultiHumanObjectContextPooling(nn.Module):
    def __init__(self, n_classes, x_so_shape):
        super(ClassifierMultiHumanObjectContextPooling, self).__init__()

        self.layer_name_fusion = 'context_fusion_%d'
        self.__init_layers(n_classes, x_so_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_so_shape):
        """
        Define model layers.
        """

        n_units = 512
        n_channels_in_action   = x_so_shape[0] 

        self.hard = False

        print('n_channels_in_action: ', n_channels_in_action)

        self.N = x_so_shape[1]

        print('Number of human-object: ', self.N)

        # sparial pooling
        self.spatial_pooling = pl.Max(dim=(3, 4))

        self.softmax = nn.Softmax(dim=1)

        # layers for pooling
        pool_layers = []
        pool_layers.append(nn.BatchNorm1d(n_units))
        pool_layers.append(nn.Dropout(0.25))
        pool_layers.append(nn.Linear(n_units, 1))
        self.pool_layers = nn.Sequential(*pool_layers)

        # layers for action
        classifier_layers_action = []
        classifier_layers_action.append(nn.BatchNorm1d(n_channels_in_action))
        classifier_layers_action.append(nn.Dropout(0.25))
        classifier_layers_action.append(nn.Linear(n_channels_in_action, n_units))
        classifier_layers_action.append(nn.BatchNorm1d(n_units))
        classifier_layers_action.append(nn.LeakyReLU(0.2))
        classifier_layers_action.append(nn.Dropout(0.25))
        self.classifier_layers_action = nn.Sequential(*classifier_layers_action)

        classifier = []
        classifier.append(nn.Linear(n_units, n_classes))
        self.classifier = nn.Sequential(*classifier) 

    def __init_optimizer(self):
        """
        Define loss, metric and optimizer.
        """

        self._loss_fn = F.binary_cross_entropy
        self._metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
        self._optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-4)
        #self._optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9)

    def forward(self, *input):
        """
        input is two features: full image feature and context feature
        :param x_so: full image feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x = input[0]
        N = self.N

        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)
        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x = x.contiguous().view(B * N, C)  # (B*N, C)

        # feed to classifier
        for l in self.classifier_layers_action:
            x = l(x)

        alpha = x
        for l in self.pool_layers:
            alpha = l(alpha) # (B*N, 1)

        #alpha = torch.sigmoid(alpha)
        alpha = alpha.view(B, N)
        alpha = self.softmax(alpha)

        alpha = alpha.view(B*N,1)

        BN, C = pytorch_utils.get_shape(x)

        # reweight features
        x = torch.mul(x, alpha) # (B*N, C)
        x = x.view(B, N, C)  # (B, N, C)
        x = torch.sum(x, 1)  # (B, 1, C)

        # Feed-to-joint-classifier
        x = self.classifier(x)
        x = torch.sigmoid(x)

        return x

    def inference(self, *input):
        """
        input is two features: full image feature and context feature
        :param x_so: full image feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x = input[0]
        N = self.N

        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)
        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x = x.contiguous().view(B * N, C)  # (B*N, C)

        # feed to classifier
        for l in self.classifier_layers_action:
            x = l(x)

        alpha = x
        for l in self.pool_layers:
            alpha = l(alpha) # (B*N, 1)

        #alpha = torch.sigmoid(alpha)
        alpha = alpha.view(B, N)
        alpha = self.softmax(alpha)

        alpha = alpha.view(B*N,1)

        BN, C = pytorch_utils.get_shape(x)

        # reweight features
        x = torch.mul(x, alpha) # (B*N, C)
        x = x.view(B, N, C)  # (B, N, C)
        x = torch.sum(x, 1)  # (B, 1, C)

        return x



class ClassifierMultiHumanObjectContextPoolingHard(nn.Module):
    def __init__(self, n_classes, x_so_shape):
        super(ClassifierMultiHumanObjectContextPoolingHard, self).__init__()

        self.layer_name_fusion = 'context_fusion_%d'
        self.__init_layers(n_classes, x_so_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_so_shape):
        """
        Define model layers.
        """

        n_units = 512
        n_channels_in_action   = x_so_shape[0] 

        self.hard = False

        print('n_channels_in_action: ', n_channels_in_action)

        self.N = x_so_shape[1]

        print('Number of human-object: ', self.N)

        # sparial pooling
        self.spatial_pooling = pl.Max(dim=(3, 4))

        self.softmax = nn.Softmax(dim=1)

        # layers for pooling
        pool_layers = []
        pool_layers.append(nn.BatchNorm1d(n_units))
        pool_layers.append(nn.Dropout(0.25))
        pool_layers.append(nn.Linear(n_units, 1))
        self.pool_layers = nn.Sequential(*pool_layers)

        # layers for action
        classifier_layers_action = []
        classifier_layers_action.append(nn.BatchNorm1d(n_channels_in_action))
        classifier_layers_action.append(nn.Dropout(0.25))
        classifier_layers_action.append(nn.Linear(n_channels_in_action, n_units))
        classifier_layers_action.append(nn.BatchNorm1d(n_units))
        classifier_layers_action.append(nn.LeakyReLU(0.2))
        classifier_layers_action.append(nn.Dropout(0.25))
        self.classifier_layers_action = nn.Sequential(*classifier_layers_action)

        classifier = []
        classifier.append(nn.Linear(n_units, n_classes))
        self.classifier = nn.Sequential(*classifier) 

    def __init_optimizer(self):
        """
        Define loss, metric and optimizer.
        """

        self._loss_fn = F.binary_cross_entropy
        self._metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
        #self._optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-4)
        self._optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9)

    def sample_gumbel_like(self, template_tensor, eps=1e-10):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumble_samples_tensor = - torch.log(eps - torch.log(uniform_samples_tensor + eps))
        return gumble_samples_tensor

    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        gumble_samples_tensor = self.sample_gumbel_like(logits.data)
        gumble_trick_log_prob_samples = logits + Variable(gumble_samples_tensor)
        soft_samples = F.softmax(gumble_trick_log_prob_samples / temperature, 1)
        return soft_samples

    def forward(self, *input):
        """
        input is two features: full image feature and context feature
        :param x_so: full image feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x = input[0]
        N = self.N

        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)
        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x = x.contiguous().view(B * N, C)  # (B*N, C)

        # feed to classifier
        for l in self.classifier_layers_action:
            x = l(x)

        alpha = x
        for l in self.pool_layers:
            alpha = l(alpha) # (B*N, 1)

        #alpha = torch.sigmoid(alpha)
        alpha = alpha.view(B, N)

        # hard attention
        alpha = self.gumbel_softmax_sample(alpha, 1)

        _, max_value_indexes = alpha.data.max(1, keepdim=True)

        alpha_hard = alpha.data.clone().zero_().scatter_(1, max_value_indexes, 1)
        alpha = Variable(alpha_hard - alpha.data) + alpha

        alpha = alpha.view(B*N,1)

        BN, C = pytorch_utils.get_shape(x)

        # reweight features
        x = torch.mul(x, alpha) # (B*N, C)
        x = x.view(B, N, C)  # (B, N, C)
        x, _= torch.sum(x, 1)  # (B, 1, C)

        # Feed-to-joint-classifier
        x = self.classifier(x)
        x = torch.sigmoid(x)

        return x
