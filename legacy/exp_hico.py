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

# endregion

# region Train Classifier: Baseline

def train_classifier_using_features_single_region():
    """
    Train model.
    """

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    features_path = Pth('Hico/features/legacy/features_images.h5')
    n_channels, n_regions, channel_side_dim = 2048, 1, 1

    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())
    input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)

    print('--- start time')
    print(datetime.datetime.now())

    print('... loading data')
    t1 = time.time()
    (img_names_tr, y_tr, y_tr_mask, img_names_te, y_te, y_te_mask) = utils.pkl_load(annot_path)
    (x_tr, x_te) = utils.h5_load_multi(features_path, ['x_tr', 'x_te'])
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)
    y_tr_mask = y_tr_mask.astype(np.float32)
    y_te_mask = y_te_mask.astype(np.float32)

    print('train_set_shape: ', x_tr.shape)
    print('test_set_shape: ', x_te.shape)

    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    model = ClassifierSimpleSingleRegion(n_classes, input_shape)
    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    pytorch_utils.model_summary(model, input_size=input_shape, batch_size=-1, device='cuda')
    print('... model built, duration (sec): %d' % (duration))

    # callbacks
    callbacks = []

    # start training
    pytorch_utils.train_model_custom_metric_mask(model, model._optimizer, model._loss_fn, model._metric_fn, x_tr, y_tr, y_tr_mask, x_te, y_te, y_te_mask, n_epochs, batch_size_tr, batch_size_te, callbacks=callbacks)

    print('--- finish time')
    print(datetime.datetime.now())

class ClassifierSimpleSingleRegion(nn.Module):
    def __init__(self, n_classes, input_shape):
        super(ClassifierSimpleSingleRegion, self).__init__()

        self.__init_layers(n_classes, input_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, input_shape):
        """
        Define model layers.
        """

        n_channels, n_regions, side_dim, side_dim = input_shape
        n_units = 512

        self.spatial_pooling = pl.Max(dim=(3, 4))
        self.squeeze_regions = pl.Squeeze(dim=2)

        # layers for classification
        classifier_layers = []
        classifier_layers.append(nn.Dropout(0.25))
        classifier_layers.append(nn.Linear(n_channels, n_units))
        classifier_layers.append(nn.BatchNorm1d(n_units))
        classifier_layers.append(nn.LeakyReLU(0.2))
        classifier_layers.append(nn.Dropout(0.25))
        classifier_layers.append(nn.Linear(n_units, n_classes))
        self.classifier_layers = nn.Sequential(*classifier_layers)

    def __init_optimizer(self):
        """
        Define loss, metric and optimizer.
        """
        self._loss_fn = torch.nn.BCELoss()
        self._metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
        self._optimizer = optim.Adam(self.parameters(), lr=0.01, eps=1e-4)

    def forward(self, x):
        # spatial pooling
        x = self.spatial_pooling(x)
        x = self.squeeze_regions(x)

        # feed to classifier
        for l in self.classifier_layers:
            x = l(x)

        x = torch.sigmoid(x)

        return x

def train_classifier_using_features_single_context():
    """
    Train model.
    """

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    features_path = Pth('Hico/features/features_context_places.h5')
    n_channels, n_regions, channel_side_dim = 512, 1, 1

    features_path = Pth('Hico/features/features_context_local_scene.h5')
    n_channels, n_regions, channel_side_dim =2048, 1, 1

    features_path = Pth('Hico/features/features_context_relashionship.h5')
    n_channels, n_regions, channel_side_dim =2048, 1, 1

    features_path = Pth('Hico/features/features_local_object.h5')
    n_channels, n_regions, channel_side_dim =6144, 1, 1

    features_path = Pth('Hico/features/features_imsitu.h5')
    n_channels, n_regions, channel_side_dim =2048, 1, 1

    features_path = Pth('Hico/features/features_part_states.h5')
    n_channels, n_regions, channel_side_dim =1032, 1, 1

    features_path = Pth('Hico/features/features_deformation.h5')
    n_channels, n_regions, channel_side_dim =512, 1, 1

    features_path = Pth('Hico/features/features_scene_places.h5')
    n_channels, n_regions, channel_side_dim =512, 1, 1

    features_path = Pth('Hico/features/features_scene_segment.h5')
    n_channels, n_regions, channel_side_dim =2048, 1, 1

    features_path = Pth('Hico/features/features_global_pose.h5')
    n_channels, n_regions, channel_side_dim =512, 1, 1

    features_path = Pth('Hico/features/legacy/features_lvis.h5')
    n_channels, n_regions, channel_side_dim =1300, 1, 1

    input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)

    print('feature_in_use: ', features_path)

    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())
    input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)

    print('--- start time')
    print(datetime.datetime.now())

    print('... loading data')
    t1 = time.time()
    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    (x_tr, x_te) = utils.h5_load_multi(features_path, ['x_tr', 'x_te'])
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)

    print('train_set_shape: ', x_tr.shape)
    print('test_set_shape: ', x_te.shape)

    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    model = ClassifierSimpleSingleContext(n_classes, input_shape)
    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    pytorch_utils.model_summary(model, input_size=input_shape, batch_size=-1, device='cuda')
    print('... model built, duration (sec): %d' % (duration))

    # callbacks
    callbacks = []

    # start training
    pytorch_utils.train_model_custom_metric(model, model._optimizer, model._loss_fn, model._metric_fn, x_tr, y_tr, x_te, y_te, n_epochs, batch_size_tr, batch_size_te, callbacks=callbacks)

    print('--- finish time')
    print(datetime.datetime.now())

class ClassifierSimpleSingleContext(nn.Module):
    def __init__(self, n_classes, input_shape):
        super(ClassifierSimpleSingleContext, self).__init__()

        self.__init_layers(n_classes, input_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, input_shape):
        """
        Define model layers.
        """

        n_channels, n_regions, side_dim, side_dim = input_shape
        print('n_channels: ', n_channels, ' side_dim: ', side_dim, ' side_dim: ', side_dim, ' input_shape: ', input_shape)
        n_units = n_channels

        self.spatial_pooling = pl.Max(dim=(3, 4))
        self.squeeze_regions = pl.Squeeze(dim=2)

        # layers for classification
        classifier_layers = []
        classifier_layers.append(nn.Dropout(0.25))
        classifier_layers.append(nn.Linear(n_channels, n_units))
        classifier_layers.append(nn.BatchNorm1d(n_units))
        classifier_layers.append(nn.LeakyReLU(0.2))
        classifier_layers.append(nn.Dropout(0.25))
        classifier_layers.append(nn.Linear(n_units, n_classes))
        self.classifier_layers = nn.Sequential(*classifier_layers)

    def __init_optimizer(self):
        """
        Define loss, metric and optimizer.
        """
        self._loss_fn = torch.nn.BCELoss()
        self._metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
        self._optimizer = optim.Adam(self.parameters(), lr=0.01, eps=1e-4)

    def forward(self, x):
        # spatial pooling
        x = self.spatial_pooling(x)
        x = self.squeeze_regions(x)

        # feed to classifier
        for l in self.classifier_layers:
            x = l(x)

        x = torch.sigmoid(x)

        return x

def train_classifier_using_features_multiple_context_early_fusion():

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    # Features of the image: f_scene
    feats_c1_path = Pth('Hico/features/features_scene_segment.h5')
    feats_c2_path =  Pth('Hico/features/features_deformation.h5')
    x_cs_shape = [(2048, 1, 1, 1), (512, 1, 1, 1)]

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

    print('train_set_shape_context-2: ', x_tr_c2.shape)
    print('test_set_shape_context-2: ',  x_te_c2.shape)

    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    model = ClassifierMultiContextFusionConcat(n_classes, x_cs_shape)
    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    input_sizes =  list(x_cs_shape)
    pytorch_utils.model_summary_multi_input(model, input_sizes=input_sizes, batch_size=-1, device='cuda')    
    print('... model built, duration (sec): %d' % (duration))

    # callbacks
    callbacks = []

    print('first_context: %s, second_context: %s' %(feats_c1_path, feats_c2_path))

    # start training
    pytorch_utils.train_model_custom_metric(model, model._optimizer, model._loss_fn, model._metric_fn, [x_tr_c1, x_tr_c2], y_tr, [x_te_c1, x_te_c2], y_te, n_epochs, batch_size_tr, batch_size_te, callbacks=callbacks)

    print('--- finish time')
    print(datetime.datetime.now())

class ClassifierMultiContextFusionConcat(nn.Module):
    def __init__(self, n_classes, x_cs_shape):
        super(ClassifierMultiContextFusionConcat, self).__init__()

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

        print('x_cs_shape: ', x_cs_shape)
        print('n_channels_out: ', n_channels_out)

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

        x_cs = input[0:]

        x_cs = torch.cat(x_cs, dim=1)
        x = x_cs

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


def train_classifier_using_features_single_region_single_context_early_fusion():

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    # Features of the image: f_interaction
    features_path_interaction = Pth('Hico/features/legacy/features_images.h5')
    n_channels, n_regions, channel_side_dim = 2048, 1, 1

    # Features of the image: f_scene
    '''
    feature_path_context = Pth('Hico/features/features_context_places.h5')
    x_cs_shape = [(512, 1, 1, 1)]
    '''

    '''
    feature_path_context = Pth('Hico/features/features_context_relashionship.h5')
    x_cs_shape = [(2048, 1, 1, 1)]
    '''

    '''
    feature_path_context = Pth('Hico/features/features_context_local_scene.h5')
    x_cs_shape = [(2048, 1, 1, 1)]
    '''

    feature_path_context = Pth('Hico/features/features_local_object.h5')
    x_cs_shape = [(6144, 1, 1, 1)]

    feature_path_context = Pth('Hico/features/features_global_pose.h5')
    x_cs_shape = [(256, 1, 1, 1)]

    # Annotation of the image
    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())
    input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)

    print('--- start time')
    print(datetime.datetime.now())

    print('... loading data')
    t1 = time.time()

    print('... interaction features')
    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    (x_tr, x_te) = utils.h5_load_multi(features_path_interaction, ['x_tr', 'x_te'])
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)

    print('... context features')
    (x_tr_c, x_te_c) = utils.h5_load_multi(feature_path_context, ['x_tr', 'x_te'])

    print('train_set_shape_interaction: ', x_tr.shape)
    print('test_set_shape_interaction: ', x_te.shape)

    print('train_set_shape_context: ', x_tr_c.shape)
    print('test_set_shape_context: ',  x_te_c.shape)

    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    model = ClassifierContextFusionConcat(n_classes, input_shape, x_cs_shape) # TODO: Change this to support multiple input
    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    input_sizes = [input_shape] + list(x_cs_shape)
    pytorch_utils.model_summary_multi_input(model, input_sizes=input_sizes, batch_size=-1, device='cuda')    
    print('... model built, duration (sec): %d' % (duration))

    # callbacks
    callbacks = []

    # start training
    pytorch_utils.train_model_custom_metric(model, model._optimizer, model._loss_fn, model._metric_fn, [x_tr, x_tr_c], y_tr, [x_te, x_te_c], y_te, n_epochs, batch_size_tr, batch_size_te, callbacks=callbacks)

    print('--- finish time')
    print(datetime.datetime.now())

class ClassifierContextFusionConcat(nn.Module):
    def __init__(self, n_classes, x_so_shape, x_cs_shape):
        super(ClassifierContextFusionConcat, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_fusion = 'context_fusion_%d'
        self.__init_layers(n_classes, x_so_shape, x_cs_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_so_shape, x_cs_shape):
        """
        Define model layers.
        """

        n_units = 512
        n_channels_out = x_so_shape[0] + np.sum([i[0] for i in x_cs_shape])
        self.N = x_so_shape[1]

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
        self._optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-4)

    def forward(self, *input):
        """
        input is two features: full image feature and context feature
        :param x_so: full image feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x_so = input[0]
        x_cs = input[1:]
        N = self.N

        x_cs = torch.cat(x_cs, dim=1)
        x_cs = x_cs.repeat(1, 1, N, 1, 1)
        x = torch.cat((x_so, x_cs), dim=1)

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



def train_classifier_using_features_single_region_single_context_late_fusion():

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    # Features of the image: f_interaction
    features_path_interaction = Pth('Hico/features/features_images.h5')
    n_channels, n_regions, channel_side_dim = 2048, 1, 1

    # Features of the image: f_scene

    feature_path_context = Pth('Hico/features/features_context_local_scene.h5')
    x_cs_shape = [(2048, 1, 1, 1)]
 
    # Annotation of the image
    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())
    input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)

    print('--- start time')
    print(datetime.datetime.now())

    print('... loading data')
    t1 = time.time()

    print('... interaction features')
    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    (x_tr, x_te) = utils.h5_load_multi(features_path_interaction, ['x_tr', 'x_te'])
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)

    print('... context features')
    (x_tr_c, x_te_c) = utils.h5_load_multi(feature_path_context, ['x_tr', 'x_te'])

    print('train_set_shape_interaction: ', x_tr.shape)
    print('test_set_shape_interaction: ', x_te.shape)

    print('train_set_shape_context: ', x_tr_c.shape)
    print('test_set_shape_context: ',  x_te_c.shape)

    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    model = ClassifierContextLateFusion(n_classes, input_shape, x_cs_shape) 
    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    input_sizes = [input_shape] + list(x_cs_shape)
    pytorch_utils.model_summary_multi_input(model, input_sizes=input_sizes, batch_size=-1, device='cuda')    
    print('... model built, duration (sec): %d' % (duration))

    # callbacks
    callbacks = []

    # start training
    pytorch_utils.train_model_custom_metric(model, model._optimizer, model._loss_fn, model._metric_fn, [x_tr, x_tr_c], y_tr, [x_te, x_te_c], y_te, n_epochs, batch_size_tr, batch_size_te, callbacks=callbacks)

    print('--- finish time')
    print(datetime.datetime.now())

class ClassifierContextLateFusion(nn.Module):
    def __init__(self, n_classes, x_so_shape, x_cs_shape):
        super(ClassifierContextLateFusion, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_fusion = 'context_fusion_%d'
        self.__init_layers(n_classes, x_so_shape, x_cs_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_so_shape, x_cs_shape):
        """
        Define model layers.
        """

        n_units = 512
        n_channels_in_action   = x_so_shape[0] 
        n_channels_in_context  =  np.sum([i[0] for i in x_cs_shape])
        self.N = x_so_shape[1]

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
        classifier_layers_action.append(nn.Linear(n_units, n_classes))
        self.classifier_layers_action = nn.Sequential(*classifier_layers_action)

        # layers for context
        classifier_layers_context = []
        classifier_layers_context.append(nn.BatchNorm1d(n_channels_in_context))
        classifier_layers_context.append(nn.Dropout(0.25))
        classifier_layers_context.append(nn.Linear(n_channels_in_context, n_units))
        classifier_layers_context.append(nn.BatchNorm1d(n_units))
        classifier_layers_context.append(nn.LeakyReLU(0.2))
        classifier_layers_context.append(nn.Dropout(0.25))
        classifier_layers_context.append(nn.Linear(n_units, n_classes))
        self.classifier_layers_context = nn.Sequential(*classifier_layers_context)


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
        :param x_so: full image feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x_so = input[0]
        x_cs = input[1:]
        N = self.N

        x_cs = torch.cat(x_cs, dim=1)
        x_cs = x_cs.repeat(1, 1, N, 1, 1)

        # Process action features
        x = x_so
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

        # max over N dimension, then apply activation
        x, _ = torch.max(x, dim=1)  # (B, C)
        x_action = torch.sigmoid(x)

        # Process context features
        x = x_cs
        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)
        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x = x.contiguous().view(B * N, C)  # (B*N, C)

        # feed to classifier
        for l in self.classifier_layers_context:
            x = l(x)

        # recover the dimension of N
        _, C = pytorch_utils.get_shape(x)
        x = x.view(B, N, C)  # (B, N, C)

        # max over N dimension, then apply activation
        x, _ = torch.max(x, dim=1)  # (B, C)
        x_context = torch.sigmoid(x)

        # Combine predictions 
        x = x_action * x_context

        return x

def train_classifier_using_features_single_region_single_context_late_early__fusion():

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    # Features of the image: f_interaction
    features_path_interaction = Pth('Hico/features/features_images.h5')
    n_channels, n_regions, channel_side_dim = 2048, 1, 1

    # Features of the image: f_scene
    feature_path_context = Pth('Hico/features/features_context_early_fusion.h5')
    x_cs_shape = [(3072, 1, 1, 1)]
 
    # Annotation of the image
    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())
    input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)

    print('--- start time')
    print(datetime.datetime.now())

    print('... loading data')
    t1 = time.time()

    print('... interaction features')
    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    (x_tr, x_te) = utils.h5_load_multi(features_path_interaction, ['x_tr', 'x_te'])
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)

    print('... context features')
    (x_tr_c, x_te_c) = utils.h5_load_multi(feature_path_context, ['x_tr', 'x_te'])

    print('train_set_shape_interaction: ', x_tr.shape)
    print('test_set_shape_interaction: ', x_te.shape)

    print('train_set_shape_context: ', x_tr_c.shape)
    print('test_set_shape_context: ',  x_te_c.shape)

    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    model = ClassifierContextLateEarlyFusion(n_classes, input_shape, x_cs_shape) 
    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    input_sizes = [input_shape] + list(x_cs_shape)
    pytorch_utils.model_summary_multi_input(model, input_sizes=input_sizes, batch_size=-1, device='cuda')    
    print('... model built, duration (sec): %d' % (duration))

    # callbacks
    callbacks = []

    # start training
    pytorch_utils.train_model_custom_metric(model, model._optimizer, model._loss_fn, model._metric_fn, [x_tr, x_tr_c], y_tr, [x_te, x_te_c], y_te, n_epochs, batch_size_tr, batch_size_te, callbacks=callbacks)

    print('--- finish time')
    print(datetime.datetime.now())

class ClassifierContextLateEarlyFusion(nn.Module):
    def __init__(self, n_classes, x_so_shape, x_cs_shape):
        super(ClassifierContextLateEarlyFusion, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_fusion = 'context_fusion_%d'
        self.__init_layers(n_classes, x_so_shape, x_cs_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_so_shape, x_cs_shape):
        """
        Define model layers.
        """

        n_units = 512
        n_channels_in_action   = x_so_shape[0] 
        n_channels_in_context  =  np.sum([i[0] for i in x_cs_shape])
        self.N = x_so_shape[1]

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

        # layers for context
        classifier_layers_context = []
        classifier_layers_context.append(nn.BatchNorm1d(n_channels_in_context))
        classifier_layers_context.append(nn.Dropout(0.25))
        classifier_layers_context.append(nn.Linear(n_channels_in_context, n_units))
        classifier_layers_context.append(nn.BatchNorm1d(n_units))
        classifier_layers_context.append(nn.LeakyReLU(0.2))
        classifier_layers_context.append(nn.Dropout(0.25))
        self.classifier_layers_context = nn.Sequential(*classifier_layers_context)

        classifier = []
        classifier.append(nn.Linear(n_units*2, n_classes))
        self.classifier = nn.Sequential(*classifier) 


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
        :param x_so: full image feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x_so = input[0]
        x_cs = input[1:]
        N = self.N

        x_cs = torch.cat(x_cs, dim=1)
        x_cs = x_cs.repeat(1, 1, N, 1, 1)

        # Process action features
        x = x_so
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
        x_feat_action = x

        # Process context features
        x = x_cs
        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)
        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x = x.contiguous().view(B * N, C)  # (B*N, C)

        # TODO: Feed interaction and context features to appropriate MLPs
        # feed to classifier
        for l in self.classifier_layers_context:
            x = l(x)

        # recover the dimension of N
        _, C = pytorch_utils.get_shape(x)
        x = x.view(B, N, C)  # (B, N, C)
        x_feat_context = x

        # Concatenate features
        x = torch.cat((x_feat_action, x_feat_context), dim=2)

        # Feed-to-joint-classifier
        x = self.classifier(x)

        # apply max ops there
        x, _ = torch.max(x, dim=1)  # (B, C)
        x = torch.sigmoid(x)

        return x


def train_classifier_using_features_pairatt_pose_late_early_fusion():

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    # Features of the image: f_interaction
    feature_path_interaction = Pth('Hico/features/legacy/features_pairattn.h5')
    n_channels, n_regions, channel_side_dim = 4096, 3, 1

    # Features of the pose: f_context
    feature_path_context = Pth('Hico/features/legacy/features_pairattn_pose.h5')
    x_cs_shape = [(4096, 3, 1, 1)]
 
    # Features of the pose: f_context
    feature_path_context = Pth('Hico/features/features_global_deformation.h5')
    x_cs_shape = [(512, 1, 1, 1)]

    # Features of the pose: f_context
    feature_path_context = Pth('Hico/features/legacy/features_context_local_scene.h5')
    x_cs_shape = [(2048, 1, 1, 1)]

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

    print('... context features')
    (x_tr_c, x_te_c) = utils.h5_load_multi(feature_path_context, ['x_tr', 'x_te'])

    '''
    x_tr_c = np.expand_dims(x_tr_c, 2)
    x_tr_c = np.expand_dims(x_tr_c, 3)
    x_tr_c = np.expand_dims(x_tr_c, 4)

    x_te_c = np.expand_dims(x_te_c, 2)
    x_te_c = np.expand_dims(x_te_c, 3)
    x_te_c = np.expand_dims(x_te_c, 4)
    '''
    
    # Fix pose features
    #x_tr_c = np.swapaxes(x_tr_c, 1,2)
    #x_te_c = np.swapaxes(x_te_c, 1,2)


    print('train_set_shape_interaction: ', x_tr.shape)
    print('test_set_shape_interaction: ', x_te.shape)

    print('train_set_shape_context: ', x_tr_c.shape)
    print('test_set_shape_context: ',  x_te_c.shape)

    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    model = ClassifierContextLateEarlyFusionPose(n_classes, input_shape, x_cs_shape) 
    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    input_sizes = [input_shape] + list(x_cs_shape)
    pytorch_utils.model_summary_multi_input(model, input_sizes=input_sizes, batch_size=-1, device='cuda')    
    print('... model built, duration (sec): %d' % (duration))

    # callbacks
    callbacks = []

    # start training
    pytorch_utils.train_model_custom_metric_mask(model, model._optimizer, model._loss_fn, model._metric_fn, [x_tr, x_tr_c], y_tr, y_tr_mask, [x_te, x_te_c], y_te, y_te_mask, n_epochs, batch_size_tr, batch_size_te, callbacks=callbacks)

    print('--- finish time')
    print(datetime.datetime.now())

class ClassifierContextLateEarlyFusionPose(nn.Module):
    def __init__(self, n_classes, x_so_shape, x_cs_shape):
        super(ClassifierContextLateEarlyFusionPose, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_fusion = 'context_fusion_%d'
        self.__init_layers(n_classes, x_so_shape, x_cs_shape)
        self.__init_optimizer()

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, mean = 0, std = 1)
            m.bias.data.fill_(0.01)

    def __init_layers(self, n_classes, x_so_shape, x_cs_shape):
        """
        Define model layers.
        """

        n_units = 512
        n_channels_in_action   = x_so_shape[0] 
        n_channels_in_context  =  np.sum([i[0] for i in x_cs_shape])

        print('n_channels_in_action: ', n_channels_in_action)
        print('n_channels_in_context: ', n_channels_in_context)

        self.N = x_so_shape[1]

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

        # layers for context
        classifier_layers_context = []
        classifier_layers_context.append(nn.BatchNorm1d(n_channels_in_context))
        classifier_layers_context.append(nn.Dropout(0.25))
        classifier_layers_context.append(nn.Linear(n_channels_in_context, n_units))
        classifier_layers_context.append(nn.BatchNorm1d(n_units))
        classifier_layers_context.append(nn.LeakyReLU(0.2))
        classifier_layers_context.append(nn.Dropout(0.25))
        self.classifier_layers_context = nn.Sequential(*classifier_layers_context)

        classifier = []
        classifier.append(nn.Linear(n_units*2, n_classes))
        self.classifier = nn.Sequential(*classifier) 

        self.init_weights(self.classifier_layers_action)
        self.init_weights(self.classifier_layers_context)

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
        :param x_so: full image feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x_so = input[0]
        x_cs = input[1:]
        N = self.N

        x_cs = torch.cat(x_cs, dim=1)
        x_cs = x_cs.repeat(1, 1, N, 1, 1)

        # Process action features
        x = x_so
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
        x_feat_action = x

        # Process context features
        x = x_cs
        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)
        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x = x.contiguous().view(B * N, C)  # (B*N, C)

        # feed to classifier
        for l in self.classifier_layers_context:
            x = l(x)

        # recover the dimension of N
        _, C = pytorch_utils.get_shape(x)
        x = x.view(B, N, C)  # (B, N, C)
        x_feat_context = x

        # Concatenate features
        x = torch.cat((x_feat_action, x_feat_context), dim=2)

        # Feed-to-joint-classifier
        x = self.classifier(x)

        # apply max ops there
        x, _ = torch.max(x, dim=1)  # (B, C)
        x = torch.sigmoid(x)

        return x

def train_classifier_using_features_pairatt_pose_late_fusion():

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    # Features of the image: f_interaction
    feature_path_interaction = Pth('Hico/features/legacy/features_pairattn.h5')
    n_channels, n_regions, channel_side_dim = 4096, 3, 1

    # Features of the image: f_interaction
    feature_path_interaction = Pth('Hico/features/features_base_subject_object.h5')
    n_channels, n_regions, channel_side_dim = 4096, 12, 1

    # Features of the pose: f_context
    feature_path_context = Pth('Hico/features/legacy/features_pairattn_pose.h5')
    x_cs_shape = [(4096, 3, 1, 1)]
 
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

    # Fix pose features
    x_tr = np.swapaxes(x_tr, 1,2)
    x_te = np.swapaxes(x_te, 1,2)

    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)

    y_tr_mask = y_tr_mask.astype(np.float32)
    y_te_mask = y_te_mask.astype(np.float32)
    print('... context features')
    (x_tr_c, x_te_c) = utils.h5_load_multi(feature_path_context, ['x_tr', 'x_te'])

    # Fix pose features
    x_tr_c = np.swapaxes(x_tr_c, 1,2)
    x_te_c = np.swapaxes(x_te_c, 1,2)

    x_tr_c = np.expand_dims(x_tr_c, 3)
    x_tr_c = np.expand_dims(x_tr_c, 4)

    x_te_c = np.expand_dims(x_te_c, 3)
    x_te_c = np.expand_dims(x_te_c, 4)

    print('train_set_shape_interaction: ', x_tr.shape)
    print('test_set_shape_interaction: ', x_te.shape)

    print('train_set_shape_context: ', x_tr_c.shape)
    print('test_set_shape_context: ',  x_te_c.shape)

    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    model = ClassifierContextLateFusionPose(n_classes, input_shape, x_cs_shape) 
    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    input_sizes = [input_shape] + list(x_cs_shape)
    pytorch_utils.model_summary_multi_input(model, input_sizes=input_sizes, batch_size=-1, device='cuda')    
    print('... model built, duration (sec): %d' % (duration))

    # callbacks
    callbacks = []

    # start training
    pytorch_utils.train_model_custom_metric_mask(model, model._optimizer, model._loss_fn, model._metric_fn, [x_tr, x_tr_c], y_tr, y_tr_mask, [x_te, x_te_c], y_te, y_te_mask, n_epochs, batch_size_tr, batch_size_te, callbacks=callbacks)

    print('--- finish time')
    print(datetime.datetime.now())

class ClassifierContextLateFusionPose(nn.Module):
    def __init__(self, n_classes, x_so_shape, x_cs_shape):
        super(ClassifierContextLateFusionPose, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_fusion = 'context_fusion_%d'
        self.__init_layers(n_classes, x_so_shape, x_cs_shape)
        self.__init_optimizer()

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, mean = 0, std = 1)
            m.bias.data.fill_(0.01)

    def __init_layers(self, n_classes, x_so_shape, x_cs_shape):
        """
        Define model layers.
        """

        n_units = 512
        n_channels_in_action   = x_so_shape[0] 
        n_channels_in_context  =  np.sum([i[0] for i in x_cs_shape])
        self.N = x_so_shape[1]

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
        classifier_layers_action.append(nn.Linear(n_units, n_classes))
        self.classifier_layers_action = nn.Sequential(*classifier_layers_action)

        # layers for context
        classifier_layers_context = []
        classifier_layers_context.append(nn.BatchNorm1d(n_channels_in_context))
        classifier_layers_context.append(nn.Dropout(0.25))
        classifier_layers_context.append(nn.Linear(n_channels_in_context, n_units))
        classifier_layers_context.append(nn.BatchNorm1d(n_units))
        classifier_layers_context.append(nn.LeakyReLU(0.2))
        classifier_layers_context.append(nn.Dropout(0.25))
        classifier_layers_context.append(nn.Linear(n_units, n_classes))
        self.classifier_layers_context = nn.Sequential(*classifier_layers_context)

        #self.init_weights(self.classifier_layers_action)
        #self.init_weights(self.classifier_layers_context)

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

        x_so = input[0]
        x_cs = input[1:]
        N = self.N

        x_cs = torch.cat(x_cs, dim=1)
        x_cs = x_cs.repeat(1, 1, N, 1, 1)

        # Process action features
        x = x_so
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

        # max over N dimension, then apply activation
        x, _ = torch.max(x, dim=1)  # (B, C)
        x_action = torch.sigmoid(x)

        # Process context features
        x = x_cs
        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)
        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x = x.contiguous().view(B * N, C)  # (B*N, C)

        # feed to classifier
        for l in self.classifier_layers_context:
            x = l(x)

        # recover the dimension of N
        _, C = pytorch_utils.get_shape(x)
        x = x.view(B, N, C)  # (B, N, C)

        # max over N dimension, then apply activation
        x, _ = torch.max(x, dim=1)  # (B, C)
        x_context = torch.sigmoid(x)

        # Combine predictions 
        x = x_action * x_context

        return x

def train_classifier_using_features_multi_region_single_context():

    n_epochs = 100000000
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    # Features of the image: f_interaction
    feature_path_interaction = Pth('Hico/features/features_base_subject_object.h5')
    n_channels, n_regions, channel_side_dim = 4096, 12, 1
 
    # Features of the pose: f_context
    feature_path_context = Pth('Hico/features/features_global_deformation.h5')
    x_cs_shape = [(512, 1, 1, 1)]

    # Features of the pose: f_context
    feature_path_context = Pth('Hico/features/features_local_locality.h5')
    x_cs_shape = [(512, 12, 1, 1)]

    # Features of the pose: f_context
    feature_path_context = Pth('Hico/features/legacy/features_context_local_scene.h5')
    x_cs_shape = [(2048, 1, 1, 1)]

    # Features of the pose: f_context
    feature_path_context = Pth('Hico/features/features_local_part_states.h5')
    x_cs_shape = [(86, 12, 1, 1)]

    # Features of the pose: f_context
    feature_path_context = Pth('Hico/features/legacy/features_lvis.h5')
    x_cs_shape = [(1300, 1, 1, 1)]

    # Features of the pose: f_context
    feature_path_context = Pth('Hico/features/legacy/features_imsitu_role.h5')
    x_cs_shape = [(1980, 1, 1, 1)]

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

    print('... context features')
    (x_tr_c, x_te_c) = utils.h5_load_multi(feature_path_context, ['x_tr', 'x_te'])
    #x_tr_c = np.swapaxes(x_tr_c, 1,2)
    #x_te_c = np.swapaxes(x_te_c, 1,2)

    '''
    x_tr_c = np.expand_dims(x_tr_c, 2)
    x_tr_c = np.expand_dims(x_tr_c, 3)
    x_tr_c = np.expand_dims(x_tr_c, 4)

    x_te_c = np.expand_dims(x_te_c, 2)
    x_te_c = np.expand_dims(x_te_c, 3)
    x_te_c = np.expand_dims(x_te_c, 4)
    '''

    print('train_set_shape_interaction: ', x_tr.shape)
    print('test_set_shape_interaction: ', x_te.shape)

    print('train_set_shape_context: ', x_tr_c.shape)
    print('test_set_shape_context: ',  x_te_c.shape)

    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    model = ClassifierContextLateEarlyFusionHumanObject(n_classes, input_shape, x_cs_shape) 
    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    input_sizes = [input_shape] + list(x_cs_shape)
    #pytorch_utils.model_summary_multi_input(model, input_sizes=input_sizes, batch_size=-1, device='cuda')    
    print('... model built, duration (sec): %d' % (duration))

    # callbacks
    callbacks = []

    print('Interaction_feat: %s, Context_feat: %s\n' %(feature_path_interaction, feature_path_context))

    # start training
    pytorch_utils.train_model_custom_metric_mask(model, model._optimizer, model._loss_fn, model._metric_fn, [x_tr, x_tr_c], y_tr, y_tr_mask, [x_te, x_te_c], y_te, y_te_mask, n_epochs, batch_size_tr, batch_size_te, callbacks=callbacks)

    print('--- finish time')
    print(datetime.datetime.now())

class ClassifierContextLateEarlyFusionHumanObject(nn.Module):
    def __init__(self, n_classes, x_so_shape, x_cs_shape):
        super(ClassifierContextLateEarlyFusionHumanObject, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_fusion = 'context_fusion_%d'
        self.__init_layers(n_classes, x_so_shape, x_cs_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_so_shape, x_cs_shape):
        """
        Define model layers.
        """

        n_units = 512
        n_channels_in_action   = x_so_shape[0] 
        n_channels_in_context  =  np.sum([i[0] for i in x_cs_shape])

        print('n_channels_in_action: ', n_channels_in_action)
        print('n_channels_in_context: ', n_channels_in_context)

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

        # layers for context
        classifier_layers_context = []
        classifier_layers_context.append(nn.BatchNorm1d(n_channels_in_context))
        classifier_layers_context.append(nn.Dropout(0.25))
        classifier_layers_context.append(nn.Linear(n_channels_in_context, n_units))
        classifier_layers_context.append(nn.BatchNorm1d(n_units))
        classifier_layers_context.append(nn.LeakyReLU(0.2))
        classifier_layers_context.append(nn.Dropout(0.25))
        self.classifier_layers_context = nn.Sequential(*classifier_layers_context)

        classifier = []
        classifier.append(nn.Linear(n_units*2, n_classes))
        self.classifier = nn.Sequential(*classifier) 

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
        :param x_so: full image feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x_so = input[0]
        x_cs = input[1:]
        N = self.N

        x_cs = torch.cat(x_cs, dim=1)
        x_cs = x_cs.repeat(1, 1, N, 1, 1)

        # Process action features
        x = x_so
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
        x_feat_action = x

        # Process context features
        x = x_cs
        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)
        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x = x.contiguous().view(B * N, C)  # (B*N, C)

        # feed to classifier
        for l in self.classifier_layers_context:
            x = l(x)

        # recover the dimension of N
        _, C = pytorch_utils.get_shape(x)
        x = x.view(B, N, C)  # (B, N, C)
        x_feat_context = x

        # Concatenate features
        x = torch.cat((x_feat_action, x_feat_context), dim=2)

        # Feed-to-joint-classifier
        x = self.classifier(x)

        # apply max ops there
        x, _ = torch.max(x, dim=1)  # (B, C)
        x = torch.sigmoid(x)

        return x


class ClassifierContextLateEarlyFusionHumanObjectMultiContext(nn.Module):
    def __init__(self, n_classes, x_so_shape, x_cs_shape):
        super(ClassifierContextLateEarlyFusionHumanObjectMultiContext, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_fusion = 'context_fusion_%d'
        self.__init_layers(n_classes, x_so_shape, x_cs_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_so_shape, x_cs_shape):
        """
        Define model layers.
        """

        n_units = 512
        n_channels_in_action   = x_so_shape[0] 
        n_channels_in_context  =  np.sum([i[0] for i in x_cs_shape])

        print('n_channels_in_action: ', n_channels_in_action)
        print('n_channels_in_context: ', n_channels_in_context)

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

        # layers for context
        classifier_layers_context = []
        classifier_layers_context.append(nn.BatchNorm1d(n_channels_in_context))
        classifier_layers_context.append(nn.Dropout(0.25))
        classifier_layers_context.append(nn.Linear(n_channels_in_context, n_units))
        classifier_layers_context.append(nn.BatchNorm1d(n_units))
        classifier_layers_context.append(nn.LeakyReLU(0.2))
        classifier_layers_context.append(nn.Dropout(0.25))
        self.classifier_layers_context = nn.Sequential(*classifier_layers_context)

        classifier = []
        classifier.append(nn.Linear(n_units*2, n_classes))
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

        x_so = input[0]
        x_cs = input[1:]
        N = self.N

        x_cs = torch.cat(x_cs, dim=1)

        # Process action features
        x = x_so
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
        x_feat_action = x

        # Process context features
        x = x_cs
        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)
        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x = x.contiguous().view(B * N, C)  # (B*N, C)

        # feed to classifier
        for l in self.classifier_layers_context:
            x = l(x)

        # recover the dimension of N
        _, C = pytorch_utils.get_shape(x)
        x = x.view(B, N, C)  # (B, N, C)
        x_feat_context = x

        # Concatenate features
        x = torch.cat((x_feat_action, x_feat_context), dim=2)

        # Feed-to-joint-classifier
        x = self.classifier(x)

        # apply max ops there
        x, _ = torch.max(x, dim=1)  # (B, C)
        x = torch.sigmoid(x)

        return x

def train_classifier_using_features_multi_region_multi_context():

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    # Features of the image: f_interaction
    feature_path_interaction = Pth('Hico/features/features_base_subject_object.h5')
    n_channels, n_regions, channel_side_dim = 4096, 12, 1

    # Features of the pose: f_context
    feature_path_c1 = Pth('Hico/features/features_global_deformation.h5')
    feature_path_c2 = Pth('Hico/features/legacy/features_context_local_scene.h5')
    feature_path_c3 = Pth('Hico/features/legacy/features_lvis.h5')
    feature_path_c4 = Pth('Hico/features/features_local_part_states.h5')

    x_cs_shape = [(512, 1, 1, 1), (2048, 1, 1, 1),(1300, 1, 1, 1), (1032, 1, 1, 1)]
 
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
    x_tr = np.swapaxes(x_tr, 1,2)
    x_te = np.swapaxes(x_te, 1,2)

    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)
    y_tr_mask = y_tr_mask.astype(np.float32)
    y_te_mask = y_te_mask.astype(np.float32)

    print('... context features')
    (x_tr_c1, x_te_c1) = utils.h5_load_multi(feature_path_c1, ['x_tr', 'x_te'])
    (x_tr_c2, x_te_c2) = utils.h5_load_multi(feature_path_c2, ['x_tr', 'x_te'])
    (x_tr_c3, x_te_c3) = utils.h5_load_multi(feature_path_c3, ['x_tr', 'x_te'])
    (x_tr_c4, x_te_c4) = utils.h5_load_multi(feature_path_c4, ['x_tr', 'x_te'])

    x_tr_c1 = np.expand_dims(x_tr_c1, 2)
    x_tr_c1 = np.expand_dims(x_tr_c1, 3)
    x_tr_c1 = np.expand_dims(x_tr_c1, 4)

    x_te_c1 = np.expand_dims(x_te_c1, 2)
    x_te_c1 = np.expand_dims(x_te_c1, 3)
    x_te_c1 = np.expand_dims(x_te_c1, 4)

    x_tr_c4 = x_tr_c4.reshape(-1, 1032, 1,1,1)
    x_te_c4 = x_te_c4.reshape(-1, 1032, 1,1,1)

    print('train_set_shape_interaction: ', x_tr.shape)
    print('test_set_shape_interaction: ', x_te.shape)

    print('train_set_shape_context-1: ', x_tr_c1.shape)
    print('test_set_shape_context-1: ',  x_te_c1.shape)

    print('train_set_shape_context-2: ', x_tr_c2.shape)
    print('test_set_shape_context-2: ',  x_te_c2.shape)

    print('train_set_shape_context-3: ', x_tr_c3.shape)
    print('test_set_shape_context-3: ',  x_te_c3.shape)

    print('train_set_shape_context-4: ', x_tr_c4.shape)
    print('test_set_shape_context-4: ',  x_te_c4.shape)

    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    model = ClassifierContextLateEarlyFusionHumanObjectMulti(n_classes, input_shape, x_cs_shape) 
    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    input_sizes = [input_shape] + list(x_cs_shape)
    pytorch_utils.model_summary_multi_input(model, input_sizes=input_sizes, batch_size=-1, device='cuda')    
    print('... model built, duration (sec): %d' % (duration))

    # callbacks
    callbacks = []

    print('Interaction_feat: %s, Context_feat-1: %s, Context_feat-2: %s\n, Context_feat-3: %s\n' %(feature_path_interaction, feature_path_c1, feature_path_c2, feature_path_c3))

    # start training
    pytorch_utils.train_model_custom_metric_mask(model, model._optimizer, model._loss_fn, model._metric_fn, [x_tr, x_tr_c1, x_tr_c2, x_tr_c3, x_tr_c4], y_tr, y_tr_mask, [x_te, x_te_c1, x_te_c2, x_te_c3, x_te_c4], y_te, y_te_mask, n_epochs, batch_size_tr, batch_size_te, callbacks=callbacks)

    print('--- finish time')
    print(datetime.datetime.now())

class ClassifierContextLateEarlyFusionHumanObjectMulti(nn.Module):
    def __init__(self, n_classes, x_so_shape, x_cs_shape):
        super(ClassifierContextLateEarlyFusionHumanObjectMulti, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_fusion = 'context_fusion_%d'
        self.__init_layers(n_classes, x_so_shape, x_cs_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_so_shape, x_cs_shape):
        """
        Define model layers.
        """

        n_units = 512*2
        n_channels_in_action   = x_so_shape[0] 
        n_channels_in_context  =  np.sum([i[0] for i in x_cs_shape])

        print('n_channels_in_action: ', n_channels_in_action)
        print('n_channels_in_context: ', n_channels_in_context)

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

        # layers for context
        classifier_layers_context = []
        classifier_layers_context.append(nn.BatchNorm1d(n_channels_in_context))
        classifier_layers_context.append(nn.Dropout(0.25))
        classifier_layers_context.append(nn.Linear(n_channels_in_context, n_units))
        classifier_layers_context.append(nn.BatchNorm1d(n_units))
        classifier_layers_context.append(nn.LeakyReLU(0.2))
        classifier_layers_context.append(nn.Dropout(0.25))
        self.classifier_layers_context = nn.Sequential(*classifier_layers_context)

        classifier = []
        classifier.append(nn.Linear(n_units*2, n_units))
        classifier.append(nn.BatchNorm1d(n_units))
        classifier.append(nn.LeakyReLU(0.2))
        classifier.append(nn.Dropout(0.25))
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

        x_so = input[0]
        x_cs = input[1:]
        N = self.N

        x_cs = torch.cat(x_cs, dim=1)
        x_cs = x_cs.repeat(1, 1, N, 1, 1)

        # Process action features
        x = x_so
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
        x_feat_action = x

        # Process context features
        x = x_cs
        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)
        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x = x.contiguous().view(B * N, C)  # (B*N, C)

        # feed to classifier
        for l in self.classifier_layers_context:
            x = l(x)

        # recover the dimension of N
        _, C = pytorch_utils.get_shape(x)
        x = x.view(B, N, C)  # (B, N, C)
        x_feat_context = x

        # Concatenate features
        x = torch.cat((x_feat_action, x_feat_context), dim=2)

        B, N, C = pytorch_utils.get_shape(x)
        x = x.contiguous().view(B * N, C)  # (B*N, C)

        # Feed-to-joint-classifier
        for l in self.classifier:
            x = l(x)

        _, C = pytorch_utils.get_shape(x)
        x = x.view(B, N, C)  # (B, N, C)

        # apply max ops there
        x, _ = torch.max(x, dim=1)  # (B, C)
        x = torch.sigmoid(x)

        return x



def train_classifier_using_features_multi_region_single_context_late_fusion():

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    # Features of the image: f_interaction
    feature_path_interaction = Pth('Hico/features/features_subject_object.h5')
    n_channels, n_regions, channel_side_dim = 4096, 12, 1

    # Features of the pose: f_context
    feature_path_context = Pth('Hico/features//features_imsitu.h5')
    x_cs_shape = [(504, 1, 1, 1)]
 
    # Annotation of the image
    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())
    input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)

    print('--- start time')
    print(datetime.datetime.now())

    print('... loading data')
    t1 = time.time()

    print('... interaction features')
    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)

    print('... context features')
    (x_tr_c, x_te_c) = utils.h5_load_multi(feature_path_context, ['x_tr', 'x_te'])

    print('train_set_shape_interaction: ', x_tr.shape)
    print('test_set_shape_interaction: ', x_te.shape)

    print('train_set_shape_context: ', x_tr_c.shape)
    print('test_set_shape_context: ',  x_te_c.shape)

    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    model = ClassifierContextLateFusionHumanObject(n_classes, input_shape, x_cs_shape) 
    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    input_sizes = [input_shape] + list(x_cs_shape)
    pytorch_utils.model_summary_multi_input(model, input_sizes=input_sizes, batch_size=-1, device='cuda')    
    print('... model built, duration (sec): %d' % (duration))

    # callbacks
    callbacks = []

    print('Interaction_feat: %s, Context_feat: %s\n' %(feature_path_interaction, feature_path_context))

    # start training
    pytorch_utils.train_model_custom_metric(model, model._optimizer, model._loss_fn, model._metric_fn, [x_tr, x_tr_c], y_tr, [x_te, x_te_c], y_te, n_epochs, batch_size_tr, batch_size_te, callbacks=callbacks)

    print('--- finish time')
    print(datetime.datetime.now())

class ClassifierContextLateFusionHumanObject(nn.Module):
    def __init__(self, n_classes, x_so_shape, x_cs_shape):
        super(ClassifierContextLateFusionHumanObject, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_fusion = 'context_fusion_%d'
        self.__init_layers(n_classes, x_so_shape, x_cs_shape)
        self.__init_optimizer()

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, mean = 0, std = 1)
            m.bias.data.fill_(0.01)

    def __init_layers(self, n_classes, x_so_shape, x_cs_shape):
        """
        Define model layers.
        """

        n_units = 512
        n_channels_in_action   = x_so_shape[0] 
        n_channels_in_context  =  np.sum([i[0] for i in x_cs_shape])
        self.N = x_so_shape[1]

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
        classifier_layers_action.append(nn.Linear(n_units, n_classes))
        self.classifier_layers_action = nn.Sequential(*classifier_layers_action)

        # layers for context
        classifier_layers_context = []
        classifier_layers_context.append(nn.BatchNorm1d(n_channels_in_context))
        classifier_layers_context.append(nn.Dropout(0.25))
        classifier_layers_context.append(nn.Linear(n_channels_in_context, n_units))
        classifier_layers_context.append(nn.BatchNorm1d(n_units))
        classifier_layers_context.append(nn.LeakyReLU(0.2))
        classifier_layers_context.append(nn.Dropout(0.25))
        classifier_layers_context.append(nn.Linear(n_units, n_classes))
        self.classifier_layers_context = nn.Sequential(*classifier_layers_context)

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
        :param x_so: full image feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x_so = input[0]
        x_cs = input[1:]
        N = self.N

        x_cs = torch.cat(x_cs, dim=1)
        x_cs = x_cs.repeat(1, 1, N, 1, 1)

        # Process action features
        x = x_so
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

        # max over N dimension, then apply activation
        x, _ = torch.max(x, dim=1)  # (B, C)
        x_action = torch.sigmoid(x)

        # Process context features
        x = x_cs
        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)
        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x = x.contiguous().view(B * N, C)  # (B*N, C)

        # feed to classifier
        for l in self.classifier_layers_context:
            x = l(x)

        # recover the dimension of N
        _, C = pytorch_utils.get_shape(x)
        x = x.view(B, N, C)  # (B, N, C)

        # max over N dimension, then apply activation
        x, _ = torch.max(x, dim=1)  # (B, C)
        x_context = torch.sigmoid(x)

        # Combine predictions 
        x = x_action * x_context

        return x


def train_classifier_using_features_single_region_single_context_context_gating():

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    # Features of the image: f_interaction
    features_path_interaction = Pth('Hico/features/features_images.h5')
    n_channels, n_regions, channel_side_dim = 2048, 1, 1

    # Features of the image: f_scene
    '''
    feature_path_context = Pth('Hico/features/features_context_places.h5')
    x_cs_shape = [(512, 1, 1, 1)]
    '''

    feature_path_context = Pth('Hico/features/features_context_relashionship.h5')
    x_cs_shape = [(2048, 1, 1, 1)]

    '''
    feature_path_context = Pth('Hico/features/features_context_local_scene.h5')
    x_cs_shape = [(2048, 1, 1, 1)]
    '''

    # Annotation of the image
    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())
    input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)

    print('--- start time')
    print(datetime.datetime.now())

    print('... loading data')
    t1 = time.time()

    print('... interaction features')
    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    (x_tr, x_te) = utils.h5_load_multi(features_path_interaction, ['x_tr', 'x_te'])
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)

    print('... context features')
    (x_tr_c, x_te_c) = utils.h5_load_multi(feature_path_context, ['x_tr', 'x_te'])

    print('train_set_shape_interaction: ', x_tr.shape)
    print('test_set_shape_interaction: ', x_te.shape)

    print('train_set_shape_context: ', x_tr_c.shape)
    print('test_set_shape_context: ',  x_te_c.shape)

    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    model = ClassifierContextGating(n_classes, input_shape, x_cs_shape) 
    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    input_sizes = [input_shape] + list(x_cs_shape)
    pytorch_utils.model_summary_multi_input(model, input_sizes=input_sizes, batch_size=-1, device='cuda')    
    print('... model built, duration (sec): %d' % (duration))

    # callbacks
    callbacks = []
    callbacks.append(SelectionRatioCallback(model = model, n_items = x_te.shape[0], batch_size = batch_size_te))

    # start training
    pytorch_utils.train_model_custom_metric(model, model._optimizer, model._loss_fn, model._metric_fn, [x_tr, x_tr_c], y_tr, [x_te, x_te_c], y_te, n_epochs, batch_size_tr, batch_size_te, callbacks=callbacks)

    print('--- finish time')
    print(datetime.datetime.now())


class ClassifierContextGating(nn.Module):
    def __init__(self, n_classes, x_so_shape, x_c_shape):
        super(ClassifierContextGating, self).__init__()

        self.__init_layers(n_classes, x_so_shape, x_c_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_so_shape, x_c_shape):
        """
        Define model layers.
        """

        n_units = 600
        n_channels = 512

        C_a, N1, H1, W1 = x_so_shape
        C_c, N2, H2, W2 = x_c_shape[0]

        # layers for input embedding
        self.dense_a = nn.Sequential(pl.Linear3d(C_a, n_channels), nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))
        self.dense_c = nn.Sequential(pl.Linear3d(C_c, n_channels), nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))

        # layer for selection
        x_so_shape = (n_channels, N1, H1, W1)
        self.feature_selection = context_fusion.ContextGatingSigmoid(x_so_shape)

        # sparial pooling
        self.spatial_pooling = pl.Max(dim=(3, 4))

        # layers for classification
        classifier_layers = []
        classifier_layers.append(nn.Dropout(0.25))
        classifier_layers.append(nn.Linear(n_channels, n_units))
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

    def forward(self, x_so, x_c):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        # input embedding
        x_c = self.dense_c(x_c)
        x_a = self.dense_a(x_so)

        # feature selection and interaction
        x = self.feature_selection(x_a, x_c)  # (B, C, N, H, W)

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

def train_classifier_using_features_single_region_single_context_context_gating_concat():

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    '''
    # Features of the image: f_interaction
    features_path_interaction = Pth('Hico/features/legacy/features_images.h5')
    n_channels, n_regions, channel_side_dim = 2048, 1, 1
    '''

    # Features of the image: f_scene
    '''
    feature_path_context = Pth('Hico/features/features_context_places.h5')
    x_cs_shape = [(512, 1, 1, 1)]
    '''

    '''
    feature_path_context = Pth('Hico/features/features_context_relashionship.h5')
    x_cs_shape = [(2048, 1, 1, 1)]
    '''

    feature_path_interaction = Pth('Hico/features/features_base_subject_object.h5')
    n_channels, n_regions, channel_side_dim = 4096, 12, 1

    feature_path_context = Pth('Hico/features/features_global_deformation.h5')
    x_cs_shape = [(512, 1, 1, 1)]

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
    print('... context features')
    (x_tr_c, x_te_c) = utils.h5_load_multi(feature_path_context, ['x_tr', 'x_te'])

    x_tr_c = np.expand_dims(x_tr_c, 2)
    x_tr_c = np.expand_dims(x_tr_c, 3)
    x_tr_c = np.expand_dims(x_tr_c, 4)

    x_te_c = np.expand_dims(x_te_c, 2)
    x_te_c = np.expand_dims(x_te_c, 3)
    x_te_c = np.expand_dims(x_te_c, 4)

    print('train_set_shape_interaction: ', x_tr.shape)
    print('test_set_shape_interaction: ', x_te.shape)

    print('train_set_shape_context: ', x_tr_c.shape)
    print('test_set_shape_context: ',  x_te_c.shape)

    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    model = ClassifierContextGatingConcat(n_classes, input_shape, x_cs_shape) 
    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    input_sizes = [input_shape] + list(x_cs_shape)
    #pytorch_utils.model_summary_multi_input(model, input_sizes=input_sizes, batch_size=-1, device='cuda')    
    print('... model built, duration (sec): %d' % (duration))

    # callbacks
    callbacks = []
    callbacks.append(SelectionRatioCallback(model = model, n_items = x_te.shape[0], batch_size = batch_size_te))

    # start training
    pytorch_utils.train_model_custom_metric_mask(model, model._optimizer, model._loss_fn, model._metric_fn, [x_tr, x_tr_c], y_tr, y_tr_mask, [x_te, x_te_c], y_te, y_te_mask, n_epochs, batch_size_tr, batch_size_te, callbacks=callbacks)

    print('--- finish time')
    print(datetime.datetime.now())

def train_classifier_using_features_multiple_region_single_context_context_gating_concat():

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    feature_path_interaction = Pth('Hico/features/features_base_subject_object.h5')
    n_channels, n_regions, channel_side_dim = 4096, 12, 1

    feature_path_context = Pth('Hico/features/features_global_deformation.h5')
    x_cs_shape = [(512, 1, 1, 1)]

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

    x_tr = np.swapaxes(x_tr, 1,2)
    x_te = np.swapaxes(x_te, 1,2)

    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)
    y_tr_mask = y_tr_mask.astype(np.float32)
    y_te_mask = y_te_mask.astype(np.float32)
    print('... context features')
    (x_tr_c, x_te_c) = utils.h5_load_multi(feature_path_context, ['x_tr', 'x_te'])

    x_tr_c = np.expand_dims(x_tr_c, 2)
    x_tr_c = np.expand_dims(x_tr_c, 3)
    x_tr_c = np.expand_dims(x_tr_c, 4)

    x_te_c = np.expand_dims(x_te_c, 2)
    x_te_c = np.expand_dims(x_te_c, 3)
    x_te_c = np.expand_dims(x_te_c, 4)

    print('train_set_shape_interaction: ', x_tr.shape)
    print('test_set_shape_interaction: ', x_te.shape)

    print('train_set_shape_context: ', x_tr_c.shape)
    print('test_set_shape_context: ',  x_te_c.shape)

    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    model = ClassifierContextGatingConcat(n_classes, input_shape, x_cs_shape) 
    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    input_sizes = [input_shape] + list(x_cs_shape)
    #pytorch_utils.model_summary_multi_input(model, input_sizes=input_sizes, batch_size=-1, device='cuda')    
    print('... model built, duration (sec): %d' % (duration))

    # callbacks
    callbacks = []
    callbacks.append(SelectionRatioCallback(model = model, n_items = x_te.shape[0], batch_size = batch_size_te))

    # start training
    pytorch_utils.train_model_custom_metric_mask(model, model._optimizer, model._loss_fn, model._metric_fn, [x_tr, x_tr_c], y_tr, y_tr_mask, [x_te, x_te_c], y_te, y_te_mask, n_epochs, batch_size_tr, batch_size_te, callbacks=callbacks)

    print('--- finish time')
    print(datetime.datetime.now())



class ClassifierContextGatingConcat(nn.Module):
    def __init__(self, n_classes, x_so_shape, x_c_shape):
        super(ClassifierContextGatingConcat, self).__init__()

        self.__init_layers(n_classes, x_so_shape, x_c_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_so_shape, x_c_shape):
        """
        Define model layers.
        """

        n_units = 600
        n_channels = 512

        C_a, N1, H1, W1 = x_so_shape
        C_c, N2, H2, W2 = x_c_shape[0]

        # layers for input embedding
        self.dense_a = nn.Sequential(pl.Linear3d(C_a, n_channels), nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))
        self.dense_c = nn.Sequential(pl.Linear3d(C_c, n_channels), nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))

        # layer for selection
        x_so_shape = (n_channels, N1, H1, W1)
        self.feature_selection = context_fusion.ContextGatingSigmoidConcatConditionSumCombination(x_so_shape, x_c_shape)

        # sparial pooling
        self.spatial_pooling = pl.Max(dim=(3, 4))

        # layers for classification
        classifier_layers = []
        classifier_layers.append(nn.Dropout(0.25))
        classifier_layers.append(nn.Linear(n_channels, n_units))
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

    def forward(self, x_so, x_c):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        # input embedding
        x_c = self.dense_c(x_c)
        x_a = self.dense_a(x_so)

        # feature selection and interaction
        x = self.feature_selection(x_a, x_c)  # (B, C, N, H, W)

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


def train_classifier_using_features_gated_late_fusion():

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    # Features of the image: f_interaction
    feature_path_interaction = Pth('Hico/features/features_base_subject_object.h5')
    n_channels, n_regions, channel_side_dim = 4096, 12, 1

    # Features of the pose: f_context
    feature_path_context = Pth('Hico/features/features_global_deformation.h5')
    x_cs_shape = [(512, 1, 1, 1)]
 
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

    # Fix pose features
    x_tr = np.swapaxes(x_tr, 1,2)
    x_te = np.swapaxes(x_te, 1,2)

    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)

    y_tr_mask = y_tr_mask.astype(np.float32)
    y_te_mask = y_te_mask.astype(np.float32)
    print('... context features')
    (x_tr_c, x_te_c) = utils.h5_load_multi(feature_path_context, ['x_tr', 'x_te'])

    x_tr_c = np.expand_dims(x_tr_c, 2)
    x_tr_c = np.expand_dims(x_tr_c, 3)
    x_tr_c = np.expand_dims(x_tr_c, 4)

    x_te_c = np.expand_dims(x_te_c, 2)
    x_te_c = np.expand_dims(x_te_c, 3)
    x_te_c = np.expand_dims(x_te_c, 4)

    print('train_set_shape_interaction: ', x_tr.shape)
    print('test_set_shape_interaction: ', x_te.shape)

    print('train_set_shape_context: ', x_tr_c.shape)
    print('test_set_shape_context: ',  x_te_c.shape)

    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    model = ClassifierContextGatedLateFusion(n_classes, input_shape, x_cs_shape) 
    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    input_sizes = [input_shape] + list(x_cs_shape)
    pytorch_utils.model_summary_multi_input(model, input_sizes=input_sizes, batch_size=-1, device='cuda')    
    print('... model built, duration (sec): %d' % (duration))

    # callbacks
    # callbacks
    callbacks = []
    callbacks.append(SelectionRatioCallback(model = model, n_items = x_te.shape[0], batch_size = batch_size_te))

    # start training
    pytorch_utils.train_model_custom_metric_mask(model, model._optimizer, model._loss_fn, model._metric_fn, [x_tr, x_tr_c], y_tr, y_tr_mask, [x_te, x_te_c], y_te, y_te_mask, n_epochs, batch_size_tr, batch_size_te, callbacks=callbacks)

    print('--- finish time')
    print(datetime.datetime.now())


class ClassifierContextGatedLateFusion(nn.Module):
    def __init__(self, n_classes, x_so_shape, x_cs_shape):
        super(ClassifierContextGatedLateFusion, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_fusion = 'context_fusion_%d'
        self.__init_layers(n_classes, x_so_shape, x_cs_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_so_shape, x_cs_shape):
        """
        Define model layers.
        """

        n_units = 512
        n_channels_in_action   = x_so_shape[0] 
        n_channels_in_context  =  np.sum([i[0] for i in x_cs_shape])
        self.N = x_so_shape[1]

        # spatial pooling
        self.spatial_pooling = pl.Max(dim=(3, 4))

        self.feature_selection = context_fusion.ContextGatingSigmoidClassifierSoft(x_so_shape, x_cs_shape)

        # layers for action
        classifier_layers_action = []
        classifier_layers_action.append(nn.BatchNorm1d(n_channels_in_action))
        classifier_layers_action.append(nn.Dropout(0.25))
        classifier_layers_action.append(nn.Linear(n_channels_in_action, n_units))
        classifier_layers_action.append(nn.BatchNorm1d(n_units))
        classifier_layers_action.append(nn.LeakyReLU(0.2))
        classifier_layers_action.append(nn.Dropout(0.25))
        classifier_layers_action.append(nn.Linear(n_units, n_classes))
        self.classifier_layers_action = nn.Sequential(*classifier_layers_action)

        # layers for context
        classifier_layers_context = []
        classifier_layers_context.append(nn.BatchNorm1d(n_channels_in_context))
        classifier_layers_context.append(nn.Dropout(0.25))
        classifier_layers_context.append(nn.Linear(n_channels_in_context, n_units))
        classifier_layers_context.append(nn.BatchNorm1d(n_units))
        classifier_layers_context.append(nn.LeakyReLU(0.2))
        classifier_layers_context.append(nn.Dropout(0.25))
        classifier_layers_context.append(nn.Linear(n_units, n_classes))
        self.classifier_layers_context = nn.Sequential(*classifier_layers_context)


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
        :param x_so: full image feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x_so = input[0]
        x_cs = input[1:]
        N = self.N

        x_cs = torch.cat(x_cs, dim=1)
        x_cs = x_cs.repeat(1, 1, N, 1, 1)

        # predict the importance of context per region
        alpha = self.feature_selection(x_so, x_cs)

        # Process action features
        x = x_so
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

        # max over N dimension, then apply activation
        x_action = x
        #x_action = torch.sigmoid(x) # (B,N,C)

        # Process context features
        x = x_cs
        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)
        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x = x.contiguous().view(B * N, C)  # (B*N, C)

        # feed to classifier
        for l in self.classifier_layers_context:
            x = l(x)

        # recover the dimension of N
        _, C = pytorch_utils.get_shape(x)
        x = x.view(B, N, C)  # (B, N, C)

        # max over N dimension, then apply activation
        x_context = x
        #x_context = torch.sigmoid(x) # (B, N, C)

        # reweigh contribution of x_context with predicted alphas
        x_context = x_context * alpha

        # Combine predictions 
        x = torch.sigmoid(x_action + x_context)

        x, _ = torch.max(x, dim=1)  # (B, C)

        return x



def train_classifier_using_features_late_fusion_multi_context():

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    # Features of the image: f_interaction
    feature_path_interaction = Pth('Hico/features/features_base_subject_object.h5')
    n_channels, n_regions, channel_side_dim = 4096, 12, 1

    # Features of the pose: f_context
    feature_path_context_1 = Pth('Hico/features/features_global_deformation.h5')
    feature_path_context_2 = Pth('Hico/features/legacy/features_context_local_scene.h5')
    feature_path_context_3 = Pth('Hico/features/legacy/features_lvis.h5')

    x_cs_shape = [(512, 1, 1, 1), (2048, 1, 1, 1), (1300, 1, 1, 1)]
 
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

    # Fix pose features
    x_tr = np.swapaxes(x_tr, 1,2)
    x_te = np.swapaxes(x_te, 1,2)

    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)

    y_tr_mask = y_tr_mask.astype(np.float32)
    y_te_mask = y_te_mask.astype(np.float32)
    print('... context features')
    (x_tr_c, x_te_c) = utils.h5_load_multi(feature_path_context_1, ['x_tr', 'x_te'])

    x_tr_c = np.expand_dims(x_tr_c, 2)
    x_tr_c = np.expand_dims(x_tr_c, 3)
    x_tr_c = np.expand_dims(x_tr_c, 4)

    x_te_c = np.expand_dims(x_te_c, 2)
    x_te_c = np.expand_dims(x_te_c, 3)
    x_te_c = np.expand_dims(x_te_c, 4)

    x_tr_c1 = x_tr_c
    x_te_c1 = x_te_c

    (x_tr_c2, x_te_c2) = utils.h5_load_multi(feature_path_context_2, ['x_tr', 'x_te'])
    (x_tr_c3, x_te_c3) = utils.h5_load_multi(feature_path_context_3, ['x_tr', 'x_te'])


    print('train_set_shape_interaction: ', x_tr.shape)
    print('test_set_shape_interaction: ', x_te.shape)

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
    model = ClassifierContextLateFusionMulti(n_classes, input_shape, x_cs_shape) 
    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    input_sizes = [input_shape] + list(x_cs_shape)
    #pytorch_utils.model_summary_multi_input(model, input_sizes=input_sizes, batch_size=-1, device='cuda')    
    print('... model built, duration (sec): %d' % (duration))

    # callbacks
    # callbacks
    callbacks = []

    # start training
    pytorch_utils.train_model_custom_metric_mask(model, model._optimizer, model._loss_fn, model._metric_fn, [x_tr, x_tr_c1, x_tr_c2, x_tr_c3], y_tr, y_tr_mask, [x_te, x_te_c1, x_te_c2, x_te_c3], y_te, y_te_mask, n_epochs, batch_size_tr, batch_size_te, callbacks=callbacks)

    print('--- finish time')
    print(datetime.datetime.now())



class ClassifierContextLateFusionMulti(nn.Module):
    def __init__(self, n_classes, x_so_shape, x_cs_shape):
        super(ClassifierContextLateFusionMulti, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_dense_context = 'dense_context_%d'
        self.__init_layers(n_classes, x_so_shape, x_cs_shape)
        self.__init_optimizer()


    def __init_layers(self, n_classes, x_so_shape, x_cs_shape):
        """
        Define model layers.
        """

        self.n_classes = 600

        n_units = 600
        n_channels = 512

        C_so, N, H, W = x_so_shape
        self.C_so = C_so
        self.N = N

        # Map so features to a smaller size
        self.dense_so = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_so, n_channels), nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))


        # Loop over existing context features: Map them into interaction categories
        for idx_context in range(self.n_contexts):
            C_c = x_cs_shape[idx_context][0]

            # embedding of multi_ context
            layer_name = self.layer_name_dense_context % (idx_context + 1)
            layer = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_c, n_channels), nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2), nn.Dropout(0.25), pl.Linear3d(n_channels, n_units))

            setattr(self, layer_name, layer)

        # spatial pooling
        self.spatial_pooling = pl.Max(dim=(3, 4))

        # layers for classification
        classifier_layers = []
        classifier_layers.append(nn.Dropout(0.25))
        classifier_layers.append(nn.Linear(n_channels, n_units))
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
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x_so = input[0]
        x_so = self.dense_so(x_so)

        B, C, N, _,_ = pytorch_utils.get_shape(x_so)

        x_cs = input[1:]

        x_cs_classes = []

        # loop on multi_contexts
        for idx_context in range(self.n_contexts):
            # embedding of context
            x_c = x_cs[idx_context]
            layer = getattr(self, self.layer_name_dense_context % (idx_context + 1))
            x_c = layer(x_c)

            # append to list of context class predictions
            x_cs_classes.append(x_c.view(1, -1, self.n_classes)) # (n_context, B, C)

        # Process action features to get action category from so features
        x_cs_classes = torch.stack(x_cs_classes, dim=0).view(-1, B, self.n_classes)

        x = x_so
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

        x, _ = torch.max(x, dim=1)  # (B, C)

        # TODO: Finally add to this feature context prediction
        x_cs= torch.sum(x_cs_classes,0)
        x = torch.sigmoid(x + x_cs)

        return x
 


class SelectionRatioCallback():
    def __init__(self, model, n_items, batch_size):

        self.__is_local_machine = configs.is_local_machine()

        self.model = model
        self.n_items = n_items
        self.batch_size = batch_size
        self.f_mean = None
        self.f_std = None
        self.alpha_ratio = None

        # f_mean

    def on_batch_ends(self, batch_num, is_training):

        # only consider test
        if is_training:
            return

        n_items = self.n_items
        batch_size = self.batch_size
        n_batches = utils.calc_num_batches(n_items, batch_size)

        # get tensor value and append it to the list
        f_mean = pytorch_utils.model_get_tensor_value(self.model, ('feature_selection', 'f_mean'))  # (B,)
        f_std = pytorch_utils.model_get_tensor_value(self.model, ('feature_selection', 'f_std'))  # (B,)
        alpha_ratio = pytorch_utils.model_get_tensor_value(self.model, ('feature_selection', 'alpha_ratio'))  # (B,)

        # clear old list from previous epoch
        if batch_num == 1:
            self.f_mean = np.zeros((n_batches,), dtype=np.float32)
            self.f_std = np.zeros((n_batches,), dtype=np.float32)
            self.alpha_ratio = np.zeros((n_batches,), dtype=np.float32)

        idx_batch = batch_num - 1
        self.f_mean[idx_batch] = f_mean  # (B,)
        self.f_std[idx_batch] = f_std  # (B,)
        self.alpha_ratio[idx_batch] = alpha_ratio  # (B,)

    def on_epoch_ends(self, epoch_num):
        """
        plot histogram of node assignments.
        :param epoch_num:
        :return:
        """
        mean = np.mean(self.f_mean)
        std = np.mean(self.f_std)
        alpha_ratio = np.mean(self.alpha_ratio)


        sys.stdout.write('\r\r      | ratio %.02f | mean %.02f | std %.02f\n' % (alpha_ratio, mean, std))
