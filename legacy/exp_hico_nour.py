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

# Training a full image classifier 
def train_classifier_using_features_single_region():
    """
    Train model.
    """

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    features_path = Pth('Hico/features/features_scene.h5')
    n_channels, n_regions, channel_side_dim = 512, 1, 1

    features_path = Pth('Hico/features/features_images.h5')
    n_channels, n_regions, channel_side_dim = 2048, 1, 1

    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())
    input_shape = (n_channels, channel_side_dim, channel_side_dim)

    print('--- start time')
    print(datetime.datetime.now())

    print('... loading data')
    t1 = time.time()
    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    (x_tr, x_te) = utils.h5_load_multi(features_path, ['x_tr', 'x_te'])
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)
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
    pytorch_utils.train_model_custom_metric(model, model._optimizer, model._loss_fn, model._metric_fn, x_tr, y_tr, x_te, y_te, n_epochs, batch_size_tr, batch_size_te, callbacks=callbacks)

    print('--- finish time')
    print(datetime.datetime.now())

# Training a subject-object classifier either our features or PairAtt features
def train_classifier_using_features_multi_region():
    """
    Train model.
    """

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    features_path = Pth('Hico/features/features_pairattn.h5')
    n_regions, n_channels, channel_side_dim = 4096, 3, 1

    features_path = Pth('Hico/features/features_subject_object.h5')
    n_channels, n_regions, channel_side_dim = 4096, 2, 1
    n_channels, n_regions, channel_side_dim = 4096, 12, 1

    features_path = Pth('Hico/features/features_images.h5')
    n_channels, n_regions, channel_side_dim = 2048, 1, 1

    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())
    input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)

    print('--- start time')
    print(datetime.datetime.now())

    print('... loading data')
    t1 = time.time()
    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    (x_tr, x_te) = utils.h5_load_multi(features_path, ['x_tr', 'x_te'])
    x_tr = x_tr[:, :, :n_regions]
    x_te = x_te[:, :, :n_regions]
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)
    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))
    print(x_tr.shape)
    print(x_te.shape)
    print(y_tr.shape)
    print(y_te.shape)

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    model = ClassifierSimpleMultiRegion(n_classes, input_shape)
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

# endregion

# region Train Classifier: Context Fusion

def train_classifier_context_fusion_using_features():
    # model = ClassifierContextAttentionV1(n_classes, x1_shape, x2_shape)
    # model = ClassifierContextAttentionV2(n_classes, x1_shape, x2_shape)
    # model = ClassifierContextEarlyFusion(n_classes, x_so_shape, x_c_shape, is_concat=True)
    pass

def train_classifier_context_fusion_multi_using_features():
    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())
    root_path = c.ROOT_PATH_TYPES.desktop_nour_data if configs.is_local_machine() else c.ROOT_PATH_TYPES.das5_nour_local

    feats_c1_path = Pth('Hico/features/features_context_relashionship.h5', root_path=root_path)
    feats_c2_path = Pth('Hico/features/features_context_local_scene.h5', root_path=root_path)
    feats_c3_path = Pth('Hico/features/features_context_segmentation.h5', root_path=root_path)
    feats_c4_path = Pth('Hico/features/features_context_places.h5', root_path=root_path)
    feats_c5_path = Pth('Hico/features/features_images.h5', root_path=root_path)
    x_cs_shape = [(2048, 1, 1, 1), (2048, 1, 1, 1), (512, 1, 1, 1), (512, 1, 1, 1), (2048, 1, 1, 1)]

    feats_so_path = Pth('Hico/features/features_subject_object.h5', root_path=root_path)
    n_regions, n_channels, channel_side_dim = 12, 4096, 1

    x_so_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)
    n_contexts = len(x_cs_shape)

    print('--- start time')
    print(datetime.datetime.now())

    print('... loading data')
    t1 = time.time()
    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    (x_so_tr, x_so_te) = utils.h5_load_multi(feats_so_path, ['x_tr', 'x_te'])
    (x_c1_tr, x_c1_te) = utils.h5_load_multi(feats_c1_path, ['x_tr', 'x_te'])
    (x_c2_tr, x_c2_te) = utils.h5_load_multi(feats_c2_path, ['x_tr', 'x_te'])
    (x_c3_tr, x_c3_te) = utils.h5_load_multi(feats_c3_path, ['x_tr', 'x_te'])
    (x_c4_tr, x_c4_te) = utils.h5_load_multi(feats_c4_path, ['x_tr', 'x_te'])
    (x_c5_tr, x_c5_te) = utils.h5_load_multi(feats_c5_path, ['x_tr', 'x_te'])
    x_so_tr = x_so_tr[:, :, :n_regions]
    x_so_te = x_so_te[:, :, :n_regions]
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)
    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))
    print(x_so_tr.shape)
    print(x_so_te.shape)
    print(x_c1_tr.shape)
    print(x_c1_te.shape)
    print(x_c2_tr.shape)
    print(x_c2_te.shape)
    print(y_tr.shape)
    print(y_te.shape)

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    model = ClassifierContextFusionMulti(n_classes, x_so_shape, x_cs_shape, is_concat=True)
    model = model.cuda()
    t2 = time.time()
    duration = t2 - t1
    input_sizes = [x_so_shape] + list(x_cs_shape)
    pytorch_utils.model_summary_multi_input(model, input_sizes=input_sizes, batch_size=-1, device='cuda')
    print('... model built, duration (sec): %d' % (duration))

    # callbacks
    callbacks = []

    # start training
    pytorch_utils.train_model_custom_metric(model, model._optimizer, model._loss_fn, model._metric_fn, [x_so_tr, x_c1_tr, x_c2_tr, x_c3_tr, x_c4_tr, x_c5_tr], y_tr, [x_so_te, x_c1_te, x_c2_te, x_c3_te, x_c4_te, x_c5_te], y_te, n_epochs, batch_size_tr, batch_size_te, callbacks=callbacks)

    print('--- finish time')
    print(datetime.datetime.now())

# endregion

# region Train Classifier: Context Gating

def train_classifier_context_gating_using_features():
    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())

    feats_c_path = Pth('Hico/features/features_scene.h5')
    x_c_shape = (512, 1, 1, 1)

    feats_c_path = Pth('Hico/features/features_scene_early_fusion.h5')
    x_c_shape = (3072, 1, 1, 1)

    feats_so_path = Pth('Hico/features/features_pairattn.h5')
    n_regions, n_channels, channel_side_dim = 3, 4096, 1

    feats_so_path = Pth('Hico/features/features_subject_object.h5')
    n_regions, n_channels, channel_side_dim = 2, 4096, 1
    n_regions, n_channels, channel_side_dim = 12, 4096, 1

    x_so_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)

    print('--- start time')
    print(datetime.datetime.now())

    print('... loading data')
    t1 = time.time()
    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    (x_so_tr, x_so_te) = utils.h5_load_multi(feats_so_path, ['x_tr', 'x_te'])
    (x_c_tr, x_c_te) = utils.h5_load_multi(feats_c_path, ['x_tr', 'x_te'])
    x_so_tr = x_so_tr[:, :, :n_regions]
    x_so_te = x_so_te[:, :, :n_regions]
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)
    n_tr = len(y_tr)
    n_te = len(y_te)
    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))
    print(x_so_tr.shape)
    print(x_so_te.shape)
    print(x_c_tr.shape)
    print(x_c_te.shape)
    print(y_tr.shape)
    print(y_te.shape)

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    # model = ClassifierContextAttentionV1(n_classes, x1_shape, x2_shape)
    # model = ClassifierContextAttentionV2(n_classes, x1_shape, x2_shape)
    # model = ClassifierContextEarlyFusion(n_classes, x_so_shape, x_c_shape, is_concat=False)
    model = ClassifierChannelGating(n_classes, x_so_shape, x_c_shape)
    model = model.cuda()
    t2 = time.time()
    duration = t2 - t1
    pytorch_utils.model_summary_multi_input(model, input_sizes=[x_so_shape, x_c_shape], batch_size=-1, device='cuda')
    print('... model built, duration (sec): %d' % (duration))

    # callbacks
    callbacks = []
    # callbacks.append(ChannelGatingPerCategoryCallback(model, [x_so_te, x_c_te], y_te, batch_size_te * 5))

    # start training
    pytorch_utils.train_model_custom_metric(model, model._optimizer, model._loss_fn, model._metric_fn, [x_so_tr, x_c_tr], y_tr, [x_so_te, x_c_te], y_te, n_epochs, batch_size_tr, batch_size_te, callbacks=callbacks)

    print('--- finish time')
    print(datetime.datetime.now())

# endregion

# region Train Classifier: Channel Gating

def train_classifier_channel_gating_using_features():
    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())

    feats_c_path = Pth('Hico/features/features_context_local_scene.h5')
    x_c_shape = (2048, 1, 1, 1)

    feats_c_path = Pth('Hico/features/features_images.h5')
    x_c_shape = (2048, 1, 1, 1)

    feats_c_path = Pth('Hico/features/features_context_early_fusion.h5')
    x_c_shape = (3072, 1, 1, 1)

    feats_c_path = Pth('Hico/features/features_context_scene.h5')
    x_c_shape = (512, 1, 1, 1)

    feats_c_path = Pth('Hico/features/features_context_places.h5')
    x_c_shape = (512, 1, 1, 1)

    feats_c_path = Pth('Hico/features/features_context_segmentation.h5')
    x_c_shape = (512, 1, 1, 1)

    feats_c_path = Pth('Hico/features/features_context_relashionship.h5')
    x_c_shape = (2048, 1, 1, 1)

    feats_so_path = Pth('Hico/features/features_subject_object.h5')
    n_regions, n_channels, channel_side_dim = 2, 4096, 1
    n_regions, n_channels, channel_side_dim = 12, 4096, 1

    x_so_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)

    print('--- start time')
    print(datetime.datetime.now())

    print('... loading data')
    t1 = time.time()
    print('... loading annotations')
    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    print('... loading subject-object features')
    (x_so_tr, x_so_te) = utils.h5_load_multi(feats_so_path, ['x_tr', 'x_te'])
    print('... loading multi-context features')
    (x_c_tr, x_c_te) = utils.h5_load_multi(feats_c_path, ['x_tr', 'x_te'])

    x_so_tr = x_so_tr[:, :, :n_regions]
    x_so_te = x_so_te[:, :, :n_regions]
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)
    n_tr = len(y_tr)
    n_te = len(y_te)
    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))
    print(x_so_tr.shape)
    print(x_so_te.shape)
    print(x_c_tr.shape)
    print(x_c_te.shape)
    print(y_tr.shape)
    print(y_te.shape)

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    model = ClassifierChannelGating(n_classes, x_so_shape, x_c_shape)
    model = model.cuda()
    t2 = time.time()
    duration = t2 - t1
    pytorch_utils.model_summary_multi_input(model, input_sizes=[x_so_shape, x_c_shape], batch_size=-1, device='cuda')
    print('... model built, duration (sec): %d' % (duration))

    # callbacks
    callbacks = []
    # callbacks.append(ChannelGatingPerCategoryCallback(model, [x_so_te, x_c_te], y_te, batch_size_te * 5))

    # start training
    pytorch_utils.train_model_custom_metric(model, model._optimizer, model._loss_fn, model._metric_fn, [x_so_tr, x_c_tr], y_tr, [x_so_te, x_c_te], y_te, n_epochs, batch_size_tr, batch_size_te, callbacks=callbacks)

    print('--- finish time')
    print(datetime.datetime.now())

def train_classifier_channel_gating_multi_using_features():
    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())
    root_path = c.ROOT_PATH_TYPES.desktop_mert_data if configs.is_local_machine() else c.ROOT_PATH_TYPES.das5_mert_data

    print('root_path: ', root_path)

    feats_c1_path = Pth('Hico/features/features_context_relashionship.h5', root_path=root_path)
    feats_c2_path = Pth('Hico/features/features_context_local_scene.h5', root_path=root_path)
    feats_c3_path = Pth('Hico/features/features_context_segmentation.h5', root_path=root_path)
    feats_c4_path = Pth('Hico/features/features_context_places.h5', root_path=root_path)
    feats_c5_path = Pth('Hico/features/features_images.h5', root_path=root_path)
    x_cs_shape = [(2048, 1, 1, 1), (2048, 1, 1, 1), (512, 1, 1, 1), (512, 1, 1, 1), (2048, 1, 1, 1)]

    feats_so_path = Pth('Hico/features/features_subject_object.h5', root_path=root_path)
    n_regions, n_channels, channel_side_dim = 12, 4096, 1

    x_so_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)
    n_contexts = len(x_cs_shape)

    print('--- start time')
    print(datetime.datetime.now())

    print('... loading data')
    t1 = time.time()
    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    (x_so_tr, x_so_te) = utils.h5_load_multi(feats_so_path, ['x_tr', 'x_te'])
    (x_c1_tr, x_c1_te) = utils.h5_load_multi(feats_c1_path, ['x_tr', 'x_te'])
    (x_c2_tr, x_c2_te) = utils.h5_load_multi(feats_c2_path, ['x_tr', 'x_te'])
    (x_c3_tr, x_c3_te) = utils.h5_load_multi(feats_c3_path, ['x_tr', 'x_te'])
    (x_c4_tr, x_c4_te) = utils.h5_load_multi(feats_c4_path, ['x_tr', 'x_te'])
    (x_c5_tr, x_c5_te) = utils.h5_load_multi(feats_c5_path, ['x_tr', 'x_te'])
    x_so_tr = x_so_tr[:, :, :n_regions]
    x_so_te = x_so_te[:, :, :n_regions]
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)
    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))
    print(x_so_tr.shape)
    print(x_so_te.shape)
    print(y_tr.shape)
    print(y_te.shape)

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
     model = ClassifierContextFusionMulti(n_classes, x_so_shape, x_cs_shape, is_concat=True)
    # model = ClassifierContextGatingMulti(n_classes, x_so_shape, x_cs_shape,)
    # model = ClassifierChannelGatingMultiSeries(n_classes, x_so_shape, x_cs_shape)
    # model = ClassifierChannelGatingMultiParallel(n_classes, x_so_shape, x_cs_shape)
    model = ClassifierContextInteraction(n_classes, x_so_shape, x_cs_shape)
    model = model.cuda()
    t2 = time.time()
    duration = t2 - t1
    input_sizes = [x_so_shape] + list(x_cs_shape)
    pytorch_utils.model_summary_multi_input(model, input_sizes=input_sizes, batch_size=-1, device='cuda')
    print('... model built, duration (sec): %d' % (duration))

    # callbacks
    callbacks = []
    # callbacks.append(SelectionRatioMultiCallback(model, n_contexts))

    # start training
    pytorch_utils.train_model_custom_metric(model, model._optimizer, model._loss_fn, model._metric_fn, [x_so_tr, x_c1_tr, x_c2_tr, x_c3_tr, x_c4_tr, x_c5_tr], y_tr, [x_so_te, x_c1_te, x_c2_te, x_c3_te, x_c4_te, x_c5_te], y_te, n_epochs, batch_size_tr, batch_size_te, callbacks=callbacks)

    print('--- finish time')
    print(datetime.datetime.now())

# endregion

# region Train Classifier: SO Fusion

def train_classifier_so_fusion_using_features():
    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())

    feats_so_path = Pth('Hico/features/features_subject_object.h5')
    n_regions, n_channels, channel_side_dim = 2, 2048, 1
    n_regions, n_channels, channel_side_dim = 12, 2048, 1
    x_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)
    C = n_channels

    print('--- start time')
    print(datetime.datetime.now())

    print('... loading data')
    t1 = time.time()
    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    (x_so_tr, x_so_te) = utils.h5_load_multi(feats_so_path, ['x_tr', 'x_te'])
    x_so_tr = x_so_tr[:, :, :n_regions]
    x_so_te = x_so_te[:, :, :n_regions]
    x_s_tr = x_so_tr[:, :C]
    x_s_te = x_so_te[:, :C]
    x_o_tr = x_so_tr[:, C:]
    x_o_te = x_so_te[:, C:]
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)
    n_tr = len(y_tr)
    n_te = len(y_te)
    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))
    print(x_s_tr.shape)
    print(x_s_te.shape)
    print(x_o_tr.shape)
    print(x_o_te.shape)
    print(y_tr.shape)
    print(y_te.shape)

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    # model = ClassifierSOAttentionV2(n_classes, x_shape)
    model = ClassifierSOAttentionV3(n_classes, x_shape)
    model = model.cuda()
    t2 = time.time()
    duration = t2 - t1
    pytorch_utils.model_summary_multi_input(model, input_sizes=[x_shape, x_shape], batch_size=-1, device='cuda')
    print('... model built, duration (sec): %d' % (duration))

    # callbacks
    callbacks = []

    # start training
    pytorch_utils.train_model_custom_metric(model, model._optimizer, model._loss_fn, model._metric_fn, [x_s_tr, x_o_tr], y_tr, [x_s_te, x_o_te], y_te, n_epochs, batch_size_tr, batch_size_te, callbacks=callbacks)

    print('--- finish time')
    print(datetime.datetime.now())

# endregion

# region Train Classifier: SO Gating

def train_classifier_so_gating_using_features():
    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())

    feats_so_path = Pth('Hico/features/features_subject_object.h5')
    n_regions, n_channels, channel_side_dim = 12, 2048, 1
    C = n_channels

    x_so_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)
    x_s_shape = x_so_shape
    x_o_shape = x_so_shape

    print('--- start time')
    print(datetime.datetime.now())

    print('... loading data')
    t1 = time.time()
    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    (x_so_tr, x_so_te) = utils.h5_load_multi(feats_so_path, ['x_tr', 'x_te'])
    x_so_tr = x_so_tr[:, :, :n_regions]
    x_so_te = x_so_te[:, :, :n_regions]
    x_s_tr = x_so_tr[:, :C]
    x_s_te = x_so_te[:, :C]
    x_o_tr = x_so_tr[:, C:]
    x_o_te = x_so_te[:, C:]
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)
    n_tr = len(y_tr)
    n_te = len(y_te)
    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))
    print(x_s_tr.shape)
    print(x_s_te.shape)
    print(x_o_tr.shape)
    print(x_o_te.shape)
    print(y_tr.shape)
    print(y_te.shape)

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    # model = ClassifierSOGating(n_classes, x_c_shape, x_s_shape, x_o_shape)
    model = ClassifierSOChannelGating(n_classes, x_s_shape, x_o_shape)
    model = model.cuda()
    t2 = time.time()
    duration = t2 - t1
    pytorch_utils.model_summary_multi_input(model, input_sizes=[x_s_shape, x_o_shape], batch_size= -1, device='cuda')
    print('... model built, duration (sec): %d' % (duration))

    # callbacks
    callbacks = []

    # start training
    pytorch_utils.train_model_custom_metric(model, model._optimizer, model._loss_fn, model._metric_fn, [x_s_tr, x_o_tr], y_tr, [x_s_te, x_o_te], y_te, n_epochs, batch_size_tr, batch_size_te, callbacks=callbacks)

    print('--- finish time')
    print(datetime.datetime.now())

# endregion

# region Model: Context Fusion

class ClassifierSimpleSingleRegion(nn.Module):
    def __init__(self, n_classes, input_shape):
        super(ClassifierSimpleSingleRegion, self).__init__()

        self.__init_layers(n_classes, input_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, input_shape):
        """
        Define model layers.
        """

        n_channels, side_dim, side_dim = input_shape
        n_units = 1200

        self.spatial_pooling = pl.Max(dim=(2, 3))

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
        #x = self.spatial_pooling(x)

        # feed to classifier
        for l in self.classifier_layers:
            x = l(x)

        x = torch.sigmoid(x)

        return x

class ClassifierSimpleMultiRegion(nn.Module):
    def __init__(self, n_classes, input_shape):
        super(ClassifierSimpleMultiRegion, self).__init__()

        self.__init_layers(n_classes, input_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, input_shape):
        """
        Define model layers.
        """

        n_channels, n_regions, side_dim, side_dim = input_shape
        n_units = 600

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

        self._loss_fn = torch.nn.BCELoss()
        self._metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
        self._optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-4)

    def forward(self, x):
        # spatial pooling
        x = self.spatial_pooling(x)

        B, C, N = pytorch_utils.get_shape(x)  # (B, C, N)

        # hide N dimension
        x = x.permute(0, 2, 1)  # (B, N, C)
        x = x.contiguous().view(B * N, C)  # (B*N, C)

        # feed to classifier
        for l in self.classifier_layers:
            x = l(x)

        # recover the dimension of N
        _, C = pytorch_utils.get_shape(x)
        x = x.view(B, N, C)

        # max over N dimension, then apply activation
        x, _ = torch.max(x, dim=1)

        x = torch.sigmoid(x)

        return x

class ClassifierContextAttentionV1(nn.Module):
    def __init__(self, n_classes, x1_shape, x2_shape):
        super(ClassifierContextAttentionV1, self).__init__()

        self.loss_gating_coeff = 1.0

        self.__init_layers(n_classes, x1_shape, x2_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x1_shape, x2_shape):
        """
        Define model layers.
        """
        N, C1, H1, W1 = x1_shape
        C2, H2, W2 = x2_shape

        n_heads = 1
        reduction_factor = 8.0
        n_units = 600
        n_channels = 1024
        non_local_input_shape = (n_channels, N + 1, H1, W1)

        self.dense_x1 = nn.Sequential(pl.Linear3d(C1, n_channels, dim=1), nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))
        self.dense_x2 = nn.Sequential(pl.Linear2d(C2, n_channels, dim=1), nn.BatchNorm2d(n_channels), nn.LeakyReLU(0.2))

        self.spatial_pooling = pl.Max(dim=(3, 4))

        # attention layer followed by BN and ReLU
        attention_layers = []
        attention_layers.append(self_attention.GlobalSelfAttentionMultiHead(non_local_input_shape, n_heads, reduction_factor))
        attention_layers.append(nn.BatchNorm3d(n_channels))
        attention_layers.append(nn.LeakyReLU(0.2))
        self.attention_layers = nn.Sequential(*attention_layers)

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

        self._loss_fn = self.__loss_function
        self._metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
        self._optimizer = optim.Adam(self.parameters(), lr=0.0001, eps=1e-4)

    def __loss_function(self, y_pred, y_true):
        """
        loss of prediction
        """

        loss_entropy = F.binary_cross_entropy(y_pred, y_true)
        loss_total = loss_entropy
        # loss_total = loss_entropy + (self.node_assignment.gating_loss * self.loss_gating_coeff) if self.training else loss_entropy

        return loss_total

    def forward(self, x1, x2):
        # input is two features: subject-object feature and context feature

        x1 = x1.permute(0, 2, 1, 3, 4)  # (B, C, N, H, W)
        x1 = self.dense_x1(x1)

        x2 = self.dense_x2(x2)  # (B, C, H, W)
        x2 = torch.unsqueeze(x2, dim=2)  # (B, C, N, H, W)

        x = torch.cat((x1, x2), dim=2)  # (B, C, N, H, W)

        # non-local attention for feature interaction
        x = self.attention_layers(x)  # (B, C, N, H, W)

        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)
        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x = x.contiguous().view(B * N, C)  # (B*N, C)

        # feed to feature selection
        # x_c = self.feature_selection(x_c)

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

class ClassifierContextAttentionV2(nn.Module):
    def __init__(self, n_classes, x1_shape, x2_shape):
        super(ClassifierContextAttentionV2, self).__init__()

        self.loss_gating_coeff = 1.0

        self.__init_layers(n_classes, x1_shape, x2_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x1_shape, x2_shape):
        """
        Define model layers.
        """
        N, C1, H1, W1 = x1_shape
        C2, H2, W2 = x2_shape

        n_heads = 1
        reduction_factor = 8.0
        n_units = 600
        n_channels = 1024
        non_local_input_shape = (n_channels, 2, H1, W1)

        self.dense_x1 = nn.Sequential(pl.Linear3d(C1, n_channels, dim=1), nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))
        self.dense_x2 = nn.Sequential(pl.Linear2d(C2, n_channels, dim=1), nn.BatchNorm2d(n_channels), nn.LeakyReLU(0.2))

        self.spatial_pooling = pl.Max(dim=(3, 4))

        # attention layer followed by BN and ReLU
        attention_layers = []
        attention_layers.append(self_attention.GlobalSelfAttentionMultiHead(non_local_input_shape, n_heads, reduction_factor))
        attention_layers.append(nn.BatchNorm3d(n_channels))
        attention_layers.append(nn.LeakyReLU(0.2))
        self.attention_layers = nn.Sequential(*attention_layers)

        # layer for selection
        # self.feature_selection = feature_selection.FeatureSelectionSigmoid(input_shape)

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

        self._loss_fn = self.__loss_function
        self._metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
        self._optimizer = optim.Adam(self.parameters(), lr=0.0001, eps=1e-4)

    def __loss_function(self, y_pred, y_true):
        """
        loss of prediction
        """

        loss_entropy = F.binary_cross_entropy(y_pred, y_true)
        loss_total = loss_entropy
        # loss_total = loss_entropy + (self.node_assignment.gating_loss * self.loss_gating_coeff) if self.training else loss_entropy

        return loss_total

    def forward(self, x1, x2):
        """
        input is two features: subject-object feature and context feature
        :param x1: pairattn feature (B, C, N, H, W)
        :param x2: scene feature (B, C, H, W)
        :return:
        """

        x1 = x1.permute(0, 2, 1, 3, 4)  # (B, C, N, H, W)
        x1 = self.dense_x1(x1)

        x2 = self.dense_x2(x2)  # (B, C, H, W)
        x2 = torch.unsqueeze(x2, dim=2)  # (B, C, N, H, W)

        n_regions = pytorch_utils.get_shape(x1)[2]

        x_items = []
        for idx_region in range(n_regions):
            # get feature of one region
            x_item = x1[:, :, idx_region:idx_region + 1]
            x_item = torch.cat((x_item, x2), dim=2)  # (B, C, N, H, W)

            # non-local attention for feature interaction
            x_item = self.attention_layers(x_item)  # (B, C, N, H, W)
            x_items.append(x_item)

        # concat all region features
        x = torch.cat(x_items, dim=2)

        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)
        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x = x.contiguous().view(B * N, C)  # (B*N, C)

        # feed to feature selection
        # x_c = self.feature_selection(x_c)

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

class ClassifierContextFusion(nn.Module):
    def __init__(self, n_classes, x_so_shape, x_c_shape, is_concat):
        super(ClassifierContextFusion, self).__init__()

        self.__init_layers(n_classes, x_so_shape, x_c_shape, is_concat)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_so_shape, x_c_shape, is_concat):
        """
        Define model layers.
        """

        n_units = 600
        n_channels_in = 512
        n_channels_out = 2 * n_channels_in if is_concat else n_channels_in

        # layer for fusion
        self.feature_fusion = context_fusion.FeatureFusionConcat(x_so_shape, x_c_shape, n_channels_in) if is_concat else context_fusion.FeatureFusionResidual(x_so_shape, x_c_shape, n_channels_in)

        # sparial pooling
        self.spatial_pooling = pl.Max(dim=(3, 4))

        # layers for classification
        classifier_layers = []
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

    def forward(self, x_so, x_c):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, H, W)
        :return:
        """

        # feature selection and interaction
        x = self.feature_fusion(x_so, x_c)  # (B, C, N, H, W)

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

class ClassifierContextFusionMulti(nn.Module):
    def __init__(self, n_classes, x_so_shape, x_cs_shape, is_concat):
        super(ClassifierContextFusionMulti, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_fusion = 'context_fusion_%d'
        self.__init_layers(n_classes, x_so_shape, x_cs_shape, is_concat)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_so_shape, x_cs_shape, is_concat):
        """
        Define model layers.
        """

        n_units = 600
        n_channels = 512
        C, N, H, W = x_so_shape
        n_channels_out = n_channels * 2 if is_concat else n_channels

        # loop on different contexts
        for idx_context in range(self.n_contexts):
            x_c_shape = x_cs_shape[idx_context]

            # fusion layer
            layer_name = self.layer_name_fusion % (idx_context + 1)
            layer = context_fusion.FeatureFusionConcat(x_so_shape, x_c_shape, n_channels) if is_concat else context_fusion.FeatureFusionResidual(x_so_shape, x_c_shape, n_channels)
            setattr(self, layer_name, layer)

            # update shape of subject_object feature
            x_so_shape = (n_channels_out, N, H, W)

        # sparial pooling
        self.spatial_pooling = pl.Max(dim=(3, 4))

        # layers for classification
        classifier_layers = []
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
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x_so = input[0]
        x_cs = input[1:]

        # loop on multi_contexts
        for idx_context in range(self.n_contexts):
            # fusion layer
            x_c = x_cs[idx_context]
            layer = getattr(self, self.layer_name_fusion % (idx_context + 1))
            x_so = layer(x_so, x_c)

        # spatial pooling
        x = self.spatial_pooling(x_so)  # (B, C, N)
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

        n_units = 600
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
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
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

class ClassifierContextInteraction(nn.Module):
    def __init__(self, n_classes, x_so_shape, x_cs_shape):
        super(ClassifierContextInteraction, self).__init__()

        self.__init_layers(n_classes, x_so_shape, x_cs_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_so_shape, x_cs_shape):
        """
        Define model layers.
        """

        n_units = 600
        n_channels_inner = 128
        n_channels_out = 512

        C_so = x_so_shape[0]
        n_channels_dense = C_so + n_channels_out

        # interaction layer
        self.context_interaction = context_fusion.ContextInteraction(x_so_shape, x_cs_shape, n_channels_inner, n_channels_out)
        self.context_activation = nn.Sequential(nn.BatchNorm3d(n_channels_out), nn.LeakyReLU(0.2))

        # sparial pooling
        self.spatial_pooling = pl.Max(dim=(3, 4))

        # layers for classification
        classifier_layers = []
        classifier_layers.append(nn.BatchNorm1d(n_channels_dense))
        classifier_layers.append(nn.Dropout(0.25))
        classifier_layers.append(nn.Linear(n_channels_dense, n_units))
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

    def forward(self, *inputs):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x_so = inputs[0]

        # interaction
        x_cs = self.context_interaction(*inputs)
        x_cs = self.context_activation(x_cs)

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

# endregion

# region Model: Context Gating

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

        C_so, N1, H1, W1 = x_so_shape
        C_c, N2, H2, W2 = x_c_shape

        C_so_half = int(C_so / 2.0)
        C_s = C_so_half
        C_o = C_so_half
        self.C_so_half = C_so_half

        # layers for input embedding
        self.dense_s = nn.Sequential(pl.Linear3d(C_s, n_channels), nn.BatchNorm3d(n_channels))
        self.dense_o = nn.Sequential(pl.Linear3d(C_o, n_channels), nn.BatchNorm3d(n_channels))
        self.dense_c = nn.Sequential(pl.Linear3d(C_c, n_channels), nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))
        self.activation_so = nn.LeakyReLU(0.2)

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

        C_so_half = self.C_so_half

        x_s = x_so[:, :C_so_half]
        x_o = x_so[:, C_so_half:]

        # input embedding
        x_c = self.dense_c(x_c)
        x_s = self.dense_s(x_s)
        x_o = self.dense_o(x_o)

        # interaction between subject and object
        x_so = x_s + x_o
        x_so = self.activation_so(x_so)

        # feature selection and interaction
        x = self.feature_selection(x_so, x_c)  # (B, C, N, H, W)

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

class ClassifierContextGatingMulti(nn.Module):
    def __init__(self, n_classes, x_so_shape, x_cs_shape):
        super(ClassifierContextGatingMulti, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_dense_context = 'dense_context_%d'
        self.layer_name_feature_selection = 'feature_selection_%d'
        self.__init_layers(n_classes, x_so_shape, x_cs_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_so_shape, x_cs_shape):
        """
        Define model layers.
        """

        n_units = 600
        n_channels = 512

        C_so, N1, H1, W1 = x_so_shape
        C_so_half = int(C_so / 2.0)
        x_so_shape = (n_channels, N1, H1, W1)
        self.C_so_half = C_so_half

        # embedding of subject and object
        self.dense_s = nn.Sequential(pl.Linear3d(C_so_half, n_channels), nn.BatchNorm3d(n_channels))
        self.dense_o = nn.Sequential(pl.Linear3d(C_so_half, n_channels), nn.BatchNorm3d(n_channels))
        self.activation_so = nn.LeakyReLU(0.2)

        # loop on different contexts
        for idx_context in range(self.n_contexts):
            C_c, N2, H2, W2 = x_cs_shape[idx_context]

            # embedding of multi_ context
            layer_name = self.layer_name_dense_context % (idx_context + 1)
            layer = nn.Sequential(pl.Linear3d(C_c, n_channels), nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))
            setattr(self, layer_name, layer)

            # layer for selection
            layer_name = self.layer_name_feature_selection % (idx_context + 1)
            layer = context_fusion.ContextGatingSigmoid(x_so_shape)
            setattr(self, layer_name, layer)

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

    def forward(self, *input):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x_so = input[0]
        x_cs = input[1:]

        C_so_half = self.C_so_half

        x_s = x_so[:, :C_so_half]
        x_o = x_so[:, C_so_half:]

        # embedding of subject_object
        x_s = self.dense_s(x_s)
        x_o = self.dense_o(x_o)

        # interaction between subject and object
        x_so = x_s + x_o
        x = self.activation_so(x_so)

        # loop on multi_contexts
        for idx_context in range(self.n_contexts):
            # embedding of context
            x_c = x_cs[idx_context]
            dense_c = getattr(self, self.layer_name_dense_context % (idx_context + 1))
            x_c = dense_c(x_c)

            # feature selection and interaction
            feature_selection = getattr(self, self.layer_name_feature_selection % (idx_context + 1))
            x = feature_selection(x, x_c)  # (B, C, N, H, W)

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

# endregion

# region Model: Channel Gating

class ClassifierChannelGatingOld(nn.Module):
    def __init__(self, n_classes, x_i_shape, x_so_shape, x_c_shape):
        super(ClassifierChannelGating, self).__init__()

        self.__init_layers(n_classes, x_i_shape, x_so_shape, x_c_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_i_shape, x_so_shape, x_c_shape):
        """
        Define model layers.
        """

        n_units = 600
        n_channels = 512

        C_so, N1, H1, W1 = x_so_shape
        C_c, N2, H2, W2 = x_c_shape

        C_so_half = int(C_so / 2.0)
        C_s = C_so_half
        C_o = C_so_half
        self.C_so_half = C_so_half
        x_s_shape = (C_so_half, N1, H1, W1)
        x_o_shape = (C_so_half, N1, H1, W1)
        self_attention_shape = (n_channels, N1, H1, W1)

        # layers for input embedding
        self.dense_s = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_s, n_channels))
        self.dense_o = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_o, n_channels))
        self.dense_c = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_c, n_channels), nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))

        # self-attention
        self.self_attention = nn.Sequential(self_attention.GlobalSelfAttentionMultiHead(self_attention_shape, n_heads=2, reduction_factor=2))
        self.activation_so = nn.Sequential(nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))
        self.activation_x = nn.Sequential(nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))

        # layer for selection
        x_so_shape = (n_channels, N1, H1, W1)
        self.channel_gating = context_fusion.ChannelGatingSigmoid(x_so_shape)

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

        C_so_half = self.C_so_half

        x_s = x_so[:, :C_so_half]
        x_o = x_so[:, C_so_half:]

        # input embedding
        x_c = self.dense_c(x_c)
        x_s = self.dense_s(x_s)
        x_o = self.dense_o(x_o)

        # interaction between subject and object
        x_so = x_s + x_o
        x_so = self.activation_so(x_so)

        # feature selection and interaction
        x_c = self.channel_gating(x_so, x_c)  # (B, C, N, H, W)

        x_s = self.self_attention(x_s)
        x_o = self.self_attention(x_o)
        x = x_s + x_o + x_c
        x = self.activation_x(x)

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

    def forward_for_gating(self, x_so, x_c):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        C_so_half = self.C_so_half

        x_s = x_so[:, :C_so_half]
        x_o = x_so[:, C_so_half:]

        # input embedding
        x_c = self.dense_c(x_c)
        x_s = self.dense_s(x_s)
        x_o = self.dense_o(x_o)

        # interaction between subject and object
        x_so = x_s + x_o
        x_so = self.activation_so(x_so)

        # feature selection and interaction
        f, alpha = self.channel_gating.forward_for_gating(x_so, x_c)  # (B, C, N, H, W)

        return f, alpha

class ClassifierChannelGating(nn.Module):
    def __init__(self, n_classes, x_so_shape, x_c_shape):
        super(ClassifierChannelGating, self).__init__()

        self.__init_layers(n_classes, x_so_shape, x_c_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_so_shape, x_c_shape):
        """
        Define model layers.
        """

        n_units = 600
        n_channels = 512

        C_so, N_so, H_so, W_so = x_so_shape
        C_c, N_c, H_c, W_c = x_c_shape
        self.C_s = self.C_o = C_s = C_o = int(C_so / 2.0)

        # layers for input embedding
        self.dense_s = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_s, n_channels))
        self.dense_o = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_o, n_channels))
        self.dense_c = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_c, n_channels), pl.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))
        self.activation_so = nn.Sequential(pl.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))

        # gating layer
        self.channel_gating = context_fusion.ChannelGatingSigmoid(n_channels)

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

    def forward(self, x_so, x_c, is_gating_only):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x_s = x_so[:, :self.C_s]
        x_o = x_so[:, self.C_s:]

        # input embedding
        x_s = self.dense_s(x_s)
        x_o = self.dense_o(x_o)
        x_c = self.dense_c(x_c)

        # interaction between subject and object
        x_so = x_s + x_o
        x_so = self.activation_so(x_so)

        if is_gating_only:
            f, alpha = self.channel_gating(x_so, x_c)  # (B, C, N, H, W)
            return f, alpha
        else:
            # feature selection
            x_c = self.channel_gating(x_so, x_c)  # (B, C, N, H, W)

        # input fusion
        x = x_so + x_c

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

    def forward_for_gating(self, x_so, x_c):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        f, alpha = self.forward(x_so, x_c, is_gating_only=True)

        return f, alpha

class ClassifierChannelGatingMultiSeries(nn.Module):
    def __init__(self, n_classes, x_so_shape, x_cs_shape):
        super(ClassifierChannelGatingMultiSeries, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_dense_context = 'dense_context_%d'
        self.layer_name_channel_gating = 'channel_gating_%d'
        self.layer_name_activation_x = 'activation_x_%d'
        self.__init_layers(n_classes, x_so_shape, x_cs_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_so_shape, x_cs_shape):
        """
        Define model layers.
        """

        n_units = 600
        n_channels = 512

        C_so, N1, H1, W1 = x_so_shape
        x_so_shape = (n_channels, N1, H1, W1)
        self.C_s = self.C_o = C_s = C_o = int(C_so / 2.0)

        # embedding of subject and object
        self.dense_s = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_s, n_channels))
        self.dense_o = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_o, n_channels))
        self.activation_so = nn.Sequential(nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))

        # loop on different contexts
        for idx_context in range(self.n_contexts):
            C_c, N2, H2, W2 = x_cs_shape[idx_context]

            # embedding of multi_ context
            layer_name = self.layer_name_dense_context % (idx_context + 1)
            layer = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_c, n_channels), nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))
            setattr(self, layer_name, layer)

            # layer for selection
            layer_name = self.layer_name_channel_gating % (idx_context + 1)
            layer = context_fusion.ChannelGatingSigmoid(n_channels)
            setattr(self, layer_name, layer)

            # layer for selection
            layer_name = self.layer_name_activation_x % (idx_context + 1)
            layer = self.activation_x = nn.Sequential(nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))
            setattr(self, layer_name, layer)

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

    def forward(self, *input):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x_so = input[0]
        x_cs = input[1:]

        x_s = x_so[:, :self.C_s]
        x_o = x_so[:, self.C_s:]

        # embedding of subject_object
        x_s = self.dense_s(x_s)
        x_o = self.dense_o(x_o)

        # interaction between subject and object
        x_so = x_s + x_o
        x_so = self.activation_so(x_so)

        # loop on multi_contexts
        for idx_context in range(self.n_contexts):
            # embedding of context
            x_c = x_cs[idx_context]
            layer = getattr(self, self.layer_name_dense_context % (idx_context + 1))
            x_c = layer(x_c)

            # feature selection and interaction
            layer = getattr(self, self.layer_name_channel_gating % (idx_context + 1))
            x_c = layer(x, x_c)  # (B, C, N, H, W)

            # input fusion
            x = x_so + x_c

            layer = getattr(self, self.layer_name_activation_x % (idx_context + 1))
            x = layer(x)

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

class ClassifierChannelGatingMultiParallel(nn.Module):
    def __init__(self, n_classes, x_so_shape, x_cs_shape):
        super(ClassifierChannelGatingMultiParallel, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_dense_context = 'dense_context_%d'
        self.layer_name_channel_gating = 'channel_gating_%d'
        self.__init_layers(n_classes, x_so_shape, x_cs_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_so_shape, x_cs_shape):
        """
        Define model layers.
        """

        n_units = 600
        n_channels = 512

        C_so, N, H, W = x_so_shape
        self.C_s = self.C_o = C_s = C_o = int(C_so / 2.0)
        self.N = N

        # embedding of subject and object
        self.dense_s = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_s, n_channels))
        self.dense_o = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_o, n_channels))
        self.activation_so = nn.Sequential(nn.LeakyReLU(0.2), nn.BatchNorm3d(n_channels))
        # self.activation_x = nn.Sequential(nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))
        self.activation_x = nn.Sequential()

        # loop on different contexts
        for idx_context in range(self.n_contexts):
            C_c = x_cs_shape[idx_context][0]

            # embedding of multi_ context
            layer_name = self.layer_name_dense_context % (idx_context + 1)
            layer = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_c, n_channels), nn.LeakyReLU(0.2), nn.BatchNorm3d(n_channels))
            setattr(self, layer_name, layer)

            # layer for selection
            layer_name = self.layer_name_channel_gating % (idx_context + 1)
            layer = context_fusion.ChannelGatingSigmoid(n_channels)
            setattr(self, layer_name, layer)

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

    def forward(self, *input):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x_so = input[0]
        x_cs = input[1:]

        x_s = x_so[:, :self.C_s]
        x_o = x_so[:, self.C_s:]

        # embedding of subject_object
        x_s = self.dense_s(x_s)
        x_o = self.dense_o(x_o)

        # interaction between subject and object
        x_so = x_s + x_o
        x = self.activation_so(x_so)

        x_cs_gated = []

        # loop on multi_contexts
        for idx_context in range(self.n_contexts):
            # embedding of context
            x_c = x_cs[idx_context]
            layer = getattr(self, self.layer_name_dense_context % (idx_context + 1))
            x_c = layer(x_c)

            # feature selection and interaction
            layer = getattr(self, self.layer_name_channel_gating % (idx_context + 1))
            x_c = layer(x, x_c)  # (B, C, N, H, W)

            # append to list of gated context features
            x_cs_gated.append(x_c)

        # input fusion
        x_cs_gated.append(x_so)
        x = torch.stack(x_cs_gated, dim=2)
        x = torch.sum(x, dim=2)
        # x = self.activation_x(x)

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

# endregion

# region Model: SO Attention

class ClassifierSOAttentionV2(nn.Module):
    def __init__(self, n_classes, x_shape):
        super(ClassifierSOAttentionV2, self).__init__()

        self.__init_layers(n_classes, x_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_shape):
        """
        Define model layers.
        """

        n_units = 600

        C, N1, H1, W1 = x_shape
        n_channels = C
        n_channels_double = int(C * 2)

        # self-attention
        self.self_attention = nn.Sequential(self_attention.GlobalSelfAttentionMultiHead(x_shape, n_heads=4, reduction_factor=8), nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))

        # sparial pooling
        self.spatial_pooling = pl.Max(dim=(3, 4))

        # layers for classification
        classifier_layers = []
        classifier_layers.append(nn.Dropout(0.25))
        classifier_layers.append(nn.Linear(n_channels_double, n_units))
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

    def forward(self, x_s, x_o):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        # attention for object and subject
        x_s = self.self_attention(x_s)
        x_o = self.self_attention(x_o)

        # concat subject and object
        x = torch.cat((x_s, x_o), dim=1)

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

        # alpha = alpha.unsqueeze(dim=2)  # (B, N, 1)
        # x = alpha * x

        # max over N dimension, then apply activation
        x, _ = torch.max(x, dim=1)  # (B, C)
        x = torch.sigmoid(x)

        return x

class ClassifierSOAttentionV3(nn.Module):
    def __init__(self, n_classes, x_shape):
        super(ClassifierSOAttentionV3, self).__init__()

        self.__init_layers(n_classes, x_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_shape):
        """
        Define model layers.
        """

        n_units = 600

        C, N, H, W = x_shape
        n_channels = C
        N = N * 2

        # self-attention
        self.self_attention = nn.Sequential(self_attention.GlobalSelfAttentionMultiHead(x_shape, n_heads=4, reduction_factor=8), nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))

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

    def forward(self, x_s, x_o):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        # stack subject and object features
        x = torch.cat((x_s, x_o), dim=2)  # (B, C, N, H, W)
        del x_s
        del x_o

        # attention for object_subject features
        x = self.self_attention(x)
        B, C, N, H, W = pytorch_utils.get_shape(x)
        N = int(N / 2.0)

        # split into subject and object, then cat subject and object features
        x = x.view(B, C, 2, N, H, W)
        x = torch.sum(x, dim=2)
        # x = x.view(B, C * 2, N, H, W)

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

        # alpha = alpha.unsqueeze(dim=2)  # (B, N, 1)
        # x = alpha * x

        # max over N dimension, then apply activation
        x, _ = torch.max(x, dim=1)  # (B, C)
        x = torch.sigmoid(x)

        return x

# endregion

# region Model: SO Gating

class ClassifierSOGating(nn.Module):
    def __init__(self, n_classes, x_c_shape, x_s_shape, x_o_shape):
        super(ClassifierSOGating, self).__init__()

        self.__init_layers(n_classes, x_c_shape, x_s_shape, x_o_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_c_shape, x_s_shape, x_o_shape):
        """
        Define model layers.
        """

        n_units = 600
        n_channels = x_s_shape[0] + x_o_shape[0]

        # self-attention
        # self.self_attention = nn.Sequential(self_attention.GlobalSelfAttentionMultiHead(x_shape, n_heads=1, reduction_factor=16), nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))

        # layer for selection
        self.so_gating = context_fusion.SOGatingSigmoid(x_c_shape, x_s_shape, x_o_shape)

        # sparial pooling
        self.spatial_pooling = pl.Max(dim=(3, 4))

        # layers for classification
        classifier_layers = []
        classifier_layers.append(nn.BatchNorm1d(n_channels))
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

    def forward(self, x_c, x_s, x_o):
        """
        input is two features: subject-object feature and context feature
        :param x_s: subject feature (B, C, N, H, W)
        :param x_o: object feature (B, C, N, H, W)
        :return:
        """

        # attention for object and subject
        # x_s = self.self_attention(x_s)
        # x_o = self.self_attention(x_o)

        # feature selection and interaction
        x = self.so_gating(x_c, x_s, x_o)  # (B, C, N, H, W)

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

class ClassifierSOChannelGating(nn.Module):
    def __init__(self, n_classes, x_s_shape, x_o_shape):
        super(ClassifierSOChannelGating, self).__init__()

        self.__init_layers(n_classes, x_s_shape, x_o_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_s_shape, x_o_shape):
        """
        Define model layers.
        """

        n_units = 600
        n_channels = x_s_shape[0] + x_o_shape[0]
        self_attention_shape = x_s_shape
        C = x_s_shape[0]

        # self-attention
        self.self_attention = nn.Sequential(self_attention.GlobalSelfAttentionMultiHead(self_attention_shape, n_heads=2, reduction_factor=16), nn.BatchNorm3d(C), nn.LeakyReLU(0.2))

        # layer for selection
        self.so_gating = context_fusion.SOChannelGatingSigmoid(x_s_shape, x_o_shape)

        # sparial pooling
        self.spatial_pooling = pl.Max(dim=(3, 4))

        # layers for classification
        classifier_layers = []
        classifier_layers.append(nn.BatchNorm1d(n_channels))
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

    def forward(self, x_s, x_o):
        """
        input is two features: subject-object feature and context feature
        :param x_s: subject feature (B, C, N, H, W)
        :param x_o: object feature (B, C, N, H, W)
        :return:
        """

        # attention for object and subject
        x_s = self.self_attention(x_s)
        x_o = self.self_attention(x_o)

        # feature selection and interaction
        x = self.so_gating(x_s, x_o)  # (B, C, N, H, W)

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

# endregion

# region Callbacks

class NodeSelectionCallback():
    def __init__(self, model, x_tr, y_tr, x_te, y_te):
        self.__is_local_machine = configs.is_local_machine()

        if not self.__is_local_machine:
            return

        plt.ion()
        self.colors = plot_utils.tableau_category20()
        fig, axes = plt.subplots(nrows=3, ncols=1, num='Node Selection', figsize=(7, 6))
        self.axes = axes
        self.model = model
        self.is_training = False
        self.x_tr = x_tr
        self.y_tr = y_tr
        self.x_te = x_te
        self.y_te = y_te

        self.idx_1 = np.where(y_te == 1)[0][0]
        self.idx_2 = np.where(y_te == 2)[0][0]

        self.node_assignment_values = None

    def on_batch_ends(self, batch_num, is_training):

        # only consider either training batch or test batch
        if self.is_training and not is_training:
            return

        if not self.is_training and is_training:
            return

        # clear old list from previous epoch
        if batch_num == 1:
            self.assignment_values = None

        # get tensor value
        assignment_values = pytorch_utils.model_get_tensor_value(self.model, ('node_assignment', 'assignment_values'))  # (None, T, N)

        # append it to the list
        self.assignment_values = assignment_values if self.assignment_values is None else np.vstack((self.assignment_values, assignment_values))  # (None, N)

    def on_epoch_ends(self, epoch_num):
        """
        plot histogram of node assignments.
        :param epoch_num:
        :return:
        """

        assignment_values = self.assignment_values  # (None, N)

        # pick up two examples
        values_before_1 = assignment_values[self.idx_1]  # (N, )
        values_before_2 = assignment_values[self.idx_2]  # (N, )
        values_before_3 = np.mean(assignment_values, axis=0)  # (N, )

        n_nodes = len(values_before_1)
        x_range = np.arange(1, n_nodes + 1)
        colors = self.colors
        axes = self.axes
        ax1, ax2, ax3 = axes

        for idx_axis, ax in enumerate(axes):
            ax.cla()
            ax.set_ylim([0, 1.0])
            ax.set_xlabel('Node Assignment')
            ax.set_ylabel('activation')
            ax.grid()

        ax1.bar(x_range, values_before_1, color=colors[0], label='After')
        ax1.legend(loc='upper right')
        ax2.bar(x_range, values_before_2, color=colors[2], label='After')
        ax2.legend(loc='upper right')
        ax3.bar(x_range, values_before_3, color=colors[4], label='After')
        ax3.legend(loc='upper right')

        plt.tight_layout()
        plt.pause(0.01)

class EvalulationCallback():
    def __init__(self, model, x_tr, y_tr, x_te, y_te):

        self.x_tr = x_tr
        self.y_tr = y_tr
        self.x_te = x_te
        self.y_te = y_te

        self.model = model

    def on_batch_ends(self, batch_num, is_training):
        pass

    def on_epoch_ends(self, epoch_num):
        """
        plot histogram of node assignments.
        :param epoch_num:
        :return:
        """

        if not (epoch_num == 1 or epoch_num == 20):
            return

        n_tr = len(self.y_tr)
        n_te = len(self.y_te)
        model = self.model

        numBatch = n_te
        allaps = np.zeros((numBatch,), dtype=np.float32)

        with torch.no_grad():
            for it in range(numBatch):
                feat, label = self.x_te[it:it + 1], self.y_te[it:it + 1]
                response = model(torch.from_numpy(feat).to('cuda'))
                ap = self.get_ap(label, response.cpu().detach().numpy())
                allaps[it] = ap

        map = allaps.mean() * 100
        print('mean average precision: %.02f' % (map))

    def get_ap(self, label, score):
        ap = sk_metrics.average_precision_score(label.reshape(-1), score.reshape(-1))
        return ap

class SelectionHistogramCallback():
    def __init__(self, model, x_tr, y_tr, x_te, y_te):
        self.__is_local_machine = configs.is_local_machine()

        if not self.__is_local_machine:
            return

        plt.ion()
        self.colors = plot_utils.tableau_category20()
        fig, axes = plt.subplots(nrows=3, ncols=2, num='Selection Histogram', figsize=(7, 6))
        self.axes = axes
        self.model = model
        self.is_training = False
        self.x_tr = x_tr
        self.y_tr = y_tr
        self.x_te = x_te
        self.y_te = y_te

        self.idx_1 = np.where(y_te == 1)[0][0]
        self.idx_2 = np.where(y_te == 2)[0][0]

        self.idxes_1 = np.where(y_te == 1)[0]
        self.idxes_2 = np.where(y_te == 2)[0]

        self.node_assignment_values = None

    def on_batch_ends(self, batch_num, is_training):

        # only consider either training batch or test batch
        if self.is_training and not is_training:
            return

        if not self.is_training and is_training:
            return

        # clear old list from previous epoch
        if batch_num == 1:
            self.values_before = None
            self.values_after = None
            self.assignment_values = None

        # get tensor value
        values_before = pytorch_utils.model_get_tensor_value(self.model, ('node_assignment', 'values_before'))  # (None, T, N)
        values_after = pytorch_utils.model_get_tensor_value(self.model, ('node_assignment', 'values_after'))  # (None, T, N)
        assignment_values = pytorch_utils.model_get_tensor_value(self.model, ('node_assignment', 'assignment_values'))  # (None, T, N)

        # append it to the list
        self.values_before = values_before if self.values_before is None else np.vstack((self.values_before, values_before))  # (None, N, T)
        self.values_after = values_after if self.values_after is None else np.vstack((self.values_after, values_after))  # (None, N, T)
        self.assignment_values = assignment_values if self.assignment_values is None else np.vstack((self.assignment_values, assignment_values))  # (None, N, T)

    def on_epoch_ends(self, epoch_num):
        """
        plot histogram of node assignments.
        :param epoch_num:
        :return:
        """

        values_before = self.values_before  # (None, N, T)
        values_after = self.values_after  # (None, N, T)
        assignment_values = self.assignment_values  # (None, N)

        values_before = np.max(values_before, axis=2)  # (None, N)
        values_after = np.max(values_after, axis=2)  # (None, N)

        # pick up two examples
        values_before_1 = values_before[self.idx_1]  # (N,)
        values_after_1 = values_after[self.idx_1]  # (N,)
        values_before_2 = values_before[self.idx_2]  # (N,)
        values_after_2 = values_after[self.idx_2]  # (N,)

        values_before_3 = np.mean(values_before, axis=0)  # (N, )
        values_after_3 = np.mean(values_after, axis=0)  # (N, )

        #############################

        n_nodes = len(values_before_1)
        colors = self.colors
        axes = self.axes
        (ax11, ax12), (ax21, ax22), (ax31, ax32) = axes
        axes = [ax11, ax12, ax21, ax22, ax31, ax32]

        for idx_axis, ax in enumerate(axes):
            ax.cla()
            ax.set_xlabel('Histogram')
            ax.set_ylabel('activation')
            ax.set_xlim(0, 1)
            ax.grid()

        ax11.hist(values_before_1, color=colors[1], label='Before')
        ax12.hist(values_after_1, color=colors[0], label='After')
        ax21.hist(values_before_2, color=colors[3], label='Before')
        ax22.hist(values_after_2, color=colors[2], label='After')
        ax31.hist(values_before_3, color=colors[5], label='Before')
        ax32.hist(values_after_3, color=colors[4], label='After')

        for idx_axis, ax in enumerate(axes):
            ax.legend(loc='upper right')

        # plt.tight_layout()
        plt.pause(0.01)

class SelectionRatioPlotCallback():
    def __init__(self, model, y_te, n_timesteps, n_classes):

        self.__is_local_machine = configs.is_local_machine()

        if not self.__is_local_machine:
            return

        plt.ion()
        self.colors = plot_utils.tableau_category20()
        fig_width = int(n_timesteps * 20 / 64.0)
        fig, ax = plt.subplots(nrows=1, ncols=1, num='Gating', figsize=(4, 4))
        self.ax = ax

        self.n_timesteps = n_timesteps
        self.n_classes = n_classes
        self.model = model
        self.y = y_te
        self.alpha_values = None

    def on_batch_ends(self, batch_num, is_training):

        # only consider test
        if is_training:
            return

        # clear old list from previous epoch
        if batch_num == 1:
            self.alpha_values = None

        # get tensor value and append it to the list
        alpha_values = pytorch_utils.model_get_tensor_value(self.model, ('temporal_selection', 'attention_values_after'))  # (None, T)
        self.alpha_values = alpha_values if self.alpha_values is None else np.vstack((self.alpha_values, alpha_values))  # (None, T)

    def on_epoch_ends(self, epoch_num):
        """
        plot histogram of node assignments.
        :param epoch_num:
        :return:
        """

        alpha_values = self.alpha_values  # (None, T)
        n_classes = self.n_classes
        n_timesteps = self.n_timesteps
        colors = self.colors
        y = self.y
        ax = self.ax
        ratios = []

        for idx_class in range(n_classes):
            # get attn values for the current class
            vals = alpha_values[np.where(y == idx_class)]  # (None, T)
            # ratio of closed gates, i.e. chosen frames
            ratio = int(100 * np.count_nonzero(vals) / np.prod(vals.shape))
            ratios.append(ratio)

        avg_ratio = int(np.mean(ratios))
        class_nums = [str(i) for i in np.arange(1, n_classes + 1)]

        ax.cla()
        ax.set_xlim([0, 100])
        ax.set_yticks(np.arange(n_classes), class_nums)
        ax.set_xlabel('Selected Frames (%d)' % (avg_ratio))
        ax.set_ylabel('Action Class')
        ax.grid()

        ratios_full = [100] * n_classes
        y_range = np.arange(1, n_classes + 1)
        ax.barh(y_range, ratios_full, color=colors[1])
        ax.barh(y_range, ratios, color=colors[0])

        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)

class TemporalSelectionCallback():
    def __init__(self, model, n_timesteps):

        self.__is_local_machine = configs.is_local_machine()

        if not self.__is_local_machine:
            return

        plt.ion()
        self.colors = plot_utils.tableau_category20()
        fig_width = int(n_timesteps * 40 / 64.0)
        fig, axes = plt.subplots(nrows=2, ncols=1, num='Gating', figsize=(fig_width, 4))
        self.axes = axes

        self.n_timesteps = n_timesteps
        self.model = model

        self.values_bef = None
        self.values_aft = None

    def on_batch_ends(self, batch_num, is_training):

        # only consider test
        if is_training:
            return

        # clear old list from previous epoch
        if batch_num == 1:
            self.values_bef = None
            self.values_aft = None

        # get tensor value
        values_bef = pytorch_utils.model_get_tensor_value(self.model, ('temporal_selection', 'attention_values_before'))  # (None, T)
        values_aft = pytorch_utils.model_get_tensor_value(self.model, ('temporal_selection', 'attention_values_after'))  # (None, T)

        # append it to the list
        self.values_bef = values_bef if self.values_bef is None else np.vstack((self.values_bef, values_bef))  # (None, T)
        self.values_aft = values_aft if self.values_aft is None else np.vstack((self.values_aft, values_aft))  # (None, T)

    def on_epoch_ends(self, epoch_num):
        """
        plot histogram of node assignments.
        :param epoch_num:
        :return:
        """

        values_bef = self.values_bef  # (None, T)
        values_aft = self.values_aft  # (None, T)

        # get attn values for class=1
        values_bef_1 = values_bef[0]  # (None, T)
        values_aft_1 = values_aft[0]  # (None, T)

        # ratio of open gates for all dataset
        ratio = int(100 * np.count_nonzero(values_aft) / np.prod(values_bef.shape))
        ratio_1 = int(100 * np.count_nonzero(values_aft_1) / np.prod(values_aft_1.shape))
        ratios = [ratio_1, ratio]

        # pool over samples
        values_bef = np.mean(values_bef, axis=0)  # (T,)
        values_aft = np.mean(values_aft, axis=0)  # (T,)

        n_timesteps = self.n_timesteps
        x_range = np.arange(1, n_timesteps + 1)
        colors = self.colors
        axes = self.axes
        ax1, ax3 = axes

        for idx_axis, ax in enumerate(axes):
            ratio = ratios[idx_axis]
            ax.cla()
            ax.set_ylim([0, 1.0])
            ax.set_xlabel('Gates [%d%%]' % (ratio))
            ax.set_ylabel('activation')
            ax.grid()

        ax1.bar(x_range, values_bef_1, color=colors[1], label='Before')
        ax1.bar(x_range, values_aft_1, color=colors[0], label='After')
        ax1.legend(loc='upper right')
        ax3.bar(x_range, values_bef, color=colors[5], label='Before')
        ax3.bar(x_range, values_aft, color=colors[4], label='After')
        ax3.legend(loc='upper right')

        plt.tight_layout()
        plt.draw()
        # figure_path = Pth('Charades/figures/%02d.png', (epoch_num))
        # plt.savefig(figure_path)
        plt.pause(0.1)

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

        # mean = np.mean(self.f_mean)
        # std = np.mean(self.f_std)
        # alpha_ratio = np.mean(self.alpha_ratio)
        mean = self.f_mean[0]
        std = self.f_std[0]
        alpha_ratio = self.alpha_ratio[0]

        sys.stdout.write('\r\r      | ratio %.02f | mean %.02f | std %.02f\n' % (alpha_ratio, mean, std))

class SelectionRatioMultiCallback():
    def __init__(self, model, n_contexts):

        self.__is_local_machine = configs.is_local_machine()

        self.model = model
        self.n_contexts = n_contexts
        self.f_values = [None] * n_contexts
        self.alpha_values = [None] * n_contexts

    def on_batch_ends(self, batch_num, is_training):

        # only consider test
        if is_training:
            return

        n_contexts = self.n_contexts

        # clear old list from previous epoch
        if batch_num == 1:
            self.f_values = [None] * n_contexts
            self.alpha_values = [None] * n_contexts

        for i in range(n_contexts):
            # get tensor value and append it to the list
            f_values = pytorch_utils.model_get_tensor_value(self.model, ('feature_selection_%d' % (i + 1), 'f_values'))  # (B, N)
            alpha_values = pytorch_utils.model_get_tensor_value(self.model, ('feature_selection_%d' % (i + 1), 'alpha_values'))  # (B, N)

            self.f_values[i] = f_values if self.f_values[i] is None else np.vstack((self.f_values[i], f_values))  # (B, 1, N, 1, 1)
            self.alpha_values[i] = alpha_values if self.alpha_values[i] is None else np.vstack((self.alpha_values[i], alpha_values))  # (B, N)

    def on_epoch_ends(self, epoch_num):
        """
        plot histogram of node assignments.
        :param epoch_num:
        :return:
        """

        f_values = self.f_values  # (B, N)
        alpha_values = self.alpha_values  # (B, N)

        stds = ['%.02f' % np.std(f) for f in f_values]
        means = ['%.02f' % np.mean(f) for f in f_values]
        ratios = ['%.02f' % (100 * np.count_nonzero(a) / np.prod(a.shape)) for a in alpha_values]

        stds = ', '.join(stds)
        means = ', '.join(means)
        ratios = ', '.join(ratios)

        sys.stdout.write('\r\r      | ratios (%s) | means (%s) | stds (%s)\n' % (ratios, means, stds))

class ChannelGatingPerCategoryCallback():
    def __init__(self, model, x, y, batch_size):
        self.model = model
        self.batch_size = batch_size
        self.x = x
        self.y = y

        plt.ion()
        self.colors = plot_utils.tableau_category20()
        self.fig, self.axes = plt.subplots(nrows=3, ncols=1, num='Channel Gating', figsize=(7, 6))

    def on_batch_ends(self, batch_num, is_training):
        pass

    def on_epoch_ends(self, epoch_num):
        """
        :param epoch_num:
        :return:
        """

        f, alpha = pytorch_utils.batched_feedforward_twin(self.model, self.x, self.batch_size, 'forward_for_gating')  # (B, C, N, 1, 1)
        utils.pkl_dump((f, alpha), '/home/nour/Downloads/dummy.pkl')
        f = np.squeeze(f)[:, :50, 0]  # (B, C)
        alpha = np.squeeze(alpha)[:, :50, 0]  # (B, C)

        std = np.std(f)
        mean = np.mean(f)
        ratio = (100 * np.count_nonzero(alpha) / np.prod(alpha.shape))
        sys.stdout.write('\r\r      | ratio %.02f | mean %.02f | std %.02f\n' % (ratio, mean, std))

        values_bef_1 = f[0]
        values_aft_1 = alpha[0]

        values_bef_2 = f[100]
        values_aft_2 = alpha[100]

        values_bef_3 = np.mean(f, axis=0)
        values_aft_3 = np.mean(alpha, axis=0)

        n_channels = f.shape[1]
        x_range = np.arange(1, n_channels + 1)
        colors = self.colors
        axes = self.axes
        ax1, ax2, ax3 = axes

        for idx_axis, ax in enumerate(axes):
            ax.cla()
            ax.set_ylim([0, 1.0])
            ax.set_xlabel('Channel Gating')
            ax.set_ylabel('Gating Value')
            ax.grid()

        ax1.bar(x_range, values_bef_1, color=colors[1], label='Before')
        ax1.bar(x_range, values_aft_1, color=colors[0], label='After')
        ax1.legend(loc='upper right')
        ax2.bar(x_range, values_bef_2, color=colors[3], label='Before')
        ax2.bar(x_range, values_aft_2, color=colors[2], label='After')
        ax2.legend(loc='upper right')
        ax3.bar(x_range, values_bef_3, color=colors[5], label='Before')
        ax3.bar(x_range, values_aft_3, color=colors[4], label='After')
        ax3.legend(loc='upper right')

        plt.tight_layout()
        plt.draw()
        figure_path = Pth('Hico/figures/%02d.png', (epoch_num))
        plt.savefig(figure_path)
        plt.pause(0.1)

# endregion
