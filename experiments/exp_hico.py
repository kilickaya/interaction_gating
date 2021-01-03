
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

from core.compact_bilinear_pooling import CompactBilinearPooling as cbp

#from datasets import ds_breakfast
from nets import resnet_torch

# region Const

N_CLASSES = 600  

def expand_feats(feat):

    feat = np.expand_dims(feat, 2)
    feat = np.expand_dims(feat, 3)
    feat = np.expand_dims(feat, 4)

    return feat

def expand_feats_(feat):

    feat = np.expand_dims(feat, 3)
    feat = np.expand_dims(feat, 4)

    return feat

def train_human_object_single_context(contextype = 'deformation'):

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    # Features of the image: f_interaction
    #feature_path_interaction = Pth('Hico/features/h5/base_human_object.h5')
    #n_channels, n_regions, channel_side_dim = 512, 1, 1
 
    feature_path_interaction = Pth('Hico/features/h5/features_base_subject_object.h5')
    n_channels, n_regions, channel_side_dim = 4096, 12, 1

    if contextype == 'deformation':
        # Features of the pose: f_context
        feature_path_context = Pth('Hico/features/h5/deformation.h5')
        x_cs_shape = [(512, 1, 1, 1)]
    elif contextype == 'lvis':
        # Features of the pose: f_context
        feature_path_context = Pth('Hico/features/h5/lvis.h5')
        x_cs_shape = [(1300, 1, 1, 1)]
    elif contextype == 'local_scene':
        feature_path_context = Pth('Hico/features/h5/local_scene.h5')
        x_cs_shape = [(2048, 1, 1, 1)]
    elif contextype == 'stuff':
        feature_path_context = Pth('Hico/features/h5/stuff.h5')
        x_cs_shape = [(649, 1, 1, 1)]
    elif contextype == 'part_states':
        # Features of the pose: f_context
        feature_path_context = Pth('Hico/features/h5/part_states.h5')
        x_cs_shape = [(1032, 1, 1, 1)]
    elif contextype == 'local_pose':
        feature_path_context = Pth('Hico/features/h5/local_pose.h5')
        x_cs_shape = [(4096, 1, 1, 1)]
    else:
        print('Unknown context type: %s' %(contextype))

    # Annotation of the image
    annot_path = Pth('Hico/features/h5/anno_hico.pkl')
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

    if contextype != 'lvis':
        x_tr_c = expand_feats(x_tr_c)
        x_te_c = expand_feats(x_te_c)

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
    model_name = 'single_context_hico_%s_%s' % (contextype,utils.timestamp())
    model_root_path = '/var/scratch/mkilicka/data/hico/models_finetuned/%s' %(model_name)
    model_save_callback =  pytorch_utils.ModelSaveCallback(model, model_root_path)

    print('Interaction_feat: %s, Context_feat: %s\n' %(feature_path_interaction, feature_path_context))

    # start training
    pytorch_utils.train_model_custom_metric_mask(model, model._optimizer, model._loss_fn, model._metric_fn, [x_tr, x_tr_c], y_tr, y_tr_mask, [x_te, x_te_c], y_te, y_te_mask, n_epochs, batch_size_tr, batch_size_te, callbacks= [model_save_callback])

    print('--- finish time')
    print(datetime.datetime.now())


def train_human_object_multiple_context(early_flag = True, backbone = 'rcnn'):

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    # Features of the image: f_interaction
    #feature_path_interaction = Pth('Hico/features/h5/base_human_object.h5')
    #n_channels, n_regions, channel_side_dim = 512, 1, 1
 
    if backbone == 'rcnn':
        print('Using backbone rcnn')
        feature_path_interaction = Pth('Hico/features/h5/features_base_subject_object.h5')
        n_channels, n_regions, channel_side_dim = 4096, 12,1
        (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
        x_tr = np.swapaxes(x_tr, 1,2)
        x_te = np.swapaxes(x_te, 1,2)
    elif backbone == 'pairatt':
        print('Using backbone pairatt')
        feature_path_interaction = Pth('Hico/features/h5/features_pairattn.h5')
        n_channels, n_regions, channel_side_dim = 4096, 3,1
        (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
    elif backbone == 'contextfusion':
        print('Using backbone contextfusion')
        feature_path_interaction = Pth('Hico/features/h5/features_contextfusion.h5')
        n_channels, n_regions, channel_side_dim = 4096, 3,1
        (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
        x_tr = np.swapaxes(x_tr, 1,2)
        x_te = np.swapaxes(x_te, 1,2)
        x_tr = expand_feats_(x_tr)
        x_te = expand_feats_(x_te)
    elif backbone == 'vgg':
        print('Using backbone VGG')
        feature_path_interaction = Pth('Hico/features/h5/features_images.h5')
        n_channels, n_regions, channel_side_dim = 2048, 1,1
        (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
    else:
        print('Unknown backbone!')

    # Features of the pose: f_context
    feature_path_c3= Pth('Hico/features/h5/deformation.h5')
    x_cs_shape = [(512, 1, 1, 1)]

    # Features of the pose: f_context
    feature_path_c1 = Pth('Hico/features/h5/lvis.h5')
    x_cs_shape = [(1300, 1, 1, 1)]

    feature_path_c2 = Pth('Hico/features/h5/local_scene.h5')
    x_cs_shape = [(2048, 1, 1, 1)]

    feature_path_context = Pth('Hico/features/h5/stuff.h5')
    x_cs_shape = [(649, 1, 1, 1)]

    # Features of the pose: f_context
    feature_path_context = Pth('Hico/features/h5/part_states.h5')
    x_cs_shape = [(1032, 1, 1, 1)]

    feature_path_c4 = Pth('Hico/features/h5/local_pose.h5')
    x_cs_shape = [(4096, 1, 1, 1)]

    x_cs_shape = [(1300, 1, 1, 1), (2048, 1, 1, 1), (512, 1, 1, 1), (4096, 1, 1, 1)]

    # Annotation of the image
    annot_path = Pth('Hico/features/h5/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())
    input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)

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
    (x_tr_c1, x_te_c1) = utils.h5_load_multi(feature_path_c1, ['x_tr', 'x_te'])
    #x_tr_c1 = expand_feats(x_tr_c1)
    #x_te_c1 = expand_feats(x_te_c1)

    (x_tr_c2, x_te_c2) = utils.h5_load_multi(feature_path_c2, ['x_tr', 'x_te'])
    x_tr_c2 = expand_feats(x_tr_c2)
    x_te_c2 = expand_feats(x_te_c2)

    (x_tr_c3, x_te_c3) = utils.h5_load_multi(feature_path_c3, ['x_tr', 'x_te'])
    x_tr_c3 = expand_feats(x_tr_c3)
    x_te_c3 = expand_feats(x_te_c3)

    (x_tr_c4, x_te_c4) = utils.h5_load_multi(feature_path_c4, ['x_tr', 'x_te'])
    x_tr_c4 = expand_feats(x_tr_c4)
    x_te_c4 = expand_feats(x_te_c4)

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
    if early_flag == False:
        print('Training late fusion model')
        acronym_fusion = 'late_fusion_%s' %(backbone)
        model = ClassifierContextLateFusionMulti(n_classes, input_shape, x_cs_shape) 
    else:
        print('Training early fusion model')
        acronym_fusion = 'early_fusion_%s' %(backbone)
        model = ClassifierContextLateEarlyFusionHumanObject(n_classes, input_shape, x_cs_shape) 

    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    input_sizes = [input_shape] + list(x_cs_shape)
    pytorch_utils.model_summary_multi_input(model, input_sizes=input_sizes, batch_size=-1, device='cuda')    
    print('... model built, duration (sec): %d' % (duration))

    # callbacks

    model_name = 'fusion_for_hico_%s_%s' % (acronym_fusion,utils.timestamp())
    model_root_path = '/var/scratch/mkilicka/data/hico/models_finetuned/%s' %(model_name)

    print('Model will be saved to: %s' %(model_root_path))

    # callbacks
    model_save_callback =  pytorch_utils.ModelSaveCallback(model, model_root_path)

    print('Interaction_feat: %s, Context_feat-1: %s, Context_feat-2: %s, Context_feat-3: %s\n' %(feature_path_interaction, feature_path_c1, feature_path_c2, feature_path_c3))

    # start training
    pytorch_utils.train_model_custom_metric_mask(model, model._optimizer, model._loss_fn, model._metric_fn, [x_tr, x_tr_c1,x_tr_c2, x_tr_c3, x_tr_c4], y_tr, y_tr_mask, [x_te, x_te_c1, x_te_c2, x_te_c3, x_te_c4], y_te, y_te_mask, n_epochs, batch_size_tr, batch_size_te, callbacks=[model_save_callback])

    print('--- finish time')
    print(datetime.datetime.now())



def train_human_object_early_fusion_efficiency(early_flag = True, backbone = 'rcnn'):

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    # Features of the image: f_interaction
    #feature_path_interaction = Pth('Hico/features/h5/base_human_object.h5')
    #n_channels, n_regions, channel_side_dim = 512, 1, 1
 
    if backbone == 'rcnn':
        print('Using backbone rcnn')
        feature_path_interaction = Pth('Hico/features/h5/features_base_subject_object.h5')
        n_channels, n_regions, channel_side_dim = 4096, 12,1
        (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
        x_tr = np.swapaxes(x_tr, 1,2)
        x_te = np.swapaxes(x_te, 1,2)
    elif backbone == 'pairatt':
        print('Using backbone pairatt')
        feature_path_interaction = Pth('Hico/features/h5/features_pairattn.h5')
        n_channels, n_regions, channel_side_dim = 4096, 3,1
        (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
    elif backbone == 'contextfusion':
        print('Using backbone contextfusion')
        feature_path_interaction = Pth('Hico/features/h5/features_contextfusion.h5')
        n_channels, n_regions, channel_side_dim = 4096, 3,1
        (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
        x_tr = np.swapaxes(x_tr, 1,2)
        x_te = np.swapaxes(x_te, 1,2)
        x_tr = expand_feats_(x_tr)
        x_te = expand_feats_(x_te)
    elif backbone == 'vgg':
        print('Using backbone VGG')
        feature_path_interaction = Pth('Hico/features/h5/features_images.h5')
        n_channels, n_regions, channel_side_dim = 2048, 1,1
        (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
    else:
        print('Unknown backbone!')

    # Features of the pose: f_context
    feature_path_c3= Pth('Hico/features/h5/deformation.h5')
    x_cs_shape = [(512, 1, 1, 1)]

    # Features of the pose: f_context
    feature_path_c1 = Pth('Hico/features/h5/lvis.h5')
    x_cs_shape = [(1300, 1, 1, 1)]

    feature_path_c2 = Pth('Hico/features/h5/local_scene.h5')
    x_cs_shape = [(2048, 1, 1, 1)]

    feature_path_context = Pth('Hico/features/h5/stuff.h5')
    x_cs_shape = [(649, 1, 1, 1)]

    # Features of the pose: f_context
    feature_path_context = Pth('Hico/features/h5/part_states.h5')
    x_cs_shape = [(1032, 1, 1, 1)]

    feature_path_c4 = Pth('Hico/features/h5/local_pose.h5')
    x_cs_shape = [(4096, 1, 1, 1)]

    # 4 context
    x_cs_shape = [(1300, 1, 1, 1), (2048, 1, 1, 1), (512, 1, 1, 1), (4096, 1, 1, 1)]
    # 3 context
    x_cs_shape = [(1300, 1, 1, 1), (2048, 1, 1, 1), (512, 1, 1, 1)]
    # 2 context
    x_cs_shape = [(1300, 1, 1, 1), (2048, 1, 1, 1)]
    # 1 context
    x_cs_shape = [(1300, 1, 1, 1)]


    # Annotation of the image
    annot_path = Pth('Hico/features/h5/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())
    input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)

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
    (x_tr_c1, x_te_c1) = utils.h5_load_multi(feature_path_c1, ['x_tr', 'x_te'])
    #x_tr_c1 = expand_feats(x_tr_c1)
    #x_te_c1 = expand_feats(x_te_c1)

    (x_tr_c2, x_te_c2) = utils.h5_load_multi(feature_path_c2, ['x_tr', 'x_te'])
    x_tr_c2 = expand_feats(x_tr_c2)
    x_te_c2 = expand_feats(x_te_c2)

    (x_tr_c3, x_te_c3) = utils.h5_load_multi(feature_path_c3, ['x_tr', 'x_te'])
    x_tr_c3 = expand_feats(x_tr_c3)
    x_te_c3 = expand_feats(x_te_c3)

    (x_tr_c4, x_te_c4) = utils.h5_load_multi(feature_path_c4, ['x_tr', 'x_te'])
    x_tr_c4 = expand_feats(x_tr_c4)
    x_te_c4 = expand_feats(x_te_c4)

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

    acronym_fusion = 'early_fusion_%s' %(backbone)
    model = ClassifierContextEarlyFusionHumanObject(n_classes, input_shape, x_cs_shape) 

    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    input_sizes = [input_shape] + list(x_cs_shape)
    pytorch_utils.model_summary_multi_input(model, input_sizes=input_sizes, batch_size=-1, device='cuda')    
    print('... model built, duration (sec): %d' % (duration))

    # callbacks

    model_name = 'fusion_for_vcoco_%s_%s' % (acronym_fusion,utils.timestamp())
    model_root_path = '/var/scratch/mkilicka/data/hico/models_finetuned/%s' %(model_name)

    # callbacks
    model_save_callback =  pytorch_utils.ModelSaveCallback(model, model_root_path)

    print('Interaction_feat: %s, Context_feat-1: %s, Context_feat-2: %s, Context_feat-3: %s\n' %(feature_path_interaction, feature_path_c1, feature_path_c2, feature_path_c3))

    # start training
    #pytorch_utils.train_model_custom_metric_mask(model, model._optimizer, model._loss_fn, model._metric_fn, [x_tr, x_tr_c1,x_tr_c2, x_tr_c3, x_tr_c4], y_tr, y_tr_mask, [x_te, x_te_c1, x_te_c2, x_te_c3, x_te_c4], y_te, y_te_mask, n_epochs, batch_size_tr, batch_size_te, callbacks=[])
    #pytorch_utils.train_model_custom_metric_mask(model, model._optimizer, model._loss_fn, model._metric_fn, [x_tr, x_tr_c1,x_tr_c2, x_tr_c3], y_tr, y_tr_mask, [x_te, x_te_c1, x_te_c2, x_te_c3], y_te, y_te_mask, n_epochs, batch_size_tr, batch_size_te, callbacks=[])
    #pytorch_utils.train_model_custom_metric_mask(model, model._optimizer, model._loss_fn, model._metric_fn, [x_tr, x_tr_c1,x_tr_c2], y_tr, y_tr_mask, [x_te, x_te_c1, x_te_c2], y_te, y_te_mask, n_epochs, batch_size_tr, batch_size_te, callbacks=[])
    pytorch_utils.train_model_custom_metric_mask(model, model._optimizer, model._loss_fn, model._metric_fn, [x_tr, x_tr_c1], y_tr, y_tr_mask, [x_te, x_te_c1], y_te, y_te_mask, n_epochs, batch_size_tr, batch_size_te, callbacks=[])

    print('--- finish time')
    print(datetime.datetime.now())




def train_human_object_multiple_context_gating_multihead(backbone = 'rcnn'):

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    # Features of the image: f_interaction
    #feature_path_interaction = Pth('Hico/features/h5/base_human_object.h5')
    #n_channels, n_regions, channel_side_dim = 512, 1, 1
 
    if backbone == 'rcnn':
        print('Using backbone rcnn')
        feature_path_interaction = Pth('Hico/features/h5/features_base_subject_object.h5')
        n_channels, n_regions, channel_side_dim = 4096, 12,1
        (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
        x_tr = np.swapaxes(x_tr, 1,2)
        x_te = np.swapaxes(x_te, 1,2)
    elif backbone == 'pairatt':
        print('Using backbone pairatt')
        feature_path_interaction = Pth('Hico/features/h5/features_pairattn.h5')
        n_channels, n_regions, channel_side_dim = 4096, 3,1
        (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
    elif backbone == 'contextfusion':
        print('Using backbone contextfusion')
        feature_path_interaction = Pth('Hico/features/h5/features_contextfusion.h5')
        n_channels, n_regions, channel_side_dim = 4096, 3,1
        (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
        x_tr = np.swapaxes(x_tr, 1,2)
        x_te = np.swapaxes(x_te, 1,2)
        x_tr = expand_feats_(x_tr)
        x_te = expand_feats_(x_te)
    elif backbone == 'vgg':
        print('Using backbone VGG')
        feature_path_interaction = Pth('Hico/features/h5/features_images.h5')
        n_channels, n_regions, channel_side_dim = 2048, 1,1
        (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
    else:
        print('Unknown backbone!')

    # Features of the pose: f_context
    feature_path_c3= Pth('Hico/features/h5/deformation.h5')
    x_cs_shape = [(512, 1, 1, 1)]

    # Features of the pose: f_context
    feature_path_c1 = Pth('Hico/features/h5/lvis.h5')
    x_cs_shape = [(1300, 1, 1, 1)]

    feature_path_c2 = Pth('Hico/features/h5/local_scene.h5')
    x_cs_shape = [(2048, 1, 1, 1)]

    feature_path_context = Pth('Hico/features/h5/stuff.h5')
    x_cs_shape = [(649, 1, 1, 1)]

    # Features of the pose: f_context
    feature_path_context = Pth('Hico/features/h5/part_states.h5')
    x_cs_shape = [(1032, 1, 1, 1)]

    feature_path_c4 = Pth('Hico/features/h5/local_pose.h5')
    x_cs_shape = [(4096, 1, 1, 1)]

    x_cs_shape = [(1300, 1, 1, 1), (2048, 1, 1, 1), (512, 1, 1, 1), (4096, 1, 1, 1)]

    # Annotation of the image
    annot_path = Pth('Hico/features/h5/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())
    input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)

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
    (x_tr_c1, x_te_c1) = utils.h5_load_multi(feature_path_c1, ['x_tr', 'x_te'])
    #x_tr_c1 = expand_feats(x_tr_c1)
    #x_te_c1 = expand_feats(x_te_c1)

    (x_tr_c2, x_te_c2) = utils.h5_load_multi(feature_path_c2, ['x_tr', 'x_te'])
    x_tr_c2 = expand_feats(x_tr_c2)
    x_te_c2 = expand_feats(x_te_c2)

    (x_tr_c3, x_te_c3) = utils.h5_load_multi(feature_path_c3, ['x_tr', 'x_te'])
    x_tr_c3 = expand_feats(x_tr_c3)
    x_te_c3 = expand_feats(x_te_c3)

    (x_tr_c4, x_te_c4) = utils.h5_load_multi(feature_path_c4, ['x_tr', 'x_te'])
    x_tr_c4 = expand_feats(x_tr_c4)
    x_te_c4 = expand_feats(x_te_c4)

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
    print('Training gating model')
    acronym_fusion = 'gating_%s' %(backbone)
    model = ClassifierContextInteraction(n_classes, input_shape, x_cs_shape) 

    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    input_sizes = [input_shape] + list(x_cs_shape)
    pytorch_utils.model_summary_multi_input(model, input_sizes=input_sizes, batch_size=-1, device='cuda')    
    print('... model built, duration (sec): %d' % (duration))

    # callbacks

    model_name = 'hico_%s_%s' % (acronym_fusion,utils.timestamp())
    model_root_path = '/var/scratch/mkilicka/data/hico/models_finetuned/%s' %(model_name)

    # callbacks
    model_save_callback =  pytorch_utils.ModelSaveCallback(model, model_root_path)

    print('Interaction_feat: %s, Context_feat-1: %s, Context_feat-2: %s, Context_feat-3: %s, , Context_feat-4: %s\n' %(feature_path_interaction, feature_path_c1, feature_path_c2, feature_path_c3, feature_path_c4))

    # start training
    pytorch_utils.train_model_custom_metric_mask(model, model._optimizer, model._loss_fn, model._metric_fn, [x_tr, x_tr_c1,x_tr_c2, x_tr_c3, x_tr_c4], y_tr, y_tr_mask, [x_te, x_te_c1, x_te_c2, x_te_c3, x_te_c4], y_te, y_te_mask, n_epochs, batch_size_tr, batch_size_te, callbacks=[model_save_callback])

    print('--- finish time')
    print(datetime.datetime.now())


def train_human_object_multiple_context_gating_efficiency( backbone = 'rcnn'):

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    # Features of the image: f_interaction
    #feature_path_interaction = Pth('Hico/features/h5/base_human_object.h5')
    #n_channels, n_regions, channel_side_dim = 512, 1, 1
 
    if backbone == 'rcnn':
        print('Using backbone rcnn')
        feature_path_interaction = Pth('Hico/features/h5/features_base_subject_object.h5')
        n_channels, n_regions, channel_side_dim = 4096, 12,1
        (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
        x_tr = np.swapaxes(x_tr, 1,2)
        x_te = np.swapaxes(x_te, 1,2)
    elif backbone == 'pairatt':
        print('Using backbone pairatt')
        feature_path_interaction = Pth('Hico/features/h5/features_pairattn.h5')
        n_channels, n_regions, channel_side_dim = 4096, 3,1
        (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
    elif backbone == 'contextfusion':
        print('Using backbone contextfusion')
        feature_path_interaction = Pth('Hico/features/h5/features_contextfusion.h5')
        n_channels, n_regions, channel_side_dim = 4096, 3,1
        (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
        x_tr = np.swapaxes(x_tr, 1,2)
        x_te = np.swapaxes(x_te, 1,2)
        x_tr = expand_feats_(x_tr)
        x_te = expand_feats_(x_te)
    elif backbone == 'vgg':
        print('Using backbone VGG')
        feature_path_interaction = Pth('Hico/features/h5/features_images.h5')
        n_channels, n_regions, channel_side_dim = 2048, 1,1
        (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
    else:
        print('Unknown backbone!')

    # Features of the pose: f_context
    feature_path_c3= Pth('Hico/features/h5/deformation.h5')
    x_cs_shape = [(512, 1, 1, 1)]

    # Features of the pose: f_context
    feature_path_c1 = Pth('Hico/features/h5/lvis.h5')
    x_cs_shape = [(1300, 1, 1, 1)]

    feature_path_c2 = Pth('Hico/features/h5/local_scene.h5')
    x_cs_shape = [(2048, 1, 1, 1)]

    feature_path_context = Pth('Hico/features/h5/stuff.h5')
    x_cs_shape = [(649, 1, 1, 1)]

    # Features of the pose: f_context
    feature_path_context = Pth('Hico/features/h5/part_states.h5')
    x_cs_shape = [(1032, 1, 1, 1)]

    feature_path_c4 = Pth('Hico/features/h5/local_pose.h5')
    x_cs_shape = [(4096, 1, 1, 1)]

    # 4 context
    x_cs_shape = [(1300, 1, 1, 1), (2048, 1, 1, 1), (512, 1, 1, 1), (4096, 1, 1, 1)]
    # 3 context
    x_cs_shape = [(1300, 1, 1, 1), (2048, 1, 1, 1), (512, 1, 1, 1)]
    # 2 context
    x_cs_shape = [(1300, 1, 1, 1), (2048, 1, 1, 1)]
    # 1 context
    x_cs_shape = [(1300, 1, 1, 1)]

    # Annotation of the image
    annot_path = Pth('Hico/features/h5/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())
    input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)

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
    (x_tr_c1, x_te_c1) = utils.h5_load_multi(feature_path_c1, ['x_tr', 'x_te'])
    #x_tr_c1 = expand_feats(x_tr_c1)
    #x_te_c1 = expand_feats(x_te_c1)

    (x_tr_c2, x_te_c2) = utils.h5_load_multi(feature_path_c2, ['x_tr', 'x_te'])
    x_tr_c2 = expand_feats(x_tr_c2)
    x_te_c2 = expand_feats(x_te_c2)

    (x_tr_c3, x_te_c3) = utils.h5_load_multi(feature_path_c3, ['x_tr', 'x_te'])
    x_tr_c3 = expand_feats(x_tr_c3)
    x_te_c3 = expand_feats(x_te_c3)

    (x_tr_c4, x_te_c4) = utils.h5_load_multi(feature_path_c4, ['x_tr', 'x_te'])
    x_tr_c4 = expand_feats(x_tr_c4)
    x_te_c4 = expand_feats(x_te_c4)

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
    print('Training gating model')
    acronym_fusion = 'gating_%s' %(backbone)
    model = ClassifierContextInteraction(n_classes, input_shape, x_cs_shape) 

    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    input_sizes = [input_shape] + list(x_cs_shape)
    pytorch_utils.model_summary_multi_input(model, input_sizes=input_sizes, batch_size=-1, device='cuda')    
    print('... model built, duration (sec): %d' % (duration))

    # callbacks

    model_name = 'hico_%s_%s' % (acronym_fusion,utils.timestamp())
    model_root_path = '/var/scratch/mkilicka/data/hico/models_finetuned/%s' %(model_name)

    # callbacks
    model_save_callback =  pytorch_utils.ModelSaveCallback(model, model_root_path)

    print('Interaction_feat: %s, Context_feat-1: %s, Context_feat-2: %s, Context_feat-3: %s, , Context_feat-4: %s\n' %(feature_path_interaction, feature_path_c1, feature_path_c2, feature_path_c3, feature_path_c4))

    # start training
    pytorch_utils.train_model_custom_metric_mask(model, model._optimizer, model._loss_fn, model._metric_fn, [x_tr, x_tr_c1,x_tr_c2, x_tr_c3, x_tr_c4], y_tr, y_tr_mask, [x_te, x_te_c1, x_te_c2, x_te_c3, x_te_c4], y_te, y_te_mask, n_epochs, batch_size_tr, batch_size_te, callbacks=[model_save_callback])

    print('--- finish time')
    print(datetime.datetime.now())




def train_human_object_multiple_context_gating_multihead_vcoco( backbone = 'rcnn'):

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    # Features of the image: f_interaction
    #feature_path_interaction = Pth('Hico/features/h5/base_human_object.h5')
    #n_channels, n_regions, channel_side_dim = 512, 1, 1
 
    if backbone == 'rcnn':
        print('Using backbone rcnn')
        feature_path_interaction = Pth('Hico/features/h5/features_base_subject_object.h5')
        n_channels, n_regions, channel_side_dim = 4096, 12,1
        (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
        x_tr = np.swapaxes(x_tr, 1,2)
        x_te = np.swapaxes(x_te, 1,2)
    elif backbone == 'pairatt':
        print('Using backbone pairatt')
        feature_path_interaction = Pth('Hico/features/h5/features_pairattn.h5')
        n_channels, n_regions, channel_side_dim = 4096, 3,1
        (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
    elif backbone == 'contextfusion':
        print('Using backbone contextfusion')
        feature_path_interaction = Pth('Hico/features/h5/features_contextfusion.h5')
        n_channels, n_regions, channel_side_dim = 4096, 3,1
        (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
        x_tr = np.swapaxes(x_tr, 1,2)
        x_te = np.swapaxes(x_te, 1,2)
        x_tr = expand_feats_(x_tr)
        x_te = expand_feats_(x_te)
    elif backbone == 'vgg':
        print('Using backbone VGG')
        feature_path_interaction = Pth('Hico/features/h5/features_images.h5')
        n_channels, n_regions, channel_side_dim = 2048, 1,1
        (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
    else:
        print('Unknown backbone!')

    # Features of the pose: f_context
    feature_path_c3= Pth('Hico/features/h5/deformation.h5')
    x_cs_shape = [(512, 1, 1, 1)]

    # Features of the pose: f_context
    feature_path_c1 = Pth('Hico/features/h5/lvis.h5')
    x_cs_shape = [(1300, 1, 1, 1)]

    feature_path_c2 = Pth('Hico/features/h5/local_scene.h5')
    x_cs_shape = [(2048, 1, 1, 1)]

    feature_path_context = Pth('Hico/features/h5/stuff.h5')
    x_cs_shape = [(649, 1, 1, 1)]

    # Features of the pose: f_context
    feature_path_context = Pth('Hico/features/h5/part_states.h5')
    x_cs_shape = [(1032, 1, 1, 1)]

    #feature_path_c4 = Pth('Hico/features/h5/local_pose.h5')
    #x_cs_shape = [(4096, 1, 1, 1)]

    x_cs_shape = [(1300, 1, 1, 1), (2048, 1, 1, 1), (512, 1, 1, 1)]

    # Annotation of the image
    annot_path = Pth('Hico/features/h5/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())
    input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)

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
    (x_tr_c1, x_te_c1) = utils.h5_load_multi(feature_path_c1, ['x_tr', 'x_te'])
    #x_tr_c1 = expand_feats(x_tr_c1)
    #x_te_c1 = expand_feats(x_te_c1)

    (x_tr_c2, x_te_c2) = utils.h5_load_multi(feature_path_c2, ['x_tr', 'x_te'])
    x_tr_c2 = expand_feats(x_tr_c2)
    x_te_c2 = expand_feats(x_te_c2)

    (x_tr_c3, x_te_c3) = utils.h5_load_multi(feature_path_c3, ['x_tr', 'x_te'])
    x_tr_c3 = expand_feats(x_tr_c3)
    x_te_c3 = expand_feats(x_te_c3)

    print('train_set_shape_interaction: ', x_tr.shape)
    print('test_set_shape_interaction: ', x_te.shape)

    print('train_set_shape_context-1: ', x_tr_c1.shape)
    print('test_set_shape_context-1: ',  x_te_c1.shape)

    print('train_set_shape_context-2: ', x_tr_c2.shape)
    print('test_set_shape_context-2: ',  x_te_c2.shape)

    print('train_set_shape_context-3: ', x_tr_c3.shape)
    print('test_set_shape_context-3: ',  x_te_c3.shape)

    #print('train_set_shape_context-4: ', x_tr_c4.shape)
    #print('test_set_shape_context-4: ',  x_te_c4.shape)


    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    print('Training gating model')
    acronym_fusion = 'gating_%s' %(backbone)
    model = ClassifierContextInteraction(n_classes, input_shape, x_cs_shape) 

    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    input_sizes = [input_shape] + list(x_cs_shape)
    pytorch_utils.model_summary_multi_input(model, input_sizes=input_sizes, batch_size=-1, device='cuda')    
    print('... model built, duration (sec): %d' % (duration))

    # callbacks

    model_name = 'vcoco_%s_%s' % (acronym_fusion,utils.timestamp())
    model_root_path = '/var/scratch/mkilicka/data/hico/models_finetuned/%s' %(model_name)

    # callbacks
    model_save_callback =  pytorch_utils.ModelSaveCallback(model, model_root_path)

    print('Interaction_feat: %s, Context_feat-1: %s, Context_feat-2: %s, Context_feat-3: %s\n' %(feature_path_interaction, feature_path_c1, feature_path_c2, feature_path_c3))

    # start training
    pytorch_utils.train_model_custom_metric_mask(model, model._optimizer, model._loss_fn, model._metric_fn, [x_tr, x_tr_c1,x_tr_c2, x_tr_c3], y_tr, y_tr_mask, [x_te, x_te_c1, x_te_c2, x_te_c3], y_te, y_te_mask, n_epochs, batch_size_tr, batch_size_te, callbacks=[model_save_callback])

    print('--- finish time')
    print(datetime.datetime.now())

def train_human_object_many_context(early_flag = True, backbone = 'rcnn'):

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    # Features of the image: f_interaction
    #feature_path_interaction = Pth('Hico/features/h5/base_human_object.h5')
    #n_channels, n_regions, channel_side_dim = 512, 1, 1
 
    if backbone == 'rcnn':
        print('Using backbone rcnn')
        feature_path_interaction = Pth('Hico/features/h5/features_base_subject_object.h5')
        n_channels, n_regions, channel_side_dim = 4096, 12,1
        (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
        x_tr = np.swapaxes(x_tr, 1,2)
        x_te = np.swapaxes(x_te, 1,2)
    elif backbone == 'pairatt':
        print('Using backbone pairatt')
        feature_path_interaction = Pth('Hico/features/h5/features_pairattn.h5')
        n_channels, n_regions, channel_side_dim = 4096, 3,1
        (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
    elif backbone == 'contextfusion':
        print('Using backbone contextfusion')
        feature_path_interaction = Pth('Hico/features/h5/features_contextfusion.h5')
        n_channels, n_regions, channel_side_dim = 4096, 3,1
        (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
        x_tr = np.swapaxes(x_tr, 1,2)
        x_te = np.swapaxes(x_te, 1,2)
        x_tr = expand_feats_(x_tr)
        x_te = expand_feats_(x_te)
    elif backbone == 'vgg':
        print('Using backbone VGG')
        feature_path_interaction = Pth('Hico/features/h5/features_images.h5')
        n_channels, n_regions, channel_side_dim = 2048, 1,1
        (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
    else:
        print('Unknown backbone!')

    # Features of the pose: f_context
    feature_path_c3= Pth('Hico/features/h5/deformation.h5')
    x_cs_shape = [(512, 1, 1, 1)]

    # Features of the pose: f_context
    feature_path_c1 = Pth('Hico/features/h5/lvis.h5')
    x_cs_shape = [(1300, 1, 1, 1)]

    feature_path_c2 = Pth('Hico/features/h5/local_scene.h5')
    x_cs_shape = [(2048, 1, 1, 1)]

    feature_path_c5 = Pth('Hico/features/h5/stuff.h5')
    x_cs_shape = [(649, 1, 1, 1)]

    # Features of the pose: f_context
    feature_path_c6 = Pth('Hico/features/h5/part_states.h5')
    x_cs_shape = [(1032, 1, 1, 1)]

    feature_path_c4 = Pth('Hico/features/h5/local_pose.h5')
    x_cs_shape = [(4096, 1, 1, 1)]

    x_cs_shape = [(1300, 1, 1, 1), (2048, 1, 1, 1), (512, 1, 1, 1), (4096, 1,1,1), (649, 1, 1, 1), (1032, 1, 1, 1)]

    # Annotation of the image
    annot_path = Pth('Hico/features/h5/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())
    input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)

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
    (x_tr_c1, x_te_c1) = utils.h5_load_multi(feature_path_c1, ['x_tr', 'x_te'])
    #x_tr_c1 = expand_feats(x_tr_c1)
    #x_te_c1 = expand_feats(x_te_c1)

    (x_tr_c2, x_te_c2) = utils.h5_load_multi(feature_path_c2, ['x_tr', 'x_te'])
    x_tr_c2 = expand_feats(x_tr_c2)
    x_te_c2 = expand_feats(x_te_c2)

    (x_tr_c3, x_te_c3) = utils.h5_load_multi(feature_path_c3, ['x_tr', 'x_te'])
    x_tr_c3 = expand_feats(x_tr_c3)
    x_te_c3 = expand_feats(x_te_c3)

    (x_tr_c4, x_te_c4) = utils.h5_load_multi(feature_path_c4, ['x_tr', 'x_te'])
    x_tr_c4 = expand_feats(x_tr_c4)
    x_te_c4 = expand_feats(x_te_c4)

    (x_tr_c5, x_te_c5) = utils.h5_load_multi(feature_path_c5, ['x_tr', 'x_te'])
    x_tr_c5 = expand_feats(x_tr_c5)
    x_te_c5 = expand_feats(x_te_c5)

    (x_tr_c6, x_te_c6) = utils.h5_load_multi(feature_path_c6, ['x_tr', 'x_te'])
    x_tr_c6 = expand_feats(x_tr_c6)
    x_te_c6 = expand_feats(x_te_c6)

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

    print('train_set_shape_context-5: ', x_tr_c5.shape)
    print('test_set_shape_context-5: ',  x_te_c5.shape)

    print('train_set_shape_context-6: ', x_tr_c6.shape)
    print('test_set_shape_context-6: ',  x_te_c6.shape)

    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    if early_flag == False:
        print('Training late fusion model')
        acronym_fusion = 'late_fusion_%s' %(backbone)
        model = ClassifierContextLateFusionMulti(n_classes, input_shape, x_cs_shape) 
    else:
        print('Training early fusion model')
        acronym_fusion = 'early_fusion_%s' %(backbone)
        model = ClassifierContextEarlyFusionHumanObject(n_classes, input_shape, x_cs_shape) 

    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    input_sizes = [input_shape] + list(x_cs_shape)
    #pytorch_utils.model_summary_multi_input(model, input_sizes=input_sizes, batch_size=-1, device='cuda')    
    print('... model built, duration (sec): %d' % (duration))

    # callbacks

    '''
    model_name = 'fusion_for_vcoco_%s_%s' % (acronym_fusion,utils.timestamp())
    model_root_path = '/var/scratch/mkilicka/data/hico/models_finetuned/%s' %(model_name)

    # callbacks
    model_save_callback =  pytorch_utils.ModelSaveCallback(model, model_root_path)
    '''

    print('Interaction_feat: %s, Context_feat-1: %s, Context_feat-2: %s, Context_feat-3: %s\n' %(feature_path_interaction, feature_path_c1, feature_path_c2, feature_path_c3))

    # start training
    pytorch_utils.train_model_custom_metric_mask(model, model._optimizer, model._loss_fn, model._metric_fn, [x_tr, x_tr_c1,x_tr_c2, x_tr_c3, x_tr_c4, x_tr_c5, x_tr_c6], y_tr, y_tr_mask, [x_te, x_te_c1, x_te_c2, x_te_c3, x_te_c4, x_te_c5, x_te_c6], y_te, y_te_mask, n_epochs, batch_size_tr, batch_size_te, callbacks=[])

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

    def inference(self, *input):
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



class ClassifierContextEarlyFusionHumanObject(nn.Module):
    def __init__(self, n_classes, x_so_shape, x_cs_shape):
        super(ClassifierContextEarlyFusionHumanObject, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_fusion = 'context_fusion_%d'
        self.__init_layers(n_classes, x_so_shape, x_cs_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_so_shape, x_cs_shape):
        """
        Define model layers.
        """

        n_units = 1024
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

class ClassifierContextLateEarlyFusionHumanObjectCBP(nn.Module):
    def __init__(self, n_classes, x_so_shape, x_cs_shape):
        super(ClassifierContextLateEarlyFusionHumanObjectCBP, self).__init__()

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

        self.cbp_layer =  cbp(n_units, n_units,  4*n_units).cuda()


        classifier = []
        classifier.append(nn.Linear(n_units*4, n_classes))
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
        #x = torch.cat((x_feat_action, x_feat_context), dim=2)

        x = self.cbp_layer(x_feat_action, x_feat_context)

        # Feed-to-joint-classifier
        x = self.classifier(x)

        # apply max ops there
        x, _ = torch.max(x, dim=1)  # (B, C)
        x = torch.sigmoid(x)

        return x

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
        #self._optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9)

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
        x_cs = torch.mean(x_cs_classes,0)
        x = torch.sigmoid(x + x_cs)

        return x

    def inference(self, *input):
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
        x_cs = torch.mean(x_cs_classes,0)
        x = torch.sigmoid(x + x_cs)

        return x

    def forward_for_extraction(self, *input):
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
        x = x.view(1, B, self.n_classes)
        preds = torch.cat((x, x_cs_classes), 0)

        preds = preds.permute(1,2,0)

        return preds

def train_human_object_multiple_context_gating(soft_flag = True, ablation_flag = False):

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    backbone = 'rcnn'

    if backbone == 'rcnn':
        print('Using backbone rcnn')
        feature_path_interaction = Pth('Hico/features/h5/features_base_subject_object.h5')
        n_channels, n_regions, channel_side_dim = 4096, 12,1
        (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
        x_tr = np.swapaxes(x_tr, 1,2)
        x_te = np.swapaxes(x_te, 1,2)
    elif backbone == 'pairatt':
        print('Using backbone pairatt')
        feature_path_interaction = Pth('Hico/features/h5/features_pairattn.h5')
        n_channels, n_regions, channel_side_dim = 4096, 3,1
        (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])

    # Features of the pose: f_context
    feature_path_c3= Pth('Hico/features/h5/deformation.h5')
    x_cs_shape = [(512, 1, 1, 1)]

    # Features of the pose: f_context
    feature_path_c1 = Pth('Hico/features/h5/lvis.h5')
    x_cs_shape = [(1300, 1, 1, 1)]

    feature_path_c2 = Pth('Hico/features/h5/local_scene.h5')
    x_cs_shape = [(2048, 1, 1, 1)]

    feature_path_context = Pth('Hico/features/h5/stuff.h5')
    x_cs_shape = [(649, 1, 1, 1)]

    # Features of the pose: f_context
    feature_path_context = Pth('Hico/features/h5/part_states.h5')
    x_cs_shape = [(1032, 1, 1, 1)]

    feature_path_c4 = Pth('Hico/features/h5/local_pose.h5')
    x_cs_shape = [(4096, 1, 1, 1)]

    x_cs_shape = [(1300, 1, 1, 1), (2048, 1, 1, 1), (512, 1, 1, 1), (4096, 1, 1, 1)]

    # Annotation of the image
    annot_path = Pth('Hico/features/h5/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())
    input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)

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
    (x_tr_c1, x_te_c1) = utils.h5_load_multi(feature_path_c1, ['x_tr', 'x_te'])
    #x_tr_c1 = expand_feats(x_tr_c1)
    #x_te_c1 = expand_feats(x_te_c1)

    (x_tr_c2, x_te_c2) = utils.h5_load_multi(feature_path_c2, ['x_tr', 'x_te'])
    x_tr_c2 = expand_feats(x_tr_c2)
    x_te_c2 = expand_feats(x_te_c2)

    (x_tr_c3, x_te_c3) = utils.h5_load_multi(feature_path_c3, ['x_tr', 'x_te'])
    x_tr_c3 = expand_feats(x_tr_c3)
    x_te_c3 = expand_feats(x_te_c3)

    (x_tr_c4, x_te_c4) = utils.h5_load_multi(feature_path_c4, ['x_tr', 'x_te'])
    x_tr_c4 = expand_feats(x_tr_c4)
    x_te_c4 = expand_feats(x_te_c4)

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
    if soft_flag == True:
        print('Training soft fusion model')
        acronym = 'soft'
        model = ClassifierContextLateFusionMultiSoftGate_v3(n_classes, input_shape, x_cs_shape) 
    else:
        if ablation_flag == False:
            print('Training full hard fusion model')
            acronym = 'hard'
            model = ClassifierContextLateFusionMultiHardGate(n_classes, input_shape, x_cs_shape)   
        else:
            print('Training ablated hard fusion model')
            acronym = 'hard_ablated'
            model = ClassifierContextLateFusionMultiHardGateAblated(n_classes, input_shape, x_cs_shape)   


    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    input_sizes = [input_shape] + list(x_cs_shape)
    pytorch_utils.model_summary_multi_input(model, input_sizes=input_sizes, batch_size=-1, device='cuda')    
    print('... model built, duration (sec): %d' % (duration))

    # callbacks
    model_name = 'late_%s_gating_for_hico_%s' % (acronym, utils.timestamp())
    model_root_path = '/var/scratch/mkilicka/data/hico/models_finetuned/%s' %(model_name)

    print('Model will be saved to: %s' %(model_root_path))

    # callbacks
    model_save_callback =  pytorch_utils.ModelSaveCallback(model, model_root_path)
    print('model will be saved to: %s' %(model_root_path))
    print('Interaction_feat: %s, Context_feat-1: %s, Context_feat-2: %s, Context_feat-3: %s\n' %(feature_path_interaction, feature_path_c1, feature_path_c2, feature_path_c3))

    # start training
    pytorch_utils.train_model_custom_metric_mask(model, model._optimizer, model._loss_fn, model._metric_fn, [x_tr, x_tr_c1,x_tr_c2, x_tr_c3, x_tr_c4], y_tr, y_tr_mask, [x_te, x_te_c1, x_te_c2, x_te_c3, x_te_c4], y_te, y_te_mask, n_epochs, batch_size_tr, batch_size_te, callbacks=[model_save_callback])

    print('--- finish time')
    print(datetime.datetime.now())

class ClassifierContextLateFusionMultiSoftGate(nn.Module):
    def __init__(self, n_classes,  x_so_shape, x_cs_shape):
        super(ClassifierContextLateFusionMultiSoftGate, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_context_emb   = 'dense_context_%d'
        self.layer_name_context_class = 'class_context_%d'
        self.layer_name_context_selection =   'imp_context'

        self.__init_layers(n_classes, x_so_shape, x_cs_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_so_shape, x_cs_shape):
        """
        Define model layers.
        """

        self.n_classes = 600

        n_units = 600
        n_channels = 512

        self.n_channels = n_channels

        C_so, N, H, W = x_so_shape
        self.C_so = C_so
        self.N = N

        self.feature_selection = context_fusion.ContextGatingClassifierSoft(x_so_shape, x_cs_shape)

        self.softmax = nn.Softmax(dim = 0)

        # Map so features to a smaller size
        self.dense_so = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_so, n_channels), nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))

        # Loop over existing context features: Map them into interaction categories
        for idx_context in range(self.n_contexts):
            C_c = x_cs_shape[idx_context][0]

            # embedding of multi_ context
            layer_name = self.layer_name_context_emb % (idx_context + 1)
            layer = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_c, n_channels), nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))

            setattr(self, layer_name, layer)

            # categories per context 

            layer_name = self.layer_name_context_class % (idx_context + 1)
            layer = nn.Sequential(nn.Dropout(0.25), nn.Linear(n_channels, n_units))

            setattr(self, layer_name, layer)

        # spatial pooling
        self.spatial_pooling = pl.Max(dim=(3, 4))

        # layers for classification
        classifier_layers = []
        classifier_layers.append(nn.Dropout(0.25))
        classifier_layers.append(nn.Linear(n_channels, n_classes))
        self.classifier_layers = nn.Sequential(*classifier_layers)

    def __init_optimizer(self):
        """
        Define loss, metric and optimizer.
        """
        self._loss_fn = F.binary_cross_entropy
        self._metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
        self._optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-4)
        #self._optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9)

    def get_context_embeddings(self, x_cs, B):

        x_cs_embed = []

        # loop on multi_contexts
        for idx_context in range(self.n_contexts):
            # embedding of context
            x_c = x_cs[idx_context] # (B, C, 1,1,1)
            x_c = x_c.repeat(1, 1, self.N,1,1) # (B, C, N, 1, 1)

            layer = getattr(self, self.layer_name_context_emb % (idx_context + 1))
            x_c = layer(x_c)

            # append to list of context embeddings
            x_cs_embed.append(x_c.view(1, B, self.n_channels, self.N)) # (n_context, B, C, N)

        # process context features to get context embedding from x_cs features
        x_cs_embed = torch.stack(x_cs_embed, dim=0).view(-1, B, self.n_channels, self.N) # (n_context, B, C, N)
        return x_cs_embed

    def get_context_class(self, x_cs, B):

        x_cs_class = []

        # loop on multi_contexts
        for idx_context in range(self.n_contexts):
            # embedding of context
            x_c = x_cs[idx_context] # (B, C, N)

            x_c = x_c.permute(0, 2, 1)  # (B, N, C)

            # hide N dimension
            B, N, C = pytorch_utils.get_shape(x_c)
            x_c = x_c.contiguous().view(B * N, C)  # (B*N, C)

            layer = getattr(self, self.layer_name_context_class % (idx_context + 1))
            x_c = layer(x_c)

            _, C = pytorch_utils.get_shape(x_c)
            x_c = x_c.view(B, N, C)  # (B, N, C)

            # append to list of context class predictions
            x_cs_class.append(x_c.view(1, B, self.N, self.n_classes)) # (1, B,N, C)

        # Process context features to get context category from x_cs features
        x_cs_class = torch.stack(x_cs_class, dim=0).view(-1, B, self.N, self.n_classes) # (n_context, B, N, C)
        return x_cs_class

    def get_context_relevance(self, x_so, x_cs):

        x_cs_value = []
        B, C, N, _,_ = pytorch_utils.get_shape(x_so)

        # loop on multi_contexts
        for idx_context in range(self.n_contexts):
            # embedding of context
            x_c = x_cs[idx_context]
            x_c = x_c.view(B,C, N,1,1) 

            x_c = self.feature_selection(x_so, x_c) # (B, N)
            x_cs_value.append(x_c.view(1, B, N)) # (1, B, C)

        x_cs_value = torch.stack(x_cs_value, dim=0).view(self.n_contexts, B, N) # (num_context, B, N)
        return x_cs_value

    def modulate_context_classifier(self, x_so, x_cs, x_cs_classes, B):

        # return context importance per-category
        x_cs_relevance = self.get_context_relevance(x_so, x_cs) # (num_context, B, N)

        # Reweigh class predictions with activated relevance scores
        x_cs_relevance = x_cs_relevance.view(self.n_contexts, B, self.N, 1) # (4, 32, 12) (32,12,4) (384, 4) 

        x_cs_relevance = self.softmax(x_cs_relevance) # which context to use? over dim=0

        # Modulate context classifiers with relevance scores
        x_cs_classes = x_cs_classes * x_cs_relevance # (nco, B, 12, 600)
        x_cs = torch.sum(x_cs_classes,dim=0) # (B, N, 600)

        return x_cs, x_cs_relevance

    def forward(self, *input):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        # return x_so embeddings
        x_so = input[0]
        x_so = self.dense_so(x_so)

        B, C, N, _,_ = pytorch_utils.get_shape(x_so)

        x_cs = input[1:]

        # return context embeddings
        x_c = self.get_context_embeddings(x_cs, B)

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

        # return context categories
        x_cs_classes = self.get_context_class(x_c, B) # (nco, B, N, C)

        x_cs_classes, _  = self.modulate_context_classifier(x_so, x_c, x_cs_classes, B) # (B, N, 600)
 
        # Add modulated response to human-object classifier and max-pool over N

        ''' v1
        x_cs= torch.sum(x_cs_classes, dim= 1) # ( B, C)
        x,_ = torch.max(x, dim=1)  # (B, C)
        x = torch.sigmoid(x + x_cs)
        v1 '''   

        ''' v2
        x_cs,_ = torch.max(x_cs_classes, dim= 1) # ( B, C)
        x,_ = torch.max(x, dim=1)  # (B, C)
        x = torch.sigmoid(x + x_cs)
        v2 '''     

        ''' v3 '''

        x = x + x_cs_classes 
        x,_ = torch.max(x, dim=1)  # (B, C)
        x = torch.sigmoid(x)

        return x 

class ClassifierContextLateFusionMultiSoftGate_v2(nn.Module):
    def __init__(self, n_classes,  x_so_shape, x_cs_shape):
        super(ClassifierContextLateFusionMultiSoftGate_v2, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_context_emb   = 'dense_context_%d'
        self.layer_name_context_class = 'class_context_%d'
        self.layer_name_context_selection =   'imp_context'

        self.__init_layers(n_classes, x_so_shape, x_cs_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_so_shape, x_cs_shape):
        """
        Define model layers.
        """

        self.n_classes = 600

        n_units = 600
        n_channels = 512

        self.n_channels = n_channels

        C_so, N, H, W = x_so_shape
        self.C_so = C_so
        self.N = N

        self.feature_selection = context_fusion.ContextGatingClassifierSoft(x_so_shape, x_cs_shape)

        self.softmax = nn.Softmax(dim = 0)

        # Map so features to a smaller size
        self.dense_so = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_so, n_channels), nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))

        # Loop over existing context features: Map them into interaction categories
        for idx_context in range(self.n_contexts):
            C_c = x_cs_shape[idx_context][0]

            # embedding of multi_ context
            layer_name = self.layer_name_context_emb % (idx_context + 1)
            layer = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_c, n_channels), nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))

            setattr(self, layer_name, layer)

            # categories per context 

            layer_name = self.layer_name_context_class % (idx_context + 1)
            layer = nn.Sequential(nn.Dropout(0.25), nn.Linear(2*n_channels, n_channels), nn.BatchNorm1d(n_channels), nn.LeakyReLU(0.2), nn.Linear(n_channels, n_units))

            setattr(self, layer_name, layer)

        # spatial pooling
        self.spatial_pooling = pl.Max(dim=(3, 4))

        # layers for classification
        classifier_layers = []
        classifier_layers.append(nn.Dropout(0.25))
        classifier_layers.append(nn.Linear(n_channels, n_classes))
        self.classifier_layers = nn.Sequential(*classifier_layers)

    def __init_optimizer(self):
        """
        Define loss, metric and optimizer.
        """
        self._loss_fn = F.binary_cross_entropy
        self._metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
        self._optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-4)
        #self._optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9)

    def get_context_embeddings(self, x_cs, B):

        x_cs_embed = []

        # loop on multi_contexts
        for idx_context in range(self.n_contexts):
            # embedding of context
            x_c = x_cs[idx_context] # (B, C, 1,1,1)
            x_c = x_c.repeat(1, 1, self.N,1,1) # (B, C, N, 1, 1)

            layer = getattr(self, self.layer_name_context_emb % (idx_context + 1))
            x_c = layer(x_c)

            # append to list of context embeddings
            x_cs_embed.append(x_c.view(1, B, self.n_channels, self.N)) # (n_context, B, C, N)

        # process context features to get context embedding from x_cs features
        x_cs_embed = torch.stack(x_cs_embed, dim=0).view(-1, B, self.n_channels, self.N) # (n_context, B, C, N)
        return x_cs_embed

    def get_context_class(self, x_cs, x_so, B):

        x_cs_class = []

        # loop on multi_contexts
        for idx_context in range(self.n_contexts):
            # embedding of context
            x_c = x_cs[idx_context] # (B, C, N)

            x_c = x_c.permute(0, 2, 1)  # (B, N, C)

            # hide N dimension
            B, N, C = pytorch_utils.get_shape(x_c)
            x_c = x_c.contiguous().view(B * N, C)  # (B*N, C)

            x_c = torch.cat((x_so, x_c), dim=1)

            layer = getattr(self, self.layer_name_context_class % (idx_context + 1))
            x_c = layer(x_c)

            _, C = pytorch_utils.get_shape(x_c)
            x_c = x_c.view(B, N, C)  # (B, N, C)

            # append to list of context class predictions
            x_cs_class.append(x_c.view(1, B, self.N, self.n_classes)) # (1, B,N, C)

        # Process context features to get context category from x_cs features
        x_cs_class = torch.stack(x_cs_class, dim=0).view(-1, B, self.N, self.n_classes) # (n_context, B, N, C)
        return x_cs_class

    def get_context_relevance(self, x_so, x_cs):

        x_cs_value = []
        B, C, N, _,_ = pytorch_utils.get_shape(x_so)

        # loop on multi_contexts
        for idx_context in range(self.n_contexts):
            # embedding of context
            x_c = x_cs[idx_context]
            x_c = x_c.view(B,C, N,1,1) 

            x_c = self.feature_selection(x_so, x_c) # (B, N)
            x_cs_value.append(x_c.view(1, B, N)) # (1, B, C)

        x_cs_value = torch.stack(x_cs_value, dim=0).view(self.n_contexts, B, N) # (num_context, B, N)
        return x_cs_value

    def modulate_context_classifier(self, x_so, x_cs, x_cs_classes, B):

        # return context importance per-category
        x_cs_relevance = self.get_context_relevance(x_so, x_cs) # (num_context, B, N)

        # Reweigh class predictions with activated relevance scores
        x_cs_relevance = x_cs_relevance.view(self.n_contexts, B, self.N, 1) # 

        #x_cs_relevance = self.softmax(x_cs_relevance) # which context to use? over dim=0
        x_cs_relevance = torch.sigmoid(x_cs_relevance)

        # Modulate context classifiers with relevance scores
        x_cs_classes = x_cs_classes * x_cs_relevance # (nco, B, 12, 600)
        x_cs = torch.sum(x_cs_classes,dim=0) # (B, N, 600)

        return x_cs, x_cs_relevance

    def forward(self, *input):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        # return x_so embeddings
        x_so = input[0]
        x_so = self.dense_so(x_so)

        B, C, N, _,_ = pytorch_utils.get_shape(x_so)

        x_cs = input[1:]

        # return context embeddings
        x_c = self.get_context_embeddings(x_cs, B)

        x = x_so
        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)
        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x_action = x.contiguous().view(B * N, C)  # (B*N, C)

        # return context categories
        x_cs_classes = self.get_context_class(x_c, x_action, B) # (nco, B, N, C)

        x, _  = self.modulate_context_classifier(x_so, x_c, x_cs_classes, B) # (B, N, 600)
 
        # Add modulated response to human-object classifier and max-pool over N

        x,_ = torch.max(x, dim=1)  # (B, C)
        x = torch.sigmoid(x)

        return x 

class ClassifierContextLateFusionMultiSoftGate_v3(nn.Module):
    def __init__(self, n_classes,  x_so_shape, x_cs_shape):
        super(ClassifierContextLateFusionMultiSoftGate_v3, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_context_emb   = 'dense_context_%d'
        self.layer_name_context_class = 'class_context_%d'
        self.layer_name_context_selection =   'imp_context'

        self.__init_layers(n_classes, x_so_shape, x_cs_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_so_shape, x_cs_shape):
        """
        Define model layers.
        """

        self.n_classes = 600

        n_units = 600
        n_channels = 512

        self.n_channels = n_channels

        C_so, N, H, W = x_so_shape
        self.C_so = C_so
        self.N = N

        self.feature_selection = context_fusion.ContextGatingClassifierSoft(x_so_shape, x_cs_shape)

        self.softmax = nn.Softmax(dim = 0)

        # Map so features to a smaller size
        self.dense_so = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_so, n_channels), nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))

        # Loop over existing context features: Map them into interaction categories
        for idx_context in range(self.n_contexts):
            C_c = x_cs_shape[idx_context][0]

            # embedding of multi_ context
            layer_name = self.layer_name_context_emb % (idx_context + 1)
            layer = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_c, n_channels), nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))

            setattr(self, layer_name, layer)

        # spatial pooling
        self.spatial_pooling = pl.Max(dim=(3, 4))

        # layers for classification
        classifier_layers = []
        classifier_layers.append(nn.Dropout(0.25))
        classifier_layers.append(nn.Linear(2*n_channels, n_channels))
        classifier_layers.append(nn.BatchNorm1d(n_channels))
        classifier_layers.append(nn.LeakyReLU(0.2))
        classifier_layers.append(nn.Linear(n_channels, n_units))
        self.classifier_layers = nn.Sequential(*classifier_layers)

    def __init_optimizer(self):
        """
        Define loss, metric and optimizer.
        """
        self._loss_fn = F.binary_cross_entropy
        self._metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
        #self._optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-4)
        self._optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9)

    def get_context_embeddings(self, x_cs, B):

        x_cs_embed = []

        # loop on multi_contexts
        for idx_context in range(self.n_contexts):
            # embedding of context
            x_c = x_cs[idx_context] # (B, C, 1,1,1)
            x_c = x_c.repeat(1, 1, self.N,1,1) # (B, C, N, 1, 1)

            layer = getattr(self, self.layer_name_context_emb % (idx_context + 1))
            x_c = layer(x_c)

            # append to list of context embeddings
            x_cs_embed.append(x_c.view(1, B, self.n_channels, self.N)) # (n_context, B, C, N)

        # process context features to get context embedding from x_cs features
        x_cs_embed = torch.stack(x_cs_embed, dim=0).view(-1, B, self.n_channels, self.N) # (n_context, B, C, N)
        return x_cs_embed

    def get_context_class(self, x_cs, x_so, B):

        x_cs_class = []

        # loop on multi_contexts
        for idx_context in range(self.n_contexts):
            # embedding of context
            x_c = x_cs[idx_context] # (B, C, N)

            x_c = x_c.permute(0, 2, 1)  # (B, N, C)

            # hide N dimension
            B, N, C = pytorch_utils.get_shape(x_c)
            x_c = x_c.contiguous().view(B * N, C)  # (B*N, C)

            x_c = torch.cat((x_so, x_c), dim=1)

            layer = self.classifier_layers
            x_c = layer(x_c)

            _, C = pytorch_utils.get_shape(x_c)
            x_c = x_c.view(B, N, C)  # (B, N, C)

            # append to list of context class predictions
            x_cs_class.append(x_c.view(1, B, self.N, self.n_classes)) # (1, B,N, C)

        # Process context features to get context category from x_cs features
        x_cs_class = torch.stack(x_cs_class, dim=0).view(-1, B, self.N, self.n_classes) # (n_context, B, N, C)
        return x_cs_class

    def get_context_relevance(self, x_so, x_cs):

        x_cs_value = []
        B, C, N, _,_ = pytorch_utils.get_shape(x_so)

        # loop on multi_contexts
        for idx_context in range(self.n_contexts):
            # embedding of context
            x_c = x_cs[idx_context]
            x_c = x_c.view(B,C, N,1,1) 

            x_c = self.feature_selection(x_so, x_c) # (B, N)
            x_cs_value.append(x_c.view(1, B, N)) # (1, B, C)

        x_cs_value = torch.stack(x_cs_value, dim=0).view(self.n_contexts, B, N) # (num_context, B, N)
        return x_cs_value

    def modulate_context_classifier(self, x_so, x_cs, x_cs_classes, B):

        # return context importance per-category
        x_cs_relevance = self.get_context_relevance(x_so, x_cs) # (num_context, B, N)

        # Reweigh class predictions with activated relevance scores
        x_cs_relevance = x_cs_relevance.view(self.n_contexts, B, self.N, 1) # 

        #x_cs_relevance = self.softmax(x_cs_relevance) # which context to use? over dim=0
        x_cs_relevance = torch.sigmoid(x_cs_relevance)
        #x_cs_relevance = torch.tanh(x_cs_relevance)

        # Modulate context classifiers with relevance scores
        x_cs_classes = x_cs_classes * x_cs_relevance # (nco, B, 12, 600)
        x_cs = torch.sum(x_cs_classes,dim=0) # (B, N, 600)

        return x_cs, x_cs_relevance

    def forward(self, *input):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        # return x_so embeddings
        x_so = input[0]
        x_so = self.dense_so(x_so)

        B, C, N, _,_ = pytorch_utils.get_shape(x_so)

        x_cs = input[1:]

        # return context embeddings
        x_c = self.get_context_embeddings(x_cs, B)

        x = x_so
        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)
        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x_action = x.contiguous().view(B * N, C)  # (B*N, C)

        # return context categories
        x_cs_classes = self.get_context_class(x_c, x_action, B) # (nco, B, N, C)

        x, _  = self.modulate_context_classifier(x_so, x_c, x_cs_classes, B) # (B, N, 600)
 
        # Add modulated response to human-object classifier and max-pool over N

        x,_ = torch.max(x, dim=1)  # (B, C)
        x = torch.sigmoid(x)

        return x 

    def return_alphas(self, *input):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        # return x_so embeddings
        x_so = input[0]
        x_so = self.dense_so(x_so)

        B, C, N, _,_ = pytorch_utils.get_shape(x_so)

        x_cs = input[1:]

        # return context embeddings
        x_c = self.get_context_embeddings(x_cs, B)

        x = x_so
        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)
        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x_action = x.contiguous().view(B * N, C)  # (B*N, C)

        # return context categories
        x_cs_classes = self.get_context_class(x_c, x_action, B) # (nco, B, N, C)

        x, alphas  = self.modulate_context_classifier(x_so, x_c, x_cs_classes, B) # (B, N, 600)
 
        alphas = torch.mean(alphas,dim=2) # (nco, B, N)
        alphas = alphas.view(B, self.n_contexts)

        return alphas 

    def return_categories(self, *input):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        # return x_so embeddings
        x_so = input[0]
        x_so = self.dense_so(x_so)

        B, C, N, _,_ = pytorch_utils.get_shape(x_so)

        x_cs = input[1:]

        # return context embeddings
        x_c = self.get_context_embeddings(x_cs, B)

        x = x_so
        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)
        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x_action = x.contiguous().view(B * N, C)  # (B*N, C)

        # return context categories
        x_cs_classes = self.get_context_class(x_c, x_action, B) # (nco, B, N, C)

        x, alphas  = self.modulate_context_classifier(x_so, x_c, x_cs_classes, B) # (B, N, 600)
 
        # Add modulated response to human-object classifier and max-pool over N

        x,_ = torch.max(x, dim=1)  # (B, C)
        x = torch.sigmoid(x)

        return x 




class ClassifierContextLateFusionMultiHardGate(nn.Module):
    def __init__(self, n_classes,  x_so_shape, x_cs_shape):
        super(ClassifierContextLateFusionMultiHardGate, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_context_emb   = 'dense_context_%d'
        self.layer_name_context_class = 'class_context_%d'
        self.layer_name_context_selection =   'imp_context'

        self.__init_layers(n_classes, x_so_shape, x_cs_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_so_shape, x_cs_shape):
        """
        Define model layers.
        """

        self.n_classes = 600

        n_units = 600
        n_channels = 512

        self.n_channels = n_channels

        C_so, N, H, W = x_so_shape
        self.C_so = C_so
        self.N = N

        self.feature_selection = context_fusion.ContextGatingClassifierSoft(x_so_shape, x_cs_shape)

        self.softmax = nn.Softmax(dim = 0)

        # Map so features to a smaller size
        self.dense_so = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_so, n_channels), nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))

        # Loop over existing context features: Map them into interaction categories
        for idx_context in range(self.n_contexts):
            C_c = x_cs_shape[idx_context][0]

            # embedding of multi_ context
            layer_name = self.layer_name_context_emb % (idx_context + 1)
            layer = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_c, n_channels), nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))

            setattr(self, layer_name, layer)

        # spatial pooling
        self.spatial_pooling = pl.Max(dim=(3, 4))

        # layers for classification
        classifier_layers = []
        classifier_layers.append(nn.Dropout(0.25))
        classifier_layers.append(nn.Linear(2*n_channels, n_channels))
        classifier_layers.append(nn.BatchNorm1d(n_channels))
        classifier_layers.append(nn.LeakyReLU(0.2))
        classifier_layers.append(nn.Linear(n_channels, n_units))
        self.classifier_layers = nn.Sequential(*classifier_layers)

    def __init_optimizer(self):
        """
        Define loss, metric and optimizer.
        """
        self._loss_fn = F.binary_cross_entropy
        self._metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
        self._optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-4)
        #self._optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9)

    def get_context_embeddings(self, x_cs, B):

        x_cs_embed = []

        # loop on multi_contexts
        for idx_context in range(self.n_contexts):
            # embedding of context
            x_c = x_cs[idx_context] # (B, C, 1,1,1)
            x_c = x_c.repeat(1, 1, self.N,1,1) # (B, C, N, 1, 1)

            layer = getattr(self, self.layer_name_context_emb % (idx_context + 1))
            x_c = layer(x_c)

            # append to list of context embeddings
            x_cs_embed.append(x_c.view(1, B, self.n_channels, self.N)) # (n_context, B, C, N)

        # process context features to get context embedding from x_cs features
        x_cs_embed = torch.stack(x_cs_embed, dim=0).view(-1, B, self.n_channels, self.N) # (n_context, B, C, N)
        return x_cs_embed

    def get_context_class(self, x_cs, x_so, B):

        x_cs_class = []

        # loop on multi_contexts
        for idx_context in range(self.n_contexts):
            # embedding of context
            x_c = x_cs[idx_context] # (B, C, N)

            x_c = x_c.permute(0, 2, 1)  # (B, N, C)

            # hide N dimension
            B, N, C = pytorch_utils.get_shape(x_c)
            x_c = x_c.contiguous().view(B * N, C)  # (B*N, C)

            x_c = torch.cat((x_so, x_c), dim=1)

            layer = self.classifier_layers
            x_c = layer(x_c)

            _, C = pytorch_utils.get_shape(x_c)
            x_c = x_c.view(B, N, C)  # (B, N, C)

            # append to list of context class predictions
            x_cs_class.append(x_c.view(1, B, self.N, self.n_classes)) # (1, B,N, C)

        # Process context features to get context category from x_cs features
        x_cs_class = torch.stack(x_cs_class, dim=0).view(-1, B, self.N, self.n_classes) # (n_context, B, N, C)
        return x_cs_class

    def get_context_relevance(self, x_so, x_cs):

        x_cs_value = []
        B, C, N, _,_ = pytorch_utils.get_shape(x_so)

        # loop on multi_contexts
        for idx_context in range(self.n_contexts):
            # embedding of context
            x_c = x_cs[idx_context]
            x_c = x_c.view(B,C, N,1,1) 

            x_c = self.feature_selection(x_so, x_c) # (B, N)
            x_cs_value.append(x_c.view(1, B, N)) # (1, B, C)

        x_cs_value = torch.stack(x_cs_value, dim=0).view(self.n_contexts, B, N) # (num_context, B, N)
        return x_cs_value

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

    def apply_gumbel_softmax(self, logits):

        alpha = self.gumbel_softmax_sample(logits, 1)
        _, max_value_indexes = alpha.data.max(1, keepdim=True)
        alpha_hard = alpha.data.clone().zero_().scatter_(1, max_value_indexes, 1)
        alpha = Variable(alpha_hard - alpha.data) + alpha

        return alpha

    def modulate_context_classifier(self, x_so, x_cs, x_cs_classes, B):

        # return context importance per-category
        x_cs_relevance = self.get_context_relevance(x_so, x_cs) # (num_context, B, N)

        # Reweigh class predictions with activated relevance scores
        x_cs_relevance = x_cs_relevance.permute(1,2,0) # (B, N, n)
        x_cs_relevance = x_cs_relevance.view(B * self.N, self.n_contexts) #  (B * N, n)
        x_cs_relevance = self.apply_gumbel_softmax(x_cs_relevance) # (B * N, n)
        x_cs_relevance = x_cs_relevance.view(B , self.N, self.n_contexts) # (B, N, n) 
        x_cs_relevance = x_cs_relevance.permute(2, 0, 1) # (n, B, N)

        # Modulate context classifiers with relevance scores
        x_cs_relevance = x_cs_relevance.view(self.n_contexts, B, self.N, 1) # (4, 32, 12) (32,12,4) (384, 4) 
        x_cs_classes = x_cs_classes * x_cs_relevance # (nco, B, 12, 600)
        x_cs = torch.sum(x_cs_classes,dim=0) # (B, N, 600)

        return x_cs, x_cs_relevance

    def forward(self, *input):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        # return x_so embeddings
        x_so = input[0]
        x_so = self.dense_so(x_so)

        B, C, N, _,_ = pytorch_utils.get_shape(x_so)

        x_cs = input[1:]

        # return context embeddings
        x_c = self.get_context_embeddings(x_cs, B)

        x = x_so
        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)
        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x_action = x.contiguous().view(B * N, C)  # (B*N, C)

        # return context categories
        x_cs_classes = self.get_context_class(x_c, x_action, B) # (nco, B, N, C)

        x, x_cs_relevance  = self.modulate_context_classifier(x_so, x_c, x_cs_classes, B) # (B, N, 600)
 
        # Add modulated response to human-object classifier and max-pool over N

        x,_ = torch.max(x, dim=1)  # (B, C)
        x = torch.sigmoid(x)

        return x 

    def inference(self, *input):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        # return x_so embeddings
        x_so = input[0]
        x_so = self.dense_so(x_so)

        B, C, N, _,_ = pytorch_utils.get_shape(x_so)

        x_cs = input[1:]

        # return context embeddings
        x_c = self.get_context_embeddings(x_cs, B)

        x = x_so
        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)
        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x_action = x.contiguous().view(B * N, C)  # (B*N, C)

        # return context categories
        x_cs_classes = self.get_context_class(x_c, x_action, B) # (nco, B, N, C)

        x, x_cs_relevance  = self.modulate_context_classifier(x_so, x_c, x_cs_classes, B) # (B, N, 600)
 
        # Add modulated response to human-object classifier and max-pool over N

        x,_ = torch.max(x, dim=1)  # (B, C)
        x = torch.sigmoid(x)

        return x 

    def forward_for_alpha(self, *input):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        # return x_so embeddings
        x_so = input[0]
        x_so = self.dense_so(x_so)

        B, C, N, _,_ = pytorch_utils.get_shape(x_so)

        x_cs = input[1:]

        # return context embeddings
        x_c = self.get_context_embeddings(x_cs, B)

        x = x_so
        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)
        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x_action = x.contiguous().view(B * N, C)  # (B*N, C)

        # return context categories
        x_cs_classes = self.get_context_class(x_c, x_action, B) # (nco, B, N, C)

        x, x_cs_relevance  = self.modulate_context_classifier(x_so, x_c, x_cs_classes, B) # (B, N, 600)

        x_cs_relevance = x_cs_relevance.permute(1,0,2,3)
        x_cs_relevance = x_cs_relevance.view(B, self.n_contexts, N)

        return x_cs_relevance


class ClassifierContextLateFusionMultiHardGateAblated(nn.Module):
    def __init__(self, n_classes,  x_so_shape, x_cs_shape):
        super(ClassifierContextLateFusionMultiHardGateAblated, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_context_emb   = 'dense_context_%d'
        self.layer_name_context_class = 'class_context_%d'
        self.layer_name_context_selection =   'imp_context'

        self.__init_layers(n_classes, x_so_shape, x_cs_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_so_shape, x_cs_shape):
        """
        Define model layers.
        """

        self.n_classes = 600

        n_units = 600
        n_channels = 512

        self.n_channels = n_channels

        C_so, N, H, W = x_so_shape
        self.C_so = C_so
        self.N = N

        self.feature_selection = context_fusion.ContextGatingClassifierSoftAblated(x_so_shape, x_cs_shape)

        self.softmax = nn.Softmax(dim = 0)

        # Map so features to a smaller size
        self.dense_so = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_so, n_channels), nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))

        # Loop over existing context features: Map them into interaction categories
        for idx_context in range(self.n_contexts):
            C_c = x_cs_shape[idx_context][0]

            # embedding of multi_ context
            layer_name = self.layer_name_context_emb % (idx_context + 1)
            layer = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_c, n_channels), nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))

            setattr(self, layer_name, layer)

        # spatial pooling
        self.spatial_pooling = pl.Max(dim=(3, 4))

        # layers for classification
        classifier_layers = []
        classifier_layers.append(nn.Dropout(0.25))
        classifier_layers.append(nn.Linear(2*n_channels, n_channels))
        classifier_layers.append(nn.BatchNorm1d(n_channels))
        classifier_layers.append(nn.LeakyReLU(0.2))
        classifier_layers.append(nn.Linear(n_channels, n_units))
        self.classifier_layers = nn.Sequential(*classifier_layers)

    def __init_optimizer(self):
        """
        Define loss, metric and optimizer.
        """
        self._loss_fn = F.binary_cross_entropy
        self._metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
        self._optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-4)
        #self._optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9)

    def get_context_embeddings(self, x_cs, B):

        x_cs_embed = []

        # loop on multi_contexts
        for idx_context in range(self.n_contexts):
            # embedding of context
            x_c = x_cs[idx_context] # (B, C, 1,1,1)
            x_c = x_c.repeat(1, 1, self.N,1,1) # (B, C, N, 1, 1)

            layer = getattr(self, self.layer_name_context_emb % (idx_context + 1))
            x_c = layer(x_c)

            # append to list of context embeddings
            x_cs_embed.append(x_c.view(1, B, self.n_channels, self.N)) # (n_context, B, C, N)

        # process context features to get context embedding from x_cs features
        x_cs_embed = torch.stack(x_cs_embed, dim=0).view(-1, B, self.n_channels, self.N) # (n_context, B, C, N)
        return x_cs_embed

    def get_context_class(self, x_cs, x_so, B):

        x_cs_class = []

        # loop on multi_contexts
        for idx_context in range(self.n_contexts):
            # embedding of context
            x_c = x_cs[idx_context] # (B, C, N)

            x_c = x_c.permute(0, 2, 1)  # (B, N, C)

            # hide N dimension
            B, N, C = pytorch_utils.get_shape(x_c)
            x_c = x_c.contiguous().view(B * N, C)  # (B*N, C)

            x_c = torch.cat((x_so, x_c), dim=1)

            layer = self.classifier_layers
            x_c = layer(x_c)

            _, C = pytorch_utils.get_shape(x_c)
            x_c = x_c.view(B, N, C)  # (B, N, C)

            # append to list of context class predictions
            x_cs_class.append(x_c.view(1, B, self.N, self.n_classes)) # (1, B,N, C)

        # Process context features to get context category from x_cs features
        x_cs_class = torch.stack(x_cs_class, dim=0).view(-1, B, self.N, self.n_classes) # (n_context, B, N, C)
        return x_cs_class

    def get_context_relevance(self, x_so, x_cs):

        x_cs_value = []
        B, C, N, _,_ = pytorch_utils.get_shape(x_so)

        # loop on multi_contexts
        for idx_context in range(self.n_contexts):
            # embedding of context
            x_c = x_cs[idx_context]
            x_c = x_c.view(B,C, N,1,1) 

            x_c = self.feature_selection(x_so, x_c) # (B, N)
            x_cs_value.append(x_c.view(1, B, N)) # (1, B, C)

        x_cs_value = torch.stack(x_cs_value, dim=0).view(self.n_contexts, B, N) # (num_context, B, N)
        return x_cs_value

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

    def apply_gumbel_softmax(self, logits):

        alpha = self.gumbel_softmax_sample(logits, 1)
        _, max_value_indexes = alpha.data.max(1, keepdim=True)
        alpha_hard = alpha.data.clone().zero_().scatter_(1, max_value_indexes, 1)
        alpha = Variable(alpha_hard - alpha.data) + alpha

        return alpha

    def modulate_context_classifier(self, x_so, x_cs, x_cs_classes, B):

        # return context importance per-category
        x_cs_relevance = self.get_context_relevance(x_so, x_cs) # (num_context, B, N)

        # Reweigh class predictions with activated relevance scores
        x_cs_relevance = x_cs_relevance.permute(1,2,0) # (B, N, n)
        x_cs_relevance = x_cs_relevance.view(B * self.N, self.n_contexts) #  (B * N, n)
        x_cs_relevance = self.apply_gumbel_softmax(x_cs_relevance) # (B * N, n)
        x_cs_relevance = x_cs_relevance.view(B , self.N, self.n_contexts) # (B, N, n) 
        x_cs_relevance = x_cs_relevance.permute(2, 0, 1) # (n, B, N)

        # Modulate context classifiers with relevance scores
        x_cs_relevance = x_cs_relevance.view(self.n_contexts, B, self.N, 1) # (4, 32, 12) (32,12,4) (384, 4) 
        x_cs_classes = x_cs_classes * x_cs_relevance # (nco, B, 12, 600)
        x_cs = torch.sum(x_cs_classes,dim=0) # (B, N, 600)

        return x_cs, x_cs_relevance

    def forward(self, *input):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        # return x_so embeddings
        x_so = input[0]
        x_so = self.dense_so(x_so)

        B, C, N, _,_ = pytorch_utils.get_shape(x_so)

        x_cs = input[1:]

        # return context embeddings
        x_c = self.get_context_embeddings(x_cs, B)

        x = x_so
        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)
        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x_action = x.contiguous().view(B * N, C)  # (B*N, C)

        # return context categories
        x_cs_classes = self.get_context_class(x_c, x_action, B) # (nco, B, N, C)

        x, x_cs_relevance  = self.modulate_context_classifier(x_so, x_c, x_cs_classes, B) # (B, N, 600)
 
        # Add modulated response to human-object classifier and max-pool over N

        x,_ = torch.max(x, dim=1)  # (B, C)
        x = torch.sigmoid(x)

        return x 

    def inference(self, *input):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        # return x_so embeddings
        x_so = input[0]
        x_so = self.dense_so(x_so)

        B, C, N, _,_ = pytorch_utils.get_shape(x_so)

        x_cs = input[1:]

        # return context embeddings
        x_c = self.get_context_embeddings(x_cs, B)

        x = x_so
        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)
        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x_action = x.contiguous().view(B * N, C)  # (B*N, C)

        # return context categories
        x_cs_classes = self.get_context_class(x_c, x_action, B) # (nco, B, N, C)

        x, x_cs_relevance  = self.modulate_context_classifier(x_so, x_c, x_cs_classes, B) # (B, N, 600)
 
        # Add modulated response to human-object classifier and max-pool over N

        x,_ = torch.max(x, dim=1)  # (B, C)
        x = torch.sigmoid(x)

        return x 

    def forward_for_alpha(self, *input):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        # return x_so embeddings
        x_so = input[0]
        x_so = self.dense_so(x_so)

        B, C, N, _,_ = pytorch_utils.get_shape(x_so)

        x_cs = input[1:]

        # return context embeddings
        x_c = self.get_context_embeddings(x_cs, B)

        x = x_so
        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)
        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x_action = x.contiguous().view(B * N, C)  # (B*N, C)

        # return context categories
        x_cs_classes = self.get_context_class(x_c, x_action, B) # (nco, B, N, C)

        x, x_cs_relevance  = self.modulate_context_classifier(x_so, x_c, x_cs_classes, B) # (B, N, 600)

        x_cs_relevance = x_cs_relevance.permute(1,0,2,3)
        x_cs_relevance = x_cs_relevance.view(B, self.n_contexts, N)

        return x_cs_relevance



class ClassifierContextLateFusionMultiSoftGate_v4(nn.Module):
    def __init__(self, n_classes,  x_so_shape, x_cs_shape):
        super(ClassifierContextLateFusionMultiSoftGate_v4, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_context_emb   = 'dense_context_%d'
        self.layer_name_context_class = 'class_context_%d'
        self.layer_name_context_selection =   'imp_context'

        self.__init_layers(n_classes, x_so_shape, x_cs_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_so_shape, x_cs_shape):
        """
        Define model layers.
        """

        self.n_classes = 600

        n_units = 600
        n_channels = 512

        self.n_channels = n_channels

        C_so, N, H, W = x_so_shape
        self.C_so = C_so
        self.N = N

        self.feature_selection = context_fusion.ContextGatingClassifierSoft(x_so_shape, x_cs_shape)

        self.cbp_layer =  cbp(self.n_channels, self.n_channels,  2*self.n_channels).cuda()

        self.softmax = nn.Softmax(dim = 0)

        # Map so features to a smaller size
        self.dense_so = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_so, n_channels), nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))

        # Loop over existing context features: Map them into interaction categories
        for idx_context in range(self.n_contexts):
            C_c = x_cs_shape[idx_context][0]

            # embedding of multi_ context
            layer_name = self.layer_name_context_emb % (idx_context + 1)
            layer = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_c, n_channels), nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))

            setattr(self, layer_name, layer)

        # spatial pooling
        self.spatial_pooling = pl.Max(dim=(3, 4))

        # layers for classification
        classifier_layers = []
        classifier_layers.append(nn.Dropout(0.25))
        classifier_layers.append(nn.Linear(2*n_channels, n_channels))
        classifier_layers.append(nn.BatchNorm1d(n_channels))
        classifier_layers.append(nn.LeakyReLU(0.2))
        classifier_layers.append(nn.Linear(n_channels, n_units))
        self.classifier_layers = nn.Sequential(*classifier_layers)

    def __init_optimizer(self):
        """
        Define loss, metric and optimizer.
        """
        self._loss_fn = F.binary_cross_entropy
        self._metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
        #self._optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-4)
        self._optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9)

    def get_context_embeddings(self, x_cs, B):

        x_cs_embed = []

        # loop on multi_contexts
        for idx_context in range(self.n_contexts):
            # embedding of context
            x_c = x_cs[idx_context] # (B, C, 1,1,1)
            x_c = x_c.repeat(1, 1, self.N,1,1) # (B, C, N, 1, 1)

            layer = getattr(self, self.layer_name_context_emb % (idx_context + 1))
            x_c = layer(x_c)

            # append to list of context embeddings
            x_cs_embed.append(x_c.view(1, B, self.n_channels, self.N)) # (n_context, B, C, N)

        # process context features to get context embedding from x_cs features
        x_cs_embed = torch.stack(x_cs_embed, dim=0).view(-1, B, self.n_channels, self.N) # (n_context, B, C, N)
        return x_cs_embed

    def get_context_class(self, x_cs, x_so, B):

        x_cs_class = []

        # loop on multi_contexts
        for idx_context in range(self.n_contexts):
            # embedding of context
            x_c = x_cs[idx_context] # (B, C, N)

            x_c = x_c.permute(0, 2, 1)  # (B, N, C)

            # hide N dimension
            B, N, C = pytorch_utils.get_shape(x_c)
            x_c = x_c.contiguous().view(B * N, C)  # (B*N, C)

            #x_c = torch.cat((x_so, x_c), dim=1)

            # apply multi-linear pooling here
            x_c = self.cbp_layer(x_so, x_c)

            layer = self.classifier_layers
            x_c = layer(x_c)

            _, C = pytorch_utils.get_shape(x_c)
            x_c = x_c.view(B, N, C)  # (B, N, C)

            # append to list of context class predictions
            x_cs_class.append(x_c.view(1, B, self.N, self.n_classes)) # (1, B,N, C)

        # Process context features to get context category from x_cs features
        x_cs_class = torch.stack(x_cs_class, dim=0).view(-1, B, self.N, self.n_classes) # (n_context, B, N, C)
        return x_cs_class

    def get_context_relevance(self, x_so, x_cs):

        x_cs_value = []
        B, C, N, _,_ = pytorch_utils.get_shape(x_so)

        # loop on multi_contexts
        for idx_context in range(self.n_contexts):
            # embedding of context
            x_c = x_cs[idx_context]
            x_c = x_c.view(B,C, N,1,1) 

            x_c = self.feature_selection(x_so, x_c) # (B, N)
            x_cs_value.append(x_c.view(1, B, N)) # (1, B, C)

        x_cs_value = torch.stack(x_cs_value, dim=0).view(self.n_contexts, B, N) # (num_context, B, N)
        return x_cs_value

    def modulate_context_classifier(self, x_so, x_cs, x_cs_classes, B):

        # return context importance per-category
        x_cs_relevance = self.get_context_relevance(x_so, x_cs) # (num_context, B, N)

        # Reweigh class predictions with activated relevance scores
        x_cs_relevance = x_cs_relevance.view(self.n_contexts, B, self.N, 1) # 

        x_cs_relevance = torch.sigmoid(x_cs_relevance)

        # Modulate context classifiers with relevance scores
        x_cs_classes = x_cs_classes * x_cs_relevance # (nco, B, 12, 600)
        x_cs = torch.sum(x_cs_classes,dim=0) # (B, N, 600)

        return x_cs, x_cs_relevance

    def forward(self, *input):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        # return x_so embeddings
        x_so = input[0]
        x_so = self.dense_so(x_so)

        B, C, N, _,_ = pytorch_utils.get_shape(x_so)

        x_cs = input[1:]

        # return context embeddings
        x_c = self.get_context_embeddings(x_cs, B)

        x = x_so
        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)
        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x_action = x.contiguous().view(B * N, C)  # (B*N, C)

        # return context categories
        x_cs_classes = self.get_context_class(x_c, x_action, B) # (nco, B, N, C)

        x, _  = self.modulate_context_classifier(x_so, x_c, x_cs_classes, B) # (B, N, 600)
 
        # Add modulated response to human-object classifier and max-pool over N

        x,_ = torch.max(x, dim=1)  # (B, C)
        x = torch.sigmoid(x)

        return x 


class ClassifierContextLateFusionMultiHardGate_v1(nn.Module):
    def __init__(self, n_classes,  x_so_shape, x_cs_shape):
        super(ClassifierContextLateFusionMultiHardGate_v1, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_context_emb   = 'dense_context_%d'
        self.layer_name_context_class = 'class_context_%d'
        self.layer_name_context_selection =   'imp_context'

        self.__init_layers(n_classes, x_so_shape, x_cs_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_so_shape, x_cs_shape):
        """
        Define model layers.
        """

        self.n_classes = 600

        n_units = 600
        n_channels = 512

        self.n_channels = n_channels

        C_so, N, H, W = x_so_shape
        self.C_so = C_so
        self.N = N

        self.feature_selection = context_fusion.ContextGatingClassifierSoft(x_so_shape, x_cs_shape)

        self.softmax = nn.Softmax(dim = 0)

        # Map so features to a smaller size
        self.dense_so = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_so, n_channels), nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))

        # Loop over existing context features: Map them into interaction categories
        for idx_context in range(self.n_contexts):
            C_c = x_cs_shape[idx_context][0]

            # embedding of multi_ context
            layer_name = self.layer_name_context_emb % (idx_context + 1)
            layer = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_c, n_channels), nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))

            setattr(self, layer_name, layer)

            # categories per context 

            layer_name = self.layer_name_context_class % (idx_context + 1)
            layer = nn.Sequential(nn.Dropout(0.25), nn.Linear(n_channels, n_units))

            setattr(self, layer_name, layer)

        # spatial pooling
        self.spatial_pooling = pl.Max(dim=(3, 4))

        # layers for classification
        classifier_layers = []
        classifier_layers.append(nn.Dropout(0.25))
        classifier_layers.append(nn.Linear(n_channels, n_classes))
        self.classifier_layers = nn.Sequential(*classifier_layers)

    def __init_optimizer(self):
        """
        Define loss, metric and optimizer.
        """
        self._loss_fn = F.binary_cross_entropy
        self._metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
        self._optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-4)
        #self._optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9)

    def get_context_embeddings(self, x_cs, B):

        x_cs_embed = []

        # loop on multi_contexts
        for idx_context in range(self.n_contexts):
            # embedding of context
            x_c = x_cs[idx_context] # (B, C, 1,1,1)
            x_c = x_c.repeat(1, 1, self.N,1,1) # (B, C, N, 1, 1)

            layer = getattr(self, self.layer_name_context_emb % (idx_context + 1))
            x_c = layer(x_c)

            # append to list of context embeddings
            x_cs_embed.append(x_c.view(1, B, self.n_channels, self.N)) # (n_context, B, C, N)

        # process context features to get context embedding from x_cs features
        x_cs_embed = torch.stack(x_cs_embed, dim=0).view(-1, B, self.n_channels, self.N) # (n_context, B, C, N)
        return x_cs_embed

    def get_context_class(self, x_cs, B):

        x_cs_class = []

        # loop on multi_contexts
        for idx_context in range(self.n_contexts):
            # embedding of context
            x_c = x_cs[idx_context] # (B, C, N)

            x_c = x_c.permute(0, 2, 1)  # (B, N, C)

            # hide N dimension
            B, N, C = pytorch_utils.get_shape(x_c)
            x_c = x_c.contiguous().view(B * N, C)  # (B*N, C)

            layer = getattr(self, self.layer_name_context_class % (idx_context + 1))
            x_c = layer(x_c)

            _, C = pytorch_utils.get_shape(x_c)
            x_c = x_c.view(B, N, C)  # (B, N, C)

            # append to list of context class predictions
            x_cs_class.append(x_c.view(1, B, self.N, self.n_classes)) # (1, B,N, C)

        # Process context features to get context category from x_cs features
        x_cs_class = torch.stack(x_cs_class, dim=0).view(-1, B, self.N, self.n_classes) # (n_context, B, N, C)
        return x_cs_class

    def get_context_relevance(self, x_so, x_cs):

        x_cs_value = []
        B, C, N, _,_ = pytorch_utils.get_shape(x_so)

        # loop on multi_contexts
        for idx_context in range(self.n_contexts):
            # embedding of context
            x_c = x_cs[idx_context]
            x_c = x_c.view(B,C, N,1,1) 

            x_c = self.feature_selection(x_so, x_c) # (B, N)
            x_cs_value.append(x_c.view(1, B, N)) # (1, B, C)

        x_cs_value = torch.stack(x_cs_value, dim=0).view(self.n_contexts, B, N) # (num_context, B, N)
        return x_cs_value

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

    def apply_gumbel_softmax(self, logits):

        alpha = self.gumbel_softmax_sample(logits, 1)
        _, max_value_indexes = alpha.data.max(1, keepdim=True)
        alpha_hard = alpha.data.clone().zero_().scatter_(1, max_value_indexes, 1)
        alpha = Variable(alpha_hard - alpha.data) + alpha

        return alpha

    def modulate_context_classifier(self, x_so, x_cs, x_cs_classes, B):

        # return context importance per-category
        x_cs_relevance = self.get_context_relevance(x_so, x_cs) # (n, B, N)

        # Reweigh class predictions with activated relevance scores
        x_cs_relevance = x_cs_relevance.permute(1,2,0) # (B, N, n)
        x_cs_relevance = x_cs_relevance.view(B * self.N, self.n_contexts) #  (B * N, n)
        x_cs_relevance = self.apply_gumbel_softmax(x_cs_relevance) # (B * N, n)
        x_cs_relevance = x_cs_relevance.view(B , self.N, self.n_contexts) # (B, N, n) 
        x_cs_relevance = x_cs_relevance.permute(2, 0, 1) # (n, B, N)

        # Modulate context classifiers with relevance scores
        x_cs_relevance = x_cs_relevance.view(self.n_contexts, B, self.N, 1) # (4, 32, 12) (32,12,4) (384, 4) 
        x_cs_classes = x_cs_classes * x_cs_relevance # (nco, B, 12, 600)
        x_cs = torch.sum(x_cs_classes,dim=0) # (B, N, 600)

        return x_cs, x_cs_relevance

    def forward(self, *input):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        # return x_so embeddings
        x_so = input[0]
        x_so = self.dense_so(x_so)

        B, C, N, _,_ = pytorch_utils.get_shape(x_so)

        x_cs = input[1:]

        # return context embeddings
        x_c = self.get_context_embeddings(x_cs, B)

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

        # return context categories
        x_cs_classes = self.get_context_class(x_c, B) # (nco, B, N, C)

        x_cs_classes, x_cs_relevance  = self.modulate_context_classifier(x_so, x_c, x_cs_classes, B) # (B, N, 600)
 
        # Add modulated response to human-object classifier and max-pool over N

        ''' v1 
        x_cs,_ = torch.max(x_cs_classes, dim= 1) # ( B, C)
        x,_ = torch.max(x, dim=1)  # (B, C)
        x = torch.sigmoid(x + x_cs)
        '''

        ''' v2 
        x = x + x_cs_classes 
        x,_ = torch.max(x, dim=1)  # (B, C)
        x = torch.sigmoid(x)
        '''

        ''' v3 
        x,_ = torch.max(x, dim=1)  # (B, C)
        x = torch.sigmoid(x)   
        x_cs_classes,_ = torch.max(x_cs_classes, dim= 1) # ( B, C)
        x_cs_classes = torch.sigmoid(x_cs_classes)   
        x = x * x_cs_classes
        '''

        x = torch.sigmoid(x)   
        x_cs_classes = torch.sigmoid(x_cs_classes)   
        x = x * x_cs_classes
        x,_ = torch.max(x, dim=1)  # (B, C)


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
        C_so = x_so_shape[0]
        n_channels_so_out = C_so
        C_s = C_o = int(x_so_shape[0] / 2.0)
        x_s_shape = x_o_shape = (C_s, 1, 1, 1)

        n_heads = 4
        n_channels_cs_out = 512
        n_channels_inner = 32
        n_channels_dense = n_channels_so_out + n_channels_cs_out

        # interaction layer for context
        #self.context_interaction = context_fusion.ContextInteractionMultiHeadConcat(x_so_shape, x_cs_shape, n_channels_inner, n_channels_cs_out, n_heads)
        self.context_interaction = context_fusion.ContextInteractionBottleneckMultiHeadSum(x_so_shape, x_cs_shape, n_channels_inner, n_channels_cs_out, n_heads)
        self.context_activation = nn.Sequential(nn.BatchNorm3d(n_channels_cs_out), nn.LeakyReLU(0.2))

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
        #self._optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9)

    def forward(self, *inputs):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x_so = inputs[0]
        x_cs = inputs[1:]

        # context interaction
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

    def inference(self, *inputs):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        x_so = inputs[0]
        x_cs = inputs[1:]

        # context interaction
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

    def forward_for_alpha(self, *inputs):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        # get alpha
        alpha = self.context_interaction.forward_for_alpha(*inputs)  # (B, M, N, K)

        # (B, M, N, K)
        # batch_size, m_contexts, n_regions, k_heads

        return alpha

# endregion

class ClassifierContextInteractionAblation(nn.Module):
    def __init__(self, n_classes, x_so_shape, x_cs_shape):
        super(ClassifierContextInteractionAblation, self).__init__()

        self.__init_layers(n_classes, x_so_shape, x_cs_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_so_shape, x_cs_shape):
        """
        Define model layers.
        """

        n_units = 600
        C_so = x_so_shape[0]
        n_channels_so_out = C_so
        C_s = C_o = int(x_so_shape[0] / 2.0)
        x_s_shape = x_o_shape = (C_s, 1, 1, 1)

        n_heads = 4
        n_channels_cs_out = 512
        n_channels_inner = 32
        n_channels_dense = n_channels_so_out + n_channels_cs_out

        # interaction layer for context
        self.context_interaction = context_fusion.ContextInteractionAblationMultiHeadSum(x_cs_shape, n_channels_inner, n_channels_cs_out, n_heads)
        self.context_activation = nn.Sequential(nn.BatchNorm3d(n_channels_cs_out), nn.LeakyReLU(0.2))

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
        x_cs = inputs[1:]

        M = pytorch_utils.get_shape(x_so)[2]

        # context interaction
        x_cs = self.context_interaction(*x_cs)
        x_cs = self.context_activation(x_cs)
        x_cs = x_cs.repeat(1, 1, M, 1, 1)

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