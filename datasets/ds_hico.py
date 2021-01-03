#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import natsort
import time
import numpy as np
import scipy.io as sio

import torch
import torchsummary
from torch import nn

from nets import resnet_torch, resnet_places_torch

from core import utils, image_utils, pytorch_utils, data_utils, configs
from core.utils import Path as Pth
import cv2

# region Const
N_CLASSES = 600

# endregion

# region Prepare Dataset

def _101_prepare_annotation():
    num_output = N_CLASSES

    annot_mat_path = Pth('Hico/annotation/anno_hico.mat')
    annot_pickle_path = Pth('Hico/annotation/anno_hico.pkl')

    annot_mat_path = Pth('Vcoco/annotation/anno_vcoco.mat')
    annot_pickle_path = Pth('Vcoco/annotation/anno_vcoco.pkl')

    annot = sio.loadmat(annot_mat_path)
    annot = annot['dataset']

    list_tr = annot['train_names'][0][0]
    list_te = annot['val_names'][0][0]

    n_tr = len(list_tr)
    n_te = len(list_te)

    # prepare list of images
    x_tr = np.array([annot['train_names'][0][0][i][0][0] for i in range(n_tr)])
    x_te = np.array([annot['val_names'][0][0][i][0][0] for i in range(n_te)])

    y_tr = np.array([annot['anno_split_train'][0][0][:, i] for i in range(n_tr)])
    y_te = np.array([annot['anno_split_val'][0][0][:, i] for i in range(n_te)])

    y_tr_mask = np.array([annot['label_mask_train'][0][0][:, i] for i in range(n_tr)])
    y_te_mask = np.array([annot['label_mask_val'][0][0][:, i] for i in range(n_te)])

    print(x_tr.shape)
    print(x_te.shape)

    print(y_tr.shape)
    print(y_te.shape)
    
    print(y_tr_mask.shape)
    print(y_te_mask.shape)

    utils.pkl_dump((x_tr, y_tr, y_tr_mask, x_te, y_te, y_te_mask), annot_pickle_path)

def _101_pickle_vanilla_rcnn_predictions():

    annot_path = Pth('Hico/features/h5/anno_hico.pkl')
    x_attn_root_path = '/var/scratch/mkilicka/code/a_closer_look/data/hico/exp1_results/rcnn/'
    features_attn_path = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/hico/results/rcnn.h5'

    (img_names_tr, y_tr ,_, img_names_te, y_te, _) = utils.pkl_load(annot_path)
    n_tr = len(img_names_tr)
    n_te = len(img_names_te)

    print('Loading features')
    x_attn_te = np.zeros((n_te, 600), dtype=np.float32)


    for index, name in enumerate(img_names_te):
        utils.print_counter(index, n_te, freq=100)
        name_no_extn = utils.remove_extension(name)

        temp = sio.loadmat(x_attn_root_path + name_no_extn)
        temp = temp['response']
        x_attn_te[index] = temp

    print(x_attn_te.shape)

    print(y_te.shape)

    utils.h5_dump(x_attn_te, 'x_te', features_attn_path)

def _101_prepare_annotation_vcoco():
    num_output = N_CLASSES

    annot_mat_path = Pth('Vcoco/annotation/anno_vcoco.mat')
    annot_pickle_path = Pth('Vcoco/annotation/anno_vcoco.pkl')

    annot = sio.loadmat(annot_mat_path)
    annot = annot['dataset']

    list_tr = annot['train_names'][0][0]
    list_te = annot['val_names'][0][0]

    n_tr = len(list_tr)
    n_te = len(list_te)

    # prepare list of images
    x_tr = np.array([annot['train_names'][0][0][i][0][0] for i in range(n_tr)])
    x_te = np.array([annot['val_names'][0][0][i][0][0] for i in range(n_te)])

    y_tr = np.array([annot['anno_split_train'][0][0][:, i] for i in range(n_tr)])
    y_te = np.array([annot['anno_split_val'][0][0][:, i] for i in range(n_te)])

    print(x_tr.shape)
    print(x_te.shape)

    print(y_tr.shape)
    print(y_te.shape)
    
    utils.pkl_dump((x_tr, y_tr, x_te, y_te), annot_pickle_path)

def _101_prepare_annotation_cint():
    num_output = N_CLASSES

    annot_mat_path = Pth('Cint/annotation/anno_cint.mat')
    annot_pickle_path = Pth('Cint/annotation/anno_cint.pkl')

    annot = sio.loadmat(annot_mat_path)
    annot = annot['dataset']

    list_tr = annot['train_names'][0][0]
    list_te = annot['val_names'][0][0]

    n_tr = len(list_tr)
    n_te = len(list_te)

    # prepare list of images
    x_tr = np.array([annot['train_names'][0][0][i][0][0] for i in range(n_tr)])
    x_te = np.array([annot['val_names'][0][0][i][0][0] for i in range(n_te)])

    y_tr = np.array([annot['anno_split_train'][0][0][:, i] for i in range(n_tr)])
    y_te = np.array([annot['anno_split_val'][0][0][:, i] for i in range(n_te)])

    print(x_tr.shape)
    print(x_te.shape)

    print(y_tr.shape)
    print(y_te.shape)
    
    utils.pkl_dump((x_tr, y_tr, x_te, y_te), annot_pickle_path)

# endregion

# region Prepare Features

def _201_pickle_features_pairattn():
    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    x_attn_root_path = Pth('Hico/features_pairattn/')
    features_attn_path = Pth('Hico/features/features_pairattn.h5')

    x_attn_root_path = '/home/mkilicka/scratch/code/context-driven-interactions/context_features/pairatt/'
    features_attn_path = Pth('Hico/features/features_pairattn_pose.h5')

    annot_path = Pth('Vcoco/annotation/anno_vcoco.pkl')
    x_attn_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/vcoco/raw_features/pairatt/'
    features_attn_path = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/vcoco/h5/features_pairattn.h5'

    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    n_tr = len(img_names_tr)
    n_te = len(img_names_te)

    print('Loading features')
    x_attn_tr = np.zeros((n_tr, 3, 4096), dtype=np.float32)
    x_attn_te = np.zeros((n_te, 3, 4096), dtype=np.float32)

    for index, name in enumerate(img_names_tr):
        utils.print_counter(index, n_tr, freq=100)
        name_no_extn = utils.remove_extension(name)

        temp = sio.loadmat(x_attn_root_path + name_no_extn)
        temp = temp['feature_local']
        x_attn_tr[index] = temp

    for index, name in enumerate(img_names_te):
        utils.print_counter(index, n_te, freq=100)
        name_no_extn = utils.remove_extension(name)

        temp = sio.loadmat(x_attn_root_path + name_no_extn)
        temp = temp['feature_local']
        x_attn_te[index] = temp

    print(x_attn_tr.shape)
    print(x_attn_te.shape)

    print(y_tr.shape)
    print(y_te.shape)

    utils.h5_dump_multi((x_attn_tr, x_attn_te), ['x_tr', 'x_te'], features_attn_path)



def _201_pickle_features_pairattn_vcoco():

    annot_path = Pth('Vcoco/annotation/anno_vcoco.pkl')
    x_attn_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/vcoco/raw_features/pairatt/'
    features_attn_path = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/vcoco/h5/features_pairattn.h5'

    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    n_tr = len(img_names_tr)
    n_te = len(img_names_te)

    print('Loading features')
    x_attn_te = np.zeros((n_te, 3, 4096), dtype=np.float32)

    for index, name in enumerate(img_names_te):
        utils.print_counter(index, n_te, freq=100)
        name_no_extn = utils.remove_extension(name)

        temp = sio.loadmat(x_attn_root_path + name_no_extn)
        temp = temp['feature']
        x_attn_te[index] = temp

    print(x_attn_te.shape)

    print(y_te.shape)

    utils.h5_dump(x_attn_te, 'x_te', features_attn_path)

def _201_pickle_features_pairattn_cint():

    annot_path = Pth('Cint/annotation/anno_cint.pkl')
    x_attn_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/cint/pairatt/'
    features_attn_path = Pth('Cint/features/h5/features_pairattn.h5')

    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    n_tr = len(img_names_tr)
    n_te = len(img_names_te)

    print('Loading features')
    x_attn_te = np.zeros((n_te, 3, 4096), dtype=np.float32)

    for index, name in enumerate(img_names_te):
        utils.print_counter(index, n_te, freq=100)
        name_no_extn = utils.remove_extension(name)

        temp = sio.loadmat(x_attn_root_path + name_no_extn)
        temp = temp['feature']
        x_attn_te[index] = temp

    print(x_attn_te.shape)

    print(y_te.shape)

    utils.h5_dump(x_attn_te, 'x_te', features_attn_path)

def _201_pickle_predictions_pairattn_cint():

    annot_path = Pth('Cint/annotation/anno_cint.pkl')
    x_attn_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/cint/pairatt/'
    features_attn_path = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/cint/results/vanilla_pairatt.h5'

    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    n_tr = len(img_names_tr)
    n_te = len(img_names_te)

    print('Loading features')
    x_attn_te = np.zeros((n_te, 600), dtype=np.float32)

    for index, name in enumerate(img_names_te):
        utils.print_counter(index, n_te, freq=100)
        name_no_extn = utils.remove_extension(name)

        temp = sio.loadmat(x_attn_root_path + name_no_extn)
        temp = temp['prediction'].reshape(600)
        x_attn_te[index] = temp

    print(x_attn_te.shape)

    print(y_te.shape)

    utils.h5_dump(x_attn_te, 'x_te', features_attn_path)

def _201_pickle_features_contfus():
    annot_path = Pth('Hico/features/h5/anno_hico.pkl')
    x_attn_root_path = Pth('Hico/features_pairattn/')
    features_attn_path = Pth('Hico/features/features_pairattn.h5')

    x_attn_root_path = '/var/scratch/mkilicka/code/a_closer_look/improvement/contextfusion/'
    features_attn_path = Pth('Hico/features/h5/features_contextfusion.h5')

    (img_names_tr, y_tr,_ ,img_names_te, y_te, _) = utils.pkl_load(annot_path)
    n_tr = len(img_names_tr)
    n_te = len(img_names_te)

    print('Loading features')
    x_attn_tr = np.zeros((n_tr, 3, 4096), dtype=np.float32)
    x_attn_te = np.zeros((n_te, 3, 4096), dtype=np.float32)

    for index, name in enumerate(img_names_tr):
        utils.print_counter(index, n_tr, freq=100)
        name_no_extn = utils.remove_extension(name)

        temp = sio.loadmat(x_attn_root_path + name_no_extn)
        temp = temp['feature_normal']
        x_attn_tr[index] = temp

    for index, name in enumerate(img_names_te):
        utils.print_counter(index, n_te, freq=100)
        name_no_extn = utils.remove_extension(name)

        temp = sio.loadmat(x_attn_root_path + name_no_extn)
        temp = temp['feature_normal']
        x_attn_te[index] = temp

    print(x_attn_tr.shape)
    print(x_attn_te.shape)

    print(y_tr.shape)
    print(y_te.shape)

    utils.h5_dump_multi((x_attn_tr, x_attn_te), ['x_tr', 'x_te'], features_attn_path)


def _201_pickle_features_contfus_vcoco():

    annot_path = Pth('Vcoco/annotation/anno_vcoco.pkl')
    x_attn_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/vcoco/raw_features/contextfusion/'
    features_attn_path = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/vcoco/h5/features_contextfusion.h5'

    (img_names_tr, y_tr ,img_names_te, y_te) = utils.pkl_load(annot_path)
    n_tr = len(img_names_tr)
    n_te = len(img_names_te)

    print('Loading features')
    x_attn_te = np.zeros((n_te, 3, 4096), dtype=np.float32)


    for index, name in enumerate(img_names_te):
        utils.print_counter(index, n_te, freq=100)
        name_no_extn = utils.remove_extension(name)

        temp = sio.loadmat(x_attn_root_path + name_no_extn)
        temp = temp['feature']
        x_attn_te[index] = temp

    print(x_attn_te.shape)

    print(y_te.shape)

    utils.h5_dump(x_attn_te, 'x_te', features_attn_path)

def _201_pickle_features_contfus_cint():

    annot_path = Pth('Cint/annotation/anno_cint.pkl')
    x_attn_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/cint/contextfusion/'
    features_attn_path = Pth('Cint/features/h5/features_contextfusion.h5')

    (img_names_tr, y_tr ,img_names_te, y_te) = utils.pkl_load(annot_path)
    n_tr = len(img_names_tr)
    n_te = len(img_names_te)

    print('Loading features')
    x_attn_te = np.zeros((n_te, 3, 4096), dtype=np.float32)


    for index, name in enumerate(img_names_te):
        utils.print_counter(index, n_te, freq=100)
        name_no_extn = utils.remove_extension(name)

        temp = sio.loadmat(x_attn_root_path + name_no_extn)
        temp = temp['feature']
        x_attn_te[index] = temp

    print(x_attn_te.shape)

    print(y_te.shape)

    utils.h5_dump(x_attn_te, 'x_te', features_attn_path)

def _202_pickle_features_context():
    annot_path = Pth('Hico/annotation/anno_hico.pkl')

    x_root_path = Pth('Hico/features_scene/')
    features_context_path = Pth('Hico/features/features_scene.h5')
    n_channels = 512

    x_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/experiments/exp_scene_feat_early_fusion_v9/features/hico/'
    features_context_path = Pth('Hico/features/features_scene_early_fusion.h5')
    n_channels = 3072

    x_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/context_features/local_scene/'
    features_context_path = Pth('Hico/features/features_local_scene.h5')
    n_channels = 2048

    x_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/context_features/local_object/'
    features_context_path = Pth('Hico/features/features_local_object.h5')
    n_channels = 6144

    x_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/context_features/part_states/'
    features_context_path = Pth('Hico/features/features_part_states.h5')
    n_channels = 86

    x_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/context_features/lvis/'
    features_context_path = Pth('Hico/features/features_lvis.h5')
    n_channels = 1300

    x_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/context_features/imsitu/'
    features_context_path = Pth('Hico/features/features_imsitu.h5')
    n_channels = 504

    x_root_path = '/var/scratch/mkilicka/code/context-aware-interaction/main/hico/scene/'
    features_context_path = Pth('Hico/features/features_scene_att.h5')
    n_channels = 102

    x_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/context_features/imsitu_role/'
    features_context_path = Pth('Hico/features/features_imsitu_role.h5')
    n_channels = 1980


    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    n_tr = len(img_names_tr)
    n_te = len(img_names_te)

    print('Loading features')
    x_scene_tr = np.zeros((n_tr, n_channels, 1, 1, 1), dtype=np.float32)
    x_scene_te = np.zeros((n_te, n_channels, 1, 1, 1), dtype=np.float32)

    for index, name in enumerate(img_names_tr):
        utils.print_counter(index, n_tr, freq=100)
        name_no_extn = utils.remove_extension(name)
        temp = sio.loadmat(x_root_path + name_no_extn)
        temp = temp['feature'].squeeze().reshape(n_channels, 1, 1, 1)
        x_scene_tr[index] = temp

    for index, name in enumerate(img_names_te):
        utils.print_counter(index, n_te, freq=100)
        name_no_extn = utils.remove_extension(name)
        temp = sio.loadmat(x_root_path + name_no_extn)
        temp = temp['feature'].squeeze().reshape(n_channels, 1, 1, 1)

        x_scene_te[index] = temp

    print(x_scene_tr.shape)
    print(x_scene_te.shape)

    print(y_tr.shape)
    print(y_te.shape)

    utils.h5_dump_multi((x_scene_tr, x_scene_te), ['x_tr', 'x_te'], features_context_path)

def _202_pickle_features_lvis_cint():
    x_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/vcoco/raw_features/lvis/'
    features_path = Pth('Cint/features/h5/extra/features_lvis.h5')
    n_channels = 1300

    annot_path = Pth('Cint/annotation/anno_cint.pkl')

    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    n_tr = len(img_names_tr)
    n_te = len(img_names_te)

    x_te = np.zeros((n_te, n_channels, 1, 1, 1), dtype=np.float32)

    for index, name in enumerate(img_names_te):
        utils.print_counter(index, n_te, freq=100)
        name_no_extn = utils.remove_extension(name)
        temp = sio.loadmat(x_root_path + name_no_extn)
        temp = temp['feature'].squeeze().reshape(n_channels, 1, 1, 1)

        x_te[index] = temp

    utils.h5_dump(x_te, 'x_te', features_path)


def _202_pickle_features_lvis_cint():
    x_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/cint/lvis/'
    features_path = Pth('Cint/features/h5/features_lvis.h5')
    n_channels = 1300

    annot_path = Pth('Cint/annotation/anno_cint.pkl')

    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    n_tr = len(img_names_tr)
    n_te = len(img_names_te)

    x_te = np.zeros((n_te, n_channels, 1, 1, 1), dtype=np.float32)

    for index, name in enumerate(img_names_te):
        utils.print_counter(index, n_te, freq=100)
        name_no_extn = utils.remove_extension(name)
        temp = sio.loadmat(x_root_path + name_no_extn)
        temp = temp['feature'].squeeze().reshape(n_channels, 1, 1, 1)

        x_te[index] = temp

    utils.h5_dump(x_te, 'x_te', features_path)

def _202_pickle_features_vgg_vcoco():
    x_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/vcoco/raw_features/resnet/'
    features_path = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/vcoco/h5/resnet.h5'
    n_channels = 2048

    annot_path = Pth('Vcoco/annotation/anno_vcoco.pkl')

    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    n_tr = len(img_names_tr)
    n_te = len(img_names_te)

    x_te = np.zeros((n_te, n_channels, 1, 1, 1), dtype=np.float32)

    for index, name in enumerate(img_names_te):
        utils.print_counter(index, n_te, freq=100)
        name_no_extn = utils.remove_extension(name)
        temp = sio.loadmat(x_root_path + name_no_extn)
        temp = temp['feature'].squeeze().reshape(n_channels, 1, 1, 1)

        x_te[index] = temp

    utils.h5_dump(x_te, 'x_te', features_path)

def _202_pickle_features_vgg_cint():
    x_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/cint/resnet/'
    features_path = Pth('Cint/features/h5/resnet.h5')
    n_channels = 2048

    annot_path = Pth('Cint/annotation/anno_cint.pkl')

    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    n_tr = len(img_names_tr)
    n_te = len(img_names_te)

    x_te = np.zeros((n_te, n_channels, 1, 1, 1), dtype=np.float32)

    for index, name in enumerate(img_names_te):
        utils.print_counter(index, n_te, freq=100)
        name_no_extn = utils.remove_extension(name)
        temp = sio.loadmat(x_root_path + name_no_extn)
        temp = temp['feature'].squeeze().reshape(n_channels, 1, 1, 1)

        x_te[index] = temp

    utils.h5_dump(x_te, 'x_te', features_path)

def _203_pickle_features_subject_object():

    annot_path = Pth('Hico/annotation/anno_hico.pkl')

    features_path = Pth('Hico/features/features_subject_object.h5')
    x_root_path = '/var/scratch/mkilicka/code/context-aware-interaction/context_shifter_human_object_selected/features/hico_selected/'

    x_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/context_features/human_object_locality/'
    features_path = Pth('Hico/features/features_local_locality.h5')

    annot_path = Pth('Vcoco/annotation/anno_vcoco.pkl')
    x_root_path = Pth('/var/scratch/mkilicka/code/context-aware-interaction/main/vcoco/features/human_object')
    features_path = Pth('Vcoco/features/features_subject_object.h5')

    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    n_tr = len(img_names_tr)
    n_te = len(img_names_te)
    n_regions = 12
    n_channels = 4096

    print('Loading features')
    x_tr = np.zeros((n_tr, n_regions, n_channels, 1, 1), dtype=np.float32)
    x_te = np.zeros((n_te, n_regions, n_channels, 1, 1), dtype=np.float32)

    for index, name in enumerate(img_names_tr):
        utils.print_counter(index, n_tr, freq=100)
        name_no_extn = utils.remove_extension(name)
        feat_path = x_root_path + name_no_extn + '.mat'
        feat = __read_feature(feat_path, n_channels, n_regions)
        x_tr[index] = feat

    for index, name in enumerate(img_names_te):
        utils.print_counter(index, n_te, freq=100)
        name_no_extn = utils.remove_extension(name)
        feat_path = x_root_path + name_no_extn + '.mat'
        feat = __read_feature(feat_path, n_channels, n_regions)
        x_te[index] = feat

    print(x_tr.shape)
    print(x_te.shape)
    print(y_tr.shape)
    print(y_te.shape)
    utils.h5_dump_multi((x_tr, x_te), ['x_tr', 'x_te'], features_path)

    print(np.shape)


def _203_pickle_features_subject_object_vcoco():

    annot_path = Pth('Hico/annotation/anno_hico.pkl')

    features_path = Pth('Hico/features/features_subject_object.h5')
    x_root_path = '/var/scratch/mkilicka/code/context-aware-interaction/context_shifter_human_object_selected/features/hico_selected/'

    x_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/context_features/human_object_locality/'
    features_path = Pth('Hico/features/features_local_locality.h5')

    annot_path = Pth('Vcoco/annotation/anno_vcoco.pkl')
    x_root_path = Pth('/var/scratch/mkilicka/code/context-aware-interaction/main/vcoco/features/human_object/')
    features_path = Pth('Vcoco/features/features_subject_object.h5')

    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    n_tr = len(img_names_tr)
    n_te = len(img_names_te)
    n_regions = 12
    n_channels = 4096

    print('Loading features')
    x_te = np.zeros((n_te, n_regions, n_channels, 1, 1), dtype=np.float32)

    for index, name in enumerate(img_names_te):
        utils.print_counter(index, n_te, freq=100)
        name_no_extn = utils.remove_extension(name)
        feat_path = x_root_path + name_no_extn + '.mat'
        feat = __read_feature(feat_path, n_channels, n_regions)
        x_te[index] = feat

    print(x_te.shape)
    print(y_te.shape)
    utils.h5_dump(x_te, 'x_te', features_path)

def _203_pickle_features_subject_object_cint():

    annot_path = Pth('Cint/annotation/anno_cint.pkl')
    x_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/cint/features/human_object/'
    features_path = Pth('Cint/features/features_subject_object.h5')

    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    n_tr = len(img_names_tr)
    n_te = len(img_names_te)
    n_regions = 12
    n_channels = 4096

    print('Loading features')
    x_te = np.zeros((n_te, n_regions, n_channels, 1, 1), dtype=np.float32)

    for index, name in enumerate(img_names_te):
        utils.print_counter(index, n_te, freq=100)
        name_no_extn = utils.remove_extension(name)
        feat_path = x_root_path + name_no_extn + '.mat'
        feat = __read_feature(feat_path, n_channels, n_regions)
        x_te[index] = feat

    print(x_te.shape)
    print(y_te.shape)
    utils.h5_dump(x_te, 'x_te', features_path)

def _203_pickle_features_subject_object_locality():
    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    features_path = Pth('Hico/features/features_subject_object.h5')
    x_root_path = '/var/scratch/mkilicka/code/context-aware-interaction/context_shifter_human_object_selected/features/hico_selected/'

    x_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/context_features/human_object_locality/'
    features_path = Pth('Hico/features/extra/features_local_locality.h5')

    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    n_tr = len(img_names_tr)
    n_te = len(img_names_te)
    n_regions = 12
    n_channels = 4096*2

    print('Loading features')
    x_tr = np.zeros((n_tr, n_regions, n_channels, 1, 1), dtype=np.float32)
    x_te = np.zeros((n_te, n_regions, n_channels, 1, 1), dtype=np.float32)

    for index, name in enumerate(img_names_tr):
        utils.print_counter(index, n_tr, freq=100)
        name_no_extn = utils.remove_extension(name)
        feat_path = x_root_path + name_no_extn + '.mat'
        feat = __read_feature_locality(feat_path, n_channels, n_regions)
        x_tr[index] = feat

    for index, name in enumerate(img_names_te):
        utils.print_counter(index, n_te, freq=100)
        name_no_extn = utils.remove_extension(name)
        feat_path = x_root_path + name_no_extn + '.mat'
        feat = __read_feature_locality(feat_path, n_channels, n_regions)
        x_te[index] = feat

    print(x_tr.shape)
    print(x_te.shape)
    print(y_tr.shape)
    print(y_te.shape)
    utils.h5_dump_multi((x_tr, x_te), ['x_tr', 'x_te'], features_path)

    print(np.shape)


def _203_pickle_features_local_scene():
    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    features_path = Pth('Hico/features/extra/features_local_scene.h5')
    x_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/submission/annotation/hico/features/local_scene/v3/'

    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    n_tr = len(img_names_tr)
    n_te = len(img_names_te)
    n_regions = 6
    n_channels = 2048

    print('Loading features')
    x_tr = np.zeros((n_tr, n_regions , n_channels, 1, 1), dtype=np.float32)
    x_te = np.zeros((n_te, n_regions , n_channels, 1, 1),dtype=np.float32)

    for index, name in enumerate(img_names_tr):
        utils.print_counter(index, n_tr, freq=100)
        name_no_extn = utils.remove_extension(name)
        feat_path = x_root_path + name_no_extn + '.mat'
        feat = __read_feature_segmentation(feat_path, n_channels, n_regions)
        x_tr[index] = feat

    for index, name in enumerate(img_names_te):
        utils.print_counter(index, n_te, freq=100)
        name_no_extn = utils.remove_extension(name)
        feat_path = x_root_path + name_no_extn + '.mat'
        feat = __read_feature_segmentation(feat_path, n_channels, n_regions)
        x_te[index] = feat

    print(x_tr.shape)
    print(x_te.shape)
    print(y_tr.shape)
    print(y_te.shape)
    utils.h5_dump_multi((x_tr, x_te), ['x_tr', 'x_te'], features_path)



def _203_pickle_features_local_scene_vcoco():
    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    features_path = Pth('Hico/features/extra/features_local_scene.h5')
    x_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/submission/annotation/hico/features/local_scene/v3/'

    annot_path = Pth('Vcoco/annotation/anno_vcoco.pkl')
    features_path = Pth('Vcoco/features/extra/features_local_scene.h5')
    x_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/vcoco/raw_features/local_scene/'

    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    n_te = len(img_names_te)
    n_regions = 6
    n_channels = 2048

    print('Loading features')
    x_te = np.zeros((n_te, n_regions , n_channels, 1, 1),dtype=np.float32)

    for index, name in enumerate(img_names_te):
        utils.print_counter(index, n_te, freq=100)
        name_no_extn = utils.remove_extension(name)
        feat_path = x_root_path + name_no_extn + '.mat'
        feat = __read_feature_segmentation(feat_path, n_channels, n_regions)
        x_te[index] = feat

    print(x_te.shape)
    print(y_te.shape)
    utils.h5_dump(x_te, 'x_te', features_path)

def _203_pickle_features_local_scene_cint():

    annot_path = Pth('Cint/annotation/anno_cint.pkl')
    features_path = Pth('Cint/features/extra/features_local_scene.h5')
    x_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/cint/local_scene/'

    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    n_te = len(img_names_te)
    n_regions = 6
    n_channels = 2048

    print('Loading features')
    x_te = np.zeros((n_te, n_regions , n_channels, 1, 1),dtype=np.float32)

    for index, name in enumerate(img_names_te):
        utils.print_counter(index, n_te, freq=100)
        name_no_extn = utils.remove_extension(name)
        feat_path = x_root_path + name_no_extn + '.mat'
        feat = __read_feature_segmentation(feat_path, n_channels, n_regions)
        x_te[index] = feat

    print(x_te.shape)
    print(y_te.shape)
    utils.h5_dump(x_te, 'x_te', features_path)

def _203_pickle_features_deformation():
    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    features_path = Pth('Hico/features/extra/features_deformation.h5')
    x_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/main/hico/objectmask/'

    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    n_tr = len(img_names_tr)
    n_te = len(img_names_te)
    n_regions = 1
    n_channels = 80

    print('Loading features')
    x_tr = np.zeros((n_tr, n_regions , n_channels, 32, 32), dtype=np.float32)
    x_te = np.zeros((n_te, n_regions , n_channels, 32, 32),dtype=np.float32)

    for index, name in enumerate(img_names_tr):
        utils.print_counter(index, n_tr, freq=100)
        name_no_extn = utils.remove_extension(name)
        feat_path = x_root_path + name_no_extn + '.mat'
        feat = sio.loadmat(feat_path)['mask']
        feat = np.swapaxes(feat, 0,3).reshape(80, 32, 32)
        x_tr[index] = feat

    for index, name in enumerate(img_names_te):
        utils.print_counter(index, n_te, freq=100)
        name_no_extn = utils.remove_extension(name)
        feat_path = x_root_path + name_no_extn + '.mat'
        feat = sio.loadmat(feat_path)['mask']
        feat = np.swapaxes(feat, 0,3).reshape(80, 32, 32)
        x_te[index] = feat

    print(x_tr.shape)
    print(x_te.shape)
    print(y_tr.shape)
    print(y_te.shape)
    utils.h5_dump_multi((x_tr, x_te), ['x_tr', 'x_te'], features_path)

def _203_pickle_features_deformation_vcoco():

    annot_path = Pth('Vcoco/annotation/anno_vcoco.pkl')
    features_path = Pth('Vcoco/features/extra/features_deformation.h5')
    x_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/vcoco/raw_features/objectmask/'

    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    n_te = len(img_names_te)
    n_regions = 1
    n_channels = 80

    print('Loading features')
    x_te = np.zeros((n_te, n_regions , n_channels, 32, 32),dtype=np.float32)

    for index, name in enumerate(img_names_te):
        utils.print_counter(index, n_te, freq=100)
        name_no_extn = utils.remove_extension(name)
        feat_path = x_root_path + name_no_extn + '.mat'
        feat = sio.loadmat(feat_path)['mask']
        feat = np.swapaxes(feat, 0,3).reshape(80, 32, 32)
        x_te[index] = feat

    print(x_te.shape)
    print(y_te.shape)
    utils.h5_dump(x_te, 'x_te', features_path)

def _203_pickle_features_deformation_cint():

    annot_path = Pth('Cint/annotation/anno_cint.pkl')
    features_path = Pth('Cint/features/extra/features_deformation.h5')
    x_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/cint/objectmask/'

    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    n_te = len(img_names_te)
    n_regions = 1
    n_channels = 80

    print('Loading features')
    x_te = np.zeros((n_te, n_regions , n_channels, 32, 32),dtype=np.float32)

    for index, name in enumerate(img_names_te):
        utils.print_counter(index, n_te, freq=100)
        name_no_extn = utils.remove_extension(name)
        feat_path = x_root_path + name_no_extn + '.mat'
        feat = sio.loadmat(feat_path)['mask']
        feat = np.swapaxes(feat, 0,3).reshape(80, 32, 32)
        x_te[index] = feat

    print(x_te.shape)
    print(y_te.shape)
    utils.h5_dump(x_te, 'x_te', features_path)

def _203_pickle_features_pose():
    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    features_path = Pth('Hico/features/extra/features_open_pose.h5')
    x_root_path_1 = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/hico/aux/open_pose/'

    x_root_path_2 = '/var/scratch/mkilicka/code/context-driven-interactions/main/hico/objectmask/'

    (img_names_tr, y_tr,_, img_names_te, y_te, _) = utils.pkl_load(annot_path)
    n_tr = len(img_names_tr)
    n_te = len(img_names_te)
    n_regions = 1
    n_channels = 81

    regionsize = 32

    print('Loading features')
    x_tr = np.zeros((n_tr, n_regions , n_channels, regionsize, regionsize), dtype=np.float32)
    x_te = np.zeros((n_te, n_regions , n_channels, regionsize, regionsize),dtype=np.float32)

    for index, name in enumerate(img_names_tr):
        utils.print_counter(index, n_tr, freq=100)
        name_no_extn = utils.remove_extension(name)
        feat_path = x_root_path_1 + name_no_extn + '.mat'
        feat = sio.loadmat(feat_path)['pose']
        feat1 = preprocess_open_pose(feat, regionsize)

        feat_path = x_root_path_2 + name_no_extn + '.mat'
        feat = sio.loadmat(feat_path)['mask']
        feat2 = np.swapaxes(feat, 0,3).reshape(80, 32, 32)

        feat = np.concatenate((feat1, feat2), 0)

        x_tr[index] = feat

    for index, name in enumerate(img_names_te):
        utils.print_counter(index, n_te, freq=100)
        name_no_extn = utils.remove_extension(name)

        feat_path = x_root_path_1 + name_no_extn + '.mat'
        feat = sio.loadmat(feat_path)['pose']
        feat1 = preprocess_open_pose(feat, regionsize)

        feat_path = x_root_path_2 + name_no_extn + '.mat'
        feat = sio.loadmat(feat_path)['mask']
        feat2 = np.swapaxes(feat, 0,3).reshape(80, 32, 32)

        feat = np.concatenate((feat1, feat2), 0)
        x_te[index] = feat

    print(x_tr.shape)
    print(x_te.shape)
    print(y_tr.shape)
    print(y_te.shape)
    utils.h5_dump_multi((x_tr, x_te), ['x_tr', 'x_te'], features_path)

def _203_pickle_features_semanticseg():
    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    features_path = Pth('Hico/features/extra/features_semantic_seg.h5')
    x_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/main/hico/semanticseg/'

    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    n_tr = len(img_names_tr)
    n_te = len(img_names_te)
    n_regions = 1
    n_channels = 150

    print('Loading features')
    x_tr = np.zeros((n_tr, n_regions , n_channels, 14, 14), dtype=np.float32)
    x_te = np.zeros((n_te, n_regions , n_channels, 14, 14),dtype=np.float32)

    for index, name in enumerate(img_names_tr):
        utils.print_counter(index, n_tr, freq=100)
        name_no_extn = utils.remove_extension(name)
        feat_path = x_root_path + name_no_extn + '.mat'
        feat = sio.loadmat(feat_path)['mask']
        feat = np.swapaxes(feat, 0,3).reshape(n_channels, 14, 14)
        x_tr[index] = feat

    for index, name in enumerate(img_names_te):
        utils.print_counter(index, n_te, freq=100)
        name_no_extn = utils.remove_extension(name)
        feat_path = x_root_path + name_no_extn + '.mat'
        feat = sio.loadmat(feat_path)['mask']
        feat = np.swapaxes(feat, 0,3).reshape(n_channels, 14, 14)
        x_te[index] = feat

    print(x_tr.shape)
    print(x_te.shape)
    print(y_tr.shape)
    print(y_te.shape)
    utils.h5_dump_multi((x_tr, x_te), ['x_tr', 'x_te'], features_path)


def _203_pickle_features_cocostuff():
    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    features_path = Pth('Hico/features/extra/features_coco_stuff.h5')
    x_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/hico/aux/cocostuff/'

    (img_names_tr, y_tr, _, img_names_te, y_te, _) = utils.pkl_load(annot_path)
    n_tr = len(img_names_tr)
    n_te = len(img_names_te)
    n_regions = 1
    n_channels = 182

    print('Loading features')
    x_tr = np.zeros((n_tr, n_regions , n_channels, 1, 1), dtype=np.float32)
    x_te = np.zeros((n_te, n_regions , n_channels, 1, 1),dtype=np.float32)

    for index, name in enumerate(img_names_tr):
        utils.print_counter(index, n_tr, freq=100)
        name_no_extn = utils.remove_extension(name)
        feat_path = x_root_path + name_no_extn + '.mat'
        feat = sio.loadmat(feat_path)['image']
        feat = get_cocostuff(feat)
        x_tr[index] = feat.reshape(1, n_channels,1,1)

    for index, name in enumerate(img_names_te):
        utils.print_counter(index, n_te, freq=100)
        name_no_extn = utils.remove_extension(name)
        feat_path = x_root_path + name_no_extn + '.mat'
        feat = sio.loadmat(feat_path)['image']
        feat = get_cocostuff(feat)
        x_te[index] = feat.reshape(1, n_channels,1,1)

    print(x_tr.shape)
    print(x_te.shape)
    print(y_tr.shape)
    print(y_te.shape)
    utils.h5_dump_multi((x_tr, x_te), ['x_tr', 'x_te'], features_path)

def _203_pickle_features_scene():
    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    features_path = Pth('Hico/features/extra/features_scene.h5')
    x_root_path = '../../context-aware-interaction/main/hico/scene/'

    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    n_tr = len(img_names_tr)
    n_te = len(img_names_te)
    n_regions = 1
    n_channels = 512

    print('Loading features')
    x_tr = np.zeros((n_tr, n_regions , n_channels, 14, 14), dtype=np.float32)
    x_te = np.zeros((n_te, n_regions , n_channels, 14, 14),dtype=np.float32)

    for index, name in enumerate(img_names_tr):
        utils.print_counter(index, n_tr, freq=100)
        name_no_extn = utils.remove_extension(name)
        feat_path = x_root_path + name_no_extn + '.mat'
        feat = sio.loadmat(feat_path)['feature']
        feat = np.swapaxes(feat, 0,2)
        x_tr[index] = feat

    for index, name in enumerate(img_names_te):
        utils.print_counter(index, n_te, freq=100)
        name_no_extn = utils.remove_extension(name)
        feat_path = x_root_path + name_no_extn + '.mat'
        feat = sio.loadmat(feat_path)['feature']
        feat = np.swapaxes(feat, 0,2)
        x_te[index] = feat

    print(x_tr.shape)
    print(x_te.shape)
    print(y_tr.shape)
    print(y_te.shape)
    utils.h5_dump_multi((x_tr, x_te), ['x_tr', 'x_te'], features_path)

def _203_pickle_features_part_states():
    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    features_path = Pth('Hico/features/features_part_states.h5')
    x_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/context_features/part_states/'

    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    n_tr = len(img_names_tr)
    n_te = len(img_names_te)
    n_regions = 12
    n_channels = 86

    print('Loading features')
    x_tr = np.zeros((n_tr, n_regions , n_channels, 1, 1), dtype=np.float32)
    x_te = np.zeros((n_te, n_regions , n_channels, 1, 1),dtype=np.float32)

    for index, name in enumerate(img_names_tr):
        utils.print_counter(index, n_tr, freq=100)
        name_no_extn = utils.remove_extension(name)
        feat_path = x_root_path + name_no_extn + '.mat'
        feat = __read_feature(feat_path, n_channels, n_regions)
        x_tr[index] = feat

    for index, name in enumerate(img_names_te):
        utils.print_counter(index, n_te, freq=100)
        name_no_extn = utils.remove_extension(name)
        feat_path = x_root_path + name_no_extn + '.mat'
        feat = __read_feature(feat_path, n_channels, n_regions)
        x_te[index] = feat

    print(x_tr.shape)
    print(x_te.shape)
    print(y_tr.shape)
    print(y_te.shape)
    utils.h5_dump_multi((x_tr, x_te), ['x_tr', 'x_te'], features_path)

def _202_pickle_features_human_pose():
    annot_path = Pth('Hico/annotation/anno_hico.pkl')

    x_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/context_features/pose_feats/'
    features_context_path = Pth('Hico/features/features_global_pose.h5')
    n_channels = 256

    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    n_tr = len(img_names_tr)
    n_te = len(img_names_te)

    print('Loading features')
    x_scene_tr = np.zeros((n_tr, n_channels, 1, 1, 1), dtype=np.float32)
    x_scene_te = np.zeros((n_te, n_channels, 1, 1, 1), dtype=np.float32)

    for index, name in enumerate(img_names_tr):
        utils.print_counter(index, n_tr, freq=100)
        name_no_extn = utils.remove_extension(name)
        temp = sio.loadmat(x_root_path + name_no_extn)
        temp = temp['features'].reshape(-1, 7*7, 256)
        temp = np.mean(temp, 1)
        temp = np.mean(temp, 0)

        temp = temp.reshape(n_channels, 1, 1, 1)
        #temp = temp['feature'].squeeze().reshape(n_channels, 1, 1, 1)
        x_scene_tr[index] = temp

    for index, name in enumerate(img_names_te):
        utils.print_counter(index, n_te, freq=100)
        name_no_extn = utils.remove_extension(name)
        temp = sio.loadmat(x_root_path + name_no_extn)
        temp = temp['features'].reshape(-1, 7*7, 256)
        temp = np.mean(temp, 1)
        temp = np.mean(temp, 0)

        temp = temp.reshape(n_channels, 1, 1, 1)
        temp = np.mean(temp, 1).reshape(n_channels, 1, 1, 1)
        x_scene_te[index] = temp

    print(x_scene_tr.shape)
    print(x_scene_te.shape)

    print(y_tr.shape)
    print(y_te.shape)

    utils.h5_dump_multi((x_scene_tr, x_scene_te), ['x_tr', 'x_te'], features_context_path)

def _204_pickle_features_images():
    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    features_path = Pth('Hico/features/features_images.h5')
    x_root_path = '/var/scratch/mkilicka/code/context-driven-interactions/context_features/vanilla_resnet/'

    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    n_tr = len(img_names_tr)
    n_te = len(img_names_te)
    n_regions = 1
    n_channels = 2048

    print('Loading features')
    x_tr = np.zeros((n_tr, n_channels, n_regions, 1, 1), dtype=np.float32)
    x_te = np.zeros((n_te, n_channels, n_regions, 1, 1), dtype=np.float32)

    for index, name in enumerate(img_names_tr):
        utils.print_counter(index, n_tr, freq=100)
        name_no_extn = utils.remove_extension(name)
        feat_path = x_root_path + name_no_extn + '.mat'
        feat = sio.loadmat(feat_path)['feature'].squeeze().reshape((n_channels, n_regions, 1, 1))
        x_tr[index] = feat

    for index, name in enumerate(img_names_te):
        utils.print_counter(index, n_te, freq=100)
        name_no_extn = utils.remove_extension(name)
        feat_path = x_root_path + name_no_extn + '.mat'
        feat = sio.loadmat(feat_path)['feature'].squeeze().reshape((n_channels, n_regions, 1, 1))
        x_te[index] = feat

    print(x_tr.shape)
    print(x_te.shape)
    print(y_tr.shape)
    print(y_te.shape)
    utils.h5_dump_multi((x_tr, x_te), ['x_tr', 'x_te'], features_path)

    print(np.shape)

def get_cocostuff(image):

    unique_segments = np.unique(image)
    imsize = np.float32(image.shape[0] * image.shape[1])

    feat = np.zeros((182), dtype = np.float32)

    for seg in unique_segments:
        mask = image == seg
        num_pixels = np.count_nonzero(mask)

        size = num_pixels / imsize
        feat[seg] = size

    return feat



def preprocess_open_pose(feat, regionsize):

    image = cv2.resize(feat, (regionsize, regionsize), interpolation = cv2.INTER_NEAREST)


    image = np.sum(image, 2) 
    image = (image > 128).astype(np.float32)

    image = image.reshape(1, regionsize, regionsize)


    return image

def __read_feature(feat_path, n_channels, n_regions):
    feat = sio.loadmat(feat_path)
    feat = feat['feature']
    feat = __sample_n_region_features(feat, n_regions)
    feat = feat.reshape((n_regions, n_channels, 1, 1))
    return feat

def __read_feature_segmentation(feat_path, n_channels, n_regions):
    feat = sio.loadmat(feat_path)
    feat = feat['feature']
    feat = __sample_n_region_features_segmentation(feat, n_regions)
    feat = feat.reshape((n_regions, n_channels, 1, 1))
    return feat

def __read_feature_locality(feat_path, n_channels, n_regions):
    feat = sio.loadmat(feat_path)

    feat = np.concatenate((feat['f_interior'], feat['f_exterior']) , 1)
    feat = __sample_n_region_features(feat, n_regions)
    feat = feat.reshape((n_regions, n_channels, 1, 1))
    return feat

def __sample_n_region_features_segmentation(features, n_regions):
    n = features.shape[0]
    if n == 1:
        idx = [0] * n_regions
        features = features[idx]
    elif n < n_regions:
        n_samples = n_regions - n
        idx = [0] * n_samples
        sampled_features = features[idx]
        features = np.vstack((features, sampled_features))    
    elif n == n_regions:
        pass
    elif n>n_regions: # subsample
        idx = np.random.randint(low=0, high=n, size = n_regions)
        features = features[idx]
    else:
        raise Exception('Sorry, should not be happening!')
    return features

def __sample_n_region_features(features, n_regions):
    n = features.shape[0]
    if n == 1:
        idx = [0] * n_regions
        features = features[idx]
    elif n < n_regions:
        n_samples = n_regions - n
        idx = [0] * n_samples
        sampled_features = features[idx]
        features = np.vstack((features, sampled_features))    
    elif n == n_regions:
        pass
    else:
        raise Exception('Sorry, should not be happening!')
    return features

def __sample_n_region_features_legacy(features, n_regions):
    n = features.shape[0]
    if n == 1:
        idx = [0] * n_regions
        features = features[idx]
    elif n < n_regions:
        n_samples = n_regions - n
        idx = np.random.randint(low=0, high=n, size=(n_samples,))
        samples_features = features[idx]
        features = np.vstack((features, samples_features))
    elif n == n_regions:
        pass
    else:
        raise Exception('Sorry, should not be happening!')
    return features

# endregion

# region Fine-tune CNNs

def _801_finetune_resnet50_imagenet():
    is_local = configs.is_local_machine()
    if is_local:
        n_gpus = 1
        batch_size_tr = 32
        batch_size_te = 32
        img_root_path = Pth('Hico/images')
        pass
    else:
        n_gpus = 1
        batch_size_tr = 32
        batch_size_te = 32
        img_root_path = '/local/nhussein/Hico/images'

    n_epochs = 100
    n_threads = 32
    batch_size_tr = batch_size_tr * n_gpus
    batch_size_te = batch_size_te * n_gpus

    model_name = 'resnet50_hico_%s' % (utils.timestamp(),)
    model_root_path = Pth('Hico/models_finetuned/%s', (model_name))

    print('--- start time %s' % utils.timestamp())
    print('... model %s' % model_name)

    # load model
    model, optimizer, loss_fn, metric_fn = resnet_torch.__get_resne50_for_finetuning_on_hico()

    # parallelize model
    model = model if is_local else nn.DataParallel(model)

    # print summary
    input_size = (3, 224, 224)  # (B, C, T, H, W)
    torchsummary.summary(model, input_size)

    # sample image pathes
    pathes_sampler = data_utils.SamplersImagePathesBreakfast(img_root_path)

    # image reader
    async_reader = image_utils.AsyncImageReaderHicoResNetTorch(pytorch_utils.RGB_MEAN, pytorch_utils.RGB_STD, n_threads)

    # callbacks
    model_save_cb = pytorch_utils.ModelSaveCallback(model, model_root_path)

    # train model
    pytorch_utils.train_model_using_async_reader_custom_metric(model, optimizer, loss_fn, metric_fn, async_reader, pathes_sampler, n_epochs, batch_size_tr, batch_size_te, callbacks=[model_save_cb])

    print('--- end time %s' % utils.timestamp())

def _802_finetune_resnet50_places_365():
    is_local = configs.is_local_machine()
    if is_local:
        # n_gpus = 1
        # batch_size_tr = 32
        # batch_size_te = 32
        # img_root_path = Pth('Hico/images')
        pass
    else:
        n_gpus = 1
        batch_size_tr = 32
        batch_size_te = 32
        img_root_path = '/local/nhussein/Hico/images'

    n_epochs = 100
    n_threads = 32
    batch_size_tr = batch_size_tr * n_gpus
    batch_size_te = batch_size_te * n_gpus

    model_name = 'resnet50_places_hico_%s' % (utils.timestamp(),)
    model_root_path = Pth('Hico/models_finetuned/%s', (model_name))

    print('--- start time %s' % utils.timestamp())
    print('... model %s' % model_name)

    # load model
    model, optimizer, loss_fn, metric_fn = resnet_places_torch.__get_resne50_for_finetuning_on_hico()

    # parallelize model
    model = model if is_local else nn.DataParallel(model)

    # print summary
    input_size = (3, 224, 224)  # (B, C, T, H, W)
    torchsummary.summary(model, input_size)

    # sample image pathes
    pathes_sampler = data_utils.SamplersImagePathesBreakfast(img_root_path)

    # image reader
    async_reader = image_utils.AsyncImageReaderHicoResNetTorch(resnet_places_torch.RGB_MEAN, resnet_places_torch.RGB_STD, n_threads)

    # callbacks
    model_save_cb = pytorch_utils.ModelSaveCallback(model, model_root_path)

    # train model
    pytorch_utils.train_model_using_async_reader_custom_metric(model, optimizer, loss_fn, metric_fn, async_reader, pathes_sampler, n_epochs, batch_size_tr, batch_size_te, callbacks=[model_save_cb])

    print('--- end time %s' % utils.timestamp())

# endregion

# region Extract Features

def _302_extract_features_resnet_152():
    """
    Extract frames from each video. Extract only 1 frame for each spf seconds.
    :param spf: How many seconds for each sampled frames.
    :return:
    """

    annot_path = Pth('Hico/annotation/anno_hico.pkl')
    root_frame_pathes_tr = Pth('Hico/images_train')
    root_frame_pathes_te = Pth('Hico/images_test')
    model_weights = Pth('Torch_Models/ResNet/resnet152-b121ed2d.pth')
    features_path = Pth('/HICO/features/feature_image.h5')

    (img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
    img_names = np.hstack((img_names_tr, img_names_te))
    n_tr = len(img_names_tr)
    n_te = len(img_names_te)
    n_imgs = len(img_names)
    C, L = 2048, 1
    batch_size = 200

    x_tr = np.zeros((n_tr, C, L, L), dtype=np.float32)
    x_te = np.zeros((n_te, C, L, L), dtype=np.float32)
    print(x_tr.shape)
    print(x_te.shape)
    n_threads = 20

    # initialize the model
    model = resnet_torch.__get_resne50_breakfast_for_testing(model_weights)

    # extract features
    x_tr = __extract_features_from_list(model, img_names_tr, root_frame_pathes_tr, batch_size)
    x_te = __extract_features_from_list(model, img_names_te, root_frame_pathes_te, batch_size)

    # finally, save the features
    utils.h5_dump_multi((x_tr, x_te), ['x_tr', 'x_te'], features_path)

def _303_extract_features_from_resnet_18():

    import os 

    img_names_tr = os.listdir('/var/scratch/mkilicka/data/mscoco/train2017/')
    img_names_te = os.listdir('/var/scratch/mkilicka/data/mscoco/val2017/')

    image_path_tr = '/var/scratch/mkilicka/data/mscoco/train2017/'
    image_path_te = '/var/scratch/mkilicka/data/mscoco/val2017/'

    n_tr = len(img_names_tr)
    n_te = len(img_names_te)
    img_names = np.hstack((img_names_tr, img_names_te))

    n_imgs = len(img_names)
    C, L = 512, 1
    batch_size = 400

    x_tr = np.zeros((n_tr, C, L, L), dtype=np.float32)
    x_te = np.zeros((n_te, C, L, L), dtype=np.float32)
    print(x_tr.shape)
    print(x_te.shape)
    n_threads = 20

    model = resnet_torch.__get_resnet18_for_feature_extraction()

    # extract features
    x_tr = __extract_features_from_list_(model, img_names_tr, image_path_tr, batch_size)
    x_te = __extract_features_from_list_(model, img_names_te, image_path_te, batch_size)

    print(x_tr.shape)
    print(x_te.shape)


def __extract_features_from_list_(model, images, root_path, batch_size):
    n_images = len(images)
    n_batches = utils.calc_num_batches(n_images, batch_size)

    batch_idx = 0
    idx_start = batch_idx * batch_size
    idx_end = (batch_idx + 1) * batch_size

    all_features = np.zeros((n_images, 512), dtype = np.float32)

    # aync reader, and get load images for the first video
    #f_pathes = np.array([Pth('Breakfast/frames/%s/%s', (video_ids[0], n)) for n in video_frames_dict[video_ids[0]]])

    f_pathes = np.array([ root_path + name for name in images[idx_start:idx_end] ])

    img_reader = image_utils.AsyncImageReaderHicoResNetTorch(pytorch_utils.RGB_MEAN, pytorch_utils.RGB_STD, n_threads=20)

    # loop on videos
    for batch_idx in range(n_batches):

        idx_start = batch_idx * batch_size
        idx_end = (batch_idx + 1) * batch_size

        batch_num = batch_idx + 1

        next_f_pathes =  np.array([ root_path + name for name in images[idx_start:idx_end] ])
        img_reader.load_batch(next_f_pathes)

        # wait untill the image_batch is loaded
        t1 = time.time()
        while img_reader.is_busy():
            time.sleep(0.1)
        t2 = time.time()
        duration_waited = t2 - t1
        print('...... batch %d/%d:, waited: %d, index %d/%d' % (batch_num, n_batches, duration_waited, idx_start, idx_end))

        # get the video frames
        image_frames = img_reader.get_batch()  # (B, H, W, C)

        # channel first for pytorch
        #video_frames = np.transpose(video_frames, (0, 3, 1, 2))  # (B, C, H, W)

        # extract features
        features = pytorch_utils.batched_feedforward(model, image_frames, batch_size, 'extract_features')


        # cast as float_16 for saving space and computation
        features = features.astype(np.float16)

        print(features.shape)
        print(next_f_pathes.shape)

        all_features[idx_start:idx_end] = features.squeeze()

    return all_features

def __extract_features_from_list(model, images, root_path, batch_size):
    n_images = len(images)
    n_batches = utils.calc_num_batches(n_images, batch_size)

    # aync reader, and get load images for the first video
    f_pathes = np.array([Pth('Breakfast/frames/%s/%s', (video_ids[0], n)) for n in video_frames_dict[video_ids[0]]])
    img_reader = image_utils.AsyncImageReaderHicoResNetTorch(pytorch_utils.RGB_MEAN, pytorch_utils.RGB_STD, n_threads=n_threads)
    img_reader.load_batch(f_pathes)

    # loop on videos
    for batch_idx in range(n_batches):

        batch_num = batch_idx + 1

        # wait untill the image_batch is loaded
        t1 = time.time()
        while img_reader.is_busy():
            time.sleep(0.1)
        t2 = time.time()
        duration_waited = t2 - t1
        print('...... batch %d/%d:, waited: %d' % (batch_num, n_batches, duration_waited))

        # get the video frames
        video_frames = img_reader.get_batch()  # (B, H, W, C)

        # channel first for pytorch
        video_frames = np.transpose(video_frames, (0, 3, 1, 2))  # (B, C, H, W)

        # pre-load for the next video
        if batch_num < n_images:
            next_f_pathes = np.array([Pth('Breakfast/frames/%s/%s', (video_ids[idx_img + 1], n)) for n in video_frames_dict[video_ids[idx_img + 1]]])
            img_reader.load_batch(next_f_pathes)

        # extract features
        features = pytorch_utils.batched_feedforward(model, video_frames, batch_size, 'forward_conv_maxpool')

        # channel first
        features = np.transpose(features, (1, 0, 2, 3))

        # cast as float_16 for saving space and computation
        features = features.astype(np.float16)

        all_features[idx] = features

    return all_features


def __extract_features_from_list(model, images, root_path, batch_size):
    n_images = len(images)
    n_batches = utils.calc_num_batches(n_images, batch_size)

    # aync reader, and get load images for the first video
    f_pathes = np.array([Pth('Breakfast/frames/%s/%s', (video_ids[0], n)) for n in video_frames_dict[video_ids[0]]])
    img_reader = image_utils.AsyncImageReaderHicoResNetTorch(pytorch_utils.RGB_MEAN, pytorch_utils.RGB_STD, n_threads=n_threads)
    img_reader.load_batch(f_pathes)

    # loop on videos
    for batch_idx in range(n_batches):

        batch_num = batch_idx + 1

        # wait untill the image_batch is loaded
        t1 = time.time()
        while img_reader.is_busy():
            time.sleep(0.1)
        t2 = time.time()
        duration_waited = t2 - t1
        print('...... batch %d/%d:, waited: %d' % (batch_num, n_batches, duration_waited))

        # get the video frames
        video_frames = img_reader.get_batch()  # (B, H, W, C)

        # channel first for pytorch
        video_frames = np.transpose(video_frames, (0, 3, 1, 2))  # (B, C, H, W)

        # pre-load for the next video
        if batch_num < n_images:
            next_f_pathes = np.array([Pth('Breakfast/frames/%s/%s', (video_ids[idx_img + 1], n)) for n in video_frames_dict[video_ids[idx_img + 1]]])
            img_reader.load_batch(next_f_pathes)

        # extract features
        features = pytorch_utils.batched_feedforward(model, video_frames, batch_size, 'forward_conv_maxpool')

        # channel first
        features = np.transpose(features, (1, 0, 2, 3))

        # cast as float_16 for saving space and computation
        features = features.astype(np.float16)

        all_features[idx] = features

    return all_features

# endregion
