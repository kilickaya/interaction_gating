
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
from core import utils, configs
from core.utils import Path as Pth
from core import const as c
from core import pytorch_utils

#from datasets import ds_breakfast, ds_hico
from datasets import ds_hico
from experiments import exp_hico, exp_context
from argparse import ArgumentParser
import scipy.io as sio

import torch
import cv2

def analysis():

    ###### Annotation labels ###########################################################################################################################
    path_anno = 'Hico/features/h5/anno_hico.pkl'
    num_class = 600
    metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
    annot_path = Pth(path_anno)

    print('... loading data')

    (img_names_tr, y_tr,_,img_names_te, y_te, y_te_mask) = utils.pkl_load(annot_path)
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)

    metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico_all
    ###### Annotation labels ###########################################################################################################################


    ##### Load result of vanilla vs. combined #########################################################################################################

    print('...Loading results...')
    path_vanilla = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/hico/results/rcnn.h5'
    (y_te_vanilla)    = utils.h5_load(path_vanilla, 'x_te')
    acc_te_vanilla = metric_fn(y_te_vanilla * y_te_mask, y_te * y_te_mask)

    # deformation result
    path_context = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/hico/results/single_context_deformation.h5'
    (y_te_context)    = utils.h5_load(path_context, 'y_pred_te')
    acc_te_deformation = metric_fn(y_te_context * y_te_mask , y_te * y_te_mask)

    # local_scene
    path_context = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/hico/results/single_context_local_scene.h5'
    (y_te_context)    = utils.h5_load(path_context, 'y_pred_te')
    acc_te_local_scene = metric_fn(y_te_context * y_te_mask , y_te * y_te_mask)

    # local_pose
    path_context = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/hico/results/single_context_local_pose.h5'
    (y_te_context)    = utils.h5_load(path_context, 'y_pred_te')
    acc_te_local_pose = metric_fn(y_te_context * y_te_mask , y_te * y_te_mask)

    # stuff
    path_context = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/hico/results/single_context_stuff.h5'
    (y_te_context)    = utils.h5_load(path_context, 'y_pred_te')
    acc_te_stuff = metric_fn(y_te_context * y_te_mask , y_te * y_te_mask)


    ##### Load interaction categories #########################################################################################################
    print('...Loading categories...')
    classes = sio.loadmat('../../where-is-interaction/main/data/anno.mat')
    nouns =   classes['objects']
    verbs   = classes['verbs']
    cats =   classes['super_category']

    verblist = []
    for v in verbs:
        verblist.append(np.squeeze(v[0][0]))

    objlist = []
    for o in nouns:
        objlist.append(np.squeeze(o[0][0]))

    verblist = np.array(verblist)
    objlist  = np.array(objlist)

    verblist = np.squeeze(verblist)
    objlist = np.squeeze(objlist)
    ##### Load interaction categories #########################################################################################################


    ##### Compute class-level performance distribution #########################################################################################################
    print('...Computing single context level performance')
    def return_indexes(values):
        return np.where(values==1)[0]

    def return_class_performance(y_te, y_pred):
        performance = np.zeros((y_te.shape[1]), dtype = np.float32)

        y_te_transposed = np.transpose(y_te)

        for i  in range(y_te.shape[1]):
            index = return_indexes(y_te_transposed[i])
            class_performance = y_pred[index]
            class_performance = class_performance.mean()
            performance[i] = class_performance

        return performance

    performance_vanilla = return_class_performance(y_te, acc_te_vanilla)
    performance_deformation = return_class_performance(y_te, acc_te_deformation)
    performance_local_scene = return_class_performance(y_te, acc_te_local_scene)
    performance_local_pose = return_class_performance(y_te, acc_te_local_pose)
    performance_stuff      = return_class_performance(y_te, acc_te_stuff)



    ##### Compare each context performance with vanilla and report improved and decreased categories #########################################################################################################
    print('...Computing difference in the performance\n\n')

    def print_class_diffs(vanilla, context, name):

        diff_vector = context - vanilla
        Ix = np.argsort(diff_vector.reshape(-1))

        for i in range(0, 15, 1):
            print('Decrease: Context: %s, (%s, %s), %0.3f' %(name, verblist[Ix[i]], objlist[Ix[i]], diff_vector[Ix[i]]))

        print('\n')

        for i in range(585, 600, 1):
            print('Increase: Context: %s, (%s, %s), %0.3f' %(name, verblist[Ix[i]], objlist[Ix[i]], diff_vector[Ix[i]]))
        

    print_class_diffs(performance_vanilla, performance_deformation, 'deformation')
    print('\n\n')
    print_class_diffs(performance_vanilla, performance_local_scene, 'local_scene')
    print('\n\n')
    print_class_diffs(performance_vanilla, performance_local_pose, 'local_pose')
    print('\n\n')
    print_class_diffs(performance_vanilla, performance_stuff, 'stuff')

