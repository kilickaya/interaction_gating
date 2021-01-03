
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
    path_anno = 'Cint/annotation/anno_cint.pkl'
    img_path = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/cint/images/'
    save_path = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/cint/sorted/'


    num_class = 600
    metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
    annot_path = Pth(path_anno)

    print('... loading data')

    (img_names_tr, y_tr,img_names_te, y_te) = utils.pkl_load(annot_path)
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)

    metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico_all
    ###### Annotation labels ###########################################################################################################################


    ##### Load result of vanilla vs. combined #########################################################################################################

    print('...Loading results...')
    path_vanilla = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/cint/results/vanilla_pairatt.h5'
    (y_te_vanilla)    = utils.h5_load(path_vanilla, 'x_te')
    acc_te_vanilla = metric_fn(y_te_vanilla, y_te)

    path_gating = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/cint/results/late_fusion_pairatt.h5'
    (y_te_gating)    = utils.h5_load(path_gating, 'y_pred_te')
    acc_te_gating = metric_fn(y_te_gating , y_te)


    ##### Load result of vanilla vs. combined #########################################################################################################

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




    ##### Based on accuracy difference, sort images and copy them to another directory #########################################################################################################

    difference = acc_te_gating - acc_te_vanilla
    Ix = np.argsort(-difference)

    for i in range(len(img_names_te)):
        source = img_path + img_names_te[Ix[i]]
        target = save_path  + str(acc_te_gating[Ix[i]]) + '_' + str(acc_te_vanilla[Ix[i]]) + '_'  + img_names_te[Ix[i]]