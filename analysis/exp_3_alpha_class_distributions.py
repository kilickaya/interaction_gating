
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


    ##### Load alpha values for analysis #########################################################################################################
    print('...Loading alpha values...')
    alpha_path = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/hico/results/gumbel_softmax_hard_gating.h5'
    (y_te_pred, alphas) = utils.h5_load_multi(alpha_path, ['y_pred_te', 'alphas']) # (B, 600), (B, M, N)

    C = y_te_pred.shape[1]
    alphas = alphas.max(2) # (B, M)

    ##### Generating alpha values per class statistics #########################################################################################################
    print('...Computing class-level alphas...')

    # per interaction
    output = np.zeros((C,4), dtype = np.float32)

    y_te_ = np.transpose(y_te)
    for i in range(C):
        index = np.where(y_te_[i] == 1)[0]

        alphas_per_class = alphas[index].mean(0)
        output[i] = alphas_per_class

    # per object
    unique_objects = np.unique(objlist)
    output_object = np.zeros((len(unique_objects),4), dtype = np.float32)

    for i in range(len(unique_objects)):

        inter_classes = np.where(objlist == unique_objects[i])[0]

        for j in inter_classes:
            index = np.where(y_te_[j] == 1)[0]
            alphas_per_class = alphas[index].mean(0)
            output_object[i] += alphas_per_class      

        output_object[i] = output_object[i] / len(inter_classes)
 

    # per verb
    unique_verbs = np.unique(verblist)
    output_verb = np.zeros((len(unique_verbs),4), dtype = np.float32)

    for i in range(len(unique_verbs)):

        inter_classes = np.where(verblist == unique_verbs[i])[0]

        for j in inter_classes:
            index = np.where(y_te_[j] == 1)[0]
            alphas_per_class = alphas[index].mean(0)
            output_verb[i] += alphas_per_class      

        output_verb[i] = output_verb[i] / len(inter_classes)


    ##### Generating alpha values per class statistics #########################################################################################################


    ##### Export alpha values to csv file for interaction #########################################################################################################
    print('...Exporting alphas...')

    import csv
    classfile = open('./analysis/per_class_alpha_analysis.csv', 'w')

    classfile.write('verb\tobject\tlvis\tlocal_scene\tdeformation\tpart\n')

    for i in range(C):

        text = verblist[i] + '\t' + objlist[i] + '\t' + str(output[i, 0]) + '\t' + str(output[i, 1]) + '\t' + str(output[i, 2]) + '\t' + str(output[i, 3]) + '\n'
        classfile.write(text)

    classfile.close()
    ##### Export alpha values to csv file for interactions #########################################################################################################


    ##### Export alpha values to csv file for objects #########################################################################################################
    print('...Exporting alphas...')

    import csv
    classfile = open('./analysis/per_class_alpha_analysis_object.csv', 'w')

    classfile.write('object\tlvis\tlocal_scene\tdeformation\tpart\n')

    for i in range(len(unique_objects)):

        text =  unique_objects[i] + '\t' + str(output_object[i, 0]) + '\t' + str(output_object[i, 1]) + '\t' + str(output_object[i, 2]) + '\t' + str(output_object[i, 3]) + '\n'
        classfile.write(text)

    classfile.close()
    ##### Export alpha values to csv file for objects #########################################################################################################

    ##### Export alpha values to csv file for objects #########################################################################################################
    print('...Exporting alphas...')

    import csv
    classfile = open('./analysis/per_class_alpha_analysis_verb.csv', 'w')

    classfile.write('verb\tlvis\tlocal_scene\tdeformation\tpart\n')

    for i in range(len(unique_verbs)):

        text =  unique_verbs[i] + '\t' + str(output_verb[i, 0]) + '\t' + str(output_verb[i, 1]) + '\t' + str(output_verb[i, 2]) + '\t' + str(output_verb[i, 3]) + '\n'
        classfile.write(text)

    classfile.close()
    ##### Export alpha values to csv file for objects #########################################################################################################

    ##### Export selected objects to create a heatmap #########################################################################################################

    query_objects = ['dining_table', 'oven', 'refrigerator', 'motorcycle', 'horse', 'car', 'snowboard', 'skis', 'skateboard', 'bowl', 'orange', 'donut']
    contexts = ['objects', 'local scene', 'deformation', 'part appearance']
    query_objects = np.array(query_objects)
    heatmap = np.zeros((query_objects.shape[0], 4), dtype = np.float32)

    for i in range(query_objects.shape[0]):
        index = np.where(unique_objects == query_objects[i])
        temp = output_object[index]
        heatmap[i] = temp

    print(heatmap)
    sio.savemat('./analysis/exp3_heatmap_object.mat', {'heatmap': heatmap, 'objects': query_objects, 'contexts': contexts})

    ##### Export selected objects to create a heatmap #########################################################################################################

    query_verbs = ['eat_at', 'clean', 'cook', 'race', 'row', 'drive', 'throw', 'stand_on', 'jump', 'cut_with', 'brush_with', 'eat']
    contexts = ['objects', 'local scene', 'deformation', 'part appearance']
    query_verbs = np.array(query_verbs)
    heatmap = np.zeros((query_verbs.shape[0], 4), dtype = np.float32)

    for i in range(query_verbs.shape[0]):
        index = np.where(unique_verbs == query_verbs[i])
        temp = output_verb[index]
        heatmap[i] = temp

    print(heatmap)
    sio.savemat('./analysis/exp3_heatmap_verb.mat', {'heatmap': heatmap, 'objects': query_verbs, 'contexts': contexts})

    ##### Export selected objects to create a heatmap #########################################################################################################



