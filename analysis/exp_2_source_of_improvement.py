
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
    acc_te_vanilla = metric_fn(y_te_vanilla, y_te)

    path_gating = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/hico/results/late_fusion.h5'
    (y_te_gating)    = utils.h5_load(path_gating, 'y_pred_te')
    acc_te_gating = metric_fn(y_te_gating * y_te_mask, y_te * y_te_mask)


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


    ##### Compute interaction size #########################################################################################################
    print('...Obtaining interaction size...')

    savepath =  './analysis/interaction_size.mat'

    def get_size(regions, numpixels):

        result = np.zeros((regions.shape[0]), dtype = np.float32)

        for index, region in enumerate( regions):
            area = (region[2]-region[0]) * (region[3] - region[1])
            area = np.float32(area)
            area = area / numpixels

            result[index] = area

        return result

    if os.path.isfile(savepath):
        # Load the file
        inter_size = sio.loadmat(savepath)['size']
    else:

        image_path_train = '/var/scratch/mkilicka/data/hico/hico_20150920/images/train2015/'
        image_path_val   = '/var/scratch/mkilicka/data/hico/hico_20150920/images/test2015/'
        anno_path_train = '/home/mkilicka/scratch/code/where-is-interaction/main/data/annopixel/train/'
        anno_path_val = '/home/mkilicka/scratch/code/where-is-interaction/main/data/annopixel/val/'

        inter_size = np.zeros(len(img_names_te), dtype = np.float32)
        for index, name in enumerate(img_names_te):

            impath = image_path_val + name
            img =  cv2.imread(impath).astype(np.float32)
            shape = img.shape
            num_pixels = shape[0] * shape[1]
            num_pixels = np.float32(num_pixels)

            annotation = sio.loadmat(anno_path_val + name[:-4] + '.mat')

            human_regions = annotation['human'].reshape(-1, 4)
            obj_regions = annotation['object'].reshape(-1, 4)

            human_size = get_size(human_regions, num_pixels)
            obj_size   = get_size(obj_regions, num_pixels)

            inter_size[index] = 0.5 * human_size.sum() + 0.5 * obj_size.sum()

            print(index, name, inter_size[index])

        sio.savemat(savepath, {'size': inter_size})

    inter_size = np.nan_to_num(inter_size)

    inter_size = np.squeeze(inter_size)
    size_threshold = inter_size.mean()
    small_images = np.where(inter_size <= size_threshold)[0]
    large_images = np.where(inter_size > size_threshold)[0]

    print('#Small_images: ', len(small_images), ' #large_images: ', len(large_images), ' threshold:', size_threshold)


    print('Vanilla: ap-small-size: %0.4f, ap-large-size: %0.4f' %(acc_te_vanilla[small_images].mean(), acc_te_vanilla[large_images].mean()))
    print('Gating: ap-small-size: %0.4f, ap-large-size: %0.4f' %(acc_te_gating[small_images].mean(), acc_te_gating[large_images].mean()))


    ##### Compute interaction size #########################################################################################################

    print('...Obtaining interaction population...')
    from sklearn import metrics
    savepath = './analysis/inter_population.mat'

    inter_population = np.zeros((len(img_names_te)), dtype = np.float32)


    if(os.path.isfile(savepath)):
        inter_population = sio.loadmat(savepath)['population']
    else:	

        labeldistance = metrics.pairwise.pairwise_distances(X = y_te, Y = y_tr, metric = 'euclidean', n_jobs = -1)

        for index, name in enumerate(img_names_te):

                #Compute pairwise operation here
                dist = labeldistance[index]
                dist = np.squeeze(dist)
                sameclass = np.where(dist == 0)[0]
                inter_population[index]  = sameclass.shape[0]


        sio.savemat(savepath, {'population': inter_population})


    inter_population = np.squeeze(inter_population)
    population_threshold = 5
    rare_images = np.where(inter_population <= population_threshold)[0]
    freq_images = np.where(inter_population >  population_threshold * 5)[0]

    print('Vanilla: ap-rare: %0.4f, ap-freq: %0.4f' %(acc_te_vanilla[rare_images].mean(), acc_te_vanilla[freq_images].mean()))
    print('Gating: ap-rare: %0.4f, ap-freq: %0.4f' %(acc_te_gating[rare_images].mean(), acc_te_gating[freq_images].mean()))


    ##### Compute interaction population #########################################################################################################



    ##### Existence of interaction #########################################################################################################
    print('...Obtaining no interaction...')
    no_interaction_indexes = np.where(verblist == 'no_interaction')[0]

    no_interaction_flag = np.zeros((len(img_names_te)), dtype = np.int8)
    interaction_flag = np.ones((len(img_names_te)), dtype = np.int8)

    for index, name in enumerate(img_names_te):

        imlabel = y_te[index]
        imindexes = np.where(imlabel == 1)[0]

        if imindexes in no_interaction_indexes:
            no_interaction_flag[index] = 1
            interaction_flag[index] = 0

    print('Vanilla: ap-no: %0.4f, ap-yes: %0.4f' %(acc_te_vanilla[no_interaction_flag>0].mean(), acc_te_vanilla[interaction_flag>0].mean()))
    print('Gating: ap-no: %0.4f, ap-yes: %0.4f' %(acc_te_gating[no_interaction_flag>0].mean(), acc_te_gating[interaction_flag>0].mean()))


    ##### Existence of interaction #########################################################################################################
