

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


import torch


parser = ArgumentParser(description= 'Example')
parser.add_argument('--gpu', help='gpu to use', type = int, default = 0)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


# Feature extraction
def expand_feats(feat):

    feat = np.expand_dims(feat, 2)
    feat = np.expand_dims(feat, 3)
    feat = np.expand_dims(feat, 4)

    return feat

def expand_feats_(feat):

    feat = np.expand_dims(feat, 3)
    feat = np.expand_dims(feat, 4)

    return feat

###### Annotation labels ###########################################################################################################################
path_anno = 'Vcoco/annotation/anno_vcoco.pkl'
num_class = 600

annot_path = Pth(path_anno)

print('... loading data')

(img_names_tr, y_tr, img_names_te, y_te) = utils.pkl_load(annot_path)
y_tr = y_tr.astype(np.float32)
y_te = y_te.astype(np.float32)

metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico

###### Annotation labels ###########################################################################################################################

'''
###### Dataset feature extraction ###########################################################################################################################

# 1) dump annotation
#ds_hico._101_prepare_annotation_vcoco()
# 2) subject-object
#ds_hico._203_pickle_features_subject_object_vcoco()
# 3) local scene 
#ds_hico._203_pickle_features_local_scene_vcoco()
# 4) deformation
# ds_hico._203_pickle_features_deformation_vcoco()
# 5) lvis
#ds_hico._202_pickle_features_lvis_vcoco()
# 6) pairatt
#ds_hico._201_pickle_features_pairattn_vcoco()
# 7) contextfusion
#ds_hico._201_pickle_features_contfus_vcoco()
# 8) vgg
#ds_hico._202_pickle_features_vgg_vcoco()

###### Dataset feature extraction ###########################################################################################################################
'''

'''
###### local scene feature extractor ###########################################################################################################################
path_feature = 'Vcoco/features/extra/features_local_scene.h5'
path_feature = Pth(path_feature)
feat_shape = (2048, 6, 1, 1)

path_model = '/var/scratch/mkilicka/data/hico/models_finetuned/local_scene/model.pt'
path_save  = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/vcoco/h5/local_scene.h5'
model = exp_context.ClassifierLocalContextPooling(num_class, feat_shape)
(x_te_c1) = utils.h5_load(path_feature, 'x_te')
x_te_c1  = np.swapaxes(x_te_c1, 1,2)
# Main inference
print('data set: ', x_te_c1.shape)
###### local scene feature extractor ###########################################################################################################################
'''

'''
###### deformation feature extractor ###########################################################################################################################
path_feature = 'Vcoco/features/extra/features_deformation.h5'
path_feature = Pth(path_feature)
feat_shape = (80, 1, 32, 32)

path_model = '/var/scratch/mkilicka/data/hico/models_finetuned/deformation/model.pt'
path_save  = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/vcoco/h5/deformation.h5'
model = exp_context.ClassifierDeformation(num_class, feat_shape)
(x_te_c1) = utils.h5_load(path_feature, 'x_te')
print('data set: ', x_te_c1.shape)
###### deformation feature extractor ###########################################################################################################################
'''
'''
###### deformation classifier performance ###########################################################################################################################
path_feature = 'Vcoco/features/extra/features_deformation.h5'
path_feature = Pth(path_feature)
feat_shape = (80, 1, 32, 32)

path_model = '/var/scratch/mkilicka/data/hico/models_finetuned/deformation/model.pt'
path_save  = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/vcoco/h5/deformation.h5'
model = exp_context.ClassifierDeformation(num_class, feat_shape)
(x_te_c1) = utils.h5_load(path_feature, 'x_te')
print('data set: ', x_te_c1.shape)


model = model.cuda()
model.load_state_dict(torch.load(path_model))
model.eval()
batch_size = 32

# Run actual feed-forward here
y_pred_te = pytorch_utils.batched_feedforward_multi(model, [x_te_c1], batch_size, func_name='inference')

# Evalaute results
acc_te = metric_fn(y_pred_te, y_te)
acc_te = 100 * acc_te

print('Result of deformation alone: %02.02f' %(acc_te))

###### deformation classifier performance ###########################################################################################################################
'''

'''
###### early fusion inference loop ############################################################################################################################
backbone = 'vgg'

if backbone == 'rcnn':
    features_interaction = Pth('Vcoco/features/features_subject_object.h5')
    n_channels, n_regions, channel_side_dim = 4096, 12,1
    input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)
    (x_te)    = utils.h5_load(features_interaction, 'x_te')
    x_te = np.swapaxes(x_te, 1,2)
    path_model = '/var/scratch/mkilicka/data/hico/models_finetuned/fusion_for_vcoco_early_fusion_rcnn/model.pt'

elif backbone == 'contextfusion':
    features_interaction = Pth('Vcoco/features/h5/features_contextfusion.h5')
    n_channels, n_regions, channel_side_dim = 4096, 3,1  
    input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)
    (x_te)    = utils.h5_load(features_interaction, 'x_te')
    x_te = np.swapaxes(x_te, 1,2)
    x_te = expand_feats_(x_te)
    path_model = '/var/scratch/mkilicka/data/hico/models_finetuned/fusion_for_vcoco_early_fusion_contextfusion/model.pt'

elif backbone == 'pairatt':
    features_interaction = Pth('Vcoco/features/h5/features_pairattn.h5')
    n_channels, n_regions, channel_side_dim = 4096, 3,1  
    input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)
    (x_te)    = utils.h5_load(features_interaction, 'x_te')
    x_te = np.swapaxes(x_te, 1,2)
    x_te = expand_feats_(x_te)
    path_model = '/var/scratch/mkilicka/data/hico/models_finetuned/fusion_for_vcoco_early_fusion_pairatt/model.pt'

elif backbone == 'vgg':

    features_interaction = Pth('Vcoco/features/h5/resnet.h5')
    n_channels, n_regions, channel_side_dim = 2048, 1,1  
    input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)
    (x_te)    = utils.h5_load(features_interaction, 'x_te')
    path_model = '/var/scratch/mkilicka/data/hico/models_finetuned/fusion_for_vcoco_early_fusion_vgg/model.pt'

print('bacbone:', backbone)

feature_path_c1 = Pth('Vcoco/features/h5/lvis.h5')
feature_path_c2 = Pth('Vcoco/features/h5/local_scene.h5')
feature_path_c3= Pth('Vcoco/features/h5/deformation.h5')

x_cs_shape = [(1300, 1, 1, 1), (2048, 1, 1, 1), (512, 1, 1, 1)]

(x_te_c1) = utils.h5_load(feature_path_c1, 'x_te')
(x_te_c2) = utils.h5_load(feature_path_c2, 'x_te')
x_te_c2 = expand_feats(x_te_c2)
(x_te_c3) = utils.h5_load(feature_path_c3, 'x_te')
x_te_c3 = expand_feats(x_te_c3)

print('test_set_shape_interaction: ', x_te.shape)
print('test_set_shape_context-1: ',  x_te_c1.shape)
print('test_set_shape_context-2: ',  x_te_c2.shape)
print('test_set_shape_context-3: ',  x_te_c3.shape)

path_save  = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/vcoco/results/early_fusion.h5'
model = exp_hico.ClassifierContextLateEarlyFusionHumanObject(num_class, input_shape, x_cs_shape)

model = model.cuda()
model.load_state_dict(torch.load(path_model))
model.eval()
batch_size = 32

# Run actual feed-forward here
y_pred_te = pytorch_utils.batched_feedforward_multi(model, [x_te, x_te_c1, x_te_c2, x_te_c3], batch_size, func_name='inference')
print('shape_of_result',  y_pred_te.shape)

utils.h5_dump(y_pred_te, 'y_pred_te', path_save)

# Evalaute results
acc_te = metric_fn(y_pred_te, y_te)
acc_te = 100 * acc_te

print('Result of early fusion exp: %02.02f' %(acc_te))

###### early fusion inference loop ###########################################################################################################################
'''

'''
###### late fusion inference loop ############################################################################################################################
backbone = 'pairatt'

if backbone == 'rcnn':
    features_interaction = Pth('Vcoco/features/features_subject_object.h5')
    n_channels, n_regions, channel_side_dim = 4096, 12,1
    input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)
    (x_te)    = utils.h5_load(features_interaction, 'x_te')
    x_te = np.swapaxes(x_te, 1,2)
    path_model = '/var/scratch/mkilicka/data/hico/models_finetuned/fusion_for_vcoco_late_fusion_rcnn/model.pt'

elif backbone == 'contextfusion':
    features_interaction = Pth('Vcoco/features/h5/features_contextfusion.h5')
    n_channels, n_regions, channel_side_dim = 4096, 3,1  
    input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)
    (x_te)    = utils.h5_load(features_interaction, 'x_te')
    x_te = np.swapaxes(x_te, 1,2)
    x_te = expand_feats_(x_te)
    path_model = '/var/scratch/mkilicka/data/hico/models_finetuned/fusion_for_vcoco_late_fusion_contextfusion/model.pt'

elif backbone == 'pairatt':
    features_interaction = Pth('Vcoco/features/h5/features_pairattn.h5')
    n_channels, n_regions, channel_side_dim = 4096, 3,1  
    input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)
    (x_te)    = utils.h5_load(features_interaction, 'x_te')
    x_te = np.swapaxes(x_te, 1,2)
    x_te = expand_feats_(x_te)
    path_model = '/var/scratch/mkilicka/data/hico/models_finetuned/fusion_for_vcoco_late_fusion_pairatt/model.pt'

elif backbone == 'vgg':

    features_interaction = Pth('Vcoco/features/h5/resnet.h5')
    n_channels, n_regions, channel_side_dim = 2048, 1,1  
    input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)
    (x_te)    = utils.h5_load(features_interaction, 'x_te')
    path_model = '/var/scratch/mkilicka/data/hico/models_finetuned/fusion_for_vcoco_late_fusion_vgg/model.pt'

print('bacbone:', backbone)

feature_path_c1 = Pth('Vcoco/features/h5/lvis.h5')
feature_path_c2 = Pth('Vcoco/features/h5/local_scene.h5')
feature_path_c3= Pth('Vcoco/features/h5/deformation.h5')

x_cs_shape = [(1300, 1, 1, 1), (2048, 1, 1, 1), (512, 1, 1, 1)]

(x_te_c1) = utils.h5_load(feature_path_c1, 'x_te')
(x_te_c2) = utils.h5_load(feature_path_c2, 'x_te')
x_te_c2 = expand_feats(x_te_c2)
(x_te_c3) = utils.h5_load(feature_path_c3, 'x_te')
x_te_c3 = expand_feats(x_te_c3)

print('test_set_shape_interaction: ', x_te.shape)
print('test_set_shape_context-1: ',  x_te_c1.shape)
print('test_set_shape_context-2: ',  x_te_c2.shape)
print('test_set_shape_context-3: ',  x_te_c3.shape)

path_save  = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/vcoco/results/late_fusion.h5'
model = exp_hico.ClassifierContextLateFusionMulti(num_class, input_shape, x_cs_shape)

model = model.cuda()
model.load_state_dict(torch.load(path_model))
model.eval()

batch_size = 32

# Run actual feed-forward here
y_pred_te = pytorch_utils.batched_feedforward_multi(model, [x_te, x_te_c1, x_te_c2, x_te_c3], batch_size, func_name='inference')
print('shape_of_result',  y_pred_te.shape)

utils.h5_dump(y_pred_te, 'y_pred_te', path_save)

# Evalaute results
acc_te = metric_fn(y_pred_te, y_te)
acc_te = 100 * acc_te

print('Result of late fusion exp: %02.02f' %(acc_te))

###### late fusion inference loop ##########################################################################################################################
'''

###### gating inference loop ############################################################################################################################
backbone = 'contextfusion'

if backbone == 'rcnn':
    features_interaction = Pth('Vcoco/features/h5/features_subject_object.h5')
    n_channels, n_regions, channel_side_dim = 4096, 12,1
    input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)
    (x_te)    = utils.h5_load(features_interaction, 'x_te')
    x_te = np.swapaxes(x_te, 1,2)
    path_model = '/var/scratch/mkilicka/data/hico/models_finetuned/vcoco_gating_rcnn/model.pt'

elif backbone == 'contextfusion':
    features_interaction = Pth('Vcoco/features/h5/features_contextfusion.h5')
    n_channels, n_regions, channel_side_dim = 4096, 3,1  
    input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)
    (x_te)    = utils.h5_load(features_interaction, 'x_te')
    x_te = np.swapaxes(x_te, 1,2)
    x_te = expand_feats_(x_te)
    path_model = '/var/scratch/mkilicka/data/hico/models_finetuned/vcoco_gating_contextfusion/model.pt'

elif backbone == 'pairatt':
    features_interaction = Pth('Vcoco/features/h5/features_pairattn.h5')
    n_channels, n_regions, channel_side_dim = 4096, 3,1  
    input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)
    (x_te)    = utils.h5_load(features_interaction, 'x_te')
    x_te = np.swapaxes(x_te, 1,2)
    x_te = expand_feats_(x_te)
    path_model = '/var/scratch/mkilicka/data/hico/models_finetuned/vcoco_gating_pairatt/model.pt'

elif backbone == 'vgg':

    features_interaction = Pth('Vcoco/features/h5/resnet.h5')
    n_channels, n_regions, channel_side_dim = 2048, 1,1  
    input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)
    (x_te)    = utils.h5_load(features_interaction, 'x_te')
    path_model = '/var/scratch/mkilicka/data/hico/models_finetuned/vcoco_gating_vgg/model.pt'

print('bacbone:', backbone)

feature_path_c1 = Pth('Vcoco/features/h5/lvis.h5')
feature_path_c2 = Pth('Vcoco/features/h5/local_scene.h5')
feature_path_c3= Pth('Vcoco/features/h5/deformation.h5')

x_cs_shape = [(1300, 1, 1, 1), (2048, 1, 1, 1), (512, 1, 1, 1)]

(x_te_c1) = utils.h5_load(feature_path_c1, 'x_te')
(x_te_c2) = utils.h5_load(feature_path_c2, 'x_te')
x_te_c2 = expand_feats(x_te_c2)
(x_te_c3) = utils.h5_load(feature_path_c3, 'x_te')
x_te_c3 = expand_feats(x_te_c3)

print('test_set_shape_interaction: ', x_te.shape)
print('test_set_shape_context-1: ',  x_te_c1.shape)
print('test_set_shape_context-2: ',  x_te_c2.shape)
print('test_set_shape_context-3: ',  x_te_c3.shape)

path_save  = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/vcoco/results/gating.h5'
model = exp_hico.ClassifierContextInteraction(num_class, input_shape, x_cs_shape)

model = model.cuda()
model.load_state_dict(torch.load(path_model))
model.eval()
batch_size = 32

# Run actual feed-forward here
y_pred_te = pytorch_utils.batched_feedforward_multi(model, [x_te, x_te_c1, x_te_c2, x_te_c3], batch_size, func_name='inference')
print('shape_of_result',  y_pred_te.shape)

utils.h5_dump(y_pred_te, 'y_pred_te', path_save)

# Evalaute results
acc_te = metric_fn(y_pred_te, y_te)
acc_te = 100 * acc_te

print('Result of gating:%s is: %02.02f' %(backbone, acc_te))

###### gating inference loop ###########################################################################################################################



'''
###### Main inference loop ###########################################################################################################################

print('running feed-forward')

# Init model
model = model.cuda()
model.load_state_dict(torch.load(path_model))
model.eval()
batch_size = 32

# Run actual feed-forward here
x_te = pytorch_utils.batched_feedforward_multi(model,[x_te_c1], batch_size, func_name='inference')

print('feed-forward finished: ', x_te.shape)
print('dumping feature: %s, model: %s, into save path: %s' %(path_feature, path_model, path_save))
utils.h5_dump(x_te, 'x_te', path_save)
###### Main inference loop ###########################################################################################################################
'''