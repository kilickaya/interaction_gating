

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

def expand_feats(feat):

    feat = np.expand_dims(feat, 2)
    feat = np.expand_dims(feat, 3)
    feat = np.expand_dims(feat, 4)

    return feat

def expand_feats_(feat):

    feat = np.expand_dims(feat, 3)
    feat = np.expand_dims(feat, 4)

    return feat
# Extraction paths
###### Annotation labels ###########################################################################################################################
path_anno = 'Hico/features/h5/anno_hico.pkl'
num_class = 600
metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
annot_path = Pth(path_anno)

print('... loading data')

(img_names_tr, y_tr,_,img_names_te, y_te, y_te_mask) = utils.pkl_load(annot_path)
y_tr = y_tr.astype(np.float32)
y_te = y_te.astype(np.float32)

metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
###### Annotation labels ###########################################################################################################################

'''
# base subject-object features
path_feature = 'Hico/features/features_base_subject_object.h5'
path_feature = Pth(path_feature)
feat_shape = (4096, 12, 1, 1)
path_model = '/var/scratch/mkilicka/data/hico/models_finetuned/base_human_object/model.pt'
path_save  = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/hico/h5/base_human_object.h5'
model = exp_context.ClassifierMultiHumanObjectContextPooling(num_class, feat_shape) 
'''

'''
# local pose features
path_feature = 'Hico/features/legacy/features_pairattn_pose.h5'
path_feature = Pth(path_feature)
feat_shape = (4096, 3, 1, 1)

path_model = '/var/scratch/mkilicka/data/hico/models_finetuned/local_pose/model.pt'
path_save  = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/hico/h5/local_pose.h5'
model = exp_context.ClassifierLocalContextPooling(num_class, feat_shape)
(x_tr_c1, x_te_c1) = utils.h5_load_multi(path_feature, ['x_tr', 'x_te'])
x_tr_c1 = np.swapaxes(x_tr_c1, 1,2)
x_te_c1 = np.swapaxes(x_te_c1, 1,2)

x_tr_c1 = np.expand_dims(x_tr_c1, 3)
x_tr_c1 = np.expand_dims(x_tr_c1, 4)

x_te_c1 = np.expand_dims(x_te_c1, 3)
x_te_c1 = np.expand_dims(x_te_c1, 4)
'''

'''
# local scene features
path_feature = 'Hico/features/extra/features_local_scene.h5'
path_feature = Pth(path_feature)
feat_shape = (2048, 6, 1, 1)

path_model = '/var/scratch/mkilicka/data/hico/models_finetuned/local_scene/model.pt'
path_save  = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/hico/h5/local_scene.h5'
model = exp_context.ClassifierLocalContextPooling(num_class, feat_shape)
'''

'''
# part states feature
path_feature = 'Hico/features/legacy2/features_local_part_states.h5'
path_feature = Pth(path_feature)
feat_shape = (1032, 1, 1, 1)
path_model = '/var/scratch/mkilicka/data/hico/models_finetuned/part_states/model.pt'
path_save  = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/hico/h5/part_states.h5'
model = exp_context.ClassifierPartState(num_class, feat_shape)

x_tr_c1 = x_tr_c1.reshape(-1, 1032, 1,1,1)
x_te_c1 = x_te_c1.reshape(-1, 1032, 1,1,1)
'''

'''
# stuff and scene and attribute features
# Features of the image: f_scene
feats_c1_path = Pth('Hico/features/extra/features_coco_stuff.h5')
feats_c2_path = Pth('Hico/features/legacy/features_scene_places.h5')
feats_c3_path = Pth('Hico/features/legacy/features_scene_att.h5')

x_cs_shape = [(182, 1, 1, 1), (365, 1, 1, 1), (102, 1, 1, 1)]

(x_tr_c1, x_te_c1) = utils.h5_load_multi(feats_c1_path, ['x_tr', 'x_te'])
(x_tr_c2, x_te_c2) = utils.h5_load_multi(feats_c2_path, ['x_tr', 'x_te'])
(x_tr_c3, x_te_c3) = utils.h5_load_multi(feats_c3_path, ['x_tr', 'x_te'])

x_tr_c1 = np.swapaxes(x_tr_c1, 1,2)
x_te_c1 = np.swapaxes(x_te_c1, 1,2)

path_feature = feats_c1_path
path_model = '/var/scratch/mkilicka/data/hico/models_finetuned/stuff/model.pt'
path_save  = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/hico/h5/stuff.h5'
model = exp_context.ClassifierCombination(num_class, x_cs_shape)
'''

'''
# deformation features
path_feature = 'Hico/features/extra/features_deformation.h5'
path_feature = Pth(path_feature)
feat_shape = (80, 1, 32, 32)

path_model = '/var/scratch/mkilicka/data/hico/models_finetuned/deformation/model.pt'
path_save  = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/hico/h5/deformation.h5'
model = exp_context.ClassifierDeformation(num_class, feat_shape)
'''

'''
# global scene features
path_feature = 'Hico/features/extra/features_scene.h5'
path_feature = Pth(path_feature)
feat_shape = (512, 1, 14, 14)

path_model = '/var/scratch/mkilicka/data/hico/models_finetuned/global_scene/model.pt'
path_save  = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/hico/h5/global_scene.h5'
model = exp_context.ClassifierScene(num_class, feat_shape)
'''

'''
# interior-exterior features
path_feature = 'Hico/features/extra/features_local_locality.h5'
path_feature = Pth(path_feature)
feat_shape = (4096*2,12,1,1)

path_model = '/var/scratch/mkilicka/data/hico/models_finetuned/aura/model.pt'
path_save  = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/hico/h5/aura.h5'
model = exp_context.ClassifierMultiHumanObjectContextPooling(num_class, feat_shape)
'''

'''
###### late fusion inference loop ############################################################################################################################
backbone = 'rcnn'

if backbone == 'rcnn':
    feature_path_interaction = Pth('Hico/features/h5/features_base_subject_object.h5')
    n_channels, n_regions, channel_side_dim = 4096, 12,1
    (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
    x_te = np.swapaxes(x_te, 1,2)
    path_model = '/var/scratch/mkilicka/data/hico/models_finetuned/fusion_for_hico_late_fusion_rcnn/model.pt'

print('backbone:', backbone)
input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)

feature_path_c1 = Pth('Hico/features/h5/lvis.h5')
feature_path_c2 = Pth('Hico/features/h5/local_scene.h5')
feature_path_c3= Pth('Hico/features/h5/deformation.h5')
feature_path_c4 = Pth('Hico/features/h5/local_pose.h5')

x_cs_shape = [(1300, 1, 1, 1), (2048, 1, 1, 1), (512, 1, 1, 1), (4096, 1, 1, 1)]

# Load context features
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

print('test_set_shape_interaction: ', x_te.shape)
print('test_set_shape_context-1: ',  x_te_c1.shape)
print('test_set_shape_context-2: ',  x_te_c2.shape)
print('test_set_shape_context-3: ',  x_te_c3.shape)
print('test_set_shape_context-3: ',  x_te_c4.shape)

path_save  = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/hico/results/late_fusion.h5'
model = exp_hico.ClassifierContextLateFusionMulti(num_class, input_shape, x_cs_shape)

model = model.cuda()
model.load_state_dict(torch.load(path_model))
model.eval()

batch_size = 32

# Run actual feed-forward here
y_pred_te = pytorch_utils.batched_feedforward_multi(model, [x_te, x_te_c1, x_te_c2, x_te_c3, x_te_c4], batch_size, func_name='inference')
print('shape_of_result',  y_pred_te.shape)

utils.h5_dump(y_pred_te, 'y_pred_te', path_save)

# Evalaute results
y_pred_te = y_te_mask * y_pred_te
y_te = y_te * y_te_mask
acc_te = metric_fn(y_pred_te, y_te)
acc_te = 100 * acc_te

print('Result of late fusion exp: %02.02f' %(acc_te))

###### late fusion inference loop ##########################################################################################################################
'''

'''
###### soft gating inference loop (for alphas) ##########################################################################################################################

backbone = 'rcnn'

if backbone == 'rcnn':
    feature_path_interaction = Pth('Hico/features/h5/features_base_subject_object.h5')
    n_channels, n_regions, channel_side_dim = 4096, 12,1
    (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
    x_te = np.swapaxes(x_te, 1,2)
    path_model = '/var/scratch/mkilicka/data/hico/models_finetuned/late_soft_gating_for_hico/model.pt'

print('backbone:', backbone)
input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)

feature_path_c1 = Pth('Hico/features/h5/lvis.h5')
feature_path_c2 = Pth('Hico/features/h5/local_scene.h5')
feature_path_c3= Pth('Hico/features/h5/deformation.h5')
feature_path_c4 = Pth('Hico/features/h5/local_pose.h5')

x_cs_shape = [(1300, 1, 1, 1), (2048, 1, 1, 1), (512, 1, 1, 1), (4096, 1, 1, 1)]

# Load context features
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

print('test_set_shape_interaction: ', x_te.shape)
print('test_set_shape_context-1: ',  x_te_c1.shape)
print('test_set_shape_context-2: ',  x_te_c2.shape)
print('test_set_shape_context-3: ',  x_te_c3.shape)
print('test_set_shape_context-3: ',  x_te_c4.shape)

path_save  = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/hico/results/soft_gating.h5'
model = exp_hico.ClassifierContextLateFusionMultiSoftGate_v3(num_class, input_shape, x_cs_shape)

model = model.cuda()
model.load_state_dict(torch.load(path_model))
model.eval()

batch_size = 32

# Run actual feed-forward here
alphas = pytorch_utils.batched_feedforward_multi(model, [x_te, x_te_c1, x_te_c2, x_te_c3, x_te_c4], batch_size, func_name='return_alphas')
y_pred_te = pytorch_utils.batched_feedforward_multi(model, [x_te, x_te_c1, x_te_c2, x_te_c3, x_te_c4], batch_size, func_name='return_categories')

print('shape_of_result',  y_pred_te.shape)
print('shape_of_alphas',  alphas.shape)

print(np.mean(alphas, 1))
print(np.var(alphas, 1))

utils.h5_dump_multi((alphas, y_pred_te), ['alphas', 'y_pred_te'], path_save)

# Evalaute results
y_pred_te = y_te_mask * y_pred_te
y_te = y_te * y_te_mask
acc_te = metric_fn(y_pred_te, y_te)
acc_te = 100 * acc_te

print('Result of soft gating exp: %02.02f' %(acc_te))


###### soft gating inference loop (for alphas) ##########################################################################################################################
'''

'''
###### multi-head gating inference loop (for alphas) ##########################################################################################################################

backbone = 'rcnn'

if backbone == 'rcnn':
    feature_path_interaction = Pth('Hico/features/h5/features_base_subject_object.h5')
    n_channels, n_regions, channel_side_dim = 4096, 12,1
    (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
    x_te = np.swapaxes(x_te, 1,2)
    path_model = '/var/scratch/mkilicka/data/hico/models_finetuned/hico_gating_rcnn/model.pt'

print('backbone:', backbone)
input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)

feature_path_c1 = Pth('Hico/features/h5/lvis.h5')
feature_path_c2 = Pth('Hico/features/h5/local_scene.h5')
feature_path_c3= Pth('Hico/features/h5/deformation.h5')
feature_path_c4 = Pth('Hico/features/h5/local_pose.h5')

x_cs_shape = [(1300, 1, 1, 1), (2048, 1, 1, 1), (512, 1, 1, 1), (4096, 1, 1, 1)]

# Load context features
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

print('test_set_shape_interaction: ', x_te.shape)
print('test_set_shape_context-1: ',  x_te_c1.shape)
print('test_set_shape_context-2: ',  x_te_c2.shape)
print('test_set_shape_context-3: ',  x_te_c3.shape)
print('test_set_shape_context-3: ',  x_te_c4.shape)

path_save  = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/hico/results/multi_head_gating.h5'
model = exp_hico.ClassifierContextInteraction(num_class, input_shape, x_cs_shape)

model = model.cuda()
model.load_state_dict(torch.load(path_model))
model.eval()

batch_size = 32

# Run actual feed-forward here
alphas = pytorch_utils.batched_feedforward_multi(model, [x_te, x_te_c1, x_te_c2, x_te_c3, x_te_c4], batch_size, func_name='forward_for_alpha')
y_pred_te = pytorch_utils.batched_feedforward_multi(model, [x_te, x_te_c1, x_te_c2, x_te_c3, x_te_c4], batch_size, func_name='inference')

print('shape_of_result',  y_pred_te.shape)
print('shape_of_alphas',  alphas.shape)

alphas = np.mean(alphas, 3) # (B, nco, N)
alphas = np.mean(alphas, 2) # (B, nco)
alphas_mean = np.mean(alphas, 0) # (nco)
alphas_var = np.var(alphas, 0)

print(alphas_mean)
print(alphas_var)

utils.h5_dump_multi((alphas, y_pred_te), ['alphas', 'y_pred_te'], path_save)

# Evalaute results
y_pred_te = y_te_mask * y_pred_te
y_te = y_te * y_te_mask
acc_te = metric_fn(y_pred_te, y_te)
acc_te = 100 * acc_te

print('Result of multi-head gating exp: %02.02f' %(acc_te))


###### multi-head gating inference loop (for alphas) ##########################################################################################################################
'''


###### multi-head gating inference loop (for alphas) ##########################################################################################################################

backbone = 'rcnn'
ablation = False

if backbone == 'rcnn':
    feature_path_interaction = Pth('Hico/features/h5/features_base_subject_object.h5')
    n_channels, n_regions, channel_side_dim = 4096, 12,1
    (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
    x_te = np.swapaxes(x_te, 1,2)
    if ablation == False:
        path_model = '/var/scratch/mkilicka/data/hico/models_finetuned/late_hard_gating_for_hico/model.pt'
        path_save  = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/hico/results/gumbel_softmax_hard_gating.h5'

    else:
        path_model = '/var/scratch/mkilicka/data/hico/models_finetuned/late_hard_ablated_gating_for_hico/model.pt'
        path_save  = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/hico/results/gumbel_softmax_hard_ablated_gating.h5'


print('backbone:', backbone)
input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)

feature_path_c1 = Pth('Hico/features/h5/lvis.h5')
feature_path_c2 = Pth('Hico/features/h5/local_scene.h5')
feature_path_c3= Pth('Hico/features/h5/deformation.h5')
feature_path_c4 = Pth('Hico/features/h5/local_pose.h5')

x_cs_shape = [(1300, 1, 1, 1), (2048, 1, 1, 1), (512, 1, 1, 1), (4096, 1, 1, 1)]

# Load context features
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

print('test_set_shape_interaction: ', x_te.shape)
print('test_set_shape_context-1: ',  x_te_c1.shape)
print('test_set_shape_context-2: ',  x_te_c2.shape)
print('test_set_shape_context-3: ',  x_te_c3.shape)
print('test_set_shape_context-3: ',  x_te_c4.shape)

if ablation == False:
    model = exp_hico.ClassifierContextLateFusionMultiHardGate(num_class, input_shape, x_cs_shape)
else:
    model = exp_hico.ClassifierContextLateFusionMultiHardGateAblated(num_class, input_shape, x_cs_shape)

model = model.cuda()
model.load_state_dict(torch.load(path_model))
model.eval()

batch_size = 32

# Run actual feed-forward here
alphas = pytorch_utils.batched_feedforward_multi(model, [x_te, x_te_c1, x_te_c2, x_te_c3, x_te_c4], batch_size, func_name='forward_for_alpha')
y_pred_te = pytorch_utils.batched_feedforward_multi(model, [x_te, x_te_c1, x_te_c2, x_te_c3, x_te_c4], batch_size, func_name='inference')
utils.h5_dump_multi((alphas, y_pred_te), ['alphas', 'y_pred_te'], path_save)

print('shape_of_result',  y_pred_te.shape)
print('shape_of_alphas',  alphas.shape)

alphas = np.mean(alphas, 2)

print('shape_of_alphas',  alphas.shape)

print('mean:', np.mean(alphas, 0))
print('var:', np.var(alphas, 0))

# Evalaute results
y_pred_te = y_te_mask * y_pred_te
y_te = y_te * y_te_mask
acc_te = metric_fn(y_pred_te, y_te)
acc_te = 100 * acc_te

print('Result of hard gumbel gating exp: %02.02f' %(acc_te))


###### multi-head gating inference loop (for alphas) ##########################################################################################################################

'''
###### single context inference loop ############################################################################################################################
backbone = 'rcnn'
contextype = 'stuff'

feature_path_interaction = Pth('Hico/features/h5/features_base_subject_object.h5')
n_channels, n_regions, channel_side_dim = 4096, 12,1
(x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
x_te = np.swapaxes(x_te, 1,2)

print('backbone:', backbone)
input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)

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

path_model = '/var/scratch/mkilicka/data/hico/models_finetuned/single_context_hico_%s/model.pt' %(contextype)


print('... context features')
(x_tr_c, x_te_c) = utils.h5_load_multi(feature_path_context, ['x_tr', 'x_te'])

if contextype != 'lvis':
    x_tr_c = expand_feats(x_tr_c)
    x_te_c = expand_feats(x_te_c)

print('test_set_shape_interaction: ', x_te.shape)
print('test_set_shape_context: ',  x_te_c.shape)

path_save  = '/var/scratch/mkilicka/code/context-driven-interactions/submission/data/hico/results/single_context_%s.h5' %(contextype)
model = exp_hico.ClassifierContextLateEarlyFusionHumanObject(num_class, input_shape, x_cs_shape)

model = model.cuda()
model.load_state_dict(torch.load(path_model))
model.eval()

batch_size = 32

# Run actual feed-forward here
y_pred_te = pytorch_utils.batched_feedforward_multi(model, [x_te, x_te_c], batch_size, func_name='inference')
print('shape_of_result',  y_pred_te.shape)

utils.h5_dump(y_pred_te, 'y_pred_te', path_save)

# Evalaute results
y_pred_te = y_te_mask * y_pred_te
y_te = y_te * y_te_mask
acc_te = metric_fn(y_pred_te, y_te)
acc_te = 100 * acc_te

print('Result of context: %s exp: %02.02f' %(contextype, acc_te))

###### single context inference loop ##########################################################################################################################
'''

''''
###### Main loop ############################################################################################################################

# Annotation of the image
annot_path = Pth(path_anno)

print('... loading data')

(img_names_tr, y_tr, y_tr_mask, img_names_te, y_te, y_te_mask) = utils.pkl_load(annot_path)
y_tr = y_tr.astype(np.float32)
y_te = y_te.astype(np.float32)

(x_tr_c1, x_te_c1) = utils.h5_load_multi(path_feature, ['x_tr', 'x_te'])
#x_tr_c1 = np.swapaxes(x_tr_c1, 1,2)
#x_te_c1 = np.swapaxes(x_te_c1, 1,2)

print('data set: ', x_tr_c1.shape, x_te_c1.shape)

print('running feed-forward')

# Init model
model = model.cuda()
model.load_state_dict(torch.load(path_model))
model.eval()

batch_size = 32

# Run actual feed-forward here
x_tr = pytorch_utils.batched_feedforward_multi(model,[x_tr_c1], batch_size, func_name='inference')
x_te = pytorch_utils.batched_feedforward_multi(model,[x_te_c1], batch_size, func_name='inference')

print('feed-forward finished: ', x_tr.shape, x_te.shape)
#print('dumping feature: %s, model: %s, into save path: %s' %(path_feature, path_model, path_save))
#utils.h5_dump_multi((x_tr, x_te), ['x_tr', 'x_te'], path_save)

x_te = x_te * y_te_mask
y_te = y_te * y_te_mask

acc_te = metric_fn(x_te, y_te)
acc_te = 100 * acc_te

print('Result of deformation alone: %02.02f' %(acc_te)
###### Main loop ############################################################################################################################
'''