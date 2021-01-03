#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Main file of the project.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
from core import utils, configs
from core.utils import Path as Pth
from core import const as c
#from analysis import exp_2_source_of_improvement as exp_analysis
#from analysis import exp_3_alpha_class_distributions as exp_analysis
from analysis import exp_3_pairatt_qualitative as exp_analysis

#from datasets import ds_breakfast, ds_hico
from datasets import ds_hico
from experiments import exp_hico, exp_context


from argparse import ArgumentParser

parser = ArgumentParser(description= 'Example')
parser.add_argument('--gpu', help='gpu to use', type = int, default = 0)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# Dataset ops
# ds_hico._201_pickle_features_contfus()
#ds_hico._101_pickle_vanilla_rcnn_predictions()
#ds_hico._201_pickle_predictions_pairattn_cint()
ds_hico._303_extract_features_from_resnet_18()

# Experiment 0: Training context and base classifiers for feature extraction
# 1) human-object softmax pooling, 512d
# exp_context.train_human_object_pooling() 

# 2) Combine interior and exterior into a compact feature: 512d
#exp_context.train_local_context() 

# 3) Deformation features obtained from Maskrcnn stacking (H,W, 80)
# exp_context.train_classifier_deformation()

# 4) Local scene segment pooling 
# exp_context.train_classifier_local_segment_pooling()

# 5) Stuff context where we combine: coco_stuff, places, attributes
#exp_context.train_stuff_context()

# 6) Part states where we concatenate all part states 86*12 (12 human-object)
#exp_context.train_part_states()

# 7) Pose pooling of PairAtt
#exp_context.train_classifier_local_pose_pooling()

# Experiment 1: Combining single context features with human-object
#exp_hico.train_human_object_single_context()
#exp_hico.train_human_object_single_context(contextype='deformation')
#exp_hico.train_human_object_single_context(contextype='local_scene')
#exp_hico.train_human_object_single_context(contextype='local_pose')
#exp_hico.train_human_object_single_context(contextype= 'stuff')


# Experiment 2: Combining multiple context features with human-object
#exp_hico.train_human_object_multiple_context(early_flag=True, backbone = 'rcnn') # Early fusion - rcnn
#exp_hico.train_human_object_multiple_context(early_flag=True, backbone='pairatt') # Early fusion - pairatt
#exp_hico.train_human_object_multiple_context(early_flag=True, backbone='contextfusion') # Early fusion - contextfusion
#exp_hico.train_human_object_multiple_context(early_flag=True, backbone='vgg') # Early fusion - vgg

# Experiment 3: Combining multiple context features with human-object
#exp_hico.train_human_object_multiple_context(early_flag=False, backbone='rcnn') # Late fusion
#exp_hico.train_human_object_multiple_context(early_flag=False, backbone='contextfusion') # Late fusion - contextfusion
#exp_hico.train_human_object_multiple_context(early_flag=False, backbone='pairatt') # Early fusion - vgg
#exp_hico.train_human_object_multiple_context(early_flag=False, backbone='vgg') # Early fusion - vgg

# Experiment 4: Gating experiments
#exp_hico.train_human_object_multiple_context_gating(soft_flag=True) # soft fusion
#exp_hico.train_human_object_multiple_context_gating(soft_flag= False, ablation_flag = True) # hard fusion, but without conditioning

# Experiment 5: Many context soft late gating
#exp_hico.train_human_object_many_context(early_flag=False, backbone='rcnn') # Late fusion

# Experiment 6: Multi-headed gating experiments for HICO
#exp_hico.train_human_object_multiple_context_gating_multihead(backbone='rcnn')
#exp_hico.train_human_object_multiple_context_gating_multihead(backbone='vgg')
#exp_hico.train_human_object_multiple_context_gating_multihead(backbone='contextfusion')
#exp_hico.train_human_object_multiple_context_gating_multihead(backbone='pairatt')

# Experiment 7: Multi-headed gating experiments for V-coco
#exp_hico.train_human_object_multiple_context_gating_multihead_vcoco(backbone='rcnn')
#exp_hico.train_human_object_multiple_context_gating_multihead_vcoco(backbone='vgg')
#exp_hico.train_human_object_multiple_context_gating_multihead_vcoco(backbone='contextfusion')
#exp_hico.train_human_object_multiple_context_gating_multihead_vcoco(backbone='pairatt')

# Experiment 8: Parameter efficiency comparison
#exp_hico.train_human_object_early_fusion_efficiency()
#exp_hico.train_human_object_multiple_context_gating_efficiency()


# Analysis 
#exp_analysis.analysis()