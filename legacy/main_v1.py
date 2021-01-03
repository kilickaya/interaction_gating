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

#from datasets import ds_breakfast, ds_hico
from datasets import ds_hico
from experiments import exp_hico, exp_context


from argparse import ArgumentParser

parser = ArgumentParser(description= 'Example')
parser.add_argument('--gpu', help='gpu to use', type = int, default = 0)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


# region Dataset: Hico

#ds_hico._101_prepare_annotation()
#ds_hico._201_pickle_features_pairattn()
#ds_hico._203_pickle_features_subject_object()
#ds_hico._203_pickle_features_subject_object_locality()
#ds_hico._203_pickle_features_local_scene()
#ds_hico._203_pickle_features_scene()
#ds_hico._203_pickle_features_semanticseg()
#ds_hico._202_pickle_features_context()
#ds_hico._203_pickle_features_deformation()
#ds_hico._203_pickle_features_part_states()
#ds_hico._203_pickle_features_pose()
#ds_hico._202_pickle_features_human_pose()
#ds_hico._203_pickle_features_cocostuff()

# ds_hico._801_finetune_resnet50_imagenet()
# ds_hico._802_finetune_resnet50_places_365()

# endregion

# region Experiment: Hico

# baselines
#exp_hico.train_classifier_using_features_single_region()
#exp_hico.train_classifier_using_features_multi_region()
#exp_hico.train_classifier_using_features_single_context()

# so gating
#exp_hico.train_classifier_so_fusion_using_features()
# exp_hico.train_classifier_so_gating_using_features()

# context fusion
# exp_hico.train_classifier_context_fusion_multi_using_features()

# channel gating
#exp_hico.train_classifier_channel_gating_using_features()
#exp_hico.train_classifier_channel_gating_multi_using_features()

#exp_hico.train_classifier_using_features_single_region_single_context_early_fusion()

# Exp 0: Local scene experiments: 1) Max-pool over classifier, 2) Learn to aggregate using sigmoid
#exp_context.train_classifier_local_segment_pooling()
#exp_context.train_classifier_deformation()
#exp_context.train_classifier_openpose()
#exp_context.train_classifier_scene()
#exp_context.train_classifier_comb()
#exp_context.train_classifier_comb_scene_deformation()
#exp_context.train_local_context()
#exp_context.train_human_object_pooling()
#exp_context.train_classifier_coco_stuff()
#exp_context.train_classifier_context()

# Exp 1: Single context performances
#exp_hico.train_classifier_using_features_single_context()

# Exp 2: Multiple context performances
#exp_hico.train_classifier_using_features_multiple_context_early_fusion()

# Exp 3: Human-object + single context experiments 
#exp_hico.train_classifier_using_features_multi_region_single_context()
#exp_hico.train_classifier_using_features_multi_region_single_context_late_fusion()

# Exp 4: Human-object + multiple context experiments
exp_hico.train_classifier_using_features_multi_region_multi_context()

# combination of full image hico_feat and places_feat
#exp_hico.train_classifier_using_features_single_region_single_context_early_fusion()
#exp_hico.train_classifier_using_features_single_region_single_context_late_fusion()
#exp_hico.train_classifier_using_features_single_region_single_context_late_early__fusion()
#exp_hico.train_classifier_using_features_pairatt_pose_late_early_fusion()
#exp_hico.train_classifier_using_features_pairatt_pose_late_fusion()
#exp_hico.train_classifier_using_features_single_region_single_context_context_gating()
#exp_hico.train_classifier_using_features_single_region_single_context_context_gating_concat()
#exp_hico.train_classifier_using_features_multiple_region_single_context_context_gating_concat()


# Multiple context late fusion experiments
exp_hico.train_classifier_using_features_late_fusion_multi_context()

# Gated late fusion experiments
exp_hico.train_classifier_using_features_gated_late_fusion()

# endregion
