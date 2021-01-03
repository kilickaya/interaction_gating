#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
We have 1712 videos
We have 52 persons/actors
10 activities
48 action units
persons: 52 total, 44 train, 8 test
videos: 1712 total, 1357 train, 335 test
"""

import os
import numpy as np
import time
import cv2
import natsort
from matplotlib import pyplot as plt
from matplotlib import lines as mlines
from optparse import OptionParser
from MulticoreTSNE import MulticoreTSNE

import torch
import torchsummary

from core import const as c
from core.utils import Path as Pth
from core.image_utils import AsyncImageReaderBreakfastI3DKerasModel, AsyncImageReaderBreakfastResNetTorch
from core import utils, image_utils, video_utils, plot_utils, clustering, sobol, pytorch_utils

from nets import resnet_torch

# region Constants
N_CLASSES_ACTIONS = 48
N_CLASSES_ACTIVITIES = 10

# endregion

# region 1.0 Prepare Annotation

def _101_prepare_action_ids():
    """
    Get list of all unit-actions and activities
    :return:
    """

    video_types = ['cam', 'webcam', 'stereo']

    videos_root_path = Pth('Breakfast/videos')
    unit_actions_path = Pth('Breakfast/annotation/unit_actions_list.pkl')
    activities_path = Pth('Breakfast/annotation/activities_list.pkl')

    person_names = utils.folder_names(videos_root_path, is_nat_sort=True)

    unit_actions = []
    activities = []

    # loop on persons
    for person_name in person_names:
        p_video_root_path = '%s/%s' % (videos_root_path, person_name)

        p_video_types = [n for n in utils.folder_names(p_video_root_path) if __check_correct_video_type(video_types, n)]
        p_video_types = np.array(p_video_types)

        # loop on videos for each person
        for p_video_type in p_video_types:
            # get file names
            instance_video_root_path = '%s/%s' % (p_video_root_path, p_video_type)
            instance_video_names = utils.file_names(instance_video_root_path, is_nat_sort=True)

            # if stereo videos, consider only the first channel
            instance_video_names = [n for n in instance_video_names if utils.get_file_extension(n) == 'avi' and ('stereo' not in p_video_type or 'ch0' in n)]

            # append relative pathes of videos
            instance_video_relative_pathes = ['Breakfast/videos/%s/%s/%s' % (person_name, p_video_type, n) for n in instance_video_names]

            # also, get ground truth for unit-actions and activities
            instance_annot_file_pathes = ['%s/%s.txt' % (instance_video_root_path, utils.remove_extension(n)) for n in instance_video_names]
            instance_unit_actions = __get_action_names_from_files(instance_annot_file_pathes)
            instance_activities = [utils.remove_extension(n).split('_')[1] for n in instance_video_relative_pathes]

            unit_actions += instance_unit_actions
            activities += instance_activities

    activities = np.unique(activities)
    activities = natsort.natsorted(activities)
    activities = np.array(activities)

    unit_actions = np.unique(unit_actions)
    unit_actions = natsort.natsorted(unit_actions)
    unit_actions = np.array(unit_actions)

    print(len(activities), len(unit_actions))
    print(activities)
    print(unit_actions)

    utils.pkl_dump(unit_actions, unit_actions_path)
    utils.pkl_dump(activities, activities_path)

    unit_actions_path = Pth('Breakfast/annotation/unit_actions_list.txt')
    activities_path = Pth('Breakfast/annotation/activities_list.txt')
    utils.txt_dump(unit_actions, unit_actions_path)
    utils.txt_dump(activities, activities_path)

def _102_prepare_video_annot():
    """
    Check ground truth of each video.
    :return:
    """

    video_types = ['cam', 'webcam', 'stereo']

    videos_root_path = Pth('Breakfast/videos')
    unit_actions_path = Pth('Breakfast/annotation/unit_actions_list.pkl')
    activities_path = Pth('Breakfast/annotation/activities_list.pkl')

    annot_unit_actions_path = Pth('Breakfast/annotation/annot_unit_actions.pkl')
    annot_activities_path = Pth('Breakfast/annotation/annot_activities.pkl')

    unit_actions = utils.pkl_load(unit_actions_path)
    activities = utils.pkl_load(activities_path)
    person_names = utils.folder_names(videos_root_path, is_nat_sort=True)

    split_ratio = 0.85
    video_relative_pathes_tr = []
    video_relative_pathes_te = []

    y_unit_actions_tr = []
    y_unit_actions_te = []

    y_activities_tr = []
    y_activities_te = []

    n_persons = len(person_names)
    n_persons_tr = int(n_persons * split_ratio)
    n_persons_te = n_persons - n_persons_tr
    person_names_tr = person_names[:n_persons_tr]
    person_names_te = person_names[n_persons_tr:]

    # loop on persons
    for person_name in person_names:
        p_video_root_path = '%s/%s' % (videos_root_path, person_name)

        p_video_types = [n for n in utils.folder_names(p_video_root_path) if __check_correct_video_type(video_types, n)]
        p_video_types = np.array(p_video_types)

        # loop on videos for each person
        for p_video_type in p_video_types:
            # get file names
            instance_video_root_path = '%s/%s' % (p_video_root_path, p_video_type)
            instance_video_names = utils.file_names(instance_video_root_path, is_nat_sort=True)

            # if stereo videos, consider only the first channel
            instance_video_names = [n for n in instance_video_names if utils.get_file_extension(n) == 'avi' and ('stereo' not in p_video_type or 'ch0' in n)]

            # append relative pathes of videos
            instance_video_relative_pathes = ['%s/%s/%s' % (person_name, p_video_type, n) for n in instance_video_names]

            # also, get ground truth for unit-actions and activities
            instance_activities_y, instance_unit_actions_y = __get_gt_activities_and_actions(instance_video_root_path, instance_video_names, activities, unit_actions)

            if person_name in person_names_tr:
                video_relative_pathes_tr += instance_video_relative_pathes
                y_unit_actions_tr += instance_unit_actions_y
                y_activities_tr += instance_activities_y
            else:
                video_relative_pathes_te += instance_video_relative_pathes
                y_unit_actions_te += instance_unit_actions_y
                y_activities_te += instance_activities_y

    video_relative_pathes_tr = np.array(video_relative_pathes_tr)
    video_relative_pathes_te = np.array(video_relative_pathes_te)

    y_activities_tr = np.array(y_activities_tr)
    y_activities_te = np.array(y_activities_te)

    y_unit_actions_tr = np.array(y_unit_actions_tr)
    y_unit_actions_te = np.array(y_unit_actions_te)

    print(video_relative_pathes_tr.shape)
    print(video_relative_pathes_te.shape)

    print(y_activities_tr.shape)
    print(y_activities_te.shape)

    print(y_unit_actions_tr.shape)
    print(y_unit_actions_te.shape)

    # finally, save video annotation ()
    annot_unit_action = (video_relative_pathes_tr, y_unit_actions_tr, video_relative_pathes_te, y_unit_actions_te)
    annot_activities = (video_relative_pathes_tr, y_activities_tr, video_relative_pathes_te, y_activities_te)
    utils.pkl_dump(annot_unit_action, annot_unit_actions_path)
    utils.pkl_dump(annot_activities, annot_activities_path)

    return

def _103_prepare_video_info():
    video_info_path = Pth('Breakfast/annotation/video_info.pkl')
    annot_activities_path = Pth('Breakfast/annotation/annot_activities.pkl')
    (video_relative_pathes_tr, _, video_relative_pathes_te, _) = utils.pkl_load(annot_activities_path)

    video_relative_pathes = np.hstack((video_relative_pathes_tr, video_relative_pathes_te))
    n_videos = len(video_relative_pathes)

    video_info = dict()
    fps, n_frames, duration = [], [], []

    # loop on the videos
    for idx_video, video_relative_path in enumerate(video_relative_pathes):

        utils.print_counter(idx_video, n_videos, 100)

        video_path = Pth('Breakfast/videos/%s', (video_relative_path,))
        video_id = __video_relative_path_to_video_id(video_relative_path)

        try:
            v_fps, v_n_frames, v_duration = video_utils.get_video_info(video_path)
        except:
            print(video_relative_path)
            continue

        fps.append(v_fps)
        n_frames.append(v_n_frames)
        duration.append(v_duration)
        video_info[video_id] = {'duration': v_duration, 'fps': v_fps, 'n_frames': v_n_frames}

    print(np.mean(fps), np.std(fps), np.min(fps), np.max(fps))
    print(np.mean(duration), np.std(duration), np.min(duration), np.max(duration))
    print(np.mean(n_frames), np.std(n_frames), np.min(n_frames), np.max(n_frames))

    # 15.0 0.0 15.0 15.0
    # 140.30865654205607 121.76493338896255 12.4 649.67
    # 2105.308995327103 1826.5189539717755 187 9746

    utils.pkl_dump(video_info, video_info_path)

def _104_prepare_video_gt():
    video_ids_path = Pth('Breakfast/annotation/video_ids_split.pkl')
    annot_activities_path = Pth('Breakfast/annotation/annot_activities.pkl')
    annot_actions_path = Pth('Breakfast/annotation/annot_unit_actions.pkl')
    gt_activities_path = Pth('Breakfast/annotation/gt_activities.pkl')
    gt_actions_path = Pth('Breakfast/annotation/gt_unit_actions.pkl')
    (video_ids_tr, video_ids_te) = utils.pkl_load(video_ids_path)

    (video_relative_pathes_tr, annot_activities_tr, video_relative_pathes_te, annot_activities_te) = utils.pkl_load(annot_activities_path)
    video_relative_pathes_tr = np.array([utils.remove_extension(p).replace('/', '_') for p in video_relative_pathes_tr])
    video_relative_pathes_te = np.array([utils.remove_extension(p).replace('/', '_') for p in video_relative_pathes_te])

    gt_activities_tr = []
    gt_activities_te = []

    gt_actions_tr = []
    gt_actions_te = []

    for video_id in video_ids_tr:
        idx = np.where(video_id == video_relative_pathes_tr)[0][0]
        gt_activities_tr.append(annot_activities_tr[idx])

    for video_id in video_ids_te:
        idx = np.where(video_id == video_relative_pathes_te)[0][0]
        gt_activities_te.append(annot_activities_te[idx])

    (video_relative_pathes_tr, annot_actions_tr, video_relative_pathes_te, annot_actions_te) = utils.pkl_load(annot_actions_path)
    video_relative_pathes_tr = np.array([utils.remove_extension(p).replace('/', '_') for p in video_relative_pathes_tr])
    video_relative_pathes_te = np.array([utils.remove_extension(p).replace('/', '_') for p in video_relative_pathes_te])

    for video_id in video_ids_tr:
        idx = np.where(video_id == video_relative_pathes_tr)[0][0]
        gt_actions_tr.append(annot_actions_tr[idx])

    for video_id in video_ids_te:
        idx = np.where(video_id == video_relative_pathes_te)[0][0]
        gt_actions_te.append(annot_actions_te[idx])

    gt_activities_tr = np.array(gt_activities_tr)
    gt_activities_te = np.array(gt_activities_te)
    gt_actions_tr = np.array(gt_actions_tr)
    gt_actions_te = np.array(gt_actions_te)

    print(gt_activities_tr.shape)
    print(gt_activities_te.shape)
    print(gt_actions_tr.shape)
    print(gt_actions_te.shape)

    gt_activities_tr = utils.debinarize_label(gt_activities_tr)
    gt_activities_te = utils.debinarize_label(gt_activities_te)
    gt_actions_tr = utils.debinarize_label(gt_actions_tr)
    gt_actions_te = utils.debinarize_label(gt_actions_te)

    utils.pkl_dump(((video_ids_tr, gt_activities_tr, video_ids_te, gt_activities_te)), gt_activities_path)
    utils.pkl_dump(((video_ids_tr, gt_actions_tr, video_ids_te, gt_actions_te)), gt_actions_path)

def _105_prepare_action_gt_timestamped():
    """
    Get ground truth of unit-actions with their timestamps.
    :return:
    """
    root_path = c.DATA_ROOT_PATH
    video_ids_path = Pth('Breakfast/annotation/video_ids_split.pkl')
    unit_actions_path = Pth('Breakfast/annotation/unit_actions_list.pkl')
    gt_actions_path = Pth('Breakfast/annotation/gt_unit_actions_timestamped.pkl')

    (video_ids_tr, video_ids_te) = utils.pkl_load(video_ids_path)
    unit_actions = utils.pkl_load(unit_actions_path)

    video_pathes_tr = ['%s/Breakfast/videos/%s' % (root_path, __video_video_id_to_video_relative_path(id, False),) for id in video_ids_tr]
    video_pathes_te = ['%s/Breakfast/videos/%s' % (root_path, __video_video_id_to_video_relative_path(id, False),) for id in video_ids_te]

    gt_actions_te = __get_gt_actions_timestamped(video_pathes_te, unit_actions)
    gt_actions_tr = __get_gt_actions_timestamped(video_pathes_tr, unit_actions)

    gt_actions_tr = np.array(gt_actions_tr)
    gt_actions_te = np.array(gt_actions_te)

    l_tr = [len(i) for i in gt_actions_tr]
    l_te = [len(i) for i in gt_actions_te]
    print('mean, std, min, max for number of nodes in each video [tr/te]')
    print(np.mean(l_tr), np.std(l_tr), np.min(l_tr), np.max(l_tr))
    print(np.mean(l_te), np.std(l_te), np.min(l_te), np.max(l_te))

    print(gt_actions_tr.shape)
    print(gt_actions_te.shape)

    utils.pkl_dump(((video_ids_tr, gt_actions_tr), (video_ids_te, gt_actions_te)), gt_actions_path)

def _106_prepare_action_graph_vector():
    """
    Each video is labled with a set of actions, we construct a graph using these actions.
    Links represent the relationship between two nodes. A node however represents one action.
    For a video, a link is only drawn between two nodes if these two nodes are neighbours.
    :return:
    """

    gt_actions_path = Pth('Breakfast/annotation/gt_unit_actions_timestamped.pkl')
    action_graph_vectors_path = Pth('Breakfast/annotation/action_graph_vectors.pkl')
    action_graph_matrices_path = Pth('Breakfast/annotation/action_graph_matrices.pkl')
    (video_ids_tr, gt_actions_tr), (video_ids_te, gt_actions_te) = utils.pkl_load(gt_actions_path)

    graph_matrices_tr = __get_action_graph_matrices(video_ids_tr, gt_actions_tr)
    graph_matrices_te = __get_action_graph_matrices(video_ids_te, gt_actions_te)

    graph_vectors_tr = __get_action_graph_vectors(video_ids_tr, gt_actions_tr)
    graph_vectors_te = __get_action_graph_vectors(video_ids_te, gt_actions_te)

    print(graph_matrices_tr.shape)
    print(graph_matrices_te.shape)
    print(graph_vectors_tr.shape)
    print(graph_vectors_te.shape)

    # save the graph data
    utils.pkl_dump((graph_matrices_tr, graph_matrices_te), action_graph_matrices_path)
    utils.pkl_dump((graph_vectors_tr, graph_vectors_te), action_graph_vectors_path)

def _107_prepare_frames_annot():
    """
    Get list of frames from each video.
    """

    frames_root_path = Pth('Breakfast/frames')
    gt_activities_path = Pth('Breakfast/annotation/gt_activities.pkl')
    frames_dict_path = Pth('Breakfast/annotation/frames_dict.pkl')

    (video_ids_tr, y_tr, video_ids_te, y_te) = utils.pkl_load(gt_activities_path)

    video_frames_dict_tr = dict()
    video_frames_dict_te = dict()

    for v_id in video_ids_tr:
        v_id = utils.byte_array_to_string(v_id)
        frames_path = '%s/%s' % (frames_root_path, v_id)
        frame_names = utils.file_names(frames_path, is_nat_sort=True)
        frame_names = np.array(['frames/%s/%s' % (v_id, n) for n in frame_names])
        video_frames_dict_tr[v_id] = frame_names

    for v_id in video_ids_te:
        v_id = utils.byte_array_to_string(v_id)
        frames_path = '%s/%s' % (frames_root_path, v_id)
        frame_names = utils.file_names(frames_path, is_nat_sort=True)
        frame_names = np.array(['frames/%s/%s' % (v_id, n) for n in frame_names])
        video_frames_dict_te[v_id] = frame_names

    data = (video_frames_dict_tr, video_frames_dict_te)
    utils.pkl_dump(data, frames_dict_path)

def __get_action_names_from_files(pathes):
    action_names = []

    for path in pathes:
        lines = utils.txt_load(path)
        for l in lines:
            action_name = l.split(' ')[1]
            action_names.append(action_name)

    return action_names

def __get_gt_activities_and_actions(root_path, video_names, activities, unit_actions):
    y_activities = []
    y_actions = []

    for video_name in video_names:
        # first, get idx of activity
        activity = utils.remove_extension(video_name).split('_')[1]
        idx_activity = np.where(activity == activities)[0][0]
        y_activity = np.zeros((N_CLASSES_ACTIVITIES,), dtype=np.int)
        y_activity[idx_activity] = 1
        y_activities.append(y_activity)

        # then, get idx of actions
        action_txt_path = '%s/%s.txt' % (root_path, utils.remove_extension(video_name))
        lines = utils.txt_load(action_txt_path)
        idx_actions = [np.where(unit_actions == l.split(' ')[1])[0][0] for l in lines]
        y_action = np.zeros((N_CLASSES_ACTIONS,), dtype=np.int)
        y_action[idx_actions] = 1
        y_actions.append(y_action)

    return y_activities, y_actions

def __get_gt_actions_timestamped(video_pathes, unit_actions):
    y_actions = []

    for video_path in video_pathes:
        # then, get idx of actions
        action_txt_path = '%s.txt' % (video_path)
        lines = utils.txt_load(action_txt_path)

        video_annot = []
        for l in lines:
            line_splits = l.split(' ')
            idx_action = np.where(unit_actions == line_splits[1])[0][0]
            frame_start, frame_end = line_splits[0].split('-')
            frame_start = int(frame_start)
            frame_end = int(frame_end)
            video_annot.append((idx_action, frame_start, frame_end))

        y_actions.append(video_annot)

    return y_actions

def __check_correct_video_type(video_types, n):
    for t in video_types:
        if t in n:
            return True
    return False

def __video_relative_path_to_video_id(relative_path):
    video_id = utils.remove_extension(relative_path).replace('/', '_')
    return video_id

def __video_video_id_to_video_relative_path(id, include_extension=True):
    splits = tuple(id.split('_'))
    s_format = '%s/%s/%s_%s' if len(splits) == 4 else '%s/%s/%s_%s_%s'
    video_path = s_format % splits
    video_path = '%s.avi' % video_path if include_extension else video_path
    return video_path

def __get_action_graph_matrices(video_ids, gt_action_timestamped):
    n_videos = len(video_ids)
    n_actions = N_CLASSES_ACTIONS
    n_neighnours = 2

    graph_matrices = np.zeros((n_videos, n_actions, n_actions), dtype=np.int)

    # loop on all videos
    for idx_video, video_id in enumerate(video_ids):

        # matrix to save distances
        graph_matrix = np.zeros((n_actions, n_actions), dtype=np.int)

        # get annotation of certain video
        video_action_labels = gt_action_timestamped[idx_video]

        n_labels = len(video_action_labels)
        n_local_windows = n_labels - n_neighnours

        for idx in range(n_local_windows):

            # get all items inside this local window, items can be either: verbs, nouns or actions
            local_action_labels = video_action_labels[idx:idx + n_neighnours]
            local_ids = np.array([l[0] for l in local_action_labels])

            # add the distances to the matrix, distances are only in this local window
            for i in range(n_neighnours):
                for j in range(i + 1, n_neighnours):
                    id_1 = local_ids[i]
                    id_2 = local_ids[j]

                    # if two nodes are the same, then don't consider
                    if id_1 == id_2:
                        continue
                        # set value = 1 to denote a link
                    graph_matrix[id_1, id_2] = 1

        # add the current matrix to list of matrices
        graph_matrices[idx_video] = graph_matrix

    return graph_matrices

def __get_action_graph_vectors(video_ids, gt_action_timestamped):
    n_actions = N_CLASSES_ACTIONS
    n_videos = len(video_ids)

    # if we have n nouns, then to save all pairwise distances between nouns, we need (n-1)*(n/2) values
    vector_dim = n_actions * (n_actions - 1) * 0.5
    assert vector_dim - int(vector_dim) == 0
    vector_dim = int(vector_dim)
    graph_vectors = np.zeros((n_videos, vector_dim), dtype=np.int)
    idx_matrix = __get_idx_matrix(n_actions)

    # number of neighbours in a local window
    n_neighnours = 2

    # loop on all videos
    for idx_video, video_id in enumerate(video_ids):

        graph_vector = np.zeros((vector_dim,), dtype=np.int)

        # get annotation of certain video
        video_action_labels = gt_action_timestamped[idx_video]

        n_labels = len(video_action_labels)
        n_local_windows = n_labels - n_neighnours

        for idx in range(n_local_windows):

            # get all items inside this local window, items can be either: verbs, nouns or actions
            local_action_labels = video_action_labels[idx:idx + n_neighnours]
            local_ids = np.array([l[0] for l in local_action_labels])

            # add the distances to the matrix, distances are only in this local window
            for i in range(n_neighnours):
                for j in range(i + 1, n_neighnours):
                    id_1 = local_ids[i]
                    id_2 = local_ids[j]
                    # if two nodes are different
                    if id_1 == id_2:
                        continue

                    # add value = 1 to denote a link between current two nodes, i.e. two actions
                    id_vector = idx_matrix[id_1, id_2]
                    graph_vector[id_vector] = 1

        # append the distance_vector
        graph_vectors[idx_video] = graph_vector

    # save distance matrix of the video
    return graph_vectors

def __get_idxes_to_convert_graph_matrix_to_vector(n_dims):
    """
    For a square matrix of size n, we return two lists, each pair in the two lists represent a position in the matrix and it's mirror.
    For example, let n = 3, then it is a 3x3 matrix. Then, the two lists are:
    List1 = [(0, 1), (0, 2), (1, 2)]
    list2 = [(1, 0), (2, 0), (2, 1)]
    :return:
    """

    idxes = []

    for i in range(n_dims):
        for j in range(i, n_dims):
            if j > i:
                idxes.append((i, j))

    idxes = np.array(idxes)
    return idxes

def __get_idx_matrix(n_ids_dict):
    idx_matrix = - 1 * np.ones((n_ids_dict, n_ids_dict), dtype=int)
    idx = -1
    for i in range(n_ids_dict):
        for j in range(n_ids_dict):
            if j > i:
                idx += 1
                idx_matrix[i, j] = idx
                idx_matrix[j, i] = idx

    return idx_matrix

# endregion

# region 2.0 Sample Frames

def _201_extract_frames_wrapper():
    parser = OptionParser()
    parser.add_option("-b", "--begin_num", dest="begin_num", help="begin_num")
    parser.add_option("-e", "--end_num", dest="end_num", help="end_num")
    (options, args) = parser.parse_args()
    begin_num = int(options.begin_num)
    end_num = int(options.end_num)

    _201_extract_frames(begin_num, end_num)

def _201_extract_frames(begin_num, end_num):
    annot_activities_path = Pth('Breakfast/annotation/annot_activities.pkl')
    (video_relative_pathes_tr, _, video_relative_pathes_te, _) = utils.pkl_load(annot_activities_path)

    video_relative_pathes = np.hstack((video_relative_pathes_tr, video_relative_pathes_te))
    n_videos = len(video_relative_pathes)

    image_name_format = '%s/%06d.jpg'

    for idx_video, video_relative_path in enumerate(video_relative_pathes):

        if idx_video < begin_num or idx_video >= end_num:
            continue

        t1 = time.time()
        video_id = __video_relative_path_to_video_id(video_relative_path)
        video_path = Pth('Breakfast/videos/%s', (video_relative_path))

        # path to to store video frames
        video_frames_root_path = Pth('Breakfast/frames/%s', (video_id))
        if not os.path.exists(video_frames_root_path):
            os.mkdir(video_frames_root_path)

        # save all frames to disc
        video_utils.video_save_frames(video_path, video_frames_root_path, image_name_format, c.RESIZE_TYPES[1])
        t2 = time.time()
        duration = t2 - t1
        print('%03d/%03d, %d sec' % (idx_video + 1, end_num, duration))

def _202_sample_frames_i3d():
    """
    Uniformly sample sequences of frames form each video. Each sequences consists of 8 successive frames.
    """

    n_frames_per_segment = 8

    n_frames = 128
    n_frames = 256
    n_frames = 512
    n_frames = 1024
    n_frames = 4096
    n_frames = 8192
    n_timesteps = int(n_frames / float(n_frames_per_segment))
    assert n_timesteps * n_frames_per_segment == n_frames

    model_type = 'i3d'
    annot_activities_path = Pth('Breakfast/annotation/annot_activities.pkl')
    frames_annot_path = Pth('Breakfast/annotation/annot_frames_i3d_%d.pkl', (n_frames,))

    (video_relative_pathes_tr, y_tr, video_relative_pathes_te, y_te) = utils.pkl_load(annot_activities_path)

    video_frames_dict_tr = __sample_frames(video_relative_pathes_tr, n_frames, model_type)
    video_frames_dict_te = __sample_frames(video_relative_pathes_te, n_frames, model_type)

    utils.pkl_dump((video_frames_dict_tr, video_frames_dict_te), frames_annot_path)

def _203_sample_frames_resnet():
    """
    Get list of frames from each video. With max 600 of each video and min 96 frames from each video.
    These frames will be used to extract features for each video.
    """

    # if required frames per video are 128, there are 51/6 out of 7986/1864 videos in training/testing splits that don't satisfy this
    n_frames_per_video = 64
    model_type = 'resnet'

    annot_activities_path = Pth('Breakfast/annotation/annot_activities.pkl')
    frames_annot_path = Pth('Breakfast/annotation/annot_frames_resnet_%d.pkl', (n_frames_per_video,))

    (video_relative_pathes_tr, _, video_relative_pathes_te, _) = utils.pkl_load(annot_activities_path)

    video_frames_dict_tr = __sample_frames(video_relative_pathes_tr, n_frames_per_video, model_type)
    video_frames_dict_te = __sample_frames(video_relative_pathes_te, n_frames_per_video, model_type)

    utils.pkl_dump((video_frames_dict_tr, video_frames_dict_te), frames_annot_path)

def _204_sample_frames_non_local():
    """
    Uniformly sample sequences of frames form each video. Each sequences consists of 8 successive frames.
    """

    n_frames_per_video = 512
    model_type = 'non_local'

    annot_activities_path = Pth('Breakfast/annotation/annot_activities.pkl')
    frames_annot_path = Pth('Breakfast/annotation/annot_frames_non_local_%d.pkl', (n_frames_per_video,))

    (video_relative_pathes_tr, _, video_relative_pathes_te, _) = utils.pkl_load(annot_activities_path)

    video_frames_dict_tr = __sample_frames(video_relative_pathes_tr, n_frames_per_video, model_type)
    video_frames_dict_te = __sample_frames(video_relative_pathes_te, n_frames_per_video, model_type)

    utils.pkl_dump((video_frames_dict_tr, video_frames_dict_te), frames_annot_path)

def __sample_frames(video_relative_pathes, n_frames_per_video, model_type):
    video_frames_dict = dict()
    n_videos = len(video_relative_pathes)

    assert model_type in ['resnet', 'i3d', 'non_local']

    for idx_video, video_relative_path in enumerate(video_relative_pathes):
        utils.print_counter(idx_video, n_videos, 100)
        video_id = __video_relative_path_to_video_id(video_relative_path)

        # get all frames of the video
        frames_root_path = Pth('Breakfast/frames/%s', (video_id,))
        video_frame_names = utils.file_names(frames_root_path, is_nat_sort=True)

        # sample from these frames
        if model_type == 'resnet':
            video_frame_names = __sample_frames_for_resnet(video_frame_names, n_frames_per_video)
        elif model_type == 'i3d':
            video_frame_names = __sample_frames_for_i3d(video_frame_names, n_frames_per_video)
        elif model_type == 'non_local':
            video_frame_names = __sample_frames_for_non_local(video_frame_names, n_frames_per_video)
        else:
            raise Exception('Unkonwn model type: %s' % (model_type))
        n_frames = len(video_frame_names)
        assert n_frames == n_frames_per_video

        video_frames_dict[video_id] = video_frame_names

    return video_frames_dict

def __sample_frames_for_i_dont_know(frames, n_required):
    # get n frames
    n_frames = len(frames)

    if n_frames < n_required:
        repeats = int(n_required / float(n_frames)) + 1
        idx = np.arange(0, n_frames).tolist()
        idx = idx * repeats
        idx = idx[:n_required]
    elif n_frames == n_required:
        idx = np.arange(n_required)
    else:
        start_idx = int((n_frames - n_required) / 2.0)
        stop_idx = start_idx + n_required
        idx = np.arange(start_idx, stop_idx)

    sampled_frames = np.array(frames)[idx]
    assert len(idx) == n_required
    assert len(sampled_frames) == n_required
    return sampled_frames

def __sample_frames_for_resnet(frames, n_required):
    # get n frames from all over the video
    n_frames = len(frames)

    if n_frames < n_required:
        step = n_frames / float(n_required)
        idx = np.arange(0, n_frames, step, dtype=np.float32).astype(np.int32)
    elif n_frames == n_required:
        idx = np.arange(n_required)
    else:
        step = n_frames / float(n_required)
        idx = np.arange(0, n_frames, step, dtype=np.float32).astype(np.int32)

    sampled_frames = np.array(frames)[idx]
    assert len(idx) == n_required
    assert len(sampled_frames) == n_required
    return sampled_frames

def __sample_frames_for_i3d(frames, n_required):
    # i3d model accepts sequence of 8 frames
    n_frames = len(frames)
    segment_length = 8
    n_segments = int(n_required / segment_length)

    assert n_required % segment_length == 0
    assert n_frames > segment_length

    if n_frames < n_required:
        step = (n_frames - segment_length) / float(n_segments)
        idces_start = np.arange(0, n_frames - segment_length, step=step, dtype=np.int)
        idx = []
        for idx_start in idces_start:
            idx += np.arange(idx_start, idx_start + segment_length, dtype=np.int).tolist()
    elif n_frames == n_required:
        idx = np.arange(n_required)
    else:
        step = n_frames / float(n_segments)
        idces_start = np.arange(0, n_frames, step=step, dtype=np.int)
        idx = []
        for idx_start in idces_start:
            idx += np.arange(idx_start, idx_start + segment_length, dtype=np.int).tolist()

    sampled_frames = np.array(frames)[idx]
    return sampled_frames

def __sample_frames_for_non_local(frames, n_required):
    # i3d model accepts sequence of 8 frames
    n_frames = len(frames)
    segment_length = 128
    n_segments = int(n_required / segment_length)

    assert n_required % segment_length == 0
    assert n_frames > segment_length

    if n_frames < n_required:
        step = (n_frames - segment_length) / float(n_segments)
        idces_start = np.arange(0, n_frames - segment_length, step=step, dtype=np.int)
        idx = []
        for idx_start in idces_start:
            idx += np.arange(idx_start, idx_start + segment_length, dtype=np.int).tolist()
    elif n_frames == n_required:
        idx = np.arange(n_required)
    else:
        step = n_frames / float(n_segments)
        idces_start = np.arange(0, n_frames, step=step, dtype=np.int)
        idx = []
        for idx_start in idces_start:
            idx += np.arange(idx_start, idx_start + segment_length, dtype=np.int).tolist()

    sampled_frames = np.array(frames)[idx]
    return sampled_frames

# endregion

# region 3.0 Extract Features

def _301_extract_features_i3d_wrapper():
    parser = OptionParser()
    parser.add_option("-b", "--begin_num", dest="begin_num", help="begin_num")
    parser.add_option("-e", "--end_num", dest="end_num", help="end_num")
    parser.add_option("-c", "--core_id", dest="core_id", help="core_id")
    (options, args) = parser.parse_args()
    begin_num = int(options.begin_num)
    end_num = int(options.end_num)
    core_id = int(options.core_id)

    _301_extract_features_i3d(begin_num, end_num, core_id)

def _301_extract_features_i3d(idx_start, idx_end, core_id):
    __config_session_for_keras(core_id)

    n_frames = 128
    n_frames = 256
    n_frames = 512
    n_frames = 1024
    n_frames = 4096
    n_frames = 8192

    n_frames_per_segment = 8
    n_timesteps = int(n_frames / n_frames_per_segment)
    n_timesteps = int(n_frames / float(n_frames_per_segment))
    assert n_timesteps * n_frames_per_segment

    # which feature to extract
    feature_name = 'softmax'
    feature_name = 'mixed_5c'
    feature_name = 'mixed_5c_maxpool'

    video_ids_path = Pth('Breakfast/annotation/video_ids.pkl')
    frames_annot_path = Pth('Breakfast/annotation/annot_frames_i3d_%d.pkl', (n_frames,))

    video_ids = utils.pkl_load(video_ids_path)
    video_ids = video_ids[idx_start:idx_end]
    (video_frames_dict_tr, video_frames_dict_te) = utils.pkl_load(frames_annot_path)

    features_root_path = Pth('Breakfast/features_i3d_%s_%d_timesteps', (feature_name, n_timesteps))
    if not os.path.exists(features_root_path):
        print('Sorry, feature path does not exist: %s' % (features_root_path))
        return

    video_frames_dict = {}
    video_frames_dict.update(video_frames_dict_tr)
    video_frames_dict.update(video_frames_dict_te)
    del video_frames_dict_tr
    del video_frames_dict_te

    n_threads = 20
    n_videos = len(video_ids)

    # aync reader, and get load images for the first video
    f_pathes = np.array([Pth('Breakfast/frames/%s/%s', (video_ids[0], n)) for n in video_frames_dict[video_ids[0]]])
    img_reader = AsyncImageReaderBreakfastI3DKerasModel(n_threads=n_threads)
    img_reader.load_batch(f_pathes)

    # initialize the model
    if feature_name == 'softmax':
        model = __get_i3d_model_softmax()
    else:
        model = __get_i3d_model_mixed_5c()

    for idx_video, video_id in enumerate(video_ids):

        video_num = idx_video + 1

        # wait untill the image_batch is loaded
        t1 = time.time()
        while img_reader.is_busy():
            time.sleep(0.1)
        t2 = time.time()
        duration_waited = t2 - t1
        print('...... video %d/%d:, waited: %d' % (video_num, n_videos, duration_waited))
        features_path = '%s/%s.pkl' % (features_root_path, video_id)

        # get the video frames
        video_frames = img_reader.get_batch()

        # reshape to get the segments in one dimension
        frames_shape = video_frames.shape
        frames_shape = [n_timesteps, n_frames_per_segment] + list(frames_shape[1:])
        video_frames = np.reshape(video_frames, frames_shape)

        # pre-load for the next video
        if video_num < n_videos:
            next_f_pathes = np.array([Pth('Breakfast/frames/%s/%s', (video_ids[idx_video + 1], n)) for n in video_frames_dict[video_ids[idx_video + 1]]])
            img_reader.load_batch(next_f_pathes)

        # extract features
        features = model.predict(video_frames, verbose=0)

        # squeeze
        if feature_name == 'mixed_5c' or feature_name == 'mixed_5c_maxpool':
            features = np.squeeze(features, axis=1)  # (B, H, W, C)

        # maxpool
        if feature_name == 'mixed_5c_maxpool':
            features = np.max(features, axis=(1, 2), keepdims=True)  # (B, 1, 1, C)

        # channel first
        features = np.transpose(features, (0, 3, 1, 2))  # (B, C, H, W)

        # cast as float_16 for saving space and computation
        features = features.astype(np.float16)

        # finally, save the features
        utils.pkl_dump(features, features_path)

def _302_extract_features_resnet_152_wrapper():
    parser = OptionParser()
    parser.add_option("-b", "--begin_num", dest="begin_num", help="begin_num")
    parser.add_option("-e", "--end_num", dest="end_num", help="end_num")
    parser.add_option("-c", "--core_id", dest="core_id", help="core_id")
    (options, args) = parser.parse_args()
    begin_num = int(options.begin_num)
    end_num = int(options.end_num)
    core_id = int(options.core_id)
    _302_extract_features_resnet_152(begin_num, end_num, core_id)

def _302_extract_features_resnet_152(idx_start, idx_end, core_id):
    """
    Extract frames from each video. Extract only 1 frame for each spf seconds.
    :param spf: How many seconds for each sampled frames.
    :return:
    """

    __config_session_for_keras(core_id)

    frames_annot_path = Pth('Breakfast/annotation/annot_frames_resnet_64.pkl')
    (video_frames_dict_tr, video_frames_dict_te) = utils.pkl_load(frames_annot_path)

    video_names_tr = video_frames_dict_tr.keys()
    video_names_te = video_frames_dict_te.keys()

    video_names = np.hstack((video_names_tr, video_names_te))
    video_names = natsort.natsorted(video_names)
    video_names = np.array(video_names)[idx_start:idx_end]
    n_videos_names = len(video_names)
    video_frames_dict = dict()
    video_frames_dict.update(video_frames_dict_tr)
    video_frames_dict.update(video_frames_dict_te)

    feature_name = 'res5c'
    n_frames_per_video = 64
    features_root_path = '/ssd/nhussein/Breakfast/features_resnet_%s_%d_frames' % (feature_name, n_frames_per_video)
    frames_root_path = '/ssd/nhussein/Breakfast/frames'
    if not os.path.exists(features_root_path):
        print(os.mkdir(features_root_path))

    batch_size = 80
    bgr_mean = np.array([103.939, 116.779, 123.680])

    # load model
    model = resnet152_keras.ResNet152(include_top=False, weights='imagenet')
    model.trainable = False
    print(model.summary())

    # loop on videos
    for idx_video, video_id in enumerate(video_names):
        video_num = idx_video + 1
        video_features_path = '%s/%s.pkl' % (features_root_path, video_id)

        # read frames of the video (in batches), and extract features accordingly

        frames_pathes = video_frames_dict[video_id]
        frames_pathes = np.array(['%s/%s/%s' % (frames_root_path, video_id, n) for n in frames_pathes])

        t1 = time.time()

        # read images
        # video_imgs = __read_and_preprocess_images(frames_pathes, bgr_mean)
        video_imgs = None

        # extract features
        video_features = model.predict(video_imgs, batch_size)

        # save features
        utils.pkl_dump(video_features, video_features_path)

        t2 = time.time()
        duration = int(t2 - t1)
        print('... %d/%d: %s, %d sec' % (video_num, idx_end, video_id, duration))

def _303_extract_features_mobilenet_wrapper():
    parser = OptionParser()
    parser.add_option("-b", "--begin_num", dest="begin_num", help="begin_num")
    parser.add_option("-e", "--end_num", dest="end_num", help="end_num")
    (options, args) = parser.parse_args()
    begin_num = int(options.begin_num)
    end_num = int(options.end_num)
    _303_extract_features_mobilenet(begin_num, end_num)

def _303_extract_features_mobilenet(idx_start, idx_end):
    n_frames_per_video = 64
    model_path = Pth('Torch_Models/MobileNet/mobilenetv3-small-c7eb32fe.pth')
    frames_annot_path = Pth('Breakfast/annotation/annot_frames_i3d_%d.pkl', (n_frames_per_video * 8,))

    # loading data annotation of i3d, but consider the middle frame for each video snippet (of 8 frames)
    (video_frames_dict_tr, video_frames_dict_te) = utils.pkl_load(frames_annot_path)

    video_names_tr = list(video_frames_dict_tr.keys())
    video_names_te = list(video_frames_dict_te.keys())

    video_names = np.hstack((video_names_tr, video_names_te))
    video_names = natsort.natsorted(video_names)
    print('Total videos: %d' % len(video_names))

    video_names = np.array(video_names)[idx_start:idx_end]
    n_videos_names = len(video_names)

    video_frames_dict = dict()
    video_frames_dict.update(video_frames_dict_tr)
    video_frames_dict.update(video_frames_dict_te)

    feature_1_name = 'conv12'
    feature_2_name = 'convpool'
    features_1_root_path = Pth('Breakfast/features_mobilenetv3_small_%s_%d_frames', (feature_1_name, n_frames_per_video))
    features_2_root_path = Pth('Breakfast/features_mobilenetv3_small_%s_%d_frames', (feature_2_name, n_frames_per_video))
    frames_root_path = Pth('Breakfast/frames')

    for p in [features_1_root_path, features_2_root_path]:
        if not os.path.exists(p):
            os.mkdir(p)
            print('Directory created: %s' % (p))

    rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    rgb_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # load model and weights
    model = MobileNetV3(mode='small')
    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict, strict=True)

    # flag the model as testing only
    model = model.cuda()
    model.eval()
    model.training = False

    # print summary
    input_size = (3, 224, 224)  # (B, C, H, W)
    torchsummary.summary(model, input_size)
    batch_size = 360

    # loop on videos
    for idx_video, video_id in enumerate(video_names):
        video_num = idx_video + 1
        video_id_encoded = video_id
        video_id = video_id_encoded.decode("utf-8")
        video_features_1_path = '%s/%s.pkl' % (features_1_root_path, video_id)
        video_features_2_path = '%s/%s.pkl' % (features_2_root_path, video_id)

        # read frames of the video (in batches), and extract features accordingly
        frames_pathes = video_frames_dict[video_id_encoded]
        frames_pathes = [p.decode("utf-8") for p in frames_pathes]
        frames_pathes = np.array(['%s/%s/%s' % (frames_root_path, video_id, n) for n in frames_pathes])

        t1 = time.time()

        # read images
        video_imgs = __read_and_preprocess_images_for_mobilenet(frames_pathes, rgb_mean, rgb_std)

        # channel first for pytorch
        video_imgs = np.transpose(video_imgs, (0, 3, 1, 2))

        # extract features
        video_features_1, video_features_2 = pytorch_utils.batched_feedforward_twin(model, video_imgs, batch_size, 'extract_features')

        # save features
        utils.pkl_dump(video_features_1, video_features_1_path)
        utils.pkl_dump(video_features_2, video_features_2_path)

        t2 = time.time()
        duration = int(t2 - t1)
        print('... %d/%d: %s, %d sec' % (video_num, idx_end, video_id, duration))

def _304_extract_features_resnet_18_wrapper():
    parser = OptionParser()
    parser.add_option("-b", "--begin_num", dest="begin_num", help="begin_num")
    parser.add_option("-e", "--end_num", dest="end_num", help="end_num")
    (options, args) = parser.parse_args()
    begin_num = int(options.begin_num)
    end_num = int(options.end_num)
    _304_extract_features_resnet_18(begin_num, end_num)

def _304_extract_features_resnet_18(idx_start, idx_end):
    n_frames_per_video = 64
    model_path = Pth('Torch_Models/ResNet/resnet18-5c106cde.pth')
    frames_annot_path = Pth('Breakfast/annotation/annot_frames_i3d_%d.pkl', (n_frames_per_video * 8,))

    # loading data annotation of i3d, but consider the middle frame for each video snippet (of 8 frames)
    (video_frames_dict_tr, video_frames_dict_te) = utils.pkl_load(frames_annot_path)

    video_names_tr = list(video_frames_dict_tr.keys())
    video_names_te = list(video_frames_dict_te.keys())

    video_names = np.hstack((video_names_tr, video_names_te))
    video_names = natsort.natsorted(video_names)
    print('Total videos: %d' % len(video_names))

    video_names = np.array(video_names)[idx_start:idx_end]
    n_videos = len(video_names)

    video_frames_dict = dict()
    video_frames_dict.update(video_frames_dict_tr)
    video_frames_dict.update(video_frames_dict_te)

    feature_1_name = 'conv5c'
    feature_2_name = 'conv5c_pool'
    features_1_root_path = Pth('Breakfast/features_resnet18_%s_%d_frames', (feature_1_name, n_frames_per_video))
    features_2_root_path = Pth('Breakfast/features_resnet18_%s_%d_frames', (feature_2_name, n_frames_per_video))
    frames_root_path = Pth('Breakfast/frames')

    for p in [features_1_root_path, features_2_root_path]:
        if not os.path.exists(p):
            os.mkdir(p)
            print('Directory created: %s' % (p))

    rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    rgb_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    n_threads = 20

    # load model and weights
    model = resnet_torch.resnet18()
    pytorch_utils.load_model_dict(model, model_path)

    # flag the model as testing only
    model = model.cuda()
    model.training = False
    model.train(False)
    model.eval()

    # print summary
    input_size = (3, 224, 224)  # (B, C, H, W)
    torchsummary.summary(model, input_size)
    batch_size = 300

    f_pathes = np.array(['%s/%s/%s' % (frames_root_path, video_names[0].decode("utf-8"), n.decode("utf-8")) for n in video_frames_dict[video_names[0]]])
    img_reader = AsyncImageReaderBreakfastResNetTorch(rgb_mean, rgb_std, n_threads)
    img_reader.load_batch(f_pathes)

    # loop on videos
    for idx_video in range(n_videos):

        video_num = idx_video + 1
        video_id = video_names[idx_video]
        video_id_encoded = video_id
        video_id = video_id_encoded.decode("utf-8")
        video_features_1_path = '%s/%s.pkl' % (features_1_root_path, video_id)
        video_features_2_path = '%s/%s.pkl' % (features_2_root_path, video_id)

        # wait untill the image_batch is loaded
        t1 = time.time()
        while img_reader.is_busy():
            time.sleep(0.1)
        t2 = time.time()
        duration_waited = t2 - t1
        print('...... video %d/%d:, waited: %d' % (video_num, n_videos, duration_waited))

        # get the video frames
        video_imgs = img_reader.get_batch()

        # pre-load for the next video
        if video_num < n_videos:
            next_video_id = video_names[idx_video + 1]
            next_video_id_decoded = next_video_id.decode("utf-8")
            next_f_pathes = np.array(['%s/%s/%s' % (frames_root_path, next_video_id_decoded, n.decode("utf-8")) for n in video_frames_dict[next_video_id]])
            img_reader.load_batch(next_f_pathes)

        t1 = time.time()

        # channel first for pytorch
        video_imgs = np.transpose(video_imgs, (0, 3, 1, 2))

        # extract features
        video_features_1, video_features_2 = pytorch_utils.batched_feedforward_twin(model, video_imgs, batch_size, 'extract_features')

        # save features
        utils.pkl_dump(video_features_1, video_features_1_path)
        utils.pkl_dump(video_features_2, video_features_2_path)

        t2 = time.time()
        duration = int(t2 - t1)
        print('... %d/%d: %s, %d sec' % (video_num, idx_end, video_id, duration))

def _304_extract_features_resnet_18_finedtuned():
    from experiments import exp_hico

    n_timesteps = 1024
    batch_size = 300

    model_path = Pth('Breakfast/basenets/resnet18_breakfast_finetuned.pt')
    video_pathes_tr, _, video_pathes_te, _ = exp_hico.__sample_frames_for_resnet(n_timesteps, is_random_tr=False, is_random_te=False)
    video_pathes = np.vstack((video_pathes_tr, video_pathes_te))
    del video_pathes_tr
    del video_pathes_te

    n_videos = len(video_pathes)
    print('Total videos: %d' % n_videos)

    feature_name = 'conv4'
    features_root_path = Pth('Breakfast/features_resnet18_v3_%s_%d_frames', (feature_name, n_timesteps))

    if not os.path.exists(features_root_path):
        os.mkdir(features_root_path)
        print('Directory created: %s' % (features_root_path))

    rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    rgb_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    n_threads = 20

    # load model and weights
    model = resnet_torch.resnet18()
    pytorch_utils.load_model_dict(model, model_path)

    # flag the model as testing only
    model = model.cuda()
    model.training = False
    model.train(False)
    model.eval()

    # print summary
    input_size = (3, 224, 224)  # (B, C, H, W)
    torchsummary.summary(model, input_size)

    # asyn reader
    img_reader = AsyncImageReaderBreakfastResNetTorch(rgb_mean, rgb_std, n_threads)

    # get first batch of frames
    img_reader.load_batch(video_pathes[0])

    # loop on videos
    for idx_video in range(n_videos):

        video_num = idx_video + 1
        v_pathes = video_pathes[idx_video]
        video_name = v_pathes[0].split('/')[7]

        video_features_path = '%s/%s' % (features_root_path, video_name)
        if not os.path.exists(video_features_path):
            os.mkdir(video_features_path)

        # wait untill the image_batch is loaded
        t1 = time.time()
        while img_reader.is_busy():
            time.sleep(0.1)
        t2 = time.time()
        duration_waited = t2 - t1
        print('...... video %d/%d:, waited: %d' % (video_num, n_videos, duration_waited))

        # get the video frames
        video_imgs = img_reader.get_batch()

        # pre-load for the next video
        if video_num < n_videos:
            next_f_pathes = video_pathes[idx_video + 1]
            img_reader.load_batch(next_f_pathes)

        t1 = time.time()

        # channel first for pytorch
        video_imgs = np.transpose(video_imgs, (0, 3, 1, 2))

        # extract features
        video_features = pytorch_utils.batched_feedforward(model, video_imgs, batch_size, 'forward_till_conv3')

        # as float 16
        video_features = video_features.astype(np.float16)

        # save feature of each frame in a separate file
        for idx_timestep in range(n_timesteps):
            timestep_num = idx_timestep + 1
            frame_feature_path = '%s/%s/%05d.pkl' % (features_root_path, video_name, timestep_num)
            frame_feature = video_features[idx_timestep]

            utils.pkl_dump(frame_feature, frame_feature_path)

        t2 = time.time()
        duration = int(t2 - t1)
        print('... %d/%d: %s, %d sec' % (video_num, n_videos, video_name, duration))

def _305_extract_features_resnet_34_wrapper():
    parser = OptionParser()
    parser.add_option("-b", "--begin_num", dest="begin_num", help="begin_num")
    parser.add_option("-e", "--end_num", dest="end_num", help="end_num")
    (options, args) = parser.parse_args()
    begin_num = int(options.begin_num)
    end_num = int(options.end_num)
    _305_extract_features_resnet_34(begin_num, end_num)

def _305_extract_features_resnet_34(idx_start, idx_end):
    n_frames_per_video = 64
    model_path = Pth('Torch_Models/ResNet/resnet34-333f7ec4.pth')
    frames_annot_path = Pth('Breakfast/annotation/annot_frames_i3d_%d.pkl', (n_frames_per_video * 8,))

    # loading data annotation of i3d, but consider the middle frame for each video snippet (of 8 frames)
    (video_frames_dict_tr, video_frames_dict_te) = utils.pkl_load(frames_annot_path)

    video_names_tr = list(video_frames_dict_tr.keys())
    video_names_te = list(video_frames_dict_te.keys())

    video_names = np.hstack((video_names_tr, video_names_te))
    video_names = natsort.natsorted(video_names)
    print('Total videos: %d' % len(video_names))

    video_names = np.array(video_names)[idx_start:idx_end]
    n_videos = len(video_names)

    video_frames_dict = dict()
    video_frames_dict.update(video_frames_dict_tr)
    video_frames_dict.update(video_frames_dict_te)

    feature_1_name = 'conv5c'
    feature_2_name = 'conv5c_pool'
    features_1_root_path = Pth('Breakfast/features_resnet18_%s_%d_frames', (feature_1_name, n_frames_per_video))
    features_2_root_path = Pth('Breakfast/features_resnet18_%s_%d_frames', (feature_2_name, n_frames_per_video))
    frames_root_path = Pth('Breakfast/frames')

    for p in [features_1_root_path, features_2_root_path]:
        if not os.path.exists(p):
            os.mkdir(p)
            print('Directory created: %s' % (p))

    rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    rgb_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    n_threads = 20

    # load model and weights
    model = resnet_torch.resnet34()
    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict, strict=True)

    # flag the model as testing only
    model = model.cuda()
    model.eval()
    model.training = False

    # print summary
    input_size = (3, 224, 224)  # (B, C, H, W)
    torchsummary.summary(model, input_size)
    batch_size = 300

    f_pathes = np.array(['%s/%s/%s' % (frames_root_path, video_names[0].decode("utf-8"), n.decode("utf-8")) for n in video_frames_dict[video_names[0]]])
    img_reader = AsyncImageReaderBreakfastResNetTorch(rgb_mean, rgb_std, n_threads)
    img_reader.load_batch(f_pathes)

    # loop on videos
    for idx_video in range(n_videos):

        video_num = idx_video + 1
        video_id = video_names[idx_video]
        video_id_encoded = video_id
        video_id = video_id_encoded.decode("utf-8")
        video_features_1_path = '%s/%s.pkl' % (features_1_root_path, video_id)
        video_features_2_path = '%s/%s.pkl' % (features_2_root_path, video_id)

        # wait untill the image_batch is loaded
        t1 = time.time()
        while img_reader.is_busy():
            time.sleep(0.1)
        t2 = time.time()
        duration_waited = t2 - t1
        print('...... video %d/%d:, waited: %d' % (video_num, n_videos, duration_waited))

        # get the video frames
        video_imgs = img_reader.get_batch()

        # pre-load for the next video
        if video_num < n_videos:
            next_video_id = video_names[idx_video + 1]
            next_video_id_decoded = next_video_id.decode("utf-8")
            next_f_pathes = np.array(['%s/%s/%s' % (frames_root_path, next_video_id_decoded, n.decode("utf-8")) for n in video_frames_dict[next_video_id]])
            img_reader.load_batch(next_f_pathes)

        t1 = time.time()

        # channel first for pytorch
        video_imgs = np.transpose(video_imgs, (0, 3, 1, 2))

        # extract features
        video_features_1, video_features_2 = pytorch_utils.batched_feedforward_twin(model, video_imgs, batch_size, 'extract_features')

        print(video_features_1.shape)
        print(video_features_2.shape)
        return

        # save features
        utils.pkl_dump(video_features_1, video_features_1_path)
        utils.pkl_dump(video_features_2, video_features_2_path)

        t2 = time.time()
        duration = int(t2 - t1)
        print('... %d/%d: %s, %d sec' % (video_num, idx_end, video_id, duration))

def _306_extract_features_resnet_50_wrapper():
    parser = OptionParser()
    parser.add_option("-b", "--begin_num", dest="begin_num", help="begin_num")
    parser.add_option("-e", "--end_num", dest="end_num", help="end_num")
    (options, args) = parser.parse_args()
    begin_num = int(options.begin_num)
    end_num = int(options.end_num)
    _306_extract_features_resnet_50(begin_num, end_num)

def _306_extract_features_resnet_50(idx_start, idx_end):
    n_segments = 64
    n_frames = n_segments * 8
    model_path = Pth('Torch_Models/ResNet/resnet50-19c8e357.pth')
    frames_annot_path = Pth('Breakfast/annotation/annot_frames_i3d_%d.pkl', (n_frames,))

    # loading data annotation of i3d, but consider the middle frame for each video snippet (of 8 frames)
    (video_frames_dict_tr, video_frames_dict_te) = utils.pkl_load(frames_annot_path)

    video_names_tr = list(video_frames_dict_tr.keys())
    video_names_te = list(video_frames_dict_te.keys())

    video_names = np.hstack((video_names_tr, video_names_te))
    video_names = natsort.natsorted(video_names)
    print('Total videos: %d' % len(video_names))

    video_names = np.array(video_names)[idx_start:idx_end]
    n_videos = len(video_names)

    video_frames_dict = dict()
    video_frames_dict.update(video_frames_dict_tr)
    video_frames_dict.update(video_frames_dict_te)

    feature_1_name = 'conv5c'
    feature_2_name = 'conv5c_pool'
    features_1_root_path = Pth('Breakfast/features_resnet50_%s_%d_frames', (feature_1_name, n_frames))
    features_2_root_path = Pth('Breakfast/features_resnet50_%s_%d_frames', (feature_2_name, n_frames))
    frames_root_path = Pth('Breakfast/frames')

    for p in [features_1_root_path, features_2_root_path]:
        if not os.path.exists(p):
            os.mkdir(p)
            print('Directory created: %s' % (p))

    rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    rgb_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    n_threads = 20

    # load model and weights
    model = resnet_torch.resnet50()
    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict, strict=True)

    # flag the model as testing only
    model = model.cuda()
    model.eval()
    model.training = False

    # print summary
    input_size = (3, 224, 224)  # (B, C, H, W)
    torchsummary.summary(model, input_size)
    batch_size = 100

    f_pathes = np.array(['%s/%s/%s' % (frames_root_path, video_names[0].decode("utf-8"), n.decode("utf-8")) for n in video_frames_dict[video_names[0]]])
    img_reader = AsyncImageReaderBreakfastResNetTorch(rgb_mean, rgb_std, n_threads)
    img_reader.load_batch(f_pathes)

    # loop on videos
    for idx_video in range(n_videos):

        video_num = idx_video + 1
        video_id = video_names[idx_video]
        video_id_encoded = video_id
        video_id = video_id_encoded.decode("utf-8")
        video_features_1_path = '%s/%s.pkl' % (features_1_root_path, video_id)
        video_features_2_path = '%s/%s.pkl' % (features_2_root_path, video_id)

        # wait untill the image_batch is loaded
        t1 = time.time()
        while img_reader.is_busy():
            time.sleep(0.1)
        t2 = time.time()
        duration_waited = t2 - t1
        print('...... video %d/%d:, waited: %d' % (video_num, n_videos, duration_waited))

        # get the video frames
        video_imgs = img_reader.get_batch()

        # pre-load for the next video
        if video_num < n_videos:
            next_video_id = video_names[idx_video + 1]
            next_video_id_decoded = next_video_id.decode("utf-8")
            next_f_pathes = np.array(['%s/%s/%s' % (frames_root_path, next_video_id_decoded, n.decode("utf-8")) for n in video_frames_dict[next_video_id]])
            img_reader.load_batch(next_f_pathes)

        t1 = time.time()

        # channel first for pytorch
        video_imgs = np.transpose(video_imgs, (0, 3, 1, 2))

        # extract features
        video_features_1, video_features_2 = pytorch_utils.batched_feedforward_twin(model, video_imgs, batch_size, 'extract_features')

        # save features
        utils.pkl_dump(video_features_1, video_features_1_path)
        utils.pkl_dump(video_features_2, video_features_2_path)

        t2 = time.time()
        duration = int(t2 - t1)
        print('... %d/%d: %s, %d sec' % (video_num, idx_end, video_id, duration))

def __read_and_preprocess_images_for_i3d(img_pathes, bgr_mean):
    n_imgs = len(img_pathes)
    imgs = np.zeros((n_imgs, 224, 224, 3), np.float32)

    for idx, img_path in enumerate(img_pathes):
        # read image
        img = cv2.imread(img_path)
        img = img.astype(np.float32)

        # subtract mean pixel from image
        img[:, :, 0] -= bgr_mean[0]
        img[:, :, 1] -= bgr_mean[1]
        img[:, :, 2] -= bgr_mean[2]

        # convert from bgr to rgb
        img = img[:, :, (2, 1, 0)]

        imgs[idx] = img

    return imgs

def __read_and_preprocess_images_for_mobilenet(img_pathes, rgb_mean, rgb_std):
    n_imgs = len(img_pathes)
    imgs = np.zeros((n_imgs, 224, 224, 3), np.float32)

    for idx, img_path in enumerate(img_pathes):
        # load test imag, and pre-process it
        img = cv2.imread(img_path)
        img = img[:, :, (2, 1, 0)]
        img = image_utils.resize_crop(img)
        img = img.astype(np.float32)

        # normalize image
        img /= 255.0
        img[:, :] -= rgb_mean
        img[:, :] /= rgb_std

        imgs[idx] = img

    return imgs

def __get_i3d_model_mixed_5c():
    NUM_CLASSES = 400
    input_shape = (8, 224, 224, 3)

    # build model for RGB data, and load pretrained weights (trained on imagenet and kinetics dataset)
    i3d_model = Inception_Inflated3d(include_top=False, weights='rgb_imagenet_and_kinetics', input_shape=input_shape, classes=NUM_CLASSES)

    # set model as non-trainable
    for layer in i3d_model.layers:
        layer.trainable = False
    i3d_model.trainable = False

    return i3d_model

def __get_i3d_model_softmax():
    NUM_CLASSES = 400
    input_shape = (8, 224, 224, 3)

    # build model for RGB data, and load pretrained weights (trained on imagenet and kinetics dataset)
    i3d_model = Inception_Inflated3d(include_top=True, weights='rgb_imagenet_and_kinetics', input_shape=input_shape, classes=NUM_CLASSES)

    # set model as non-trainable
    for layer in i3d_model.layers:
        layer.trainable = False
    i3d_model.trainable = False

    return i3d_model

def __config_session_for_keras(gpu_core_id):
    from keras import backend as K
    import tensorflow as tf

    K.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(gpu_core_id)
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)

# endregion

# region 4.0 Stats

def _401_durations_stats():
    video_info_path = Pth('Breakfast/annotation/video_info.pkl')

    video_info = utils.pkl_load(video_info_path)
    video_ids = video_info.keys()

    durations = [video_info[id]['duration'] for id in video_ids]
    durations = np.sort(durations)

    plot_utils.plot(durations)

# endregion

# region 5.0 Analysis

def _501_tsne_of_actions_gt(is_2d=True, is_multi_core=True):
    import seaborn as sns

    sns.set()
    # sns.set(style="whitegrid")

    gt_activities_path = Pth('Breakfast/annotation/gt_activities.pkl')
    gt_actions_path = Pth('Breakfast/annotation/gt_unit_actions.pkl')
    activity_names_path = Pth('Breakfast/annotation/activities_list.pkl')
    (_, y_tr, _, y_te) = utils.pkl_load(gt_activities_path)
    (_, x_tr, _, x_te) = utils.pkl_load(gt_actions_path)

    x = np.vstack((x_tr, x_te))
    y = np.vstack((y_tr, y_te))
    activity_names = utils.pkl_load(activity_names_path)

    n_classes_actions = N_CLASSES_ACTIONS
    n_classes_activities = N_CLASSES_ACTIVITIES
    class_nums = np.arange(0, n_classes_activities)

    print(x.shape)
    print(y.shape)

    # embed the features into a low-dim manifold
    n_components = 2 if is_2d else 3
    manifold_type = c.MANIFOLD_TYPES[0]

    # learn manifold
    print('... learning manifold embedding')
    t1 = time.time()
    if is_2d and is_multi_core and manifold_type == c.MANIFOLD_TYPES[0]:
        tsne = MulticoreTSNE(n_jobs=32, n_components=n_components)
        x = x.astype(np.float64)
        x_fitted = tsne.fit_transform(x)
    else:
        x_fitted = utils.learn_manifold(manifold_type, x, n_components)
    t2 = time.time()
    duration_min = int((t2 - t1) / 60.0)
    print('... learned in %d min' % (duration_min))

    # split features from centroids
    print(x_fitted.shape)

    colors = plot_utils.tableau_category10()
    fig = plt.figure(1, (10, 8))
    ax = fig.add_subplot(111, projection='3d') if not is_2d else fig.gca()

    alpha = 0.5
    marker_size = 50
    legend_handles = []

    # loop on all the features and plot them
    for i in range(n_classes_activities):

        # name of the current activity
        a_name = activity_names[i]

        # get only the features that correspond to the current cluster
        idx = np.where(y == class_nums[i])[0]
        a_feats = x_fitted[idx]

        # custom legend
        legend_handles.append(mlines.Line2D([], [], color=colors[i], marker='o', linestyle='None', markersize=6, label=a_name))

        # plot features assigned to the current category
        if is_2d:
            ax.scatter(a_feats[:, 0], a_feats[:, 1], s=marker_size, c=colors[i], lw=0, alpha=alpha)
        else:
            ax.scatter(a_feats[:, 0], a_feats[:, 1], a_feats[:, 2], s=marker_size, c=colors[i], lw=0, alpha=alpha, label=a_name)

    if is_2d:
        # ax.axis('off')
        ax.grid(True)
    else:
        ax.grid(True)

    plt.legend(loc='best', fancybox=True, framealpha=1.0, handles=legend_handles)
    plt.tight_layout()
    plt.show()

def _502_graph_based_representation():
    """
    Get all activites in a video, represent them as a graph. See how graph-based representation can used as feature to classify videos.
    """

    gt_activities_path = Pth('Breakfast/annotation/gt_activities.pkl')
    gt_actions_path = Pth('Breakfast/annotation/gt_unit_actions.pkl')

    # (video_relative_pathes_tr, annot_activities_tr, video_relative_pathes_te, annot_activities_te) = utils.pkl_load(annot_activities_path)

    pass

# endregion

# region 6.0 Clustering

def _601_kmeans_clusteting():
    # load features
    n_timesteps = 512
    n_centroids = 200
    n_iterations = 1000
    n_samples_max = 10 * 1000

    features_path = Pth('Breakfast/features/features_i3d_mixed_5c_%d_frames.h5', (n_timesteps,))
    centroids_path = Pth('Breakfast/features/centroids_i3d_mixed_5c_%d_frames_%d_centroids.pkl', (n_timesteps, n_centroids))

    x, = utils.h5_load_multi(features_path, ['x_tr', ])  # (None, 64, 7, 7, 1024)

    # max-pool the features
    x = np.max(x, axis=(2, 3))
    print(x.shape)

    n_samples, n_timesteps, feat_dim = x.shape

    # reshape to hide time dimension
    x = np.reshape(x, (-1, feat_dim)).astype(np.float32)
    print(x.shape)

    # sample from features
    idx = np.random.randint(0, n_samples * n_timesteps, (n_samples_max,))
    x = x[idx]
    print(x.shape)

    # cluster the features
    centroids = clustering.kmeans_clustering(x, n_centroids, n_iterations)
    print(x.shape)

    # save as pickle
    utils.pkl_dump(centroids, centroids_path)

def _602_generate_centroids(n_centroids, n_dims):
    pass

    c1_path = Pth('Breakfast/features/centroids_random_%d_centroids.pkl', (n_centroids,))
    c2_path = Pth('Breakfast/features/centroids_sobol_%d_centroids.pkl', (n_centroids,))

    # centroids as random vectors
    c1 = np.random.rand(n_centroids, n_dims)

    # centroids as sobol sequence
    c2 = sobol.sobol_generate(n_dims, n_centroids)
    c2 = np.array(c2)

    print(c1.shape)
    print(c2.shape)

    # save centroids
    utils.pkl_dump(c1, c1_path)
    utils.pkl_dump(c2, c2_path)

# endregion

# region 7.0 Pickle Features

def pickle_features_i3d_mixed_5c():
    n_frames_per_video = 512
    features_root_path = Pth('Breakfast/features_i3d_mixed_5c_%d_frames', (n_frames_per_video,))
    features_path = Pth('Breakfast/features/features_i3d_mixed_5c_%d_frames.h5', (n_frames_per_video,))
    video_ids_path = Pth('Breakfast/annotation/video_ids_split.pkl')

    (video_ids_tr, video_ids_te) = utils.pkl_load(video_ids_path)

    n_tr = len(video_ids_tr)
    n_te = len(video_ids_te)

    n_frames_per_segment = 8
    n_segments = int(n_frames_per_video / n_frames_per_segment)
    assert n_segments * n_frames_per_segment == n_frames_per_video

    f_tr = np.zeros((n_tr, n_segments, 7, 7, 1024), dtype=np.float16)
    f_te = np.zeros((n_te, n_segments, 7, 7, 1024), dtype=np.float16)

    for i in range(n_tr):
        utils.print_counter(i, n_tr, 100)
        p = '%s/%s.pkl' % (features_root_path, video_ids_tr[i])
        f = utils.pkl_load(p)  # (T, 7, 7, 2048)
        f_tr[i] = f

    for i in range(n_te):
        utils.print_counter(i, n_te, 100)
        p = '%s/%s.pkl' % (features_root_path, video_ids_te[i])
        f = utils.pkl_load(p)  # (T, 7, 7, 2048)
        f_te[i] = f

    print(f_tr.shape)
    print(f_te.shape)

    print(utils.get_size_in_gb(utils.get_array_memory_size(f_tr)))
    print(utils.get_size_in_gb(utils.get_array_memory_size(f_te)))

    data_names = ['x_tr', 'x_te']
    utils.h5_dump_multi((f_tr, f_te), data_names, features_path)

def pickle_features_i3d_mixed_5c_maxpool():
    n_timesteps = 1024
    n_timesteps = 128
    features_root_path = Pth('Breakfast/features_i3d_mixed_5c_maxpool_%d_timesteps', (n_timesteps,))
    features_path = Pth('Breakfast/features/features_i3d_mixed_5c_maxpool_%d_timesteps.h5', (n_timesteps,))
    video_ids_path = Pth('Breakfast/annotation/video_ids_split.pkl')

    (video_ids_tr, video_ids_te) = utils.pkl_load(video_ids_path)
    n_tr = len(video_ids_tr)
    n_te = len(video_ids_te)

    f_tr = np.zeros((n_tr, 1024, n_timesteps, 1, 1), dtype=np.float16)  # (B, C, T, H, W)
    f_te = np.zeros((n_te, 1024, n_timesteps, 1, 1), dtype=np.float16)  # (B, C, T, H, W)

    for i in range(n_tr):
        utils.print_counter(i, n_tr, 100)
        p = '%s/%s.pkl' % (features_root_path, video_ids_tr[i])
        f = utils.pkl_load(p)  # (T, C, H, W)
        f = np.transpose(f, (1, 0, 2, 3))  # (C, T, H, W)
        f_tr[i] = f

    for i in range(n_te):
        utils.print_counter(i, n_te, 100)
        p = '%s/%s.pkl' % (features_root_path, video_ids_te[i])
        f = utils.pkl_load(p)  # (T, C, H, W)
        f = np.transpose(f, (1, 0, 2, 3))  # (C, T, H, W)
        f_te[i] = f

    print(f_tr.shape)
    print(f_te.shape)

    print(utils.get_size_in_gb(utils.get_array_memory_size(f_tr)))
    print(utils.get_size_in_gb(utils.get_array_memory_size(f_te)))

    data_names = ['x_tr', 'x_te']
    utils.h5_dump_multi((f_tr, f_te), data_names, features_path)

def pickle_features_i3d_softmax():
    n_frames_per_video = 512
    features_root_path = Pth('Breakfast/features_i3d_softmax_%d_frames', (n_frames_per_video,))
    features_path = Pth('Breakfast/features/features_i3d_softmax_%d_frames.h5', (n_frames_per_video,))
    video_ids_path = Pth('Breakfast/annotation/video_ids_split.pkl')

    (video_ids_tr, video_ids_te) = utils.pkl_load(video_ids_path)

    n_tr = len(video_ids_tr)
    n_te = len(video_ids_te)

    n_frames_per_segment = 8
    n_segments = int(n_frames_per_video / n_frames_per_segment)
    assert n_segments * n_frames_per_segment == n_frames_per_video

    f_tr = np.zeros((n_tr, n_segments, 400), dtype=np.float16)
    f_te = np.zeros((n_te, n_segments, 400), dtype=np.float16)

    for i in range(n_tr):
        utils.print_counter(i, n_tr, 100)
        p = '%s/%s.pkl' % (features_root_path, video_ids_tr[i])
        f = utils.pkl_load(p)  # (T, 400)
        f_tr[i] = f

    for i in range(n_te):
        utils.print_counter(i, n_te, 100)
        p = '%s/%s.pkl' % (features_root_path, video_ids_te[i])
        f = utils.pkl_load(p)  # (T, 400)
        f_te[i] = f

    print(f_tr.shape)
    print(f_te.shape)

    print(utils.get_size_in_gb(utils.get_array_memory_size(f_tr)))
    print(utils.get_size_in_gb(utils.get_array_memory_size(f_te)))

    data_names = ['x_tr', 'x_te']
    utils.h5_dump_multi((f_tr, f_te), data_names, features_path)

def pickle_features_resnet152():
    n_frames_per_video = 512
    features_root_path = '/ssd/nhussein/Breakfast/features_resnet152_res5c_64_frames'
    features_path = '/ssd/nhussein/Breakfast/features/features_resnet152_res5c_64_frames.h5'
    video_ids_path = Pth('Breakfast/annotation/video_ids_split.pkl')

    (video_ids_tr, video_ids_te) = utils.pkl_load(video_ids_path)

    n_tr = len(video_ids_tr)
    n_te = len(video_ids_te)

    f_tr = np.zeros((n_tr, 64, 7, 7, 2048), dtype=np.float16)
    f_te = np.zeros((n_te, 64, 7, 7, 2048), dtype=np.float16)

    for i in range(n_tr):
        utils.print_counter(i, n_tr, 100)
        p = '%s/%s.pkl' % (features_root_path, video_ids_tr[i])
        f = utils.pkl_load(p)  # (T, 400)
        f_tr[i] = f

    for i in range(n_te):
        utils.print_counter(i, n_te, 100)
        p = '%s/%s.pkl' % (features_root_path, video_ids_te[i])
        f = utils.pkl_load(p)  # (T, 400)
        f_te[i] = f

    print(f_tr.shape)
    print(f_te.shape)

    print(utils.get_size_in_gb(utils.get_array_memory_size(f_tr)))
    print(utils.get_size_in_gb(utils.get_array_memory_size(f_te)))

    data_names = ['x_tr', 'x_te']
    utils.h5_dump_multi((f_tr, f_te), data_names, features_path)

def pickle_features_mobilenet():
    n_frames = 512
    n_frames_per_segment = 8
    n_segments = int(n_frames / n_frames_per_segment)
    assert n_segments * n_frames_per_segment == n_frames

    video_ids_path = Pth('Breakfast/annotation/video_ids_split.pkl')
    features_1_root_path = Pth('Breakfast/features_mobilenetv3_small_conv12_%d_frames', (n_frames,))
    features_2_root_path = Pth('Breakfast/features_mobilenetv3_small_convpool_%d_frames', (n_frames,))
    features_1_path = Pth('Breakfast/features/features_mobilenetv3_small_conv12_%d_frames.h5', (n_segments,))
    features_2_path = Pth('Breakfast/features/features_mobilenetv3_small_convpool_%d_frames.h5', (n_segments,))

    (video_ids_tr, video_ids_te) = utils.pkl_load(video_ids_path)
    video_ids_tr = [s.decode('utf-8') for s in video_ids_tr]
    video_ids_te = [s.decode('utf-8') for s in video_ids_te]

    n_tr = len(video_ids_tr)
    n_te = len(video_ids_te)

    f1_tr = np.zeros((n_tr, n_segments, 576, 7, 7), dtype=np.float16)
    f1_te = np.zeros((n_te, n_segments, 576, 7, 7), dtype=np.float16)
    f2_tr = np.zeros((n_tr, n_segments, 576, 1, 1), dtype=np.float16)
    f2_te = np.zeros((n_te, n_segments, 576, 1, 1), dtype=np.float16)

    print(utils.get_size_in_gb(utils.get_array_memory_size(f1_tr)))
    print(utils.get_size_in_gb(utils.get_array_memory_size(f1_te)))
    print(utils.get_size_in_gb(utils.get_array_memory_size(f2_tr)))
    print(utils.get_size_in_gb(utils.get_array_memory_size(f2_te)))

    for i in range(n_tr):
        utils.print_counter(i, n_tr, 100)
        p1 = '%s/%s.pkl' % (features_1_root_path, video_ids_tr[i])
        p2 = '%s/%s.pkl' % (features_2_root_path, video_ids_tr[i])
        f1 = utils.pkl_load(p1)
        f2 = utils.pkl_load(p2)
        f1 = __pick_up_middle_frame_from_segment(f1)
        f2 = __pick_up_middle_frame_from_segment(f2)
        f1_tr[i] = f1
        f2_tr[i] = f2

    for i in range(n_te):
        utils.print_counter(i, n_te, 100)
        p1 = '%s/%s.pkl' % (features_1_root_path, video_ids_te[i])
        p2 = '%s/%s.pkl' % (features_2_root_path, video_ids_te[i])
        f1 = utils.pkl_load(p1)
        f2 = utils.pkl_load(p2)
        f1 = __pick_up_middle_frame_from_segment(f1)
        f2 = __pick_up_middle_frame_from_segment(f2)
        f1_te[i] = f1
        f2_te[i] = f2

    print(f1_tr.shape)
    print(f1_te.shape)
    print(f2_tr.shape)
    print(f2_te.shape)

    print(utils.get_size_in_gb(utils.get_array_memory_size(f1_tr)))
    print(utils.get_size_in_gb(utils.get_array_memory_size(f1_te)))
    print(utils.get_size_in_gb(utils.get_array_memory_size(f2_tr)))
    print(utils.get_size_in_gb(utils.get_array_memory_size(f2_te)))

    data_names = ['x_tr', 'x_te']
    utils.h5_dump_multi((f1_tr, f1_te), data_names, features_1_path)
    utils.h5_dump_multi((f2_tr, f2_te), data_names, features_2_path)

def pickle_features_resnet18():
    n_frames = 512
    n_frames_per_segment = 8
    n_segments = int(n_frames / n_frames_per_segment)
    assert n_segments * n_frames_per_segment == n_frames

    model_name = c.CNN_MODEL_TYPES.resnet18
    feature_1_type = c.CNN_FEATURE_TYPES.conv5c
    feature_2_type = c.CNN_FEATURE_TYPES.conv5c_pool

    video_ids_path = Pth('Breakfast/annotation/video_ids_split.pkl')
    features_1_root_path = Pth('Breakfast/features_%s_%s_%d_frames', (model_name, feature_1_type, n_frames,))
    features_2_root_path = Pth('Breakfast/features_%s_%s_%d_frames', (model_name, feature_2_type, n_frames,))
    features_1_path = Pth('Breakfast/features/features_%s_%s_%d_timesteps.h5', (model_name, feature_1_type, n_segments,))
    features_2_path = Pth('Breakfast/features/features_%s_%s_%d_timesteps.h5', (model_name, feature_2_type, n_segments,))

    (video_ids_tr, video_ids_te) = utils.pkl_load(video_ids_path)
    video_ids_tr = [s.decode('utf-8') for s in video_ids_tr]
    video_ids_te = [s.decode('utf-8') for s in video_ids_te]

    n_tr = len(video_ids_tr)
    n_te = len(video_ids_te)

    f1_tr = np.zeros((n_tr, 512, n_segments, 7, 7), dtype=np.float16)
    f1_te = np.zeros((n_te, 512, n_segments, 7, 7), dtype=np.float16)
    f2_tr = np.zeros((n_tr, 512, n_segments, 1, 1), dtype=np.float16)
    f2_te = np.zeros((n_te, 512, n_segments, 1, 1), dtype=np.float16)

    print(utils.get_size_in_gb(utils.get_array_memory_size(f1_tr)))
    print(utils.get_size_in_gb(utils.get_array_memory_size(f1_te)))
    print(utils.get_size_in_gb(utils.get_array_memory_size(f2_tr)))
    print(utils.get_size_in_gb(utils.get_array_memory_size(f2_te)))

    for i in range(n_tr):
        utils.print_counter(i, n_tr, 100)
        p1 = '%s/%s.pkl' % (features_1_root_path, video_ids_tr[i])
        p2 = '%s/%s.pkl' % (features_2_root_path, video_ids_tr[i])
        f1 = utils.pkl_load(p1)
        f2 = utils.pkl_load(p2)
        f1 = __pick_up_middle_frame_from_segment(f1)
        f2 = __pick_up_middle_frame_from_segment(f2)
        f1_tr[i] = f1
        f2_tr[i] = f2

    for i in range(n_te):
        utils.print_counter(i, n_te, 100)
        p1 = '%s/%s.pkl' % (features_1_root_path, video_ids_te[i])
        p2 = '%s/%s.pkl' % (features_2_root_path, video_ids_te[i])
        f1 = utils.pkl_load(p1)
        f2 = utils.pkl_load(p2)
        f1 = __pick_up_middle_frame_from_segment(f1)
        f2 = __pick_up_middle_frame_from_segment(f2)
        f1_te[i] = f1
        f2_te[i] = f2

    print(f1_tr.shape)
    print(f1_te.shape)
    print(f2_tr.shape)
    print(f2_te.shape)

    print(utils.get_size_in_gb(utils.get_array_memory_size(f1_tr)))
    print(utils.get_size_in_gb(utils.get_array_memory_size(f1_te)))
    print(utils.get_size_in_gb(utils.get_array_memory_size(f2_tr)))
    print(utils.get_size_in_gb(utils.get_array_memory_size(f2_te)))

    data_names = ['x_tr', 'x_te']
    utils.h5_dump_multi((f1_tr, f1_te), data_names, features_1_path)
    utils.h5_dump_multi((f2_tr, f2_te), data_names, features_2_path)

def pickle_features_resnet50():
    n_frames = 512
    n_frames_per_segment = 8
    n_segments = int(n_frames / n_frames_per_segment)
    assert n_segments * n_frames_per_segment == n_frames

    model_name = c.CNN_MODEL_TYPES.resnet50
    feature_1_type = c.CNN_FEATURE_TYPES.conv5c
    feature_2_type = c.CNN_FEATURE_TYPES.conv5c_pool

    video_ids_path = Pth('Breakfast/annotation/video_ids_split.pkl')
    features_1_root_path = Pth('Breakfast/features_%s_%s_%d_frames', (model_name, feature_1_type, n_frames,))
    features_2_root_path = Pth('Breakfast/features_%s_%s_%d_frames', (model_name, feature_2_type, n_frames,))
    features_1_path = Pth('Breakfast/features/features_%s_%s_%d_timesteps.h5', (model_name, feature_1_type, n_segments,))
    features_2_path = Pth('Breakfast/features/features_%s_%s_%d_timesteps.h5', (model_name, feature_2_type, n_segments,))

    (video_ids_tr, video_ids_te) = utils.pkl_load(video_ids_path)
    video_ids_tr = [s.decode('utf-8') for s in video_ids_tr]
    video_ids_te = [s.decode('utf-8') for s in video_ids_te]

    n_tr = len(video_ids_tr)
    n_te = len(video_ids_te)

    f1_tr = np.zeros((n_tr, 2048, n_segments, 7, 7), dtype=np.float16)
    f1_te = np.zeros((n_te, 2048, n_segments, 7, 7), dtype=np.float16)
    f2_tr = np.zeros((n_tr, 2048, n_segments, 1, 1), dtype=np.float16)
    f2_te = np.zeros((n_te, 2048, n_segments, 1, 1), dtype=np.float16)

    print(utils.get_size_in_gb(utils.get_array_memory_size(f1_tr)))
    print(utils.get_size_in_gb(utils.get_array_memory_size(f1_te)))
    print(utils.get_size_in_gb(utils.get_array_memory_size(f2_tr)))
    print(utils.get_size_in_gb(utils.get_array_memory_size(f2_te)))

    for i in range(n_tr):
        utils.print_counter(i, n_tr, 100)
        p1 = '%s/%s.pkl' % (features_1_root_path, video_ids_tr[i])
        p2 = '%s/%s.pkl' % (features_2_root_path, video_ids_tr[i])
        f1 = utils.pkl_load(p1)
        f2 = utils.pkl_load(p2)
        f1 = __pick_up_middle_frame_from_segment(f1)
        f2 = __pick_up_middle_frame_from_segment(f2)
        f1_tr[i] = f1
        f2_tr[i] = f2

    for i in range(n_te):
        utils.print_counter(i, n_te, 100)
        p1 = '%s/%s.pkl' % (features_1_root_path, video_ids_te[i])
        p2 = '%s/%s.pkl' % (features_2_root_path, video_ids_te[i])
        f1 = utils.pkl_load(p1)
        f2 = utils.pkl_load(p2)
        f1 = __pick_up_middle_frame_from_segment(f1)
        f2 = __pick_up_middle_frame_from_segment(f2)
        f1_te[i] = f1
        f2_te[i] = f2

    print(f1_tr.shape)
    print(f1_te.shape)
    print(f2_tr.shape)
    print(f2_te.shape)

    print(utils.get_size_in_gb(utils.get_array_memory_size(f1_tr)))
    print(utils.get_size_in_gb(utils.get_array_memory_size(f1_te)))
    print(utils.get_size_in_gb(utils.get_array_memory_size(f2_tr)))
    print(utils.get_size_in_gb(utils.get_array_memory_size(f2_te)))

    data_names = ['x_tr', 'x_te']
    utils.h5_dump_multi((f1_tr, f1_te), data_names, features_1_path)
    utils.h5_dump_multi((f2_tr, f2_te), data_names, features_2_path)

def __pick_up_middle_frame_from_segment(feature):
    n_frames_per_segment = 8
    f_shape = feature.shape
    T, C, H, W = f_shape

    n_frames = f_shape[0]
    n_segments = int(n_frames / n_frames_per_segment)

    # reshape
    feature = np.reshape(feature, (n_segments, n_frames_per_segment, C, H, W))

    # pick up only one feature from each segment
    feature = feature[:, 3]

    # channel first
    feature = np.transpose(feature, (1, 0, 2, 3))

    return feature

# endregion

# region Misc

def __dummy():
    pass

# endregion
