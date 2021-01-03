import numpy as np
import pickle as pkl
import h5py
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn import preprocessing, manifold

import os
import json
import natsort
import random
from multiprocessing.dummy import Pool
from core import utils, plot_utils, const

from core.utils import Path as Pth

# region Readers: Feature

class ReaderVideoFeatureBreakfast():
    def __init__(self, frame_feature_shape, video_feature_shape, n_timesteps, n_threads=20):
        random.seed(101)
        np.random.seed(101)

        self.__is_busy = False
        self.__features = None
        self.__frame_feature_shape = frame_feature_shape
        self.__video_feature_shape = video_feature_shape
        self.__n_timesteps = n_timesteps
        self.__n_threads = n_threads

        self.__pool = Pool(self.__n_threads)

    def load_batch(self, feature_pathes):
        self.__is_busy = True

        n_features = len(feature_pathes)
        idx_features = np.arange(0, n_features)

        # parameters passed to the reading function
        params = []
        for idx_timestep in range(self.__n_timesteps):
            p = [[feature_pathes[idx_feature][idx_timestep], idx_feature, idx_timestep] for idx_feature in idx_features]
            params += p

        # set list of features before start reading
        all_features_shape = [n_features] + list(self.__video_feature_shape)
        self.__features = np.zeros(all_features_shape, dtype=np.float32)  # (B, T, C, H, W)

        # start pool of threads
        self.__pool.map_async(self.__read_features_wrapper, params, callback=self.__thread_pool_callback)

    def get_batch(self):
        if self.__is_busy:
            raise Exception('Sorry, you can\'t get features while threads are running!')
        else:
            # convert to channel first
            features = self.__features  # (B, T, C, H, W)
            features = np.transpose(features, (0, 2, 1, 3, 4))  # (B, C, T, H, W)
            return features

    def is_busy(self):
        return self.__is_busy

    def __thread_pool_callback(self, args):
        self.__is_busy = False

    def __read_features_wrapper(self, params):
        try:
            self.__read_features(params)
        except Exception as exp:
            print('Error in __read_features')
            print(exp)

    def __read_features(self, params):

        features_path, idx_feature, idx_timestep = params
        frame_feature = utils.pkl_load(features_path)

        # add current feature to the list
        self.__features[idx_feature, idx_timestep] = frame_feature

    def close(self):
        self.__pool.close()
        self.__pool.terminate()

class ReaderFrameFeaturesBreakfast():
    def __init__(self, __feature_shape, n_threads=20):
        random.seed(101)
        np.random.seed(101)

        self.__is_busy = False
        self.__features = None
        self.__feature_shape = __feature_shape
        self.__n_threads = n_threads

        self.__pool = Pool(self.__n_threads)

    def load_batch(self, feature_pathes):
        self.__is_busy = True

        n_features = len(feature_pathes)
        idxces = np.arange(0, n_features)

        # parameters passed to the reading function
        params = [data_item for data_item in zip(idxces, feature_pathes)]

        # set list of features before start reading
        all_features_shape = [n_features] + list(self.__feature_shape)
        self.__features = np.zeros(all_features_shape, dtype=np.float32)  # (B, C, T, H, W)

        # start pool of threads
        self.__pool.map_async(self.__read_features_wrapper, params, callback=self.__thread_pool_callback)

    def get_batch(self):
        if self.__is_busy:
            raise Exception('Sorry, you can\'t get features while threads are running!')
        else:
            return self.__features

    def is_busy(self):
        return self.__is_busy

    def __thread_pool_callback(self, args):
        self.__is_busy = False

    def __read_features_wrapper(self, params):
        try:
            self.__read_features(params)
        except Exception as exp:
            print('Error in __read_features')
            print(exp)

    def __read_features(self, params):

        idx = params[0]
        path = params[1]

        # load feature
        print(path)
        feature = utils.pkl_load(path)

        # add current feature to the list
        self.__features[idx] = feature

    def close(self):
        self.__pool.close()
        self.__pool.terminate()

# endregion

# region Samplers: Feature Pathes

class SamplersFeaturePathesBreakfast():
    """
    Select feature pathes based on selection scores.
    """

    def __init__(self, features_root_path, n_timesteps, n_timesteps_total, is_random_tr, is_random_te):

        gt_activities_path = Pth('Breakfast/annotation/gt_activities.pkl')
        (self.__video_ids_tr, self.__y_tr, self.__video_ids_te, self.__y_te) = utils.pkl_load(gt_activities_path)

        self.__feature_root_path = features_root_path
        self.__n_timesteps_total = n_timesteps_total
        self.__n_timesteps = n_timesteps
        self.__is_random_tr = is_random_tr
        self.__is_random_te = is_random_te

    def sample_train(self):

        # shuffle training
        self.__video_ids_tr, self.__y_tr = shuffle_bi(self.__video_ids_tr, self.__y_tr)

        # sample feature pathes
        x_tr, y_tr = self.__sample_features(self.__video_ids_tr, self.__y_tr, self.__is_random_tr)
        return x_tr, y_tr

    def sample_test(self):

        # sample feature pathes
        x_te, y_te = self.__sample_features(self.__video_ids_te, self.__y_te, self.__is_random_te)
        return x_te, y_te

    def __sample_features(self, video_names, y, is_random):
        features_pathes = []

        n_timesteps = self.__n_timesteps
        n_timesteps_total = self.__n_timesteps_total
        features_root_path = self.__feature_root_path

        # loop on video names and select the timesteps, either uniform or random
        for idx_video, v_name in enumerate(video_names):

            if is_random:
                # random sampling
                idxes = np.random.randint(0, n_timesteps_total, (n_timesteps,))
                idxes = np.sort(idxes)
            else:
                # uniform sampling
                step = n_timesteps_total / float(n_timesteps)
                idxes = np.arange(0, n_timesteps_total, step, dtype=np.float32).astype(np.int32)

            # convert idxes to frame feature pathes
            video_feature_pathes = np.array(['%s/%s/%05d.pkl' % (features_root_path, v_name, idx + 1) for idx in idxes])
            features_pathes.append(video_feature_pathes)

        features_pathes = np.array(features_pathes)
        return features_pathes, y

# endregion

# region Sampler: Image Pathes

class SamplersImagePathesBreakfast():
    """
    Select feature pathes based on selection scores.
    """

    def __init__(self, img_root_path, is_shuffle_tr=True, is_shuffle_te=False):
        annot_path = Pth('Hico/annotation/anno_hico.pkl')

        (self.img_names_tr, self.y_tr, self.img_names_te, self.y_te) = utils.pkl_load(annot_path)

        self.y_tr = self.y_tr.astype(np.float32)
        self.y_te = self.y_te.astype(np.float32)

        self.is_shuffle_tr = is_shuffle_tr
        self.is_shuffle_te = is_shuffle_te

        self.img_names_tr = np.array(['%s/%s' % (img_root_path, n) for n in self.img_names_tr])
        self.img_names_te = np.array(['%s/%s' % (img_root_path, n) for n in self.img_names_te])

    def sample_train(self):
        data = self.__sample_images(self.img_names_tr, self.y_tr, self.is_shuffle_tr)
        return data

    def sample_test(self):
        data = self.__sample_images(self.img_names_te, self.y_te, self.is_shuffle_te)
        return data

    def __sample_images(self, img_names, y, is_shuffle):
        data = shuffle_bi(img_names, y) if is_shuffle else (img_names, y)
        return data

# endregion

# region Samplers: VideoFrame Pathes

class SamplerVideoFramePathesCharades():
    def __init__(self, n_timesteps, is_random_tr=True, is_random_te=False, is_shuffle_tr=True, is_shuffle_te=False):
        """
        :param n_timesteps:  How many timesteps per video.
        :param is_random_tr: Sample random or uniform frames.
        :param is_random_te: Sample random or uniform frames.
        :param is_shuffle_tr: To shuffle data or not.
        :param is_shuffle_te: To shuffle data or not.
        """

        self.__is_random_tr = is_random_tr
        self.__is_random_te = is_random_te
        self.__is_shuffle_tr = is_shuffle_tr
        self.__is_shuffle_te = is_shuffle_te
        self.__n_timesteps = n_timesteps

        frames_dict_path = Pth('Charades/annotation/frames_dict_all_frames.pkl')
        annotation_path = Pth('Charades/annotation/video_annotation.pkl')

        (self.__video_frames_dict_tr, self.__video_frames_dict_te) = utils.pkl_load(frames_dict_path)
        (self.__video_ids_tr, self.__y_tr, self.__video_ids_te, self.__y_te) = utils.pkl_load(annotation_path)

        self.__y_tr = self.__y_tr.astype(np.float32)
        self.__y_te = self.__y_te.astype(np.float32)

    def sample_train(self):

        # sample training data
        x_tr = self.__sample_frames_from_videos(self.__video_frames_dict_tr, self.__video_ids_tr, self.__n_timesteps, self.__is_random_tr)
        y_tr = self.__y_tr

        # shuffle training data
        if self.__is_shuffle_tr:
            x_tr, y_tr = shuffle_bi(x_tr, y_tr)

        return x_tr, y_tr

    def sample_test(self):

        # sample test data
        x_te = self.__sample_frames_from_videos(self.__video_frames_dict_te, self.__video_ids_te, self.__n_timesteps, self.__is_random_te)
        y_te = self.__y_te

        # shuffle test data
        if self.__is_shuffle_te:
            x_te, y_te = shuffle_bi(x_te, y_te)

        return x_te, y_te

    def __sample_frames_from_videos(self, video_frames_dict, video_ids, n_frames_required, is_random):
        frame_pathes = []
        frames_root_path = Pth('Charades/Charades_v1_rgb')

        for idx_video, video_id in enumerate(video_ids):
            # get all frames of the video
            video_frames = np.array(video_frames_dict[video_id])
            n_frames_total = len(video_frames)

            # sample from these frames. Random or uniform
            if is_random:
                idx = np.random.randint(0, n_frames_total, (n_frames_required,))
                idx = np.sort(idx)
            else:
                step = n_frames_total / float(n_frames_required)
                idx = np.arange(0, n_frames_total, step, dtype=np.float32).astype(np.int32)

            sampled_video_frames = np.array(video_frames)[idx]

            assert len(sampled_video_frames) == n_frames_required

            # get full path
            sampled_video_frames = np.array(['%s/%s/%s' % (frames_root_path, video_id, f) for f in sampled_video_frames])
            frame_pathes.append(sampled_video_frames)

        frame_pathes = np.array(frame_pathes)
        return frame_pathes

class SamplerVideoFramePathesI3dCharades():
    """
    Sample n timesteps, each consists of n_successive frames.
    """

    def __init__(self, n_timesteps, is_random_tr=True, is_random_te=False, is_shuffle_tr=True, is_shuffle_te=False):
        """
        :param n_timesteps:  How many timesteps per video.
        :param is_random_tr: Sample random or uniform frames.
        :param is_random_te: Sample random or uniform frames.
        :param is_shuffle_tr: To shuffle data or not.
        :param is_shuffle_te: To shuffle data or not.
        """

        frames_dict_path = Pth('Charades/annotation/frames_dict_all_frames.pkl')
        annotation_path = Pth('Charades/annotation/video_annotation.pkl')

        self.__is_random_tr = is_random_tr
        self.__is_random_te = is_random_te
        self.__is_shuffle_tr = is_shuffle_tr
        self.__is_shuffle_te = is_shuffle_te
        self.__n_timesteps = n_timesteps

        self.__n_frames_per_segment = 8
        self.__n_frames = self.__n_timesteps * self.__n_frames_per_segment

        (self.__video_frames_dict_tr, self.__video_frames_dict_te) = utils.pkl_load(frames_dict_path)
        (self.__video_ids_tr, self.__y_tr, self.__video_ids_te, self.__y_te) = utils.pkl_load(annotation_path)

        self.current_train = None
        self.current_test = None

    def sample_train(self):

        # sample training data
        x_tr = self.__sample_frames_from_videos(self.__video_frames_dict_tr, self.__video_ids_tr, self.__n_timesteps, self.__is_random_tr)
        y_tr = self.__y_tr

        # shuffle training data
        if self.__is_shuffle_tr:
            x_tr, y_tr = shuffle_bi(x_tr, y_tr)

        data = (x_tr, y_tr)
        self.current_train = data

        return data

    def sample_test(self):

        # sample test data
        x_te = self.__sample_frames_from_videos(self.__video_frames_dict_te, self.__video_ids_te, self.__n_timesteps, self.__is_random_te)
        y_te = self.__y_te

        # shuffle test data
        if self.__is_shuffle_te:
            x_te, y_te = shuffle_bi(x_te, y_te)

        data = (x_te, y_te)
        self.current_test = data

        return data

    def __sample_frames_from_videos(self, video_frames_dict, video_ids, n_frames_required, is_random):
        frame_pathes = []
        frames_root_path = Pth('Charades/Charades_v1_rgb')

        n_frames_per_segment = self.__n_frames_per_segment
        n_segments = int(n_frames_required / n_frames_per_segment)

        for idx_video, video_id in enumerate(video_ids):

            # get all frames of the video
            video_frames = np.array(video_frames_dict[video_id])
            n_frames_total = len(video_frames)

            if is_random:
                idxes_start = np.random.randint(0, n_frames_total - n_frames_per_segment - 1, (n_segments,))
                idxes_start = np.sort(idxes_start)
            else:
                step = (n_frames_total - n_frames_per_segment) / float(n_segments)
                idxes_start = np.arange(0, n_frames_total - n_frames_per_segment - 1, step=step, dtype=np.float32).astype(np.int32)
            idx = []
            for idx_start in idxes_start:
                idx += np.arange(idx_start, idx_start + n_frames_per_segment, dtype=np.int32).tolist()

            sampled_video_frames = video_frames[idx]

            assert len(sampled_video_frames) == n_frames_required

            # get full path
            sampled_video_frames = np.array(['%s/%s/%s' % (frames_root_path, video_id, f) for f in sampled_video_frames])
            frame_pathes.append(sampled_video_frames)

        frame_pathes = np.array(frame_pathes)
        return frame_pathes

class SamplerVideoFramePathesBreakfast():
    def __init__(self, n_timesteps, is_random_tr=True, is_random_te=False, is_shuffle_tr=True, is_shuffle_te=False):
        """
        :param n_timesteps:  How many timesteps per video.
        :param is_random_tr: Sample random or uniform frames.
        :param is_random_te: Sample random or uniform frames.
        :param is_shuffle_tr: To shuffle data or not.
        :param is_shuffle_te: To shuffle data or not.
        """

        self.__is_random_tr = is_random_tr
        self.__is_random_te = is_random_te
        self.__is_shuffle_tr = is_shuffle_tr
        self.__is_shuffle_te = is_shuffle_te
        self.__n_timesteps = n_timesteps

        gt_activities_path = Pth('Breakfast/annotation/gt_activities.pkl')
        frames_dict_path = Pth('Breakfast/annotation/frames_dict.pkl')

        (self.__video_frames_dict_tr, self.__video_frames_dict_te) = utils.pkl_load(frames_dict_path)
        (self.__video_ids_tr, self.__y_tr, self.__video_ids_te, self.__y_te) = utils.pkl_load(gt_activities_path)

    def sample_train(self):

        # sample training data
        x_tr = self.__sample_frames_from_videos(self.__video_frames_dict_tr, self.__video_ids_tr, self.__n_timesteps, self.__is_random_tr)
        y_tr = self.__y_tr

        # shuffle training data
        if self.__is_shuffle_tr:
            x_tr, y_tr = shuffle_bi(x_tr, y_tr)

        return x_tr, y_tr

    def sample_test(self):

        # sample test data
        x_te = self.__sample_frames_from_videos(self.__video_frames_dict_te, self.__video_ids_te, self.__n_timesteps, self.__is_random_te)
        y_te = self.__y_te

        # shuffle test data
        if self.__is_shuffle_te:
            x_te, y_te = shuffle_bi(x_te, y_te)

        return x_te, y_te

    def __sample_frames_from_videos(self, video_frames_dict, video_ids, n_frames_required, is_random):
        frame_pathes = []
        frames_root_path = Pth('Breakfast')

        for idx_video, video_id in enumerate(video_ids):
            # get all frames of the video
            video_frames = video_frames_dict[video_id]
            n_frames_total = len(video_frames)

            # sample from these frames. Random or uniform
            if is_random:
                idx = np.random.randint(0, n_frames_total, (n_frames_required,))
                idx = np.sort(idx)
            else:
                step = n_frames_total / float(n_frames_required)
                idx = np.arange(0, n_frames_total, step, dtype=np.float32).astype(np.int32)

            sampled_video_frames = np.array(video_frames)[idx]

            assert len(sampled_video_frames) == n_frames_required

            # get full path
            sampled_video_frames = np.array(['%s/%s' % (frames_root_path, f) for f in sampled_video_frames])
            frame_pathes.append(sampled_video_frames)

        frame_pathes = np.array(frame_pathes)
        return frame_pathes

class SamplerVideoFramePathesI3dBreakfast():
    """
    Sample n timesteps, each consists of n_successive frames.
    """

    def __init__(self, n_timesteps, is_random_tr=True, is_random_te=False, is_shuffle_tr=True, is_shuffle_te=False, is_binarize_output=False):
        """
        :param n_timesteps:  How many timesteps per video.
        :param is_random_tr: Sample random or uniform frames.
        :param is_random_te: Sample random or uniform frames.
        :param is_shuffle_tr: To shuffle data or not.
        :param is_shuffle_te: To shuffle data or not.
        """

        self.__is_random_tr = is_random_tr
        self.__is_random_te = is_random_te
        self.__is_shuffle_tr = is_shuffle_tr
        self.__is_shuffle_te = is_shuffle_te
        self.__n_timesteps = n_timesteps

        self.__n_frames_per_segment = 8
        self.__n_frames = self.__n_timesteps * self.__n_frames_per_segment

        gt_activities_path = Pth('Breakfast/annotation/gt_activities.pkl')
        frames_dict_path = Pth('Breakfast/annotation/frames_dict.pkl')

        (self.__video_frames_dict_tr, self.__video_frames_dict_te) = utils.pkl_load(frames_dict_path)
        (self.__video_ids_tr, self.__y_tr, self.__video_ids_te, self.__y_te) = utils.pkl_load(gt_activities_path)

        if is_binarize_output:
            classes = np.arange(0, 10)
            self.__y_tr = utils.binarize_label(self.__y_tr, classes)
            self.__y_te = utils.binarize_label(self.__y_te, classes)

    def sample_train(self):

        # sample training data
        x_tr = self.__sample_frames_from_videos(self.__video_frames_dict_tr, self.__video_ids_tr, self.__n_timesteps, self.__is_random_tr)
        y_tr = self.__y_tr

        # shuffle training data
        if self.__is_shuffle_tr:
            x_tr, y_tr = shuffle_bi(x_tr, y_tr)

        return x_tr, y_tr

    def sample_test(self):

        # sample test data
        x_te = self.__sample_frames_from_videos(self.__video_frames_dict_te, self.__video_ids_te, self.__n_timesteps, self.__is_random_te)
        y_te = self.__y_te

        # shuffle test data
        if self.__is_shuffle_te:
            x_te, y_te = shuffle_bi(x_te, y_te)

        return x_te, y_te

    def __sample_frames_from_videos(self, video_frames_dict, video_ids, n_frames_required, is_random):
        frame_pathes = []
        frames_root_path = Pth('Breakfast')

        for idx_video, video_id in enumerate(video_ids):

            # get all frames of the video
            video_frames = video_frames_dict[video_id]
            n_frames_total = len(video_frames)

            # get starting point, random or middle
            if is_random:
                idx_start = np.random.randint(0, n_frames_total - n_frames_required - 1)
            else:
                idx_start = int((n_frames_total - n_frames_required) / 2.0) - 1

            idx_stop = idx_start + n_frames_required
            sampled_video_frames = video_frames[idx_start:idx_stop]

            assert len(sampled_video_frames) == n_frames_required

            # get full path
            sampled_video_frames = np.array(['%s/%s' % (frames_root_path, f) for f in sampled_video_frames])
            frame_pathes.append(sampled_video_frames)

        frame_pathes = np.array(frame_pathes)
        return frame_pathes

class SamplerVideoFramePathesAndFeaturesBreakfast():
    def __init__(self, n_timesteps, n_timesteps_total, featurenet_type, x_heavy_path, is_random_tr=True, is_random_te=False, is_shuffle_tr=True, is_shuffle_te=False):
        """
        :param n_timesteps:  How many timesteps per video.
        :param is_random_tr: Sample random or uniform frames.
        :param is_random_te: Sample random or uniform frames.
        :param is_shuffle_tr: To shuffle data or not.
        :param is_shuffle_te: To shuffle data or not.
        """

        self.__is_random_tr = is_random_tr
        self.__is_random_te = is_random_te
        self.__is_shuffle_tr = is_shuffle_tr
        self.__is_shuffle_te = is_shuffle_te
        self.__n_timesteps = n_timesteps
        self.__n_timesteps_total = n_timesteps_total

        n_frames_per_segment = utils.get_model_n_frames_per_segment(featurenet_type)
        n_frames = n_timesteps_total * n_frames_per_segment

        gt_activities_path = Pth('Breakfast/annotation/gt_activities.pkl')
        frames_annot_path = Pth('Breakfast/annotation/annot_frames_%s_%d.pkl', (featurenet_type, n_frames,))

        (self.__video_ids_tr, self.__y_tr, self.__video_ids_te, self.__y_te) = utils.pkl_load(gt_activities_path)

        (x_heavy_tr, x_heavy_te) = utils.h5_load_multi(x_heavy_path, ['x_tr', 'x_te'])  # (B, C, T, H, W)
        self.__x_heavy_tr = x_heavy_tr
        self.__x_heavy_te = x_heavy_te

        # select middle frame from each snippet
        (frames_dict_tr, frames_dict_te) = utils.pkl_load(frames_annot_path)
        frames_dict_tr = self.__select_middle_frame(frames_dict_tr, n_frames_per_segment)
        frames_dict_te = self.__select_middle_frame(frames_dict_te, n_frames_per_segment)
        self.__frames_dict_tr = frames_dict_tr
        self.__frames_dict_te = frames_dict_te

    def sample_train(self):

        # sample training data
        path_tr, x_heavy_tr = self.__sample_frames_from_videos(self.__frames_dict_tr, self.__video_ids_tr, self.__x_heavy_tr, self.__is_random_tr)
        y_tr = self.__y_tr

        # shuffle training data
        if self.__is_shuffle_tr:
            path_tr, x_heavy_tr, y_tr = shuffle_tri(path_tr, x_heavy_tr, y_tr)

        return path_tr, x_heavy_tr, y_tr

    def sample_test(self):

        # sample test data
        path_te, x_heavy_te = self.__sample_frames_from_videos(self.__frames_dict_te, self.__video_ids_te, self.__x_heavy_te, self.__is_random_te)
        y_te = self.__y_te

        # shuffle test data
        if self.__is_shuffle_te:
            path_te, x_heavy_te, y_te = shuffle_tri(path_te, x_heavy_te, y_te)

        return path_te, x_heavy_te, y_te

    def __sample_frames_from_videos(self, video_frames_dict, video_ids, x_heavy, is_random):
        frame_pathes = []

        n_timesteps_total = self.__n_timesteps_total
        n_timesteps = self.__n_timesteps

        B, C, _, H, W = x_heavy.shape
        T = self.__n_timesteps
        x_selected = np.zeros((B, C, T, H, W), dtype=np.float32)

        for idx_video, video_id in enumerate(video_ids):
            # get all frames of the video
            video_frames = video_frames_dict[video_id]

            # sample from these frames. Random or uniform
            if is_random:
                idx = np.random.randint(0, n_timesteps_total, (n_timesteps,))
                idx = np.sort(idx)
            else:
                step = n_timesteps_total / float(n_timesteps)
                idx = np.arange(0, n_timesteps_total, step, dtype=np.float32).astype(np.int32)

            x_video = x_heavy[idx_video]  # (C, T, H, W)
            x_video = x_video[:, idx]  # (C, T, H, W)
            x_selected[idx_video] = x_video
            sampled_video_frames = np.array(video_frames)[idx]

            assert len(sampled_video_frames) == n_timesteps

            frame_pathes.append(sampled_video_frames)

        frame_pathes = np.array(frame_pathes)
        return frame_pathes, x_selected

    def __select_middle_frame(self, frame_dict, n_frames_per_segment):

        n_per_segment = n_frames_per_segment
        frames_root_path = Pth('Breakfast/frames')
        idx_middle = int(n_frames_per_segment / float(2.0)) - 1

        for video_name, frames in frame_dict.items():
            frames = np.reshape(frames, (-1, n_per_segment))

            # pick up the middle frame of each segment
            frames = frames[:, idx_middle]

            # get full path
            frames = np.array(['%s/%s/%s' % (frames_root_path, video_name, f) for f in frames])

            frame_dict[video_name] = frames

        return frame_dict

# endregion

# region Samplers: Memory Features

class SamplerMemoryFeatureUniform():
    """
    Select feature pathes based on selection scores.
    """

    def __init__(self, features_path, n_timesteps, n_timesteps_total, dataset_type=None):

        if dataset_type == const.DATASET_TYPES.breakfast:
            gt_activities_path = Pth('Breakfast/annotation/gt_activities.pkl')
            (_, self.__y_tr, _, self.__y_te) = utils.pkl_load(gt_activities_path)
        elif dataset_type == const.DATASET_TYPES.charades:
            gt_activities_path = Pth('Charades/annotation/video_annotation.pkl')
            (_, self.__y_tr, _, self.__y_te) = utils.pkl_load(gt_activities_path)
            self.__y_tr = self.__y_tr.astype(np.float32)
            self.__y_te = self.__y_te.astype(np.float32)
        else:
            raise Exception('Unknown Dataset Type: %s' % (dataset_type))

        (x_tr, x_te) = utils.h5_load_multi(features_path, ['x_tr', 'x_te'])
        step = n_timesteps_total / float(n_timesteps)
        idxes = np.arange(0, n_timesteps_total, step, dtype=np.float32).astype(np.int32)
        x_tr = x_tr[:, :, idxes]
        x_te = x_te[:, :, idxes]

        self.__x_tr = x_tr.astype(np.float32)
        self.__x_te = x_te.astype(np.float32)

    def sample_train(self):
        # shuffle training
        self.__x_tr, self.__y_tr = shuffle_bi(self.__x_tr, self.__y_tr)

        # sample features
        x_tr = self.__x_tr
        y_tr = self.__y_tr

        return x_tr, y_tr

    def sample_test(self):
        # sample features
        x_te = self.__x_te
        y_te = self.__y_te

        return x_te, y_te

class SamplersMemoryFeaturesRandom():
    """
    Select feature pathes based on selection scores.
    """

    def __init__(self, features_path, n_timesteps, n_timesteps_total, is_random_tr=True, is_random_te=False, dataset_type=None):

        if dataset_type == const.DATASET_TYPES.breakfast:
            gt_activities_path = Pth('Breakfast/annotation/gt_activities.pkl')
            (_, self.__y_tr, _, self.__y_te) = utils.pkl_load(gt_activities_path)
        elif dataset_type == const.DATASET_TYPES.charades:
            gt_activities_path = Pth('Charades/annotation/video_annotation.pkl')
            (_, self.__y_tr, _, self.__y_te) = utils.pkl_load(gt_activities_path)
            self.__y_tr = self.__y_tr.astype(np.float32)
            self.__y_te = self.__y_te.astype(np.float32)
        else:
            raise Exception('Unknown Dataset Type: %s' % (dataset_type))

        (self.__x_tr, self.__x_te) = utils.h5_load_multi(features_path, ['x_tr', 'x_te'])

        self.__feature_root_path = features_path
        self.__n_timesteps_total = n_timesteps_total
        self.__n_timesteps = n_timesteps

        self.__is_random_tr = is_random_tr
        self.__is_random_te = is_random_te

    def sample_train(self):

        # shuffle training
        self.__x_tr, self.__y_tr = shuffle_bi(self.__x_tr, self.__y_tr)

        # sample features
        x_tr = self.__sample_features(self.__x_tr, is_random=self.__is_random_tr)
        y_tr = self.__y_tr
        return x_tr, y_tr

    def sample_test(self):

        # sample features
        x_te = self.__sample_features(self.__x_te, is_random=self.__is_random_te)
        y_te = self.__y_te
        return x_te, y_te

    def __sample_features(self, x, is_random):
        """
        x is of shape (B, C, T, H, W)
        Selection_score is of shape (B, T)
        """

        n_timesteps = self.__n_timesteps
        n_timesteps_total = self.__n_timesteps_total

        input_shape = x.shape
        n_x, C, T, H, W = input_shape
        output_shape = (n_x, C, n_timesteps, H, W)
        x_out = np.zeros(output_shape, dtype=np.float32)

        # loop on video names and select the timesteps, either uniform or random
        for idx_item in range(n_x):

            # either random or uniform sampling
            if is_random:
                idxes = np.random.randint(0, n_timesteps_total, (n_timesteps,))
                idxes = np.sort(idxes)
            else:
                step = n_timesteps_total / float(n_timesteps)
                idxes = np.arange(0, n_timesteps_total, step, dtype=np.float32).astype(np.int32)

            x_item = x[idx_item]  # (C, T, H, W)
            x_item_out = x_item[:, idxes]
            x_out[idx_item] = x_item_out

        return x_out

# endregion

# region Functions: Shuffle

def shuffle_uni(x):
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    x = x[idx]

    return x

def shuffle_bi(x, y):
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    x = x[idx]
    y = y[idx]

    return x, y

def shuffle_tri(x, y, z):
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    x = x[idx]
    y = y[idx]
    z = z[idx]

    return x, y, z

# endregion
