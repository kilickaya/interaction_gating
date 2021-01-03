import time
import h5py
import numpy as np
import pickle as pkl
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing, manifold
import scipy.io as sio

import os
import json
import natsort
import random
from multiprocessing.dummy import Pool

from core import const, plot_utils, sobol

# region Load and Dump

def pkl_load(path):
    with open(path, 'rb') as f:
        data = pkl.load(f, encoding='bytes')
    return data

def txt_load(path):
    with open(path, 'r') as f:
        lines = f.read().splitlines()
    lines = np.array(lines)
    return lines

def byte_load(path):
    with open(path, 'rb') as f:
        data = f.read()
    return data

def json_load(path):
    with open(path, 'r') as f:
        data = json.load(f)

    return data

def h5_load(path, dataset_name='data'):
    h5_file = h5py.File(path, 'r')
    data = h5_file[dataset_name].value
    h5_file.close()
    return data

def h5_load_multi(path, dataset_names):
    h5_file = h5py.File(path, 'r')
    data = [h5_file[name].value for name in dataset_names]
    h5_file.close()
    return data

def txt_dump(data, path):
    l = len(data) - 1
    with open(path, 'w') as f:
        for i, k in enumerate(data):
            if i < l:
                k = ('%s\n' % k)
            else:
                k = ('%s' % k)
            f.writelines(k)

def byte_dump(data, path):
    with open(path, 'wb') as f:
        f.write(data)

def pkl_dump(data, path, is_highest=True):
    with open(path, 'wb') as f:
        if not is_highest:
            pkl.dump(data, f)
        else:
            pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)

def json_dump(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

def h5_dump(data,  dataset_name, path):
    h5_file = h5py.File(path, 'w')
    h5_file.create_dataset(dataset_name, data=data, dtype=data.dtype)
    h5_file.close()

def h5_dump_multi(data, dataset_names, path):
    h5_file = h5py.File(path, 'w')
    n_items = len(data)
    for i in range(n_items):
        item_data = data[i]
        item_name = dataset_names[i]
        h5_file.create_dataset(item_name, data=item_data, dtype=item_data.dtype)
    h5_file.close()

def csv_load(path, sep=',', header='infer'):
    df = pd.read_csv(path, sep=sep, header=header)
    data = df.values
    return data

def mat_load(path, m_dict=None):
    """
    Load mat files.
    :param path:
    :return:
    """
    if m_dict is None:
        data = sio.loadmat(path)
    else:
        data = sio.loadmat(path, m_dict)

    return data

# endregion

# region File/Folder Names/Pathes

def file_names(path, is_nat_sort=False):
    if not os.path.exists(path):
        exp_msg = 'Sorry, folder path does not exist: %s' % (path)
        raise Exception(exp_msg)

    names = os.walk(path).__next__()[2]

    if is_nat_sort:
        names = natsort.natsorted(names)

    return names

def file_pathes(path, is_nat_sort=False):
    if not os.path.exists(path):
        exp_msg = 'Sorry, folder path does not exist: %s' % (path)
        raise Exception(exp_msg)
    names = os.walk(path).__next__()[2]

    if is_nat_sort:
        names = natsort.natsorted(names)

    pathes = ['%s/%s' % (path, n) for n in names]
    return pathes

def folder_names(path, is_nat_sort=False):
    if not os.path.exists(path):
        exp_msg = 'Sorry, folder path does not exist: %s' % (path)
        raise Exception(exp_msg)

    names = os.walk(path).__next__()[1]

    if is_nat_sort:
        names = natsort.natsorted(names)

    return names

def folder_pathes(path, is_nat_sort=False):
    if not os.path.exists(path):
        exp_msg = 'Sorry, folder path does not exist: %s' % (path)
        raise Exception(exp_msg)

    names = os.walk(path).__next__()[1]

    if is_nat_sort:
        names = natsort.natsorted(names)

    pathes = ['%s/%s' % (path, n) for n in names]
    return pathes

# endregion

# region Normalization

def normalize_mean_std(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    x -= mean
    x /= std
    return x

def normalize_mean(x):
    mean = np.mean(x, axis=0)
    x /= mean
    return x

def normalize_sum(x):
    sum = np.sum(x, axis=1)
    x = np.array([x_i / sum_i for x_i, sum_i in zip(x, sum)])
    return x

def normalize_l2(x):
    return preprocessing.normalize(x)

def normalize_l1(x):
    return preprocessing.normalize(x, norm='l1')

def normalize_range_0_to_1(x):
    x = np.add(x, -x.min())
    x = np.divide(x, x.max())
    return x

# endregion

# region Array Helpers

def array_to_text(a, separator=', '):
    text = separator.join([str(s) for s in a])
    return text

def get_size_in_kb(size):
    size /= float(1024)
    return size

def get_size_in_mb(size):
    size /= float(1024 * 1024)
    return size

def get_size_in_gb(size):
    size /= float(1024 * 1024 * 1024)
    return size

def get_array_memory_size(a):
    if type(a) is not np.ndarray:
        raise Exception('Sorry, input is not numpy array!')

    dtype = a.dtype
    if dtype == np.float16:
        n_bytes = 2
    elif dtype == np.float32:
        n_bytes = 4
    else:
        raise Exception('Sorry, unsupported dtype:', dtype)

    s = a.size
    size = s * n_bytes
    return size

def get_expected_memory_size(array_shape, array_dtype):
    dtype = array_dtype
    if dtype == np.float16:
        n_bytes = 2
    elif dtype == np.float32:
        n_bytes = 4
    else:
        raise Exception('Sorry, unsupported dtype:', dtype)

    s = 1
    for dim_size in array_shape:
        s *= dim_size

    size = s * n_bytes
    return size

def print_array(a):
    for item in a:
        print(item)

def print_array_joined(a):
    s = ', '.join([str(i) for i in a])
    print(s)

# endregion

# region Model Training/Testing

def debinarize_label(labels):
    debinarized = np.array([np.where(l == 1)[0][0] for l in labels])
    return debinarized

def get_model_feat_maps_info(model_type, feature_type):
    ex_feature = Exception('Sorry, unsupported feature type: %s' % (feature_type))
    ex_model = Exception('Sorry, unsupported model type: %s' % (model_type))
    info = None

    if model_type in ['vgg']:
        if feature_type in ['pool5']:
            info = 512, 7
        elif feature_type in ['conv5_3']:
            info = 512, 14
        else:
            raise ex_feature
    elif model_type in ['resnet18', 'resnet34']:
        if feature_type in ['conv5c']:
            info = 512, 7
        elif feature_type in ['conv5c_pool']:
            info = 512, 1
        else:
            raise ex_feature
    elif model_type in ['resnet3d', 'resnet152', 'resnet101', 'resnet50']:
        if feature_type in ['res4b35']:
            info = 1024, 14
        elif feature_type in ['res5c', 'res52', 'conv5c']:
            info = 2048, 7
        elif feature_type in ['pool5', 'conv5c_pool']:
            info = 2048, 1
        else:
            raise ex_feature
    elif model_type in ['i3d']:
        if feature_type in ['mixed_5c']:
            info = 1024, 7
        elif feature_type in ['mixed_5c_maxpool']:
            info = 1024, 1
        elif feature_type in ['softmax']:
            info = 400, 1
        else:
            raise ex_feature
    elif model_type in ['i3d_kinetics_keras']:
        if feature_type in ['mixed_4f']:
            info = 832, 7
        else:
            raise ex_feature
    elif model_type in ['mobilenetv3_small']:
        if feature_type in ['conv12']:
            info = 576, 7
        elif feature_type in ['convpool']:
            info = 576, 1
        else:
            raise ex_feature
    else:
        raise ex_model

    return info

def calc_num_batches(n_samples, batch_size):
    n_batch = int(n_samples / float(batch_size))
    n_batch = n_batch if n_samples % batch_size == 0 else n_batch + 1
    return n_batch

# endregion

# region Misc

def byte_array_to_string(value):
    decoded = string_decode(value)
    return decoded

def string_to_byte_array(value):
    decoded = string_encode(value)
    return decoded

def string_decode(value, coding_type='utf-8'):
    """
    Convert from byte array to string.
    :param value:
    :param coding_type:
    :return:
    """
    decoded = value.decode(coding_type)
    return decoded

def string_encode(value, coding_type='utf-8'):
    """
    Convert from byte string to array.
    :param value:
    :param coding_type:
    :return:
    """
    encoded = value.encode(coding_type)
    return encoded

def is_list(value):
    input_type = type(value)
    result = input_type is list
    return result

def is_tuple(value):
    input_type = type(value)
    result = input_type is tuple
    return result

def is_enumerable(value):
    input_type = type(value)
    result = (input_type is list or input_type is tuple)
    return result

def learn_manifold(manifold_type, feats, n_components=2):
    if manifold_type == 'tsne':
        feats_fitted = manifold.TSNE(n_components=n_components, random_state=0).fit_transform(feats)
    elif manifold_type == 'isomap':
        feats_fitted = manifold.Isomap(n_components=n_components).fit_transform(feats)
    elif manifold_type == 'mds':
        feats_fitted = manifold.MDS(n_components=n_components).fit_transform(feats)
    elif manifold_type == 'spectral':
        feats_fitted = manifold.SpectralEmbedding(n_components=n_components).fit_transform(feats)
    else:
        raise Exception('wrong maniford type!')

    # methods = ['standard', 'ltsa', 'hessian', 'modified']
    # feats_fitted = manifold.LocallyLinearEmbedding(n_components=n_components, method=methods[0]).fit_transform(pred)

    return feats_fitted

def timestamp():
    time_stamp = "{0:%y}.{0:%m}.{0:%d}-{0:%I}:{0:%M}:{0:%S}".format(datetime.now())
    return time_stamp

def remove_extension(name):
    name = name.split('.')
    name = name[:-1]
    name = ''.join(name)
    return name

def get_file_extension(name):
    name = name.split('.')[-1]
    return name

def print_counter(num, total, freq=None):
    if freq is None:
        print('... %d/%d' % (num, total))
    elif num % freq == 0:
        print('... %d/%d' % (num, total))

def generate_centroids(n_centroids, n_dims, is_sobol=True):
    if is_sobol:
        # centroids as sobol sequence
        c = sobol.sobol_generate(n_dims, n_centroids)
        c = np.array(c)
    else:
        # centroids as random vectors
        c = np.random.rand(n_centroids, n_dims)
        c = c.astype(np.float32)

    return c

# endregion

# region Classes

class Path(str):
    def __new__(self, relative_path, args=None, root_path=const.DATA_ROOT_PATH):
        relative_path = relative_path % args if args is not None else relative_path
        path = os.path.join(root_path, relative_path)

        self.__path = path
        return self.__path

    def __str__(self):
        return self.__path

    def __repr__(self):
        return self.__path

class DurationTimer(object):
    def __init__(self):
        self.start_time = time.time()

    def duration(self, is_string=True):
        stop_time = time.time()
        durtation = stop_time - self.start_time
        if is_string:
            durtation = self.format_duration(durtation)
        return durtation

    def format_duration(self, duration):
        if duration < 60:
            return str(duration) + " sec"
        elif duration < (60 * 60):
            return str(duration / 60) + " min"
        else:
            return str(duration / (60 * 60)) + " hr"

class Obj():
    """
    Dummy object so you can add properties and functions to it.
    """

    def __init__(self):
        pass

# endregion
