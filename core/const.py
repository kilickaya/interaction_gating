import os
import platform
import getpass
import numpy as np

# region Constants

DL_FRAMEWORK = None
PLATFORM = None
GPU_CORE_ID = 0
DATA_ROOT_PATH = None
USER_ROOT_PATH = None
MACHINE_NAME = platform.node()
USER_NAME = getpass.getuser()

# endregion

# region Lists

class MACHINE_NAMES:
    desktop_mert = 'mert'
    desktop_nour = 'U036713'
    das5_fs4 = 'fs4'
    das5_node = 'node'

class USER_NAMES:
    desktop_nour = 'nour'
    desktop_mert = 'mert'
    das5_nour = 'nhussein'
    das5_mert = 'mkilicka'

class DL_FRAMEWORKS:
    keras = 'keras'
    caffe = 'caffe'
    caffe2 = 'caffe2'
    pytorch = 'pytorch'
    tensorflow = 'tensorflow'

class CNN_MODEL_TYPES:
    i3d = 'i3d'
    googlenet1k = 'googlenet1k'
    googlenet13k = 'googlenet13k'
    mobilenetv3_small = 'mobilenetv3_small'
    places365_resnet152 = 'places365_resnet152'
    places365_vgg = 'laces365_vgg'
    resnet = 'resnet'
    resnet18 = 'resnet18'
    resnet34 = 'resnet34'
    resnet50 = 'resnet50'
    resnet101 = 'resnet101'
    resnet152 = 'resnet152'
    vgg = 'vgg'
    vgg16 = 'vgg16'

class CNN_FEATURE_TYPES:
    fc6 = 'fc6'
    fc7 = 'fc7'
    fc1000 = 'fc1000'
    fc1024 = 'fc1024'
    fc365 = 'fc365'
    prob = 'prob'
    pool5 = 'pool5'
    fc8a = 'fc8a'
    res3b7 = 'res3b7'
    res4b35 = 'res4b35'
    res5c = 'res5c'
    softmax = 'softmax'
    mixed_5c = 'mixed_5c'
    mixed_5c_maxpool = 'mixed_5c_maxpool'
    conv5_3 = 'conv5_3'
    conv12 = 'conv12'
    convpool = 'convpool'
    conv5c = 'conv5c'
    conv5c_pool = 'conv5c_pool'

class CNN_FEATURE_SIZES:
    _576 = '576'
    _1000 = '1000'
    _1024 = '1024'
    _2048 = '2048'

class SIMILARITY_TYPES:
    kl = 'kl'
    cosine = 'cosine'
    euclidean = 'euclidean'

class MANIFOLD_TYPES:
    tsne = 'tsne'
    isomap = 'isomap'
    mds = 'mds'
    spectral = 'spectral'

class NLP_METHOD_TYPES:
    lda = 'lda'
    lsi = 'lsi'
    st = 'st'
    d2v = 'd2v'
    bt = 'bt'

class D2V_MODEL_TYPES:
    enwiki = 'enwiki'
    apnews = 'apnews'

class VECTOR_ENCODING_TYPES:
    vlad = 'vlad'
    fisher = 'fisher'

class RESIZE_TYPES:
    resize = 'resize'
    resize_crop = 'resize_crop'
    resize_crop_scaled = 'resize_crop_scaled'
    resize_keep_aspect_ratio_padded = 'resize_keep_aspect_ratio_padded'

class DEEP_XPLAIN_TYPES:
    grad_input = 'grad_input'
    saliency = 'saliency'
    intgrad = 'intgrad'
    deeplift = 'deeplift'
    elrp = 'elrp'
    occlusion = 'occlusion'

class ROOT_PATH_TYPES:
    desktop_mert_data = '/home/mert/Datasets'
    dekstop_mert_user = '/home/mert/PyCharmProjects'
    desktop_nour_data = '/home/nour/Documents/Datasets'
    dekstop_nour_user = '/home/nour/Documents/PyCharmProjects'
    das5_nour_local = '/local/nhussein'
    das5_nour_data = '/var/scratch/nhussein/Datasets'
    das5_nour_user = '/var/scratch/nhussein/PyCharmProjects'
    das5_mert_local = '/local/mert'
    das5_mert_data = '/var/scratch/mkilicka/Datasets'
    das5_mert_user = '/var/scratch/mkilicka/PyCharmProjects'

# endregion

# region Pathes of Root Directories


if MACHINE_NAME == MACHINE_NAMES.desktop_mert:
    DATA_ROOT_PATH = ROOT_PATH_TYPES.desktop_mert_data
    USER_ROOT_PATH = ROOT_PATH_TYPES.dekstop_mert_user
elif MACHINE_NAME == MACHINE_NAMES.desktop_nour:
    DATA_ROOT_PATH = ROOT_PATH_TYPES.desktop_nour_data
    USER_ROOT_PATH = ROOT_PATH_TYPES.dekstop_nour_user
elif MACHINE_NAME == MACHINE_NAMES.das5_fs4 or MACHINE_NAMES.das5_node in MACHINE_NAME:
    if USER_NAME == USER_NAMES.das5_nour:
        DATA_ROOT_PATH = ROOT_PATH_TYPES.das5_nour_data
        USER_ROOT_PATH = ROOT_PATH_TYPES.das5_nour_user
    elif USER_NAME == USER_NAMES.das5_mert:
        DATA_ROOT_PATH = ROOT_PATH_TYPES.das5_mert_local
        USER_ROOT_PATH = ROOT_PATH_TYPES.das5_mert_user

    else:
        raise Exception('Sorry, user name: %s' % (USER_NAME))
else:
    raise Exception('Sorry, unknown machine: %s' % (MACHINE_NAME))


# endregion
