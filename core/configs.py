import os
import platform
import argparse
import sys
import tensorboardX

from core import const as c

# TORCH_LOGGER = tensorboardX.SummaryWriter(logdir='./logs/')

def get_machine_name():
    return platform.node()

def is_local_machine():
    machine_name = platform.node()
    return machine_name in [c.MACHINE_NAMES.desktop_nour, c.MACHINE_NAMES.desktop_mert]

def import_dl_platform():
    if c.DL_FRAMEWORK == 'tensorflow':
        import tensorflow as tf
    elif c.DL_FRAMEWORK == 'pytorch':
        import torch
    elif c.DL_FRAMEWORK == 'caffe':
        import caffe
    elif c.DL_FRAMEWORK == 'keras':
        from keras import backend as K

def __config_python_version():
    if sys.version_info[0] < 3:
        raise Exception("Must be using Python 3")
    else:
        # print('correct python version')
        pass

def __config_gpu():
    if c.DL_FRAMEWORK == 'tensorflow':
        __config_gpu_for_tensorflow()
    elif c.DL_FRAMEWORK == 'pytorch':
        __config_gpu_for_pytorch()
    elif c.DL_FRAMEWORK == 'keras':
        __config_gpu_for_keras()
    elif c.DL_FRAMEWORK == 'caffe':
        __config_gpu_for_caffe()
    else:
        # print('gpu is not configed yet, no deep learning framework is selected')
        pass

def __config_gpu_for_tensorflow():
    import tensorflow as tf

    _is_local_machine = is_local_machine()

    # import os
    # import tensorflow as tf
    # set the logging level of tensorflow
    # 1: filter out INFO
    # 2: filter out WARNING
    # 3: filter out ERROR
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
    pass

def __config_gpu_for_keras():
    import tensorflow as tf
    from keras import backend as K

    _is_local_machine = is_local_machine()

    if _is_local_machine:
        gpu_core_id = 0
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--gpu_core_id', default='-1', type=int)
        args = parser.parse_args()
        gpu_core_id = args.gpu_core_id

        if gpu_core_id < 0 or gpu_core_id > 3:
            msg = 'Please specify a correct GPU core!!!'
            raise Exception(msg)

    K.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(gpu_core_id)
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)

    # set which device to be used
    c.GPU_CORE_ID = gpu_core_id

def __config_gpu_for_pytorch():
    import torch

    _is_local_machine = is_local_machine()

    if _is_local_machine:
        gpu_core_id = 0
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--gpu_core_id', default='-1', type=int)
        args = parser.parse_args()
        gpu_core_id = args.gpu_core_id

        if gpu_core_id < 0 or gpu_core_id > 3:
            msg = 'Please specify a correct GPU core!!!'
            raise Exception(msg)

    torch.cuda.set_device(gpu_core_id)

    # set which device to be used
    c.GPU_CORE_ID = gpu_core_id

def __config_gpu_for_caffe():
    import os

    _is_local_machine = is_local_machine()

    if _is_local_machine:
        gpu_core_id = 0
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--gpu_core_id', default='-1', type=int)
        args = parser.parse_args()
        gpu_core_id = args.gpu_core_id

        if gpu_core_id < 0 or gpu_core_id > 3:
            msg = 'Please specify a correct GPU core!!!'
            raise Exception(msg)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_core_id)

    # set which device to be used
    c.GPU_CORE_ID = gpu_core_id

# run configs
__config_python_version()
__config_gpu()
