#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
####################
# Feature Selection
####################
Here, we define modules to perform the binary selection of features based on
well-know gating mechanism, as discussed in
https://arxiv.org/abs/1711.11503
"""

import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data
from torch import distributions
from torch.autograd import Variable
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter

from core import pytorch_utils, configs, sobol, utils
from modules import layers_pytorch as pl
from modules import functions_pytorch as pf

from core import pytorch_utils
from modules import layers_pytorch as pl

# region Feature Fusion

class FeatureFusionResidualV1(nn.Module):
    def __init__(self, x_so_shape, x_c_shape, n_channels):
        super(FeatureFusionResidualV1, self).__init__()

        C1, N1, H1, W1 = x_so_shape
        C2, N2, H2, W2 = x_c_shape

        # layers for input embedding
        self.dense_so = nn.Sequential(pl.Linear3d(C1, n_channels, dim=1), nn.BatchNorm3d(n_channels))
        self.dense_c = nn.Sequential(pl.Linear3d(C2, n_channels, dim=1), nn.BatchNorm3d(n_channels))

        # activation after fusion
        self.activation = nn.Sequential(nn.LeakyReLU(0.2))

    def forward(self, x_so, x_c):
        # input is of shape (None, C, H, W)

        # pairwise interaction between x_so and x_c
        x_so = self.dense_so(x_so)  # (B, C, N, H, W)
        x_c = self.dense_c(x_c)  # (B, C, N, H, W)

        # interaction between subject_object feature (x_so) and context feature (x_c)
        x = x_so + x_c
        x = self.activation(x)

        return x

class FeatureFusionResidual(nn.Module):
    def __init__(self, x_so_shape, x_c_shape, n_channels):
        super(FeatureFusionResidual, self).__init__()

        C1, N1, H1, W1 = x_so_shape
        C2, N2, H2, W2 = x_c_shape

        # layers for input embedding
        self.dense_so = nn.Sequential(nn.BatchNorm3d(C1), pl.Linear3d(C1, n_channels, dim=1))
        self.dense_c = nn.Sequential(nn.BatchNorm3d(C2), pl.Linear3d(C2, n_channels, dim=1))

        # activation after fusion
        self.activation = nn.Sequential(nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))

    def forward(self, x_so, x_c):
        # input is of shape (None, C, H, W)

        # pairwise interaction between x_so and x_c
        x_so = self.dense_so(x_so)  # (B, C, N, H, W)
        x_c = self.dense_c(x_c)  # (B, C, N, H, W)

        # interaction between subject_object feature (x_so) and context feature (x_c)
        x = x_so + x_c
        x = self.activation(x)

        return x

class FeatureFusionConcat(nn.Module):
    def __init__(self, x_so_shape, x_c_shape, n_channels):
        super(FeatureFusionConcat, self).__init__()

        C1, N1, H1, W1 = x_so_shape
        C2, N2, H2, W2 = x_c_shape

        # layers for input embedding
        self.dense_so = nn.Sequential(nn.BatchNorm3d(C1), pl.Linear3d(C1, n_channels))
        self.dense_c = nn.Sequential(nn.BatchNorm3d(C2), pl.Linear3d(C2, n_channels))

        # activation after fusion
        self.activation = nn.Sequential(nn.BatchNorm3d(n_channels * 2), nn.LeakyReLU(0.2))

    def forward(self, x_so, x_c):
        # input is of shape (None, C, H, W)

        N = pytorch_utils.get_shape(x_so)[2]

        # pairwise interaction between x_so and x_c
        x_so = self.dense_so(x_so)  # (B, C, N, H, W)
        x_c = self.dense_c(x_c)  # (B, C, N, H, W)
        x_c = x_c.repeat((1, 1, N, 1, 1))  # (B, C, N, H, W)

        # concat
        x = torch.cat((x_so, x_c), dim=1)
        x = self.activation(x)

        return x

# endregion

# region SO Gating

class SOGatingSigmoidV1(nn.Module):
    def __init__(self, x_so_shape):
        super(SOGatingSigmoidV1, self).__init__()

        n_channels, N, H, W = x_so_shape
        n_channels_half = 256

        # layers for gating
        fs_layers = []
        fs_layers.append(pl.Linear3d(n_channels, n_channels_half))
        fs_layers.append(nn.BatchNorm3d(n_channels_half))
        fs_layers.append(nn.LeakyReLU(0.2))
        fs_layers.append(pl.Linear3d(n_channels_half, 1))
        fs_layers.append(pl.Squeeze(dim=1))
        fs_layers.append(pl.Mean(dim=(2, 3)))
        self.fs_layers = nn.Sequential(*fs_layers)

        fo_layers = []
        fo_layers.append(pl.Linear3d(n_channels, n_channels_half))
        fo_layers.append(nn.BatchNorm3d(n_channels_half))
        fo_layers.append(nn.LeakyReLU(0.2))
        fo_layers.append(pl.Linear3d(n_channels_half, 1))
        fo_layers.append(pl.Squeeze(dim=1))
        fo_layers.append(pl.Mean(dim=(2, 3)))
        self.fo_layers = nn.Sequential(*fo_layers)

        threshold_value = 0.5
        pf.Threshold.THRESHOLD_VALUE = threshold_value
        pf.BinaryThreshold.THRESHOLD_VALUE = threshold_value

        self.f_threshold = pf.Threshold.apply
        self.f_binary_threshold = pf.BinaryThreshold.apply

        self.f_sigmoid = nn.Sigmoid()
        self.f_gumbel_noise = pl.GumbelNoise(loc=0.5, scale=1.0)
        self.f_gaussian_noise = pl.GaussianNoise(loc=0.0, scale=1.0)

    def forward(self, x_s, x_o):
        # gating
        x_s = self.__gating(x_s, self.fs_layers)
        x_o = self.__gating(x_o, self.fo_layers)

        # fuse subject and object
        x = x_s + x_o

        return x

    def __gating(self, x, layers):
        f = layers(x)  # (B, N)
        f = self.f_gumbel_noise(f) if self.training else f  # (B, N)
        f = self.f_sigmoid(f)  # (B, N)
        alpha = self.f_threshold(f) if self.training else self.f_binary_threshold(f)  # (B, N)

        B, N = pytorch_utils.get_shape(f)
        alpha = alpha.view(B, 1, N, 1, 1)  # (B, 1, N, 1, 1)

        x = x * alpha

        return x

class SOGatingSigmoidV2(nn.Module):
    def __init__(self, x_so_shape):
        super(SOGatingSigmoidV2, self).__init__()

        n_channels, N, H, W = x_so_shape
        n_channels_half = 256

        # self.bn_fs = nn.BatchNorm3d(n_channels)
        # self.bn_fo = nn.BatchNorm3d(n_channels)
        # self.bn_x = nn.BatchNorm3d(n_channels * 2)

        # layers for gating
        f_layers = []
        f_layers.append(nn.BatchNorm3d(n_channels * 2))
        f_layers.append(pl.Linear3d(n_channels * 2, n_channels_half))
        f_layers.append(nn.BatchNorm3d(n_channels_half))
        f_layers.append(nn.LeakyReLU(0.2))
        f_layers.append(pl.Linear3d(n_channels_half, 1))
        f_layers.append(pl.Mean(dim=(3, 4), keepdim=True))
        self.f_layers = nn.Sequential(*f_layers)

        threshold_value = 0.5
        pf.Threshold.THRESHOLD_VALUE = threshold_value
        pf.BinaryThreshold.THRESHOLD_VALUE = threshold_value

        self.f_threshold = pf.Threshold.apply
        self.f_binary_threshold = pf.BinaryThreshold.apply

        self.f_sigmoid = nn.Sigmoid()
        self.f_gumbel_noise = pl.GumbelNoise(loc=0.5, scale=1.0)
        self.f_gaussian_noise = pl.GaussianNoise(loc=0.0, scale=1.0)

    def forward(self, x_s, x_o):
        # fuse subject and object
        # f_s = self.bn_fs(x_s)
        # f_o = self.bn_fo(x_o)
        # f = f_s + f_o
        f = torch.cat((x_s, x_o), dim=1)

        # gating
        f = self.f_layers(f)  # (B, N)
        f = self.f_gumbel_noise(f) if self.training else f  # (B, N)
        f = self.f_sigmoid(f)  # (B, N)
        alpha = self.f_threshold(f) if self.training else self.f_binary_threshold(f)  # (B, N)

        # gating
        # x_o = x_o * alpha
        x_s = x_s * alpha

        # x = torch.cat((x_s, x_o), dim=1)
        # x = self.bn_x(x)
        # return x

        return x_s, x_o

class SOGatingSigmoid(nn.Module):
    def __init__(self, x_c_shape, x_s_shape, x_o_shape):
        super(SOGatingSigmoid, self).__init__()

        n_channels, N, H, W = x_c_shape
        C_so = x_s_shape[0] + x_o_shape[0]
        n_channels_half = 256

        self.dense_so = nn.Sequential(nn.BatchNorm3d(C_so), pl.Linear3d(C_so, n_channels))
        self.bn_c = nn.BatchNorm3d(n_channels)

        # layers for gating
        f_layers = []
        f_layers.append(nn.BatchNorm3d(n_channels))
        f_layers.append(pl.Linear3d(n_channels, n_channels_half))
        f_layers.append(nn.BatchNorm3d(n_channels_half))
        f_layers.append(nn.LeakyReLU(0.2))
        f_layers.append(pl.Linear3d(n_channels_half, 1))
        f_layers.append(pl.Mean(dim=(3, 4), keepdim=True))
        self.f_layers = nn.Sequential(*f_layers)

        threshold_value = 0.5
        pf.Threshold.THRESHOLD_VALUE = threshold_value
        pf.BinaryThreshold.THRESHOLD_VALUE = threshold_value

        self.f_threshold = pf.Threshold.apply
        self.f_binary_threshold = pf.BinaryThreshold.apply

        self.f_sigmoid = nn.Sigmoid()
        self.f_gumbel_noise = pl.GumbelNoise(loc=0.5, scale=1.0)
        self.f_gaussian_noise = pl.GaussianNoise(loc=0.0, scale=1.0)

    def forward(self, x_c, x_s, x_o):
        # gating

        # gating is conditioned on both the context and subject_object pair
        x_so = torch.cat((x_s, x_o), dim=1)
        x_so = self.dense_so(x_so)
        x_c = self.bn_c(x_c)
        f = x_so + x_c

        f = self.f_layers(f)  # (B, N)
        f = self.f_gumbel_noise(f) if self.training else f  # (B, N)
        f = self.f_sigmoid(f)  # (B, N)
        alpha = self.f_threshold(f) if self.training else self.f_binary_threshold(f)  # (B, N)
        # alpha_s = alpha[:, 0:1]
        # alpha_o = alpha[:, 1:2]

        # gating
        # x_o = x_o * alpha
        x_s = x_s * alpha

        # cat subject and object
        x = torch.cat((x_s, x_o), dim=1)

        return x

class SOChannelGatingSigmoid(nn.Module):
    def __init__(self, x_s_shape, x_o_shape):
        super(SOChannelGatingSigmoid, self).__init__()

        C_so = x_s_shape[0] + x_o_shape[0]
        n_channels = 512
        n_channels_half = 256

        # self.dense_so = nn.Sequential(nn.BatchNorm3d(C_so), pl.Linear3d(C_so, n_channels))
        # self.bn_c = nn.BatchNorm3d(n_channels)

        # layers for gating
        f_layers = []
        f_layers.append(nn.BatchNorm3d(C_so))
        f_layers.append(pl.Linear3d(C_so, n_channels))
        f_layers.append(nn.BatchNorm3d(n_channels))
        f_layers.append(nn.LeakyReLU(0.2))
        f_layers.append(pl.Linear3d(n_channels, C_so))
        f_layers.append(pl.Mean(dim=(3, 4), keepdim=True))
        self.f_layers = nn.Sequential(*f_layers)

        threshold_value = 0.5
        pf.Threshold.THRESHOLD_VALUE = threshold_value
        pf.BinaryThreshold.THRESHOLD_VALUE = threshold_value

        self.f_threshold = pf.Threshold.apply
        self.f_binary_threshold = pf.BinaryThreshold.apply

        self.f_sigmoid = nn.Sigmoid()
        self.f_gumbel_noise = pl.GumbelNoise(loc=0.5, scale=1.0)
        self.f_gaussian_noise = pl.GaussianNoise(loc=0.0, scale=1.0)

    def forward(self, x_s, x_o):
        # gating

        # gating is conditioned on both the context and subject_object pair
        f = torch.cat((x_s, x_o), dim=1)
        f = self.f_layers(f)  # (B, N)
        f = self.f_gumbel_noise(f) if self.training else f  # (B, N)
        f = self.f_sigmoid(f)  # (B, N)
        alpha = self.f_threshold(f) if self.training else self.f_binary_threshold(f)  # (B, N)

        # cat subject and object
        x = torch.cat((x_s, x_o), dim=1)

        # gating
        x = x * alpha

        return x

# endregion

# region Context Gating

class ContextGatingSoftmax(nn.Module):
    def __init__(self, x_so_shape, x_c_shape):
        super(ContextGatingSoftmax, self).__init__()

        # n_channels, _, _ = input_shape
        # n_channels_half = int(n_channels / 2.0)

        N, C1, H1, W1 = x_so_shape
        C2, H2, W2 = x_c_shape
        n_channels = 1024
        n_channels_half = int(n_channels / 2.0)

        # layers for input embedding
        # self.dense_so = nn.Sequential(pl.Linear3d(C1, n_channels, dim=1), nn.BatchNorm3d(n_channels))
        # self.dense_c = nn.Sequential(pl.Linear2d(C2, n_channels, dim=1), nn.BatchNorm2d(n_channels))

        self.dense_so = nn.Sequential(nn.BatchNorm3d(C1), pl.Linear3d(C1, n_channels, dim=1))
        self.dense_c = nn.Sequential(nn.BatchNorm2d(C2), pl.Linear2d(C2, n_channels, dim=1))

        self.bn_so = nn.BatchNorm3d(n_channels)
        self.bn_c = nn.BatchNorm3d(n_channels)

        # layers for gating
        alpha_layers = []
        alpha_layers.append(nn.BatchNorm3d(n_channels))
        alpha_layers.append(pl.Linear3d(n_channels, n_channels_half))
        alpha_layers.append(nn.BatchNorm3d(n_channels_half))
        alpha_layers.append(nn.LeakyReLU(0.2))
        alpha_layers_last = pl.Linear3d(n_channels_half, 2)
        alpha_layers.append(alpha_layers_last)
        alpha_layers.append(pl.Mean(dim=(3, 4)))
        self.alpha_layers = nn.Sequential(*alpha_layers)

        # activation and thresholding for the gate
        self.alpha_hardmax = pl.HardMax()
        self.alpha_softmax = nn.Softmax(dim=-1)

        # stochasticity (i.e. noise) helps in learning
        self.alpha_gumbel_noise = pl.GumbelNoise(loc=0.0, scale=1.0)
        self.alpha_gaussian_noise = pl.GaussianNoise(loc=0.0, scale=1.0)

        # # initialize the bias with opening rate of ??%
        alpha_layers_last.layer.bias.data[0] = 0.3
        alpha_layers_last.layer.bias.data[1] = 0.7

    def forward(self, x_so, x_c):
        # input is of shape (None, C, H, W)

        # pairwise interaction between x_so and x_c
        x_so = self.dense_so(x_so)  # (B, C, N, H, W)
        x_c = self.dense_c(x_c)  # (B, C, N, H, W)
        x_c = torch.unsqueeze(x_c, dim=2)  # (B, C, N, H, W)
        x = x_so + x_c  # (B, C, N, H, W)

        # gating
        f = self.alpha_layers(x)  # (B, 2, N)
        B, C, N = pytorch_utils.get_shape(f)
        f = f.permute(0, 2, 1)  # (B, N, 2)
        f = f.contiguous().view(B * N, C)  # (B*N, C)

        # noise
        f = self.alpha_gumbel_noise(f) if self.training else f  # (B*N, 2)

        # activation of logits
        f = self.alpha_softmax(f)  # (B*N, 2)
        alpha = self.alpha_hardmax(f)  # (B*N, 2)
        # alpha = self.__hard_max(f)

        # get only the second of the 2 neurons
        alpha = alpha[:, 1]  # (B*N)
        alpha = alpha.view(B, 1, N, 1, 1)  # (B, 1, N, 1, 1)
        f = f[:, 1]
        f = f.view(B, 1, N, 1, 1)  # (B, 1, N, 1, 1)

        # gating loss
        # self.gating_loss = torch.mean(f) if self.training else None

        # save values for debugging
        self.f_values = np.array(f.tolist())
        self.alpha_values = np.array(alpha.tolist())

        # bn of features before interaction
        x_so = self.bn_so(x_so)
        x_c = self.bn_c(x_c)

        # multiply the gating value by the context feature
        x_c = x_c * alpha

        # interaction between subject_object feature (x_so) and context feature (x_c)
        x = x_so + x_c

        return x

    def __hard_max(self, f):
        _, max_value_indexes = f.data.max(1, keepdim=True)
        alpha = torch.zeros_like(f).scatter_(1, max_value_indexes, 1)
        return alpha

class ContextGatingSigmoidV1(nn.Module):
    def __init__(self, x_so_shape, x_c_shape, n_channels):
        super(ContextGatingSigmoidV1, self).__init__()

        N, C1, H1, W1 = x_so_shape
        C2, H2, W2 = x_c_shape
        n_channels_half = int(n_channels / 2.0)
        self.gating_loss = 0.0

        # layers for input embedding
        self.dense_so = nn.Sequential(nn.BatchNorm3d(C1), pl.Linear3d(C1, n_channels, dim=1))
        self.dense_c = nn.Sequential(nn.BatchNorm2d(C2), pl.Linear2d(C2, n_channels, dim=1))

        # layers for gating
        alpha_layers = []
        alpha_layers.append(nn.BatchNorm3d(n_channels))
        alpha_layers.append(pl.Linear3d(n_channels, n_channels_half))
        alpha_layers.append(nn.BatchNorm3d(n_channels_half))
        alpha_layers.append(nn.LeakyReLU(0.2))
        alpha_layers.append(pl.Linear3d(n_channels_half, 1))
        alpha_layers.append(pl.Squeeze(dim=1))
        alpha_layers.append(pl.Mean(dim=(2, 3)))
        self.alpha_layers = nn.Sequential(*alpha_layers)

        threshold_value = 0.5
        pf.Threshold.THRESHOLD_VALUE = threshold_value
        pf.BinaryThreshold.THRESHOLD_VALUE = threshold_value

        self.alpha_threshold = pf.Threshold.apply
        self.alpha_binary_threshold = pf.BinaryThreshold.apply

        self.alpha_sigmoid = nn.Sigmoid()
        self.alpha_gumbel_noise = pl.GumbelNoise(loc=0.5, scale=1.0)
        self.alpha_gaussian_noise = pl.GaussianNoise(loc=0.0, scale=1.0)

    def forward(self, x_so, x_c):
        # pairwise interaction between x_so and x_c
        x_so = self.dense_so(x_so)  # (B, C, N, H, W)
        x_c = self.dense_c(x_c)  # (B, C, N, H, W)
        x_c = torch.unsqueeze(x_c, dim=2)  # (B, C, N, H, W)

        # gating
        x = x_so + x_c  # (B, C, N, H, W)
        f = self.alpha_layers(x)  # (B, N)

        # noise
        f = self.alpha_gumbel_noise(f) if self.training else f  # (B, N)

        # activation
        f = self.alpha_sigmoid(f)  # (B, N)

        # thresholding
        alpha = self.alpha_threshold(f) if self.training else self.alpha_binary_threshold(f)  # (B, N)

        # gating loss
        self.gating_loss = torch.mean(f) if self.training else None

        # save values for debugging
        self.f_values = np.array(f.tolist())
        self.alpha_values = np.array(alpha.tolist())

        # multiply the gating value by the context feature
        B, N = pytorch_utils.get_shape(alpha)
        alpha = alpha.view(B, 1, N, 1, 1)  # (B, 1, N, 1, 1)
        x_c = x_c * alpha

        # interaction between subject_object feature (x_so) and context feature (x_c)
        x = x_so + x_c

        return x

class ContextGatingSigmoidV2(nn.Module):
    def __init__(self, x_so_shape, x_c_shape, n_channels):
        super(ContextGatingSigmoidV2, self).__init__()

        C_so, N, H1, W1 = x_so_shape
        C_c, _, H2, W2 = x_c_shape

        n_channels_f = 128
        n_channels_f_half = int(n_channels_f / 2.0)

        # layers for input embedding
        self.f_dense_so = nn.Sequential(nn.BatchNorm3d(C_so), pl.Linear3d(C_so, n_channels_f), nn.LeakyReLU(0.2))
        self.f_dense_c = nn.Sequential(nn.BatchNorm3d(C_c), pl.Linear3d(C_c, n_channels_f), nn.LeakyReLU(0.2))

        # layers for input embedding
        self.x_dense_so = nn.Sequential(nn.BatchNorm3d(C_so), pl.Linear3d(C_so, n_channels))
        self.x_dense_c = nn.Sequential(nn.BatchNorm3d(C_c), pl.Linear3d(C_c, n_channels))

        self.output_activation = nn.Sequential(nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))

        # layers for gating
        f_layers = []
        f_layers.append(nn.BatchNorm3d(n_channels_f))
        f_layers.append(pl.Linear3d(n_channels_f, n_channels_f_half))
        f_layers.append(nn.BatchNorm3d(n_channels_f_half))
        f_layers.append(nn.LeakyReLU(0.2))
        f_layers.append(pl.Linear3d(n_channels_f_half, 1))
        f_layers.append(pl.Squeeze(dim=1))
        f_layers.append(pl.Mean(dim=(2, 3)))
        self.f_layers = nn.Sequential(*f_layers)

        threshold_value = 0.5
        pf.Threshold.THRESHOLD_VALUE = threshold_value
        pf.BinaryThreshold.THRESHOLD_VALUE = threshold_value

        self.f_threshold = pf.Threshold.apply
        self.f_binary_threshold = pf.BinaryThreshold.apply

        self.f_sigmoid = nn.Sigmoid()
        self.f_gumbel_noise = pl.GumbelNoise(loc=0.5, scale=1.0)
        self.f_gaussian_noise = pl.GaussianNoise(loc=0.0, scale=1.0)

    def forward(self, x_so, x_c):
        # pairwise interaction between x_so and x_c
        f_so = self.f_dense_so(x_so)  # (B, C, N, H, W)
        f_c = self.f_dense_c(x_c)  # (B, C, N, H, W)

        # gating
        f = f_so + f_c  # (B, C, N, H, W)
        f = self.f_layers(f)  # (B, N)

        # noise
        f = self.f_gumbel_noise(f) if self.training else f  # (B, N)

        # activation
        f = self.f_sigmoid(f)  # (B, N)

        # thresholding
        alpha = self.f_threshold(f) if self.training else self.f_binary_threshold(f)  # (B, N)

        # save values for debugging
        self.f_values = np.array(f.tolist())
        self.alpha_values = np.array(alpha.tolist())

        # embedding for output
        x_so = self.x_dense_so(x_so)
        x_c = self.x_dense_c(x_c)

        # multiply the gating value by the context feature
        B, N = pytorch_utils.get_shape(alpha)
        alpha = alpha.view(B, 1, N, 1, 1)  # (B, 1, N, 1, 1)
        x_c = x_c * alpha

        # interaction between subject_object feature (x_so) and context feature (x_c)
        x = x_so + x_c
        x = self.output_activation(x)

        return x

class ContextGatingSigmoid(nn.Module):
    def __init__(self, x_so_shape):
        super(ContextGatingSigmoid, self).__init__()

        n_channels, N, H1, W1 = x_so_shape
        n_channels_half = int(n_channels / 2.0)

        self.output_activation = nn.Sequential(nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))

        # layers for gating
        f_layers = []
        f_layers.append(nn.BatchNorm3d(n_channels))
        f_layers.append(pl.Linear3d(n_channels, n_channels_half))
        f_layers.append(nn.BatchNorm3d(n_channels_half))
        f_layers.append(nn.LeakyReLU(0.2))
        f_layers.append(pl.Linear3d(n_channels_half, 1))
        f_layers.append(pl.Squeeze(dim=1))
        f_layers.append(pl.Mean(dim=(2, 3)))
        self.f_layers = nn.Sequential(*f_layers)

        threshold_value = 0.5
        pf.Threshold.THRESHOLD_VALUE = threshold_value
        pf.BinaryThreshold.THRESHOLD_VALUE = threshold_value

        self.f_threshold = pf.Threshold.apply
        self.f_binary_threshold = pf.BinaryThreshold.apply

        self.f_sigmoid = nn.Sigmoid()
        self.f_gumbel_noise = pl.GumbelNoise(loc=0.5, scale=1.0)
        self.f_gaussian_noise = pl.GaussianNoise(loc=0.0, scale=1.0)

    def forward(self, x_so, x_c):
        # pairwise interaction between x_so and x_c
        f_so = x_so  # (B, C, N, H, W)
        f_c = x_c  # (B, C, 1, H, W)
        f = torch.add(f_so, f_c)  # (B, C, N, H, W)

        # gating
        f = self.f_layers(f)  # (B, N)

        # noise
        f = self.f_gumbel_noise(f) if self.training else f  # (B, N)

        # activation
        f = self.f_sigmoid(f)  # (B, N)

        # thresholding
        alpha = self.f_threshold(f) if self.training else self.f_binary_threshold(f)  # (B, N)

        # save values for debugging
        self.__save_values_for_debugging(f, alpha)

        # multiply the gating value by the context feature
        B, N = pytorch_utils.get_shape(alpha)
        alpha = alpha.view(B, 1, N, 1, 1)  # (B, 1, N, 1, 1)
        x_c = x_c * alpha

        # interaction between subject_object feature (x_so) and context feature (x_c)
        x = torch.add(x_so, x_c)
        x = self.output_activation(x)

        return x

    def __save_values_for_debugging(self, f, alpha):
        is_training = self.training
        if is_training:
            return

        self.f_mean = torch.mean(f)
        self.f_std = torch.std(f)

        non_zero = torch.sum(alpha).item()
        sum = np.prod(pytorch_utils.get_shape(alpha))
        ratio = non_zero / sum
        self.alpha_ratio = ratio

# endregion



class ContextGatingSigmoidClassifier(nn.Module):
    def __init__(self, x_so_shape, x_c_shape):
        super(ContextGatingSigmoidClassifier, self).__init__()

        n_channels, N, H1, W1 = x_so_shape
        n_channels_half = int(n_channels / 2.0)

        self.output_activation = nn.Sequential(nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))

        C_so, N, H1, W1 = x_so_shape
        C_c, _, H2, W2 = x_c_shape[0]

        n_channels = 512
        n_channels_half = 256

        # layers for input embedding
        self.f_dense_so = nn.Sequential(nn.BatchNorm3d(C_so), pl.Linear3d(C_so, n_channels), nn.LeakyReLU(0.2))
        self.f_dense_c = nn.Sequential(nn.BatchNorm3d(C_c), pl.Linear3d(C_c, n_channels), nn.LeakyReLU(0.2))

        # layers for gating
        f_layers = []
        f_layers.append(nn.BatchNorm3d(n_channels*2))
        f_layers.append(pl.Linear3d(n_channels*2, n_channels_half))
        f_layers.append(nn.BatchNorm3d(n_channels_half))
        f_layers.append(nn.LeakyReLU(0.2))
        f_layers.append(pl.Linear3d(n_channels_half, 1))
        f_layers.append(pl.Squeeze(dim=1))
        f_layers.append(pl.Mean(dim=(2, 3)))
        self.f_layers = nn.Sequential(*f_layers)

        threshold_value = 0.5
        pf.Threshold.THRESHOLD_VALUE = threshold_value
        pf.BinaryThreshold.THRESHOLD_VALUE = threshold_value

        self.f_threshold = pf.Threshold.apply
        self.f_binary_threshold = pf.BinaryThreshold.apply

        self.f_sigmoid = nn.Sigmoid()
        self.f_gumbel_noise = pl.GumbelNoise(loc=0.5, scale=1.0)
        self.f_gaussian_noise = pl.GaussianNoise(loc=0.0, scale=1.0)

    def forward(self, x_so, x_c):
        # pairwise interaction between x_so and x_c
        f_so = self.f_dense_so(x_so)  # (B, n_channels, N, H, W)
        f_c = self.f_dense_c(x_c)  # (B, n_channels, N, H, W)
        f = torch.cat((f_so, f_c), dim = 1)  # (B, C, N, H, W)

        # gating
        f = self.f_layers(f)  # (B, N)

        # noise
        f = self.f_gumbel_noise(f) if self.training else f  # (B, N)

        # activation
        f = self.f_sigmoid(f)  # (B, N)

        # thresholding
        alpha = self.f_threshold(f) if self.training else self.f_binary_threshold(f)  # (B, N)

        # save values for debugging
        self.__save_values_for_debugging(f, alpha)

        # multiply the gating value by the context feature
        B, N = pytorch_utils.get_shape(alpha)
        alpha = alpha.view(B, N, 1)  # (B, N, 1)

        return alpha

    def __save_values_for_debugging(self, f, alpha):
        is_training = self.training
        if is_training:
            return

        self.f_mean = torch.mean(f)
        self.f_std = torch.std(f)

        non_zero = torch.sum(alpha).item()
        sum = np.prod(pytorch_utils.get_shape(alpha))
        ratio = non_zero / sum
        self.alpha_ratio = ratio



class ContextGatingSigmoidClassifierSoft(nn.Module):
    def __init__(self, x_so_shape, x_c_shape):
        super(ContextGatingSigmoidClassifierSoft, self).__init__()

        n_channels, N, H1, W1 = x_so_shape
        n_channels_half = int(n_channels / 2.0)

        self.output_activation = nn.Sequential(nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))

        C_so, N, H1, W1 = x_so_shape
        C_c, _, H2, W2 = x_c_shape[0]

        n_channels = 512
        n_channels_half = 256

        # layers for input embedding
        self.f_dense_so = nn.Sequential(nn.BatchNorm3d(C_so), pl.Linear3d(C_so, n_channels), nn.LeakyReLU(0.2))
        self.f_dense_c = nn.Sequential(nn.BatchNorm3d(C_c), pl.Linear3d(C_c, n_channels), nn.LeakyReLU(0.2))

        # layers for gating
        f_layers = []
        f_layers.append(nn.BatchNorm3d(n_channels*2))
        f_layers.append(pl.Linear3d(n_channels*2, n_channels_half))
        f_layers.append(nn.BatchNorm3d(n_channels_half))
        f_layers.append(nn.LeakyReLU(0.2))
        f_layers.append(pl.Linear3d(n_channels_half, 1))
        f_layers.append(pl.Squeeze(dim=1))
        f_layers.append(pl.Mean(dim=(2, 3)))
        self.f_layers = nn.Sequential(*f_layers)
        
        self.f_sigmoid = nn.Sigmoid()


    def forward(self, x_so, x_c):
        # pairwise interaction between x_so and x_c
        f_so = self.f_dense_so(x_so)  # (B, n_channels, N, H, W)
        f_c = self.f_dense_c(x_c)  # (B, n_channels, N, H, W)
        f = torch.cat((f_so, f_c), dim = 1)  # (B, C, N, H, W)

        # gating
        f = self.f_layers(f)  # (B, N)

        # activation
        f = self.f_sigmoid(f)  # (B, N)

        alpha = f

        # save values for debugging
        self.__save_values_for_debugging(f, alpha)

        # multiply the gating value by the context feature
        B, N = pytorch_utils.get_shape(alpha)
        alpha = alpha.view(B, N, 1)  # (B, N, 1)

        return alpha

    def __save_values_for_debugging(self, f, alpha):
        is_training = self.training
        if is_training:
            return

        self.f_mean = torch.mean(f)
        self.f_std = torch.std(f)

        non_zero = torch.sum(alpha).item()
        sum = np.prod(pytorch_utils.get_shape(alpha))
        ratio = non_zero / sum
        self.alpha_ratio = ratio


class ContextGatingClassifierSoft(nn.Module):
    def __init__(self, x_so_shape, x_c_shape):
        super(ContextGatingClassifierSoft, self).__init__()

        n_channels, N, H1, W1 = x_so_shape
        n_channels_half = int(n_channels / 2.0)


        C_so, N, H1, W1 = x_so_shape
        C_c, _, H2, W2 = x_c_shape[0]

        n_channels = 512
        n_channels_half = 256

        # layers for gating
        f_layers = []
        f_layers.append(nn.BatchNorm3d(n_channels*2))
        f_layers.append(pl.Linear3d(n_channels*2, n_channels_half))
        f_layers.append(nn.BatchNorm3d(n_channels_half))
        f_layers.append(nn.LeakyReLU(0.2))
        f_layers.append(pl.Linear3d(n_channels_half, 1))
        f_layers.append(pl.Squeeze(dim=1))
        f_layers.append(pl.Mean(dim=(2, 3)))
        self.f_layers = nn.Sequential(*f_layers)

    def forward(self, x_so, x_c):
        # pairwise interaction between x_so and x_c

        f = torch.cat((x_so, x_c), dim = 1)  # (B, C, N, H, W)

        # gating
        f = self.f_layers(f)  # (B, N)

        alpha = f

        # save values for debugging
        self.__save_values_for_debugging(f, alpha)

        # multiply the gating value by the context feature
        B, N = pytorch_utils.get_shape(alpha)
        alpha = alpha.view(B, N, 1)  # (B, N, 1)

        return alpha

    def __save_values_for_debugging(self, f, alpha):
        is_training = self.training
        if is_training:
            return

        self.f_mean = torch.mean(f)
        self.f_std = torch.std(f)

        non_zero = torch.sum(alpha).item()
        sum = np.prod(pytorch_utils.get_shape(alpha))
        ratio = non_zero / sum
        self.alpha_ratio = ratio

class ContextGatingClassifierSoftAblated(nn.Module):
    def __init__(self, x_so_shape, x_c_shape):
        super(ContextGatingClassifierSoftAblated, self).__init__()

        n_channels, N, H1, W1 = x_so_shape
        n_channels_half = int(n_channels / 2.0)


        C_so, N, H1, W1 = x_so_shape
        C_c, _, H2, W2 = x_c_shape[0]

        n_channels = 512
        n_channels_half = 256

        # layers for gating
        f_layers = []
        f_layers.append(nn.BatchNorm3d(n_channels))
        f_layers.append(pl.Linear3d(n_channels, n_channels_half))
        f_layers.append(nn.BatchNorm3d(n_channels_half))
        f_layers.append(nn.LeakyReLU(0.2))
        f_layers.append(pl.Linear3d(n_channels_half, 1))
        f_layers.append(pl.Squeeze(dim=1))
        f_layers.append(pl.Mean(dim=(2, 3)))
        self.f_layers = nn.Sequential(*f_layers)

    def forward(self, x_so, x_c):
        # pairwise interaction between x_so and x_c

        f = x_c # (B, C, N, H, W)

        # gating
        f = self.f_layers(f)  # (B, N)

        alpha = f

        # save values for debugging
        self.__save_values_for_debugging(f, alpha)

        # multiply the gating value by the context feature
        B, N = pytorch_utils.get_shape(alpha)
        alpha = alpha.view(B, N, 1)  # (B, N, 1)

        return alpha

    def __save_values_for_debugging(self, f, alpha):
        is_training = self.training
        if is_training:
            return

        self.f_mean = torch.mean(f)
        self.f_std = torch.std(f)

        non_zero = torch.sum(alpha).item()
        sum = np.prod(pytorch_utils.get_shape(alpha))
        ratio = non_zero / sum
        self.alpha_ratio = ratio



class LocalContextGatingSigmoid(nn.Module):
    def __init__(self, x_so_shape):
        super(LocalContextGatingSigmoid, self).__init__()

        n_channels, N, H1, W1 = x_so_shape
        n_channels_half = int(n_channels / 2.0)

        self.output_activation = nn.Sequential(nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))

        # layers for gating
        f_layers = []
        f_layers.append(nn.BatchNorm3d(n_channels))
        f_layers.append(pl.Linear3d(n_channels, n_channels_half))
        f_layers.append(nn.BatchNorm3d(n_channels_half))
        f_layers.append(nn.LeakyReLU(0.2))
        f_layers.append(pl.Linear3d(n_channels_half, 1))
        self.f_layers = nn.Sequential(*f_layers)

        threshold_value = 0.5
        pf.Threshold.THRESHOLD_VALUE = threshold_value
        pf.BinaryThreshold.THRESHOLD_VALUE = threshold_value

        self.f_threshold = pf.Threshold.apply
        self.f_binary_threshold = pf.BinaryThreshold.apply

        self.f_sigmoid = nn.Sigmoid()
        self.f_gumbel_noise = pl.GumbelNoise(loc=0.5, scale=1.0)
        self.f_gaussian_noise = pl.GaussianNoise(loc=0.0, scale=1.0)

    def forward(self, x_so):
        # pairwise interaction between x_so and x_c

        f = x_so # (B, C, N, H, W)

        # gating

        for l in self.f_layers:
            f = l(f)

        f = torch.squeeze(f)

        # noise
        f = self.f_gumbel_noise(f) if self.training else f  # (B, N)

        # activation
        f = self.f_sigmoid(f)  # (B, N)

        # thresholding
        alpha = self.f_threshold(f) if self.training else self.f_binary_threshold(f)  # (B, N)

        # save values for debugging
        self.__save_values_for_debugging(f, alpha)

        # multiply the gating value by the context feature
        B, N = pytorch_utils.get_shape(alpha)
        alpha = alpha.view(B, 1, N, 1, 1)  # (B, 1, N, 1, 1)
        x = x_so * alpha

        x = self.output_activation(x)

        return x

    def __save_values_for_debugging(self, f, alpha):
        is_training = self.training
        if is_training:
            return

        self.f_mean = torch.mean(f)
        self.f_std = torch.std(f)

        non_zero = torch.sum(alpha).item()
        sum = np.prod(pytorch_utils.get_shape(alpha))
        ratio = non_zero / sum
        self.alpha_ratio = ratio

# endregion




class ContextGatingSigmoidConcatSoft(nn.Module):
    def __init__(self, x_so_shape):
        super(ContextGatingSigmoidConcatSoft, self).__init__()

        n_channels, N, H1, W1 = x_so_shape
        n_channels_half = int(n_channels / 2.0)

        self.N = N

        self.output_activation = nn.Sequential(nn.BatchNorm3d(2*n_channels), nn.LeakyReLU(0.2))

        # layers for gating
        f_layers = []
        f_layers.append(nn.BatchNorm3d(2*n_channels))
        f_layers.append(pl.Linear3d(2*n_channels, n_channels_half))
        f_layers.append(nn.BatchNorm3d(n_channels_half))
        f_layers.append(nn.LeakyReLU(0.2))
        f_layers.append(pl.Linear3d(n_channels_half, 1))
        f_layers.append(pl.Squeeze(dim=1))
        f_layers.append(pl.Mean(dim=(2, 3)))
        self.f_layers = nn.Sequential(*f_layers)

        self.f_sigmoid = nn.Sigmoid()

        self.f_gumbel_noise = pl.GumbelNoise(loc=0.5, scale=1.0)
        self.f_gaussian_noise = pl.GaussianNoise(loc=0.0, scale=1.0)

    def forward(self, x_so, x_c):
        # pairwise interaction between x_so and x_c
        f_so = x_so  # (B, C, N, H, W)

        x_c = x_c.repeat(1, 1, self.N, 1, 1)

        f_c = x_c  # (B, C, 1, H, W)
        f = torch.cat((f_so, f_c), dim=1)  # (B, C, N, H, W)

        # gating
        f = self.f_layers(f)  # (B, N)

        # noise
        f = self.f_gumbel_noise(f) if self.training else f  # (B, N)

        # activation
        f = self.f_sigmoid(f)  # (B, N)

        # thresholding
        alpha = f

        # save values for debugging
        self.__save_values_for_debugging(f, alpha)

        # multiply the gating value by the context feature
        B, N = pytorch_utils.get_shape(alpha)
        alpha = alpha.view(B, 1, N, 1, 1)  # (B, 1, N, 1, 1)
        x_c = x_c * alpha

        # interaction between subject_object feature (x_so) and context feature (x_c)
        x = torch.cat((x_so, x_c), dim=1)
        x = self.output_activation(x)

        return x

    def __save_values_for_debugging(self, f, alpha):
        is_training = self.training
        if is_training:
            return

        self.f_mean = torch.mean(f)
        self.f_std = torch.std(f)

        alpha = alpha > 0.5

        non_zero = torch.sum(alpha).item()
        sum = np.prod(pytorch_utils.get_shape(alpha))
        ratio = non_zero / sum
        self.alpha_ratio = ratio


# region Channel Gating


class ContextGatingSigmoidConcat(nn.Module):
    def __init__(self, x_so_shape):
        super(ContextGatingSigmoidConcat, self).__init__()

        n_channels, N, H1, W1 = x_so_shape
        n_channels_half = int(n_channels / 2.0)

        self.N = N

        self.output_activation = nn.Sequential(nn.BatchNorm3d(2*n_channels), nn.LeakyReLU(0.2))

        # layers for gating
        f_layers = []
        f_layers.append(nn.BatchNorm3d(2*n_channels))
        f_layers.append(pl.Linear3d(2*n_channels, n_channels_half))
        f_layers.append(nn.BatchNorm3d(n_channels_half))
        f_layers.append(nn.LeakyReLU(0.2))
        f_layers.append(pl.Linear3d(n_channels_half, 1))
        f_layers.append(pl.Squeeze(dim=1))
        f_layers.append(pl.Mean(dim=(2, 3)))
        self.f_layers = nn.Sequential(*f_layers)

        threshold_value = 0.5
        pf.Threshold.THRESHOLD_VALUE = threshold_value
        pf.BinaryThreshold.THRESHOLD_VALUE = threshold_value

        self.f_threshold = pf.Threshold.apply
        self.f_binary_threshold = pf.BinaryThreshold.apply

        self.f_sigmoid = nn.Sigmoid()
        self.f_gumbel_noise = pl.GumbelNoise(loc=0.5, scale=1.0)
        self.f_gaussian_noise = pl.GaussianNoise(loc=0.0, scale=1.0)

    def forward(self, x_so, x_c):
        # pairwise interaction between x_so and x_c
        f_so = x_so  # (B, C, N, H, W)

        x_c = x_c.repeat(1, 1, self.N, 1, 1)

        f_c = x_c  # (B, C, 1, H, W)

        f = torch.cat((f_so, f_c), dim=1)  # (B, C, N, H, W)

        # gating
        f = self.f_layers(f)  # (B, N)

        # noise
        f = self.f_gumbel_noise(f) if self.training else f  # (B, N)

        # activation
        f = self.f_sigmoid(f)  # (B, N)

        # thresholding
        alpha = self.f_threshold(f) if self.training else self.f_binary_threshold(f)  # (B, N)

        # save values for debugging
        self.__save_values_for_debugging(f, alpha)

        # multiply the gating value by the context feature
        B, N = pytorch_utils.get_shape(alpha)
        alpha = alpha.view(B, 1, N, 1, 1)  # (B, 1, N, 1, 1)
        x_c = x_c * alpha

        # interaction between subject_object feature (x_so) and context feature (x_c)
        x = torch.cat((x_so, x_c), dim=1)
        x = self.output_activation(x)

        return x

    def __save_values_for_debugging(self, f, alpha):
        is_training = self.training
        if is_training:
            return

        self.f_mean = torch.mean(f)
        self.f_std = torch.std(f)

        non_zero = torch.sum(alpha).item()
        sum = np.prod(pytorch_utils.get_shape(alpha))
        ratio = non_zero / sum
        self.alpha_ratio = ratio


# region Channel Gating

class ChannelGatingSigmoid(nn.Module):
    def __init__(self, n_channels):
        super(ChannelGatingSigmoid, self).__init__()

        # layers for gating
        f_layers = []
        f_layers.append(nn.BatchNorm3d(n_channels))
        f_layers.append(nn.Dropout(0.25))
        f_layers.append(pl.Linear3d(n_channels, n_channels))
        f_layers.append(nn.LeakyReLU(0.2))
        f_layers.append(nn.BatchNorm3d(n_channels))
        f_layers.append(nn.Dropout(0.25))
        f_layers.append(pl.Linear3d(n_channels, n_channels))
        f_layers.append(pl.Mean(dim=(3, 4), keepdim=True))
        self.f_layers = nn.Sequential(*f_layers)

        threshold_value = 0.5
        pf.Threshold.THRESHOLD_VALUE = threshold_value
        pf.BinaryThreshold.THRESHOLD_VALUE = threshold_value

        self.f_threshold = pf.Threshold.apply
        self.f_binary_threshold = pf.BinaryThreshold.apply

        self.f_sigmoid = nn.Sigmoid()
        self.f_gumbel_noise = pl.GumbelNoise(loc=0.5, scale=1.0)
        self.f_gaussian_noise = pl.GaussianNoise(loc=0.0, scale=1.0)

    def forward(self, x_so, x_c, is_agating_only=False):
        # pairwise interaction between x_so and x_c

        f = x_so + x_c

        # gating
        f = self.f_layers(f)  # (B, C, N, 1, 1)

        # noise
        f = self.f_gumbel_noise(f) if self.training else f  # (B, C, N, 1, 1)

        # activation
        f = self.f_sigmoid(f)  # (B, C, N, 1, 1)

        # thresholding
        alpha = self.f_threshold(f) if self.training else self.f_binary_threshold(f)  # (B, C, N, 1, 1)

        if is_agating_only:
            return f, alpha
        else:
            # multiply the gating value by the context feature
            x_c = x_c * alpha
            return x_c

    def forward_for_gating(self, x_so, x_c):

        f, alpha = self.forward(x_so, x_c, is_agating_only=True)
        return f, alpha

    def __save_values_for_debugging(self, f, alpha):
        is_training = self.training
        if is_training:
            return

        self.f_mean = torch.mean(f)
        self.f_std = torch.std(f)

        non_zero = torch.sum(alpha).item()
        sum = np.prod(pytorch_utils.get_shape(alpha))
        ratio = non_zero / sum
        self.alpha_ratio = ratio

# endregion

# region Context Interaction

class ContextInteraction(nn.Module):
    def __init__(self, x_so_shape, x_cs_shape, n_channels_inner, n_channels_out):
        super(ContextInteraction, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.n_channels_inner = n_channels_inner
        self.n_channels_out = n_channels_out
        self.layer_name_key = 'key_%d'
        self.layer_name_value = 'value_%d'
        self.__define_layers(x_so_shape, x_cs_shape, n_channels_inner, n_channels_out)

    def __define_layers(self, x_so_shape, x_cs_shape, n_channels_inner, n_channels_out):
        """
        # values: embedding of x
        # query: embedding of x
        # keys: embedding of contexts
        :param x_so_shape:
        :param x_cs_shape:
        :param n_channels_inner:
        :return:
        """

        C_so, N, H, W = x_so_shape
        M = len(x_cs_shape)

        for idx_context in range(self.n_contexts):
            C_c = x_cs_shape[idx_context][0]

            layer_name = self.layer_name_key % (idx_context + 1)
            layer = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_c, n_channels_inner))
            setattr(self, layer_name, layer)

            layer_name = self.layer_name_value % (idx_context + 1)
            layer = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_c, n_channels_out))
            setattr(self, layer_name, layer)

        self.query_embedding = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_so, n_channels_inner))
        self.alpha_layers = nn.Sequential(nn.BatchNorm1d(M), pl.Linear1d(M, M), nn.LeakyReLU(0.2), nn.BatchNorm1d(M), pl.Linear1d(M, M), nn.Sigmoid())

    def forward(self, *inputs):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_cs: comtext features [(B, C, N, H, W), (), ...]
        :return:
        """

        x_so = inputs[0]
        x_cs = inputs[1:]

        batch_size = pytorch_utils.get_shape(x_so)[0]
        n_channels_inner = self.n_channels_inner
        n_channels_out = self.n_channels_out

        # value embedding
        value = [getattr(self, self.layer_name_value % (idx_context + 1))(x_cs[idx_context]) for idx_context in range(self.n_contexts)]
        value = torch.cat(value, dim=2)  # (B, C, M, H, W)
        value = value.view(batch_size, n_channels_out, -1)  # (B, C, M*H*W)

        # key embedding
        key = [getattr(self, self.layer_name_key % (idx_context + 1))(x_cs[idx_context]) for idx_context in range(self.n_contexts)]
        key = torch.cat(key, dim=2)  # (B, C, M, H, W)
        key = key.view(batch_size, n_channels_inner, -1)  # (B, C, M*H*W)

        # query embedding
        query = self.query_embedding(x_so)  # (B, C, N, H, W)
        query = query.view(batch_size, n_channels_inner, -1)  # (B, C, N*H*W)
        query = query.permute(0, 2, 1)  # (B, N*H*W, C)

        # attention
        alpha = torch.matmul(query, key)  # (B, N, M)
        alpha = alpha.permute(0, 2, 1)  # (B, M, N)
        alpha = self.alpha_layers(alpha)  # (B, M, N)

        # then, multiply alpha times the value * alpha
        y = torch.matmul(value, alpha)  # (B, C, N)

        B, C, N = pytorch_utils.get_shape(y)
        y = y.view(B, C, N, 1, 1)  # (B, C, N, H, W)

        return y

# endregion
class ContextInteractionBottleneckHead(nn.Module):
    def __init__(self, x_so_shape, x_cs_shape, n_channels_inner, n_channels_out):
        super(ContextInteractionBottleneckHead, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.n_channels_inner = n_channels_inner
        self.n_channels_out = n_channels_out
        self.layer_name_key = 'key_%d'
        self.layer_name_value = 'value_%d'
        self.__define_layers(x_so_shape, x_cs_shape, n_channels_inner, n_channels_out)

    def __define_layers(self, x_so_shape, x_cs_shape, n_channels_inner, n_channels_out):
        """
        # values: embedding of x
        # query: embedding of x
        # keys: embedding of contexts
        :param x_so_shape:
        :param x_cs_shape:
        :param n_channels_inner:
        :return:
        """

        C_so, N, H, W = x_so_shape
        M = len(x_cs_shape)

        for idx_context in range(self.n_contexts):
            C_c = x_cs_shape[idx_context][0]

            layer_name = self.layer_name_key % (idx_context + 1)
            layer = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_c, n_channels_inner))
            setattr(self, layer_name, layer)

            layer_name = self.layer_name_value % (idx_context + 1)
            layer = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(n_channels_inner, n_channels_out))
            setattr(self, layer_name, layer)

        self.query_embedding = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_so, n_channels_inner))
        self.alpha_layers = nn.Sequential(nn.Sigmoid())

    def forward(self, *inputs):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_cs: comtext features [(B, C, N, H, W), (), ...]
        :return:
        """

        x_so = inputs[0]
        x_cs = inputs[1:]

        batch_size = pytorch_utils.get_shape(x_so)[0]
        n_channels_inner = self.n_channels_inner
        n_channels_out = self.n_channels_out
        M = self.n_contexts

        # query embedding
        query = self.query_embedding(x_so)  # (B, C, N, H, W)
        query = query.view(batch_size, n_channels_inner, -1)  # (B, C, N*H*W)
        query = query.permute(0, 2, 1)  # (B, N*H*W, C)

        # key embedding
        key = [getattr(self, self.layer_name_key % (idx_context + 1))(x_cs[idx_context]) for idx_context in range(self.n_contexts)]

        # value embedding
        value = [getattr(self, self.layer_name_value % (idx_context + 1))(key[idx_context]) for idx_context in range(self.n_contexts)]

        # reshape for dot product
        key = torch.cat(key, dim=2)  # (B, C, M, H, W)
        value = torch.cat(value, dim=2)  # (B, C, M, H, W)
        key = key.view(batch_size, n_channels_inner, M)  # (B, C, M*H*W)
        value = value.view(batch_size, n_channels_out, M)  # (B, C, M*H*W)

        # attention
        alpha = torch.matmul(query, key)  # (B, N, M)
        alpha = alpha.permute(0, 2, 1)  # (B, M, N)
        alpha = self.alpha_layers(alpha)  # (B, M, N)

        # then, multiply alpha times the value * alpha
        y = torch.matmul(value, alpha)  # (B, C, N)

        B, C, N = pytorch_utils.get_shape(y)
        y = y.view(B, C, N, 1, 1)  # (B, C, N, H, W)

        return y

    def forward_for_alpha(self, *inputs):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_cs: comtext features [(B, C, N, H, W), (), ...]
        :return:
        """

        x_so = inputs[0]
        x_cs = inputs[1:]

        batch_size = pytorch_utils.get_shape(x_so)[0]
        n_channels_inner = self.n_channels_inner
        n_channels_out = self.n_channels_out
        M = self.n_contexts

        # query embedding
        query = self.query_embedding(x_so)  # (B, C, N, H, W)
        query = query.view(batch_size, n_channels_inner, -1)  # (B, C, N*H*W)
        query = query.permute(0, 2, 1)  # (B, N*H*W, C)

        # key embedding
        key = [getattr(self, self.layer_name_key % (idx_context + 1))(x_cs[idx_context]) for idx_context in range(self.n_contexts)]

        # reshape for dot product
        key = torch.cat(key, dim=2)  # (B, C, M, H, W)
        key = key.view(batch_size, n_channels_inner, M)  # (B, C, M*H*W)

        # attention
        alpha = torch.matmul(query, key)  # (B, N, M)
        alpha = alpha.permute(0, 2, 1)  # (B, M, N)
        alpha = self.alpha_layers(alpha)  # (B, M, N)

        return alpha

class ContextInteractionBottleneckMultiHeadSum(nn.Module):
    """
    MultiHead for Context Interaction.
    """

    def __init__(self, x_so_shape, x_cs_shape, n_channels_inner, n_channels_out, n_heads):
        """
        Initialize the module.
        """
        super(ContextInteractionBottleneckMultiHeadSum, self).__init__()

        self.n_heads = n_heads

        # we use n heads, each has inner dim
        for idx_head in range(n_heads):
            head_num = idx_head + 1
            interaction_head_name = 'interaction_head_%d' % (head_num)
            interaction_head = ContextInteractionBottleneckHead(x_so_shape, x_cs_shape, n_channels_inner, n_channels_out)
            setattr(self, interaction_head_name, interaction_head)

    def forward(self, *inputs):

        z = []
        K = self.n_heads

        # feed to to interaction_head, multi-heads
        for idx_head in range(K):
            head_num = idx_head + 1
            interaction_head_name = 'interaction_head_%d' % (head_num)
            interaction_head = getattr(self, interaction_head_name)
            z_head = interaction_head(*inputs)  # (B, C, N, H, W)
            z.append(z_head)

        # sum over the head dimension
        z = torch.stack(z, dim=1)  # (B, 1, C, N, H, W)
        z = torch.mean(z, dim=1)  # (B, C, N, H, W)

        return z

    def forward_for_alpha(self, *inputs):

        alpha = []
        K = self.n_heads

        # feed to to interaction_head, multi-heads
        for idx_head in range(K):
            head_num = idx_head + 1
            interaction_head_name = 'interaction_head_%d' % (head_num)
            interaction_head = getattr(self, interaction_head_name)
            alpha_head = interaction_head.forward_for_alpha(*inputs)  # (B, M, N)
            alpha.append(alpha_head)

        # stack
        alpha = torch.stack(alpha, dim=3)  # (B, M, N, K)

        return alpha

# endregion

# region Context Interaction: Ablation

class ContextInteractionAblationHead(nn.Module):
    def __init__(self, x_cs_shape, n_channels_inner, n_channels_out):
        super(ContextInteractionAblationHead, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.n_channels_inner = n_channels_inner
        self.n_channels_out = n_channels_out
        self.layer_name_key = 'key_%d'
        self.layer_name_value = 'value_%d'
        self.__define_layers(x_cs_shape, n_channels_inner, n_channels_out)

    def __define_layers(self, x_cs_shape, n_channels_inner, n_channels_out):
        """
        # values: embedding of x
        # query: embedding of x
        # keys: embedding of contexts
        :param x_so_shape:
        :param x_cs_shape:
        :param n_channels_inner:
        :return:
        """

        M = len(x_cs_shape)

        for idx_context in range(self.n_contexts):
            C_c = x_cs_shape[idx_context][0]

            layer_name = self.layer_name_key % (idx_context + 1)
            layer = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_c, n_channels_inner))
            setattr(self, layer_name, layer)

            layer_name = self.layer_name_value % (idx_context + 1)
            layer = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(n_channels_inner, n_channels_out))
            setattr(self, layer_name, layer)

        self.query_embedding = nn.Sequential(pl.Linear1d(n_channels_inner * M, n_channels_inner, dim=2))
        self.alpha_layers = nn.Sequential(nn.Sigmoid())

    def forward(self, *inputs):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_cs: comtext features [(B, C, N, H, W), (), ...]
        :return:
        """

        x_cs = inputs

        batch_size = pytorch_utils.get_shape(x_cs[0])[0]
        n_channels_inner = self.n_channels_inner
        n_channels_out = self.n_channels_out
        M = self.n_contexts

        # key embedding
        key = [getattr(self, self.layer_name_key % (idx_context + 1))(x_cs[idx_context]) for idx_context in range(self.n_contexts)]

        # value embedding
        value = [getattr(self, self.layer_name_value % (idx_context + 1))(key[idx_context]) for idx_context in range(self.n_contexts)]

        # reshape for dot product
        key = torch.cat(key, dim=2)  # (B, C, M, H, W)
        key = key.view(batch_size, n_channels_inner, M)  # (B, C, M*H*W)
        value = torch.cat(value, dim=2)  # (B, C, M, H, W)
        value = value.view(batch_size, n_channels_out, M)  # (B, C, M*H*W)

        # the concat of keys will be the query
        query = key.view(batch_size, 1, n_channels_inner * M)  # (B, 1, C)

        # query embedding
        query = self.query_embedding(query)  # (B, 1, C)

        # attention
        alpha = torch.matmul(query, key)  # (B, 1, M)
        alpha = alpha.permute(0, 2, 1)  # (B, M, 1)
        alpha = self.alpha_layers(alpha)  # (B, M, 1)

        # then, multiply alpha times the value * alpha
        y = torch.matmul(value, alpha)  # (B, C, 1)
        y = y.view(batch_size, n_channels_out, 1, 1, 1)  # (B, C, 1, H, W)

        return y


class ContextGatingSigmoidConcatConditionSumCombination(nn.Module):
    def __init__(self, x_so_shape, x_cs_shape):
        super(ContextGatingSigmoidConcatConditionSumCombination, self).__init__()

        n_channels_so, N, H1, W1 = x_so_shape
        n_channels_cs, _, _, _ = x_cs_shape[0]

        n_channels_f = 512

        n_channels_half = int(n_channels_f / 2.0)

        self.N = N

        self.output_activation = nn.Sequential(nn.BatchNorm3d(n_channels_f), nn.LeakyReLU(0.2))

        # layers for input embedding
        #self.f_dense_so = nn.Sequential(nn.BatchNorm3d(n_channels_so), pl.Linear3d(n_channels_so, n_channels_f), nn.LeakyReLU(0.2))
        #self.f_dense_c = nn.Sequential(nn.BatchNorm3d(n_channels_cs), pl.Linear3d(n_channels_cs, n_channels_f), nn.LeakyReLU(0.2))

        # layers for input embedding
        #self.x_dense_so = nn.Sequential(nn.BatchNorm3d(n_channels_so), pl.Linear3d(n_channels_so, n_channels_f))
        #self.x_dense_c = nn.Sequential(nn.BatchNorm3d(n_channels_cs), pl.Linear3d(n_channels_cs, n_channels_f))

        # layers for gating
        f_layers = []
        f_layers.append(nn.BatchNorm3d(n_channels_f))
        f_layers.append(pl.Linear3d(n_channels_f, n_channels_half))
        f_layers.append(nn.BatchNorm3d(n_channels_half))
        f_layers.append(nn.LeakyReLU(0.2))
        f_layers.append(pl.Linear3d(n_channels_half, 1))
        f_layers.append(pl.Squeeze(dim=1))
        f_layers.append(pl.Mean(dim=(2, 3)))
        self.f_layers = nn.Sequential(*f_layers)

        threshold_value = 0.5
        pf.Threshold.THRESHOLD_VALUE = threshold_value
        pf.BinaryThreshold.THRESHOLD_VALUE = threshold_value

        self.f_threshold = pf.Threshold.apply
        self.f_binary_threshold = pf.BinaryThreshold.apply

        self.f_sigmoid = nn.Sigmoid()
        self.f_gumbel_noise = pl.GumbelNoise(loc=0.5, scale=1.0)
        self.f_gaussian_noise = pl.GaussianNoise(loc=0.0, scale=1.0)

    def forward(self, x_so, x_c):

        # pairwise interaction between x_so and x_c
        f_so = x_so # (B, C, N, H, W)

        x_c = x_c.repeat(1, 1, self.N, 1, 1)

        f_c = x_c # (B, C, N, H, W)

        f = torch.add(f_so, f_c)

        # gating
        f = self.f_layers(f)  # (B, N)

        # noise
        f = self.f_gumbel_noise(f) if self.training else f  # (B, N)

        # activation
        f = self.f_sigmoid(f)  # (B, N)

        # thresholding
        alpha = self.f_threshold(f) if self.training else self.f_binary_threshold(f)  # (B, N)

        # save values for debugging
        self.__save_values_for_debugging(f, alpha)

        # multiply the gating value by the context feature
        B, N = pytorch_utils.get_shape(alpha)
        alpha = alpha.view(B, 1, N, 1, 1)  # (B, 1, N, 1, 1)

        # embed features before we modulate them with alpha

        x_c = x_c * alpha

        # interaction between subject_object feature (x_so) and context feature (x_c)
        x = torch.add(x_so, x_c)

        x = self.output_activation(x)

        return x

    def __save_values_for_debugging(self, f, alpha):
        is_training = self.training
        if is_training:
            return

        self.f_mean = torch.mean(f)
        self.f_std = torch.std(f)

        non_zero = torch.sum(alpha).item()
        sum = np.prod(pytorch_utils.get_shape(alpha))
        ratio = non_zero / sum
        self.alpha_ratio = ratio

