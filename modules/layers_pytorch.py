#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import math
import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data
from torch import distributions
from torch.autograd import Variable
from torchvision import datasets, transforms

from modules import functions_pytorch as pf
from core import pytorch_utils

# region Basic Layers

class Max(nn.Module):
    def __init__(self, dim, keepdim=False):
        super(Max, self).__init__()

        self.dim = dim
        self.keepdim = keepdim

    def forward(self, input):
        # input is of shape (None, C, T, H, W)

        dim_type = type(self.dim)
        dims = self.dim if (dim_type is list or dim_type is tuple) else [self.dim]
        dims = np.sort(dims)[::-1]
        dims = [int(d) for d in dims]
        tensor = input
        for d in dims:
            tensor, _ = torch.max(tensor, dim=d, keepdim=self.keepdim)
        return tensor

class Mean(nn.Module):
    def __init__(self, dim, keepdim=False):
        super(Mean, self).__init__()

        self.dim = dim
        self.keepdim = keepdim

    def forward(self, input):
        # input is of shape (None, C, T, H, W)

        dim_type = type(self.dim)
        dims = self.dim if (dim_type is list or dim_type is tuple) else [self.dim]
        dims = np.sort(dims)[::-1]
        dims = [int(d) for d in dims]
        tensor = input
        for d in dims:
            tensor = torch.mean(tensor, dim=d, keepdim=self.keepdim)
        return tensor

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        pass

    def forward(self, input):
        batch_size = input.size(0)
        output = input.view(batch_size, -1)
        return output

class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super(Squeeze, self).__init__()
        self.dim = dim
        pass

    def forward(self, input):

        if self.dim is None:
            output = torch.squeeze(input)
        else:
            dim_type = type(self.dim)
            dims = self.dim if (dim_type is list or dim_type is tuple) else [self.dim]

            output = input
            for d in dims:
                output = torch.squeeze(output, dim=d)

        return output

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape
        pass

    def forward(self, input):
        B = pytorch_utils.get_shape(input)[0]
        new_shape = [B] + list(self.shape)
        output = input.view(*new_shape)

        return output

# endregion

# region BatchNorm with Axis

class BatchNorm(nn.Module):
    """
    Batch norm of features for linear layer, with ability to specify axis.
    """

    def __init__(self, num_features, dim=1):
        super(BatchNorm, self).__init__()

        assert dim in [0, 1]

        self.dim = dim
        self.num_features = num_features
        self.layer = nn.BatchNorm1d(num_features)

    def forward(self, input):

        input_shape = pytorch_utils.get_shape(input)
        assert len(input_shape) == 2

        dim = self.dim

        # permute to put the required dimension in the 2nd dimension
        if dim == 0:
            x = input.permute(1, 0)
        else:
            x = input

        # apply batch_norm
        num_features = pytorch_utils.get_shape(x)[1]
        assert num_features == self.num_features
        x = self.layer(x)

        # permute back to the original view
        if dim == 0:
            x = x.permute(1, 0)

        x_shape = pytorch_utils.get_shape(x)
        assert input_shape == x_shape

        return x

class BatchNorm1d(nn.Module):

    def __init__(self, num_features, dim=1):
        super(BatchNorm1d, self).__init__()

        assert dim in [1, 2]

        self.dim = dim
        self.num_features = num_features
        self.layer = nn.BatchNorm1d(num_features)

    def forward(self, input):

        input_shape = pytorch_utils.get_shape(input)
        assert len(input_shape) == 3

        dim = self.dim

        # permute to put the required dimension in the 2nd dimension
        if dim == 1:
            x = input
        elif dim == 2:
            x = input.permute(0, 2, 1)

        # apply batch_norm
        num_features = pytorch_utils.get_shape(x)[1]
        assert num_features == self.num_features
        x = self.layer(x)

        # permute back to the original view
        if dim == 2:
            x = x.permute(0, 2, 1)

        x_shape = pytorch_utils.get_shape(x)
        assert input_shape == x_shape

        return x

class BatchNorm2d(nn.Module):

    def __init__(self, num_features, dim=1):
        super(BatchNorm2d, self).__init__()

        assert dim in [1, 2, 3]

        self.dim = dim
        self.num_features = num_features
        self.layer = nn.BatchNorm3d(num_features)

    def forward(self, input):

        input_shape = pytorch_utils.get_shape(input)
        assert len(input_shape) == 4

        dim = self.dim

        # permute to put the required dimension in the 2nd dimension
        if dim == 1:
            x = input
        elif dim == 2:
            x = input.permute(0, 2, 3, 1)
        elif dim == 3:
            x = input.permute(0, 1, 3, 2)

        # apply batch_norm
        num_features = pytorch_utils.get_shape(x)[1]
        assert num_features == self.num_features
        x = self.layer(x)

        # permute back to the original view
        if dim == 2:
            x = x.permute(0, 2, 3, 1)
        elif dim == 3:
            x = x.permute(0, 1, 3, 2)

        x_shape = pytorch_utils.get_shape(x)
        assert input_shape == x_shape

        return x

class BatchNorm3d(nn.Module):

    def __init__(self, num_features, dim=1):
        super(BatchNorm3d, self).__init__()

        assert dim in [1, 2, 3, 4]

        self.dim = dim
        self.num_features = num_features
        self.layer = nn.BatchNorm3d(num_features)

    def forward(self, input):

        input_shape = pytorch_utils.get_shape(input)
        assert len(input_shape) == 5

        dim = self.dim

        # permute to put the required dimension in the 2nd dimension
        if dim == 1:
            x = input
        elif dim == 2:
            x = input.permute(0, 2, 1, 3, 4)
        elif dim == 3:
            x = input.permute(0, 3, 2, 1, 4)
        elif dim == 4:
            x = input.permute(0, 4, 2, 3, 1)

        # apply batch_norm
        num_features = pytorch_utils.get_shape(x)[1]
        assert num_features == self.num_features
        x = self.layer(x)

        # permute back to the original view
        if dim == 2:
            x = x.permute(0, 2, 1, 3, 4)
        elif dim == 3:
            x = x.permute(0, 3, 2, 1, 4)
        elif dim == 4:
            x = x.permute(0, 4, 2, 3, 1)

        x_shape = pytorch_utils.get_shape(x)
        assert input_shape == x_shape

        return x

# endregion

# region LayerNorm with Axis

class LayerNorm3d(nn.Module):

    def __init__(self, num_features, dim=1):
        super(LayerNorm3d, self).__init__()

        assert dim in [1, 2, 3, 4]

        self.dim = dim
        self.num_features = num_features
        self.layer = nn.LayerNorm(num_features)

    def forward(self, input):

        input_shape = pytorch_utils.get_shape(input)
        assert len(input_shape) == 5

        dim = self.dim

        # permute to put the required dimension in the 2nd dimension
        if dim == 4:
            x = input
        elif dim == 3:
            x = input.permute(0, 1, 2, 4, 3)
        elif dim == 2:
            x = input.permute(0, 1, 4, 3, 2)
        elif dim == 1:
            x = input.permute(0, 4, 2, 3, 1)
        else:
            x = None

        B, d1, d2, d3, d4 = pytorch_utils.get_shape(x)
        assert d4 == self.num_features

        # reshape
        x = x.view(B, d1 * d2 * d3, d4)

        # apply layer_norm
        x = self.layer(x)

        # reshape back to the original view
        x = x.view(B, d1, d2, d3, d4)

        # permute back to the original view
        if dim == 3:
            x = x.permute(0, 1, 2, 4, 3)
        elif dim == 2:
            x = x.permute(0, 1, 4, 3, 2)
        elif dim == 1:
            x = x.permute(0, 4, 2, 3, 1)

        x_shape = pytorch_utils.get_shape(x)
        assert input_shape == x_shape

        return x

# endregion

# region Linear 1D, 2D, 3D

class Linear1d(nn.Module):

    def __init__(self, in_features, out_features, dim=1):
        super(Linear1d, self).__init__()

        message = 'Sorry, unsupported dimension for linear layer: %d' % (dim)
        assert dim in [1, 2], message

        # weather to permute the dimesions or not
        self.is_permute = dim in [2]

        # permutation according to the dim
        if dim == 2:
            permutation_in = (0, 2, 1)
            permutation_out = (0, 2, 1)
        else:
            permutation_in = None
            permutation_out = None

        self.permutation_in = permutation_in
        self.permutation_out = permutation_out
        kernel_size = 1

        self.layer = nn.Conv1d(in_features, out_features, kernel_size)

    def forward(self, x_in):

        # in permutation
        x_out = x_in.permute(self.permutation_in) if self.is_permute else x_in

        # linear layer
        x_out = self.layer(x_out)

        # out permutation
        x_out = x_out.permute(self.permutation_out) if self.is_permute else x_out

        return x_out

class Linear2d(nn.Module):

    def __init__(self, in_features, out_features, dim=1, **kwargs):
        super(Linear2d, self).__init__()

        message = 'Sorry, unsupported dimension for linear layer: %d' % (dim)
        assert dim in [1, 2, 3], message

        # weather to permute the dimesions or not
        self.is_permute = dim in [2, 3]

        # permutation according to the dim
        if dim == 2:
            permutation_in = (0, 2, 1, 3)
            permutation_out = (0, 2, 1, 3)
        elif dim == 3:
            permutation_in = (0, 3, 2, 1)
            permutation_out = (0, 3, 2, 1)
        else:
            permutation_in = None
            permutation_out = None

        self.permutation_in = permutation_in
        self.permutation_out = permutation_out
        kernel_size = (1, 1)

        self.layer = nn.Conv2d(in_features, out_features, kernel_size, **kwargs)

    def forward(self, x_in):

        # in permutation
        x_out = x_in.permute(self.permutation_in) if self.is_permute else x_in

        # linear layer
        x_out = self.layer(x_out)

        # out permutation
        x_out = x_out.permute(self.permutation_out) if self.is_permute else x_out

        return x_out

class Linear3d(nn.Module):

    def __init__(self, in_features, out_features, dim=1, **kwargs):
        super(Linear3d, self).__init__()

        message = 'Sorry, unsupported dimension for linear layer: %d' % (dim)
        assert dim in [1, 2, 3, 4], message

        # weather to permute the dimesions or not
        self.is_permute = dim in [2, 3, 4]

        # permutation according to the dim
        if dim == 2:
            permutation_in = (0, 2, 1, 3, 4)
            permutation_out = (0, 2, 1, 3, 4)
        elif dim == 3:
            permutation_in = (0, 3, 2, 1, 4)
            permutation_out = (0, 3, 2, 1, 4)
        elif dim == 4:
            permutation_in = (0, 4, 2, 3, 1)
            permutation_out = (0, 4, 2, 3, 1)
        else:
            permutation_in = None
            permutation_out = None

        self.permutation_in = permutation_in
        self.permutation_out = permutation_out
        kernel_size = (1, 1, 1)

        self.layer = nn.Conv3d(in_features, out_features, kernel_size, **kwargs)

    def forward(self, x_in):

        # in permutation
        x_out = x_in.permute(self.permutation_in) if self.is_permute else x_in

        # linear layer
        x_out = self.layer(x_out)

        # out permutation
        x_out = x_out.permute(self.permutation_out) if self.is_permute else x_out

        return x_out

# endregion

# region Conv Same Padding

class Pad1d(nn.Module):

    def __init__(self, kernel_size, T, stride=1, dilation=1):
        super(Pad1d, self).__init__()
        F = kernel_size
        S = stride
        D = dilation

        T = math.ceil(T / S)
        Pt = (T - 1) * S + (F - 1) * D + 1 - T
        pad_list = (Pt // 2, Pt - Pt // 2)
        self.pad = nn.ConstantPad1d(pad_list, 0.0)

    def forward(self, x_in):
        x_out = self.pad(x_in)
        return x_out

class Pad2d(nn.Module):

    def __init__(self, kernel_size, H, W, stride=1, dilation=1):
        super(Pad2d, self).__init__()
        F = kernel_size
        S = stride
        D = dilation

        H2 = math.ceil(H / S)
        W2 = math.ceil(W / S)
        Pr = (H2 - 1) * S + (F[0] - 1) * D + 1 - H
        Pc = (W2 - 1) * S + (F[1] - 1) * D + 1 - W
        pad_list = (Pr // 2, Pr - Pr // 2, Pc // 2, Pc - Pc // 2)
        self.pad = nn.ConstantPad2d(pad_list, 0.0)

    def forward(self, x_in):
        x_out = self.pad(x_in)
        return x_out

class Pad3d(nn.Module):

    def __init__(self, kernel_size, T, H, W, stride=1, dilation=1):
        super(Pad3d, self).__init__()
        F = kernel_size
        S = stride
        D = dilation

        T2 = math.ceil(T / S)
        H2 = math.ceil(H / S)
        W2 = math.ceil(W / S)

        Pt = (T2 - 1) * S + (F[0] - 1) * D + 1 - T
        Pr = (H2 - 1) * S + (F[1] - 1) * D + 1 - H
        Pc = (W2 - 1) * S + (F[2] - 1) * D + 1 - W

        # padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back
        pad_list = (Pr // 2, Pr - Pr // 2, Pc // 2, Pc - Pc // 2, Pt // 2, Pt - Pt // 2)
        self.pad = nn.ConstantPad3d(pad_list, 0.0)

    def forward(self, x_in):
        x_out = self.pad(x_in)
        return x_out

class Conv1dPaded(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, T, stride=1, dilation=1, groups=1):
        super(Conv1dPaded, self).__init__()

        self.pad = Pad1d(kernel_size, T, stride, dilation)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation=dilation, groups=groups)

    def forward(self, x_in):
        x_out = self.pad(x_in)
        x_out = self.conv(x_out)
        return x_out

class Conv2dPaded(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, H, W, stride=1, dilation=1, groups=1):
        super(Conv2dPaded, self).__init__()

        self.pad = Pad2d(kernel_size, H, W, stride, dilation)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation=dilation, groups=groups)

    def forward(self, x_in):
        x_out = self.pad(x_in)
        x_out = self.conv(x_out)
        return x_out

class Conv3dPaded(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, T, H, W, stride=1, dilation=1, groups=1):
        super(Conv3dPaded, self).__init__()

        self.pad = Pad3d(kernel_size, T, H, W, stride, dilation)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, dilation=dilation, groups=groups)

    def forward(self, x_in):
        x_out = self.pad(x_in)
        x_out = self.conv(x_out)
        return x_out

# endregion

# region Straight-Through Estimator

class Threshold(nn.Module):
    def __init__(self, value):
        super(Threshold, self).__init__()

        self.__value = value

    def forward(self, input):
        idx = input < self.__value
        y_hard = input.clone()

        # zero below threshold
        y_hard[idx] = 0.0

        # Set gradients w.r.t. y_hard to gradients w.r.t. input
        y_hard = (y_hard - input).detach() + input

        return y_hard

class ThresholdModified(nn.Module):
    def __init__(self, value):
        super(ThresholdModified, self).__init__()

        self.__value = value

    def forward(self, input):
        idx = input < self.__value

        # zero below threshold
        y_hard = torch.ones_like(input)
        y_hard[idx] = 0.0
        y_hard = y_hard.detach()

        y_hard = y_hard * input

        return y_hard

class HardMax(nn.Module):
    def __init__(self):
        super(HardMax, self).__init__()

    def forward(self, x):
        x_shape = pytorch_utils.get_shape(x)  # (None, 2)
        assert len(x_shape) == 2
        assert x_shape[1] == 2

        # x_hard as zero list
        x_hard = torch.zeros_like(x)

        # find index of max value
        _, idx = torch.max(x, dim=1, keepdim=True)

        # set max value to one
        x_hard.scatter_(1, idx, 1)

        # ser gradients to be w.r.t x instead of being w.r.t x_hard
        y = (x_hard - x).detach() + x

        return y

# endregion

# region Conv Misc

class DepthwiseConv1d(nn.Module):
    def __init__(self, n_channels, kernel_size, T, stride=1):
        super(DepthwiseConv1d, self).__init__()

        self.stride = stride
        self.kernel_size = kernel_size
        self.n_channels = n_channels

        self.depthwise_conv = nn.Conv1d(self.n_channels, self.n_channels, self.kernel_size, groups=self.n_channels)
        self.padding = Pad1d(kernel_size, T)

    def forward(self, input):
        # input is of shape (None, C, T, H, W)

        input_shape = pytorch_utils.get_shape(input)
        n, c, t, h, w = input_shape

        assert len(input_shape) == 5

        # transpose and reshape to hide the spatial dimension, only expose the temporal dimension for depthwise conv
        tensor = input.permute(0, 3, 4, 1, 2)  # (None, H, W, C, T)
        tensor = tensor.contiguous().view(n * h * w, c, t)  # (None*H*W, C, T)

        # depthwise conv on the temporal dimension
        tensor = self.padding(tensor)
        tensor = self.depthwise_conv(tensor)  # (None*H*W, C, T)

        # reshape to get the spatial dimensions
        tensor = tensor.view(n, h, w, c, t)  # (None, H, W, C, T)

        # finally, transpose to get the desired output shape
        tensor = tensor.permute(0, 3, 4, 1, 2)  # (None, C, T, H, W)

        return tensor

# endregion

# region Model Layers: Gating

class GumbelSigmoidOld(nn.Module):
    def __init__(self, temperature=0.67):
        super(GumbelSigmoidOld, self).__init__()

        self.temperature = temperature

        self.gumbel_sampler = distributions.gumbel.Gumbel(0, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # input is of shape (None, H, W, N, T)

        input_shape = pytorch_utils.get_shape(input)
        b, h, w, n, t = input_shape

        assert len(input_shape) == 5

        # reshape
        tensor = input.permute(0, 1, 2, 4, 3)  # (None, H, W, T. N)
        tensor = tensor.contiguous().view(b * h * w * t, n)  # (None*H*W*T, N)

        # sample gumbel noise
        gumbel_shape = tensor.size()
        gumbel_noise = self.gumbel_sampler.sample(gumbel_shape).cuda()
        # gumbel_noise = self.sample_gumbel(gumbel_shape)

        # gumbel sigmoid trick
        tensor = (tensor + gumbel_noise) / self.temperature
        tensor = self.sigmoid(tensor)

        # get original size and permutation
        tensor = tensor.view(b, h, w, t, n)  # (None, H, W, T, N)
        tensor = tensor.permute(0, 1, 2, 4, 3)  # (None, H, W, N, T)

        return tensor

    def sample_gumbel(self, shape, eps=1e-10):
        """Sample from Gumbel(0, 1)"""
        noise = torch.rand(shape)
        noise.add_(eps).log_().neg_()
        noise.add_(eps).log_().neg_()
        noise = noise.cuda()

        return noise

class GumbelSigmoid(nn.Module):
    def __init__(self, temperature=0.67):
        super(GumbelSigmoid, self).__init__()

        self.temperature = temperature

        self.gumbel_sampler = distributions.gumbel.Gumbel(0, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # input is of shape (None, H, W, N, T)

        input_shape = pytorch_utils.get_shape(input)
        b, h, w, n, t = input_shape

        assert len(input_shape) == 5

        # sample gumbel noise
        gumbel_shape = input.size()
        gumbel_noise = self.gumbel_sampler.sample(gumbel_shape).cuda()

        # gumbel sigmoid trick
        tensor = (input + gumbel_noise) / self.temperature
        tensor = self.sigmoid(tensor)

        return tensor

class GumbelNoise(nn.Module):
    def __init__(self, loc=0, scale=1):
        super(GumbelNoise, self).__init__()
        self.gumbel_sampler = distributions.gumbel.Gumbel(loc, scale=scale)

    def forward(self, input):
        # sample gumbel noise
        gumbel_shape = input.size()
        gumbel_noise = self.gumbel_sampler.sample(gumbel_shape).cuda()

        # add gumbel noise
        tensor = input + gumbel_noise

        return tensor

class GumbelNoiseSampler(nn.Module):
    def __init__(self, loc=0, scale=1.0):
        super(GumbelNoiseSampler, self).__init__()

        self.gumbel_sampler = distributions.gumbel.Gumbel(loc, scale)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # sample gumbel noise
        gumbel_shape = input.size()
        gumbel_noise = self.gumbel_sampler.sample(gumbel_shape).cuda()

        return gumbel_noise

class LZeroNorm(nn.Module):

    def __init__(self, loc_mean=0, loc_sdev=0.01, beta=2 / 3.0, gamma=-0.1, zeta=1.1, fix_temp=True):
        """
        L0 Norm regularization.
        :param loc_mean: mean of the normal distribution which generates initial location parameters
        :param loc_sdev: standard deviation of the normal distribution which generates initial location parameters
        :param beta: initial temperature parameter
        :param gamma: lower bound of "stretched" s
        :param zeta: upper bound of "stretched" s
        :param fix_temp: True if temperature is fixed
        """

        super(LZeroNorm, self).__init__()

        self.penalty = None
        self.gamma = gamma
        self.zeta = zeta
        self.loc_mean = loc_mean
        self.loc_sdev = loc_sdev
        self.gamma_zeta_ratio = math.log(-gamma / zeta)

        # either fixed or learnable temperature
        self.temp = beta if fix_temp else nn.Parameter(torch.zeros(1).fill_(beta).cuda())

    def forward(self, input):

        size = input.size()

        # location is sampled from mean and std
        loc = torch.zeros(size).normal_(self.loc_mean, self.loc_sdev)
        loc = nn.Parameter(loc.cuda())

        if self.training:

            # u is sampled from uniform
            u = torch.zeros(size).uniform_()
            u = Variable(u).cuda()

            s = F.sigmoid((torch.log(u) - torch.log(1 - u) + loc) / self.temp)
            s = s * (self.zeta - self.gamma) + self.gamma
            penalty = F.sigmoid(loc - self.temp * self.gamma_zeta_ratio).sum()
        else:
            s = F.sigmoid(loc) * (self.zeta - self.gamma) + self.gamma
            penalty = 0

        # clipping values by zero and one
        gates = self.__hard_sigmoid(s)

        self.gate_values = gates.tolist()

        # multiply input times the gates
        output = input * gates

        # don't forget to register the pentalty
        self.penalty = penalty

        return output

    def __hard_sigmoid(self, x):
        """
        Returns either zeros or ones.
        :param x:
        :return:
        """

        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)

        y = torch.min(torch.max(x, zeros), ones)

        return y

class GaussianNoise(nn.Module):
    def __init__(self, loc=0, scale=1):
        super(GaussianNoise, self).__init__()
        self.sampler = distributions.normal.Normal(loc=loc, scale=scale)

    def forward(self, input):
        # sample gumbel noise
        input_shape = input.size()
        noise = self.sampler.sample(input_shape).cuda()

        # add gumbel noise
        tensor = (input + noise)

        return tensor

# endregion
