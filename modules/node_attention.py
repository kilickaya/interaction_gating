#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""

####################
# Node Attention
####################
We have a set of pre-defined nodes. They are on the level of the dataset.
So, here we represent each timestep as a weighted sum of all pre-defined notes.
This is vlad style. It helps to better represent the timestep, and also in convolving it over time.

"""

import math
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

# region Local Node Attention [Sum Heads, Global Keys]

class LocalNodeAttentionHeadSumV1(nn.Module):
    """
    Original Implementation on the Self-Attention Head.
    """

    def __init__(self, n_channels_in, n_channels_inter, nodes):
        """
        Initialize the module.
        """
        super(LocalNodeAttentionHeadSumV1, self).__init__()

        self.n_channels_in = n_channels_in
        self.n_channels_inter = n_channels_inter
        self.nodes = nodes
        dropout_ratio = 0.25

        query_linear = pl.Linear3d(self.n_channels_in, self.n_channels_inter)
        key_linear = nn.Linear(self.n_channels_inter, self.n_channels_inter)
        value_linear = pl.Linear3d(self.n_channels_in, self.n_channels_inter)
        output_linear = pl.Linear3d(self.n_channels_inter, self.n_channels_in)

        # query embedding
        self.query_embedding = nn.Sequential(nn.Dropout(dropout_ratio), query_linear)

        # key embedding
        self.key_embedding = nn.Sequential(nn.Dropout(dropout_ratio), key_linear)

        # value embedding
        self.value_embedding = nn.Sequential(nn.Dropout(dropout_ratio), value_linear)

        # output embedding
        self.output_embedding = nn.Sequential(nn.Dropout(dropout_ratio), output_linear)

    def forward(self, x_window):
        """
        :param x: (B, C, T, H, W)
        :return:
        """

        nodes = self.nodes
        B, C, T, H, W = pytorch_utils.get_shape(x_window)
        batch_size = x_window.size(0)
        assert T % 2 == 1

        # get middle item of the window
        idx_item = int(T / 2.0)
        x_item = x_window[:, :, idx_item:idx_item + 1]  # (B, C, 1, H, W)

        # query embedding
        query = self.query_embedding(x_item)  # (B, C, 1, H, W)

        # key embedding
        key = self.key_embedding(nodes)  # (T, C)

        # value embedding
        value = self.value_embedding(x_window)  # (B, C, T, H, W)

        # attention
        alpha = self.__tensor_product_1(key, query)  # (B, 1, T, H, W)

        # scale and softmax
        alpha = F.softmax(alpha, dim=2)  # (B, 1, T, H, W)

        # scale and sigmoid
        # alpha = alpha / np.sqrt(self.n_channels_inter)  # (B, 1*H*W, T*H*W)
        # alpha = alpha / alpha.size(-1)  # (B, 1*H*W, T*H*W)
        # alpha = F.sigmoid(alpha)  # (B, 1*H*W, T*H*W)

        # multiply alpha with values
        y = self.__tensor_product_2(alpha, value)  # (B, C, 1, H, W)

        # output embedding
        y = self.output_embedding(y)

        return y

    def __tensor_product_1(self, key, query):
        """
        :param key: # (T, C)
        :param query: # (B, C, 1, H, W)
        :return: # (B, 1, T, H, W)
        """

        B, C, _, H, W = pytorch_utils.get_shape(query)
        T, _ = pytorch_utils.get_shape(key)

        query = query.permute(0, 2, 3, 4, 1)  # (B, 1, H, W, C)
        query = query.view(B * 1 * H * W, C)  # (B*1*H*W, C)
        key = key.permute(1, 0)

        alpha = torch.matmul(query, key)  # (B*1*H*W. T)
        alpha = alpha.view(B, 1, H, W, T)  # (B, 1, H, W. T)
        alpha = alpha.permute(0, 1, 4, 2, 3)  # (B, 1, T, H, W)

        return alpha

    def __tensor_product_2(self, alpha, value):
        """
        :param alpha: # (B, 1, T, H, W)
        :param value: # (B, C, T, H, W)
        :return: # (B, C, 1, H, W)
        """
        B, C, T, H, W = pytorch_utils.get_shape(value)

        alpha = alpha.permute(0, 1, 3, 4, 2)  # (B, 1, H, W, T)
        alpha = alpha.view(B, 1 * H * W, T)  # (B, 1*H*W, T)

        value = value.permute(0, 1, 3, 4, 2)  # (B, C, H, W, T)
        value = value.view(B, C * H * W, T)  # (B, C*H*W, T)
        value = value.permute(0, 2, 1)  # (B, T, C*H*W)

        y = torch.matmul(alpha, value)  # (B, 1*H*W, C*H*W)
        y = y.view(B, C, 1, H, W)
        return y

class LocalNodeAttentionMultiHeadSumV1(nn.Module):
    """
    MultiHead for Self-Attention.
    """

    def __init__(self, input_shape, window_size, n_heads):
        """
        Initialize the module.
        """
        super(LocalNodeAttentionMultiHeadSumV1, self).__init__()

        C, T, H, W = input_shape
        n_channels_in = C

        assert n_channels_in % n_heads == 0

        self.n_channels_in = n_channels_in
        self.n_heads = n_heads
        self.window_size = window_size

        n_channels_inter = int(n_channels_in / n_heads)

        # padding
        self.padding = pl.Pad3d((window_size, 1, 1), T, H, W)

        # we use n heads, each has inner dim
        for idx_head in range(n_heads):
            head_num = idx_head + 1
            attention_head_name = 'attention_head_%d' % (head_num)
            nodes = self.__generate_nodes(window_size, n_channels_inter)
            attention_head = LocalNodeAttentionHeadSumV1(n_channels_in, n_channels_inter, nodes)
            setattr(self, attention_head_name, attention_head)

    def forward(self, x):
        """
        :param x: (B, C, T, H, W)
        :return:
        """

        K = self.window_size

        # padd the input
        x_padded = self.padding(x)

        # get how many local windows or slices (S)
        B, C, T, H, W = pytorch_utils.get_shape(x_padded)
        S = T - K + 1
        N = self.n_heads

        tensors = []

        # loop on windows, and get them
        for idx_slice in range(S):
            idx_start = idx_slice
            idx_stop = idx_start + K

            # slice to get the window
            x_window = x_padded[:, :, idx_start:idx_stop]
            tensors.append(x_window)

        # now that you get the windows, stack them into a new dimension
        y = torch.stack(tensors, dim=1)  # (B, S, C, T, H, W)

        # reshape to hide the slices inside the batch dimension
        y = y.view(B * S, C, K, H, W)  # (B*S, C, T, H, W)

        z = []

        # feed to to local-attentions block, multi-heads
        for idx_head in range(N):
            head_num = idx_head + 1
            attention_head_name = 'attention_head_%d' % (head_num)
            attention_head = getattr(self, attention_head_name)
            z_head = attention_head(y)  # (B*S, C, T, H, W)
            z.append(z_head)

        # sum over the head dimension
        z = torch.stack(z, dim=1)  # (B*S, N, C, 1, H, W)
        z = torch.mean(z, dim=1)  # (B*S, C, 1, H, W)
        z = torch.squeeze(z, 2)  # (B*S, C, H, W)

        # reshape to get back slices
        z = z.view(B, S, C, H, W)  # (B*S, C, H, W)

        # permute to put slices in the temporal dimension
        z = z.permute(0, 2, 1, 3, 4)

        # residual
        z += x

        return z

    def __generate_nodes(self, n_nodes, feature_dim):

        nodes = utils.generate_centroids(n_nodes, feature_dim, is_sobol=False)
        nodes = torch.from_numpy(nodes).cuda()
        return nodes

# endregion

# region Local Node Attention [Sum Heads, Global Values]

class LocalNodeAttentionHeadSumV2(nn.Module):
    """
    Original Implementation on the Self-Attention Head.
    """

    def __init__(self, n_channels_in, n_channels_inter, nodes):
        """
        Initialize the module.
        """
        super(LocalNodeAttentionHeadSumV2, self).__init__()

        self.n_channels_in = n_channels_in
        self.n_channels_inter = n_channels_inter
        self.nodes = nodes
        dropout_ratio = 0.25

        query_linear = pl.Linear3d(self.n_channels_in, self.n_channels_inter)
        key_linear = pl.Linear3d(self.n_channels_in, self.n_channels_inter)
        value_linear = nn.Linear(self.n_channels_inter, self.n_channels_inter)
        output_linear = pl.Linear3d(self.n_channels_inter, self.n_channels_in)

        # query embedding
        self.query_embedding = nn.Sequential(nn.Dropout(dropout_ratio), query_linear)

        # key embedding
        self.key_embedding = nn.Sequential(nn.Dropout(dropout_ratio), key_linear)

        # value embedding
        self.value_embedding = nn.Sequential(nn.Dropout(dropout_ratio), value_linear)

        # output embedding
        self.output_embedding = nn.Sequential(nn.Dropout(dropout_ratio), output_linear)

    def forward(self, x_window):
        """
        :param x: (B, C, T, H, W)
        :return:
        """

        nodes = self.nodes
        B, C, T, H, W = pytorch_utils.get_shape(x_window)
        batch_size = x_window.size(0)
        assert T % 2 == 1

        # get middle item of the window
        idx_item = int(T / 2.0)
        x_item = x_window[:, :, idx_item:idx_item + 1]  # (B, C, 1, H, W)

        # query embedding
        query = self.query_embedding(x_item)  # (B, C, 1, H, W)

        # key embedding
        key = self.key_embedding(x_window)  # (B, C, T, H, W)

        # value embedding
        value = self.value_embedding(nodes)  # (T, C)

        # attention
        alpha = self.__tensor_product_1(key, query)  # (B, T, H, W)

        # scale and softmax
        alpha = F.softmax(alpha, dim=1)  # (B, T, H, W)

        # scale and sigmoid
        # alpha = alpha / np.sqrt(self.n_channels_inter)  # (B, 1*H*W, T*H*W)
        # alpha = alpha / alpha.size(-1)  # (B, 1*H*W, T*H*W)
        # alpha = F.sigmoid(alpha)  # (B, 1*H*W, T*H*W)

        y = self.__tensor_product_2(alpha, value)  # (B, C, 1, H, W)

        # output embedding
        y = self.output_embedding(y)

        return y

    def __tensor_product_1(self, key, query):
        """
        :param key: # (B, C, T, H, W)
        :param query: # (B, C, 1, H, W)
        :return: # (B, T, H, W)
        """

        B, C, T, H, W = pytorch_utils.get_shape(key)

        query = query.view(B, C, -1)  # (B, C, 1*H*W)

        key = key.view(B, C, -1)  # (B, C, T*H*W)
        key = key.permute(0, 2, 1)  # (B, T*H*W, C)

        # attention
        alpha = torch.matmul(key, query)  # (B, T*H*W, 1*H*W)
        alpha = alpha.view(B, T, H, W)  # (B, T, H, W)

        return alpha

    def __tensor_product_2(self, alpha, value):
        """
        :param alpha: #  (B, T, H, W)
        :param value: # (T, C)
        :return: # (B, C, H, W)
        """

        B, _, H, W = pytorch_utils.get_shape(alpha)
        T, C = pytorch_utils.get_shape(value)

        alpha = alpha.permute(0, 2, 3, 1)  # (B, H, W, T)
        alpha = alpha.view(B * H * W, T)  # (B*H*W, T)

        alpha = torch.matmul(alpha, value)  # (B*H*W. C)
        alpha = alpha.view(B, C, 1, H, W)  # (B, C, H, W)

        return alpha

class LocalNodeAttentionMultiHeadSumV2(nn.Module):
    """
    MultiHead for Self-Attention.
    """

    def __init__(self, input_shape, window_size, n_heads):
        """
        Initialize the module.
        """
        super(LocalNodeAttentionMultiHeadSumV2, self).__init__()

        C, T, H, W = input_shape
        n_channels_in = C

        assert n_channels_in % n_heads == 0

        self.n_channels_in = n_channels_in
        self.n_heads = n_heads
        self.window_size = window_size

        n_channels_inter = int(n_channels_in / n_heads)

        # padding
        self.padding = pl.Pad3d((window_size, 1, 1), T, H, W)

        # we use n heads, each has inner dim
        for idx_head in range(n_heads):
            head_num = idx_head + 1
            attention_head_name = 'attention_head_%d' % (head_num)
            nodes = self.__generate_nodes(window_size, n_channels_inter)
            attention_head = LocalNodeAttentionHeadSumV2(n_channels_in, n_channels_inter, nodes)
            setattr(self, attention_head_name, attention_head)

    def forward(self, x):
        """
        :param x: (B, C, T, H, W)
        :return:
        """

        K = self.window_size

        # padd the input
        x_padded = self.padding(x)

        # get how many local windows or slices (S)
        B, C, T, H, W = pytorch_utils.get_shape(x_padded)
        S = T - K + 1
        N = self.n_heads

        tensors = []

        # loop on windows, and get them
        for idx_slice in range(S):
            idx_start = idx_slice
            idx_stop = idx_start + K

            # slice to get the window
            x_window = x_padded[:, :, idx_start:idx_stop]
            tensors.append(x_window)

        # now that you get the windows, stack them into a new dimension
        y = torch.stack(tensors, dim=1)  # (B, S, C, T, H, W)

        # reshape to hide the slices inside the batch dimension
        y = y.view(B * S, C, K, H, W)  # (B*S, C, T, H, W)

        z = []

        # feed to to local-attentions block, multi-heads
        for idx_head in range(N):
            head_num = idx_head + 1
            attention_head_name = 'attention_head_%d' % (head_num)
            attention_head = getattr(self, attention_head_name)
            z_head = attention_head(y)  # (B*S, C, T, H, W)
            z.append(z_head)

        # sum over the head dimension
        z = torch.stack(z, dim=1)  # (B*S, N, C, 1, H, W)
        z = torch.mean(z, dim=1)  # (B*S, C, 1, H, W)
        z = torch.squeeze(z, 2)  # (B*S, C, H, W)

        # reshape to get back slices
        z = z.view(B, S, C, H, W)  # (B*S, C, H, W)

        # permute to put slices in the temporal dimension
        z = z.permute(0, 2, 1, 3, 4)

        # residual
        z += x

        return z

    def __generate_nodes(self, n_nodes, feature_dim):

        nodes = utils.generate_centroids(n_nodes, feature_dim, is_sobol=False)
        nodes = torch.from_numpy(nodes).cuda()
        return nodes

# endregion

# region Local Node Attention [Concat Heads, Global Keys]

class LocalNodeAttentionHeadConcatV1(nn.Module):
    """
    Original Implementation on the Self-Attention Head.
    """

    def __init__(self, n_channels_in, n_channels_inter, nodes):
        """
        Initialize the module.
        """
        super(LocalNodeAttentionHeadConcatV1, self).__init__()

        self.n_channels_in = n_channels_in
        self.n_channels_inter = n_channels_inter
        self.nodes = nodes
        dropout_ratio = 0.25

        query_linear = pl.Linear3d(self.n_channels_in, self.n_channels_inter)
        key_linear = nn.Linear(self.n_channels_inter, self.n_channels_inter)
        value_linear = pl.Linear3d(self.n_channels_in, self.n_channels_inter)

        # query embedding
        self.query_embedding = nn.Sequential(nn.Dropout(dropout_ratio), query_linear)

        # key embedding
        self.key_embedding = nn.Sequential(nn.Dropout(dropout_ratio), key_linear)

        # value embedding
        self.value_embedding = nn.Sequential(nn.Dropout(dropout_ratio), value_linear)

    def forward(self, x_window):
        """
        :param x: (B, C, T, H, W)
        :return:
        """

        nodes = self.nodes
        B, C, T, H, W = pytorch_utils.get_shape(x_window)
        batch_size = x_window.size(0)
        assert T % 2 == 1

        # get middle item of the window
        idx_item = int(T / 2.0)
        x_item = x_window[:, :, idx_item:idx_item + 1]  # (B, C, 1, H, W)

        # query embedding
        query = self.query_embedding(x_item)  # (B, C, 1, H, W)

        # key embedding
        key = self.key_embedding(nodes)  # (T, C)

        # value embedding
        value = self.value_embedding(x_window)  # (B, C, T, H, W)

        # attention
        alpha = self.__tensor_product_1(key, query)  # (B, 1, T, H, W)

        # scale and softmax
        alpha = F.softmax(alpha, dim=2)  # (B, 1, T, H, W)

        # scale and sigmoid
        # alpha = alpha / np.sqrt(self.n_channels_inter)  # (B, 1*H*W, T*H*W)
        # alpha = alpha / alpha.size(-1)  # (B, 1*H*W, T*H*W)
        # alpha = F.sigmoid(alpha)  # (B, 1*H*W, T*H*W)

        y = self.__tensor_product_2(alpha, value)  # (B, C, H, W)

        return y

    def __tensor_product_1(self, key, query):
        """
        :param key: # (T, C)
        :param query: # (B, C, 1, H, W)
        :return: # (B, 1, T, H, W)
        """

        B, C, _, H, W = pytorch_utils.get_shape(query)
        T, _ = pytorch_utils.get_shape(key)

        query = query.permute(0, 2, 3, 4, 1)  # (B, 1, H, W, C)
        query = query.view(B * 1 * H * W, C)  # (B*1*H*W, C)
        key = key.permute(1, 0)

        alpha = torch.matmul(query, key)  # (B*1*H*W. T)
        alpha = alpha.view(B, 1, H, W, T)  # (B, 1, H, W. T)
        alpha = alpha.permute(0, 1, 4, 2, 3)  # (B, 1, T, H, W)

        return alpha

    def __tensor_product_2(self, alpha, value):
        """
        :param alpha: # (B, 1, T, H, W)
        :param value: # (B, C, T, H, W)
        :return: # (B, C, 1, H, W)
        """
        B, C, T, H, W = pytorch_utils.get_shape(value)

        alpha = alpha.permute(0, 1, 3, 4, 2)  # (B, 1, H, W, T)
        alpha = alpha.view(B, 1 * H * W, T)  # (B, 1*H*W, T)

        value = value.permute(0, 1, 3, 4, 2)  # (B, C, H, W, T)
        value = value.view(B, C * H * W, T)  # (B, C*H*W, T)
        value = value.permute(0, 2, 1)  # (B, T, C*H*W)

        y = torch.matmul(alpha, value)  # (B, 1*H*W, C*H*W)
        y = y.view(B, C, 1, H, W)
        return y

class LocalNodeAttentionMultiHeadConcatV1(nn.Module):
    """
    MultiHead for Self-Attention.
    """

    def __init__(self, input_shape, window_size, n_heads):
        """
        Initialize the module.
        """
        super(LocalNodeAttentionMultiHeadConcatV1, self).__init__()

        C, T, H, W = input_shape
        n_channels_in = C

        assert n_channels_in % n_heads == 0

        self.n_channels_in = n_channels_in
        self.n_heads = n_heads
        self.window_size = window_size

        n_channels_inter = int(n_channels_in / n_heads)

        # padding
        self.padding = pl.Pad3d((window_size, 1, 1), T, H, W)

        # we use n heads, each has inner dim
        for idx_head in range(n_heads):
            head_num = idx_head + 1
            attention_head_name = 'attention_head_%d' % (head_num)
            nodes = self.__generate_nodes(window_size, n_channels_inter)
            attention_head = LocalNodeAttentionHeadConcatV1(n_channels_in, n_channels_inter, nodes)
            setattr(self, attention_head_name, attention_head)

    def forward(self, x):
        """
        :param x: (B, C, T, H, W)
        :return:
        """

        K = self.window_size

        # padd the input
        x_padded = self.padding(x)

        # get how many local windows or slices (S)
        B, C, T, H, W = pytorch_utils.get_shape(x_padded)
        S = T - K + 1
        N = self.n_heads

        tensors = []

        # loop on windows, and get them
        for idx_slice in range(S):
            idx_start = idx_slice
            idx_stop = idx_start + K

            # slice to get the window
            x_window = x_padded[:, :, idx_start:idx_stop]
            tensors.append(x_window)

        # now that you get the windows, stack them into a new dimension
        y = torch.stack(tensors, dim=1)  # (B, S, C, T, H, W)

        # reshape to hide the slices inside the batch dimension
        y = y.view(B * S, C, K, H, W)  # (B*S, C, T, H, W)

        z = []

        # feed to to local-attentions block, multi-heads
        for idx_head in range(N):
            head_num = idx_head + 1
            attention_head_name = 'attention_head_%d' % (head_num)
            attention_head = getattr(self, attention_head_name)
            z_head = attention_head(y)  # (B*S, C, T, H, W)
            z.append(z_head)

        # concat
        z = torch.cat(z, dim=1)  # (B*S, C, H, W)

        # reshape to get back slices
        z = z.view(B, S, C, H, W)  # (B*S, C, H, W)

        # permute to put slices in the temporal dimension
        z = z.permute(0, 2, 1, 3, 4)

        # residual
        z += x

        return z

    def __generate_nodes(self, n_nodes, feature_dim):

        nodes = utils.generate_centroids(n_nodes, feature_dim, is_sobol=False)
        nodes = torch.from_numpy(nodes).cuda()
        return nodes

# endregion

# region Local Node Attention [Concat Heads, Global Values]

class LocalNodeAttentionHeadConcatV2(nn.Module):
    """
    Original Implementation on the Self-Attention Head.
    """

    def __init__(self, n_channels_in, n_channels_inter, nodes):
        """
        Initialize the module.
        """
        super(LocalNodeAttentionHeadConcatV2, self).__init__()

        self.n_channels_in = n_channels_in
        self.n_channels_inter = n_channels_inter
        self.nodes = nodes
        dropout_ratio = 0.25

        query_linear = pl.Linear3d(self.n_channels_in, self.n_channels_inter)
        key_linear = pl.Linear3d(self.n_channels_in, self.n_channels_inter)
        value_linear = nn.Linear(self.n_channels_inter, self.n_channels_inter)

        # query embedding
        self.query_embedding = nn.Sequential(nn.Dropout(dropout_ratio), query_linear)

        # key embedding
        self.key_embedding = nn.Sequential(nn.Dropout(dropout_ratio), key_linear)

        # value embedding
        self.value_embedding = nn.Sequential(nn.Dropout(dropout_ratio), value_linear)

    def forward(self, x_window):
        """
        :param x: (B, C, T, H, W)
        :return:
        """

        nodes = self.nodes
        B, C, T, H, W = pytorch_utils.get_shape(x_window)
        batch_size = x_window.size(0)
        assert T % 2 == 1

        # get middle item of the window
        idx_item = int(T / 2.0)
        x_item = x_window[:, :, idx_item:idx_item + 1]  # (B, C, 1, H, W)

        # query embedding
        query = self.query_embedding(x_item)  # (B, C, 1, H, W)

        # key embedding
        key = self.key_embedding(x_window)  # (B, C, T, H, W)

        # value embedding
        value = self.value_embedding(nodes)  # (T, C)

        # attention
        alpha = self.__tensor_product_1(key, query)  # (B, T, H, W)

        # scale and softmax
        alpha = F.softmax(alpha, dim=1)  # (B, T, H, W)

        # scale and sigmoid
        # alpha = alpha / np.sqrt(self.n_channels_inter)  # (B, 1*H*W, T*H*W)
        # alpha = alpha / alpha.size(-1)  # (B, 1*H*W, T*H*W)
        # alpha = F.sigmoid(alpha)  # (B, 1*H*W, T*H*W)

        y = self.__tensor_product_2(alpha, value)  # (B, C, H, W)

        return y

    def __tensor_product_1(self, key, query):
        """
        :param key: # (B, C, T, H, W)
        :param query: # (B, C, 1, H, W)
        :return: # (B, T, H, W)
        """

        B, C, T, H, W = pytorch_utils.get_shape(key)

        query = query.view(B, C, -1)  # (B, C, 1*H*W)

        key = key.view(B, C, -1)  # (B, C, T*H*W)
        key = key.permute(0, 2, 1)  # (B, T*H*W, C)

        # attention
        alpha = torch.matmul(key, query)  # (B, T*H*W, 1*H*W)
        alpha = alpha.view(B, T, H, W)  # (B, T, H, W)

        return alpha

    def __tensor_product_2(self, alpha, value):
        """
        :param alpha: #  (B, T, H, W)
        :param value: # (T, C)
        :return: # (B, C, H, W)
        """

        B, _, H, W = pytorch_utils.get_shape(alpha)
        T, C = pytorch_utils.get_shape(value)

        alpha = alpha.permute(0, 2, 3, 1)  # (B, H, W, T)
        alpha = alpha.view(B * H * W, T)  # (B*H*W, T)

        alpha = torch.matmul(alpha, value)  # (B*H*W. C)
        alpha = alpha.view(B, C, 1, 1)  # (B, C, H, W)

        return alpha

class LocalNodeAttentionMultiHeadConcatV2(nn.Module):
    """
    MultiHead for Self-Attention.
    """

    def __init__(self, input_shape, window_size, n_heads):
        """
        Initialize the module.
        """
        super(LocalNodeAttentionMultiHeadConcatV2, self).__init__()

        C, T, H, W = input_shape
        n_channels_in = C

        assert n_channels_in % n_heads == 0

        self.n_channels_in = n_channels_in
        self.n_heads = n_heads
        self.window_size = window_size

        n_channels_inter = int(n_channels_in / n_heads)

        # padding
        self.padding = pl.Pad3d((window_size, 1, 1), T, H, W)

        # we use n heads, each has inner dim
        for idx_head in range(n_heads):
            head_num = idx_head + 1
            attention_head_name = 'attention_head_%d' % (head_num)
            nodes = self.__generate_nodes(window_size, n_channels_inter)
            attention_head = LocalNodeAttentionHeadConcatV2(n_channels_in, n_channels_inter, nodes)
            setattr(self, attention_head_name, attention_head)

    def forward(self, x):
        """
        :param x: (B, C, T, H, W)
        :return:
        """

        K = self.window_size

        # padd the input
        x_padded = self.padding(x)

        # get how many local windows or slices (S)
        B, C, T, H, W = pytorch_utils.get_shape(x_padded)
        S = T - K + 1
        N = self.n_heads

        tensors = []

        # loop on windows, and get them
        for idx_slice in range(S):
            idx_start = idx_slice
            idx_stop = idx_start + K

            # slice to get the window
            x_window = x_padded[:, :, idx_start:idx_stop]
            tensors.append(x_window)

        # now that you get the windows, stack them into a new dimension
        y = torch.stack(tensors, dim=1)  # (B, S, C, T, H, W)

        # reshape to hide the slices inside the batch dimension
        y = y.view(B * S, C, K, H, W)  # (B*S, C, T, H, W)

        z = []

        # feed to to local-attentions block, multi-heads
        for idx_head in range(N):
            head_num = idx_head + 1
            attention_head_name = 'attention_head_%d' % (head_num)
            attention_head = getattr(self, attention_head_name)
            z_head = attention_head(y)  # (B*S, C, T, H, W)
            z.append(z_head)

        # concat
        z = torch.cat(z, dim=1)  # (B*S, C, H, W)

        # reshape to get back slices
        z = z.view(B, S, C, H, W)  # (B*S, C, H, W)

        # permute to put slices in the temporal dimension
        z = z.permute(0, 2, 1, 3, 4)

        # residual
        z += x

        return z

    def __generate_nodes(self, n_nodes, feature_dim):

        nodes = utils.generate_centroids(n_nodes, feature_dim, is_sobol=False)
        nodes = torch.from_numpy(nodes).cuda()
        return nodes

# endregion

# region Local Node Attention (Old)

class LocalNodeAttentionHead(nn.Module):
    """
    Original Implementation on the Self-Attention Block
    """

    def __init__(self, nodes, n_channels_in, n_channels_inter):
        """
        Initialize the module.
        """
        super(LocalNodeAttentionHead, self).__init__()

        self.n_channels_in = n_channels_in
        self.n_channels_inter = n_channels_inter
        dropout_ratio = 0.25

        query_linear = pl.Linear3d(self.n_channels_in, self.n_channels_inter)
        key_linear = pl.Linear3d(self.n_channels_in, self.n_channels_inter)
        value_linear = nn.Linear(self.n_channels_inter, self.n_channels_inter)
        output_linear = pl.Linear3d(self.n_channels_inter, self.n_channels_in)

        # query embedding
        self.query_embedding = nn.Sequential(nn.Dropout(dropout_ratio), query_linear)

        # key embedding
        self.key_embedding = nn.Sequential(nn.Dropout(dropout_ratio), key_linear)

        # value embedding
        self.value_embedding = nn.Sequential(nn.Dropout(dropout_ratio), value_linear)

        # output embedding
        self.output_embedding = nn.Sequential(nn.Dropout(dropout_ratio), output_linear)

        # special weight initialization
        nn.init.constant_(output_linear.layer.weight, 0)
        nn.init.constant_(output_linear.layer.bias, 0)

    def forward(self, x_window):
        """
        :param x: (B, C, T, H, W)
        :return:
        """

        B, C, T, H, W = pytorch_utils.get_shape(x_window)
        batch_size = x_window.size(0)
        assert T % 2 == 1

        # get middle item of the window
        idx_item = int(T / 2.0)
        x_item = x_window[:, :, idx_item:idx_item + 1]  # (B, C, 1, H, W)

        # query embedding
        query = self.query_embedding(x_window)  # (B, C, T, H, W)

        # key embedding
        key = self.key_embedding(x_window)  # (B, N)

        # value embedding
        value = self.value_embedding(x_window)  # (B, C, T, H, W)

        # attention
        alpha = torch.matmul(key, query)  # (B, T*H*W, 1*H*W)
        alpha = alpha.permute(0, 2, 1)  # (B, 1*H*W, Y*H*W)
        alpha = F.softmax(alpha, dim=-1)  # (B, 1*H*W, T*H*W)

        # multiply alpha with values
        y = torch.matmul(alpha, value)  # (B, 1*H*W, C)
        y = y.permute(0, 2, 1).contiguous()  # (B, C, 1*H*W)
        y = y.view(batch_size, self.n_channels_inter, 1, H, W)  # (B, C, 1, H, W)

        # output embedding
        y = self.output_embedding(y)

        # residual
        y += x_item

        return y

class LocalNodeAttentionMultiHead(nn.Module):
    def __init__(self, input_shape, window_size, n_heads, n_nodes):
        """
        Initialize the module.
        """
        super(LocalNodeAttentionMultiHead, self).__init__()

        C, T, H, W = input_shape
        n_channels_in = C

        assert n_channels_in % n_heads == 0

        self.n_channels_in = n_channels_in
        self.n_heads = n_heads
        self.window_size = window_size

        n_channels_inter = int(n_channels_in / n_heads)
        nodes = utils.generate_centroids(n_nodes, n_channels_inter)

        # convert to tensor and move to gpu
        nodes = torch.from_numpy(nodes).cuda()

        # we use n heads, each has inner dim
        for idx_head in range(n_heads):
            head_num = idx_head + 1
            attention_head_name = 'attention_head_%d' % (head_num)
            attention_head = LocalNodeAttentionHead(nodes, n_channels_in, n_channels_inter)
            setattr(self, attention_head_name, attention_head)

    def forward(self, x):
        """
        :param x: (B, C, T, H, W)
        :return:
        """

        K = self.window_size

        # get how many local windows or slices (S)
        B, C, T, H, W = pytorch_utils.get_shape(x)
        S = T - K + 1

        tensors = []

        # loop on windows, and get them
        for idx_window in range(S):
            idx_start = idx_window
            idx_stop = idx_start + K

            # slice to get the window
            x_window = x[:, :, idx_start:idx_stop]
            tensors.append(x_window)

        # now that you get the windows, stack them into a new dimension
        y = torch.stack(tensors, dim=1)  # (B, S, C, T, H, W)

        # reshape to hide the slices inside the batch dimension
        y = y.view(B * S, C, K, H, W)  # (B*S, C, T, H, W)

        z = []

        # feed to to local-attentions block, multi-heads
        for idx_head in range(self.n_heads):
            head_num = idx_head + 1
            attention_head_name = 'attention_head_%d' % (head_num)
            attention_head = getattr(self, attention_head_name)
            z_head = attention_head(y)  # (B*S, C, T, H, W)
            z.append(z_head)

        # sum over the channel dimension
        z = torch.stack(z, dim=1)  # (B*S, 1, C, 1, H, W)
        z = torch.sum(z, dim=1)  # (B*S, C, 1, H, W)
        z = torch.squeeze(z, 2)  # (B*S, C, H, W)

        # reshape to get back slices
        z = z.view(B, S, C, H, W)  # (B*S, C, H, W)

        # permute to put slices in the temporal dimension
        z = z.permute(0, 2, 1, 3, 4)

        return z

# endregion

# region Node Attention Multi-Head: Original Implementation

class NodeAttentionHead(nn.Module):
    def __init__(self, nodes, n_channels_in, n_channels_inter):
        """
        Initialize the module.
        """
        super(NodeAttentionHead, self).__init__()

        self.nodes = nodes
        self.n_channels_in = n_channels_in
        self.n_channels_inter = n_channels_inter
        self.is_softmax_activation = True
        self.dropout_ratio = 0.25

        self.__define_layers()
        self.__init_layers()

    def __define_layers(self):
        dropout_ratio = self.dropout_ratio

        query_linear = nn.Linear(self.n_channels_inter, self.n_channels_inter)
        value_linear = nn.Linear(self.n_channels_inter, self.n_channels_inter)
        key_linear = pl.Linear3d(self.n_channels_in, self.n_channels_inter)
        output_linear = pl.Linear3d(self.n_channels_inter, self.n_channels_in)

        # query embedding
        self.query_embedding = nn.Sequential(nn.Dropout(dropout_ratio), query_linear)

        # key embedding
        self.key_embedding = nn.Sequential(nn.Dropout(dropout_ratio), key_linear)

        # value embedding
        self.value_embedding = nn.Sequential(nn.Dropout(dropout_ratio), value_linear)

        # output embedding
        self.output_embedding = nn.Sequential(nn.Dropout(dropout_ratio), output_linear)

        # for alpha
        self.alpha_bn = pl.BatchNorm3d(self.n_channels_inter, dim=3)

    def __init_layers(self):
        # special weight initialization
        nn.init.constant_(self.output_embedding[0].layer.weight, 0)
        nn.init.constant_(self.output_embedding[0].layer.bias, 0)

    def forward(self, x):
        """
        :param x: (B, C, T, H, W)
        :return:
        """

        nodes = self.nodes  # (N, C)

        batch_size = x.size(0)
        B, C, T, H, W = pytorch_utils.get_shape(x)

        # key embedding
        key = self.key_embedding(x)  # (B, C, T, H, W)

        # query embedding
        query = self.query_embedding(nodes)  # (N, C)

        # value embedding
        value = self.value_embedding(nodes)  # (N, C)

        # attention
        alpha = self.__tensor_product_1(key, query)  # (B, C, T, H, W) * # (N, C) => (B, H, W, N, T)

        # use softmax or sigmoid
        if self.is_softmax_activation:
            alpha = F.softmax(alpha, dim=3)  # (B, H, W, N, T)
        else:
            alpha = alpha / alpha.size(-1)  # (B, H, W, N, T)
            alpha = F.sigmoid(alpha)  # (B, H, W, N, T)

        # multiply alpha with values
        y = self.__tensor_product_2(alpha, value)  # (None, H, W, N, T) * (N, C) => (None, C, T, H, W)

        # output embedding
        y = self.output_embedding(y)

        # residual connection
        y += x

        return y

    def __tensor_product_1(self, x, nodes):
        """
        Takes two input tensors and does matrix multiplication across channel dimension (c).
        :param x: # (None, C, T, H, W)
        :param nodes: # (N, C)
        :return: f # (None, H, W, N, T)
        """

        n, c, t, h, w = pytorch_utils.get_shape(x)
        n_nodes, node_dim = pytorch_utils.get_shape(nodes)

        assert node_dim == c

        # reshape phi
        x = x.permute(0, 2, 3, 4, 1)  # (None, T, H, W, C)
        x = x.contiguous().view(n * t * h * w, c)  # (None*T*H*W, C)

        # transpose for matrix multiplication
        nodes = nodes.permute(1, 0)  # (C, N)

        f = torch.matmul(x, nodes)  # (None*T*H*W, C) x (C, N) = (None*T*H*W, N)
        f = f.view(n, t, h, w, n_nodes)  # (None, T, H ,W, N)
        f = f.permute(0, 2, 3, 4, 1)  # (None, H, W, N, T)

        return f

    def __tensor_product_2(self, alpha, nodes):
        """
        Takes two input tensors and does matrix multiplication across node dimension (n).
        :param alpha: (None, H, W, N, T)
        :param nodes:  (N, C)
        :return: y # (None, C, T, H, W)
        """

        n, h, w, n_c, t = pytorch_utils.get_shape(alpha)
        n_nodes, node_dim = pytorch_utils.get_shape(nodes)

        assert n_nodes == n_c

        # reshape f
        alpha = alpha.permute(0, 1, 2, 4, 3)  # (None, H, W, T, N)
        alpha = alpha.contiguous().view(n * h * w * t, n_nodes)  # (None*H*W*T, N)

        y = torch.matmul(alpha, nodes)  # (None*H*W*T, C)
        y = y.view(n, h, w, t, node_dim)  # (None, H, W, T, C)
        y = y.permute(0, 4, 3, 1, 2)  # (None, C, T, H, W)

        return y

class NodeAttentionMultiHead(nn.Module):
    def __init__(self, input_shape, nodes, n_heads):
        """
        Initialize the module.
        """
        super(NodeAttentionMultiHead, self).__init__()

        C, T, H, W = input_shape
        n_channels_in = C
        assert n_channels_in % n_heads == 0

        self.n_channels_in = n_channels_in
        self.n_heads = n_heads

        # node feature dimension is the same as inner dimension
        n_channels_inter = int(n_channels_in / n_heads)
        nodes = nodes[:, :n_channels_inter]

        # convert to tensor and move to gpu
        nodes = torch.from_numpy(nodes).cuda()

        # we use n heads, each has inner dim
        for idx_head in range(n_heads):
            head_num = idx_head + 1
            attention_head_name = 'attention_head_%d' % (head_num)
            attention_head = NodeAttentionHead(nodes, n_channels_in, n_channels_inter)
            setattr(self, attention_head_name, attention_head)

    def forward(self, x):
        """
        :param x: (B, C, T, H, W)
        :return:
        """

        y = []

        # feed to to local-attentions block, multi-heads
        for idx_head in range(self.n_heads):
            head_num = idx_head + 1
            attention_head_name = 'attention_head_%d' % (head_num)
            attention_head = getattr(self, attention_head_name)
            y_head = attention_head(x)  # (B, C, T, H, W)
            y.append(y_head)

        # sum over the channel dimension
        y = torch.stack(y, dim=1)  # (B, 1, C, T, H, W)
        y = torch.sum(y, dim=1)  # (B, C, T, H, W)

        return y

# endregion

# region Node Attention Multi-Head: My Implementation

class NodeAttentionPoolHead(nn.Module):
    def __init__(self, n_channels_in, n_channels_inter, nodes):
        super(NodeAttentionPoolHead, self).__init__()

        self.n_channels = n_channels_in
        self.nodes = nodes

        n_nodes, node_dim = pytorch_utils.get_shape(nodes)
        self.n_nodes = n_nodes
        self.node_dim = node_dim

        self.x_bn = nn.BatchNorm3d(n_channels_in)

        self.nodes_bn_1 = nn.BatchNorm1d(n_channels_inter)
        self.nodes_linear_1 = nn.Linear(n_channels_inter, n_channels_in)
        self.nodes_linear_2 = nn.Linear(n_channels_inter, n_channels_in)
        self.nodes_bn_2 = pl.BatchNorm(n_nodes, dim=0)
        self.nodes_bn_3 = nn.BatchNorm1d(n_channels_in)

        self.alpha_linear = pl.Linear3d(n_nodes, n_nodes, dim=3)
        self.alpha_bn = pl.BatchNorm3d(n_nodes, dim=3)

    def forward(self, x):
        # input is of shape (None, C, T, H, W)

        input_shape = pytorch_utils.get_shape(x)
        n, c, t, h, w = input_shape
        nodes = self.nodes

        assert len(input_shape) == 5

        # x
        x = self.x_bn(x)  # (None, C, T, H, W)

        # node embedding
        nodes = self.nodes_bn_1(nodes)  # (N, C)
        nodes = self.nodes_linear_1(nodes)  # (N, C)
        nodes = self.nodes_bn_2(nodes)  # (N, C)

        # alpha
        alpha = self.__tensor_product_1(x, nodes)  # (None, H, W, N, T)
        alpha = self.alpha_bn(alpha)

        # attend to node using alpha
        nodes = self.nodes_linear_2(self.nodes)
        nodes = self.nodes_bn_3(nodes)  # (N, C)
        y = self.__tensor_product_2(alpha, nodes)  # (None, C, T, H, W)

        return y

    def __tensor_product_1(self, x, nodes):
        """
        Takes two input tensors and does matrix multiplication across channel dimension (c).
        :param x: # (None, C, T, H, W)
        :param nodes: # (N, C)
        :return: f # (None, H, W, N, T)
        """

        n, c, t, h, w = pytorch_utils.get_shape(x)
        n_nodes, node_dim = pytorch_utils.get_shape(nodes)

        assert node_dim == c

        # reshape phi
        x = x.permute(0, 2, 3, 4, 1)  # (None, T, H, W, C)
        x = x.contiguous().view(n * t * h * w, c)  # (None*T*H*W, C)

        # transpose for matrix multiplication
        nodes = nodes.permute(1, 0)  # (C, N)

        alpha = torch.matmul(x, nodes)  # (None*T*H*W, C) x (C, N) = (None*T*H*W, N)
        alpha = alpha.view(n, t, h, w, n_nodes)  # (None, T, H ,W, N)
        alpha = alpha.permute(0, 2, 3, 4, 1)  # (None, H, W, N, T)

        return alpha

    def __tensor_product_2(self, alpha, nodes):
        """
        Takes two input tensors and does matrix multiplication across node dimension (n).
        :param alpha: (None, H, W, N, T)
        :param nodes:  (N, C)
        :return: y # (None, C, T, H, W)
        """

        n, h, w, n_c, t = pytorch_utils.get_shape(alpha)
        n_nodes, node_dim = pytorch_utils.get_shape(nodes)

        assert n_nodes == n_c

        # reshape f
        alpha = alpha.permute(0, 1, 2, 4, 3)  # (None, H, W, T, N)
        alpha = alpha.contiguous().view(n * h * w * t, n_nodes)  # (None*H*W*T, N)

        y = torch.matmul(alpha, nodes)  # (None*H*W*T, C)
        y = y.view(n, h, w, t, node_dim)  # (None, H, W, T, C)
        y = y.permute(0, 4, 3, 1, 2)  # (None, C, T, H, W)

        return y

class NodeAttentionPoolMultiHead(nn.Module):
    def __init__(self, input_shape, nodes, n_heads):
        """
        Initialize the module.
        """
        super(NodeAttentionPoolMultiHead, self).__init__()

        C, T, H, W = input_shape
        n_channels_in = C
        n_channels_inter = int(n_channels_in / n_heads)
        n_nodes = pytorch_utils.get_shape(nodes)[0]

        nodes = utils.generate_centroids(n_nodes, n_channels_inter, is_sobol=False)
        nodes = torch.from_numpy(nodes).cuda()

        assert n_channels_in % n_heads == 0

        self.n_channels_in = n_channels_in
        self.n_heads = n_heads

        # we use n heads, each has inner dim
        for idx_head in range(n_heads):
            head_num = idx_head + 1
            attention_head_name = 'attention_head_%d' % (head_num)
            attention_head = NodeAttentionPoolHead(n_channels_in, n_channels_inter, nodes)
            setattr(self, attention_head_name, attention_head)

    def forward(self, x):
        """
        :param x: (B, C, T, H, W)
        :return:
        """

        y = []

        # feed to to local-attentions block, multi-heads
        for idx_head in range(self.n_heads):
            head_num = idx_head + 1
            attention_head_name = 'attention_head_%d' % (head_num)
            attention_head = getattr(self, attention_head_name)
            y_head = attention_head(x)  # (B, C, T, H, W)
            y.append(y_head)

        # sum over the channel dimension
        y = torch.stack(y, dim=1)  # (B, 1, C, T, H, W)
        y = torch.sum(y, dim=1)  # (B, C, T, H, W)

        return y

# endregion

# region Node Attention

class NodeAttentionPool(nn.Module):
    def __init__(self, n_channels, n_nodes, nodes):
        super(NodeAttentionPool, self).__init__()

        self.n_channels = n_channels
        self.n_nodes = n_nodes
        self.nodes = nodes

        self.query_embedding = nn.BatchNorm3d(n_channels)
        self.key_embedding = nn.Sequential(nn.BatchNorm1d(n_channels), nn.Linear(n_channels, n_channels), nn.LayerNorm(n_channels))
        self.value_embedding = nn.BatchNorm1d(n_channels)

        self.alpha_bn = pl.BatchNorm3d(n_nodes, dim=3)

        # TODO: nn.init.kaiming_normal_(self.fc1_0st_gate_net.weight)

    def forward(self, x):
        # input is of shape (None, C, T, H, W)

        input_shape = pytorch_utils.get_shape(x)
        n, c, t, h, w = input_shape

        assert len(input_shape) == 5

        # query is the input features
        query = self.query_embedding(x)  # (None, C, T, H, W)

        # keys are embedding of nodes
        key = self.key_embedding(self.nodes)  # (N, C)

        # value is batch_norm of nodes
        value = self.value_embedding(self.nodes)  # (N, C)

        # alpha is the similarity between query and key
        alpha = self.__tensor_product_1(query, key)  # (None, H, W, N, T)
        alpha = self.alpha_bn(alpha)

        # weighted sum over nodes
        y = self.__tensor_product_2(alpha, value)  # (None, C, T, H, W)

        return y

    def __tensor_product_1(self, query, key):
        """
        Takes two input tensors and does matrix multiplication across channel dimension (c).
        :param query: # (None, C, T, H, W)
        :param key: # (N, C)
        :return: f # (None, H, W, N, T)
        """

        n, c, t, h, w = pytorch_utils.get_shape(query)
        n_nodes, node_dim = pytorch_utils.get_shape(key)

        assert node_dim == c

        # reshape phi
        query = query.permute(0, 2, 3, 4, 1)  # (None, T, H, W, C)
        query = query.contiguous().view(n * t * h * w, c)  # (None*T*H*W, C)

        # transpose for matrix multiplication
        key = key.permute(1, 0)  # (C, N)

        f = torch.matmul(query, key)  # (None*T*H*W, C) x (C, N) = (None*T*H*W, N)
        f = f.view(n, t, h, w, n_nodes)  # (None, T, H ,W, N)
        f = f.permute(0, 2, 3, 4, 1)  # (None, H, W, N, T)

        return f

    def __tensor_product_2(self, f, g):
        """
        Takes two input tensors and does matrix multiplication across node dimension (n).
        :param f: (None, H, W, N, T)
        :param g:  (N, C)
        :return: y # (None, C, T, H, W)
        """

        n, h, w, n_c, t = pytorch_utils.get_shape(f)
        n_nodes, node_dim = pytorch_utils.get_shape(g)

        assert n_nodes == n_c

        # reshape f
        f = f.permute(0, 1, 2, 4, 3)  # (None, H, W, T, N)
        f = f.contiguous().view(n * h * w * t, n_nodes)  # (None*H*W*T, N)

        y = torch.matmul(f, g)  # (None*H*W*T, C)
        y = y.view(n, h, w, t, node_dim)  # (None, H, W, T, C)
        y = y.permute(0, 4, 3, 1, 2)  # (None, C, T, H, W)

        return y

class NodeAttentionHardmax(nn.Module):
    def __init__(self, n_channels, n_nodes, nodes):
        super(NodeAttentionHardmax, self).__init__()

        self.n_channels = n_channels
        self.n_nodes = n_nodes
        self.nodes = nodes

        self.bn_1 = nn.BatchNorm3d(n_channels)
        self.bn_2 = nn.BatchNorm1d(n_channels)
        self.bn_3 = pl.BatchNorm(n_nodes, dim=0)
        self.bn_4 = pl.BatchNorm3d(n_nodes, dim=3)
        self.bn_5 = nn.BatchNorm1d(n_channels)
        self.bn_6 = nn.BatchNorm3d(n_channels)

        self.linear_1 = nn.Linear(n_channels, n_channels)
        self.conv3d = nn.Conv3d(n_nodes, n_nodes, (1, 1, 1))

        self.relu_1 = nn.LeakyReLU(negative_slope=0.2)
        self.softmax_1 = nn.Softmax(dim=3)
        self.hardmax_1 = pf.Hardmax.apply

    def forward(self, input):
        # input is of shape (None, C, T, H, W)

        input_shape = pytorch_utils.get_shape(input)
        n, c, t, h, w = input_shape

        assert len(input_shape) == 5

        # phi path (Q) or (x)
        x = self.bn_1(input)
        phi = x  # (None, C, T, H, W)

        # theta path (K) or (c)
        theta = self.bn_2(self.nodes)  # (N, C)
        theta = self.linear_1(theta)  # (N, C)
        theta = self.bn_3(theta)  # (N, C)

        # f is the similarity between (theta and phi) or (Q and K)
        f = self.__tensor_product_1(phi, theta)  # (None, H, W, N, T)
        # batchnorm across the node dimension
        f = self.bn_4(f)  # (None, H, W, N, T)
        # softmax to select only one node
        f = self.softmax_1(f)  # (None, H, W, N, T)
        # now, hardmax to select only one node for each timestep
        f = self.hardmax_1(f)  # (None, H, W, N, T)

        # g path (V)
        g = self.bn_5(self.nodes)  # (N, C)

        y = self.__tensor_product_2(f, g)  # (None, C, T, H, W)
        y = self.bn_6(y)
        y = self.relu_1(y)

        return y

    def __tensor_product_1(self, phi, theta):
        """
        Takes two input tensors and does matrix multiplication across channel dimension (c).
        :param phi: # (None, C, T, H, W)
        :param theta: # (N, C)
        :return: f # (None, H, W, N, T)
        """

        n, c, t, h, w = pytorch_utils.get_shape(phi)
        n_nodes, node_dim = pytorch_utils.get_shape(theta)

        assert node_dim == c

        # reshape phi
        phi = phi.permute(0, 2, 3, 4, 1)  # (None, T, H, W, C)
        phi = phi.contiguous().view(n * t * h * w, c)  # (None*T*H*W, C)

        # transpose for matrix multiplication
        theta = theta.permute(1, 0)  # (C, N)

        f = torch.matmul(phi, theta)  # (None*T*H*W, C) x (C, N) = (None*T*H*W, N)
        f = f.view(n, t, h, w, n_nodes)  # (None, T, H ,W, N)
        f = f.permute(0, 2, 3, 4, 1)  # (None, H, W, N, T)

        return f

    def __tensor_product_2(self, f, g):
        """
        Takes two input tensors and does matrix multiplication across node dimension (n).
        :param f: (None, H, W, N, T)
        :param g:  (N, C)
        :return: y # (None, C, T, H, W)
        """

        n, h, w, n_c, t = pytorch_utils.get_shape(f)
        n_nodes, node_dim = pytorch_utils.get_shape(g)

        assert n_nodes == n_c

        # reshape f
        f = f.permute(0, 1, 2, 4, 3)  # (None, H, W, T, N)
        f = f.contiguous().view(n * h * w * t, n_nodes)  # (None*H*W*T, N)

        y = torch.matmul(f, g)  # (None*H*W*T, C)
        y = y.view(n, h, w, t, node_dim)  # (None, H, W, T, C)
        y = y.permute(0, 4, 3, 1, 2)  # (None, C, T, H, W)

        return y

class NodeAttentionHardSigmoid(nn.Module):
    def __init__(self, n_channels, n_nodes, nodes):
        super(NodeAttentionHardSigmoid, self).__init__()

        self.n_channels = n_channels
        self.n_nodes = n_nodes
        self.nodes = nodes

        self.bn_1 = nn.BatchNorm3d(n_channels)
        self.bn_2 = nn.BatchNorm1d(n_channels)
        self.bn_3 = pl.BatchNorm(n_nodes, dim=0)
        self.bn_4 = pl.BatchNorm3d(n_nodes, dim=3)
        self.bn_5 = nn.BatchNorm1d(n_channels)
        self.bn_6 = nn.BatchNorm3d(n_channels)

        self.linear_1 = nn.Linear(n_channels, n_channels)
        self.conv3d = nn.Conv3d(n_nodes, n_nodes, (1, 1, 1))

        self.sigmoid_1 = nn.Sigmoid()
        self.threshold = pf.Threshold.apply

        self.relu_1 = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input):
        # input is of shape (None, C, T, H, W)

        input_shape = pytorch_utils.get_shape(input)
        n, c, t, h, w = input_shape

        assert len(input_shape) == 5

        # phi path (Q) or (x)
        x = self.bn_1(input)
        phi = x  # (None, C, T, H, W)

        # theta path (K) or (c)
        theta = self.bn_2(self.nodes)  # (N, C)
        theta = self.linear_1(theta)  # (N, C)
        theta = self.bn_3(theta)  # (N, C)

        # f is the similarity between (theta and phi) or (Q and K)
        f = self.__tensor_product_1(phi, theta)  # (None, H, W, N, T)

        # batchnorm across the node dimension
        f = self.bn_4(f)  # (None, H, W, N, T)

        # sigmoid + threshold to select few nodes for teach timestep
        f = self.sigmoid_1(f)  # (None, H, W, N, T)
        f = self.threshold(f)  # (None, H, W, N, T)

        # g path (V)
        g = self.bn_5(self.nodes)  # (N, C)

        y = self.__tensor_product_2(f, g)  # (None, C, T, H, W)
        y = self.bn_6(y)
        y = self.relu_1(y)

        return y

    def __tensor_product_1(self, phi, theta):
        """
        Takes two input tensors and does matrix multiplication across channel dimension (c).
        :param phi: # (None, C, T, H, W)
        :param theta: # (N, C)
        :return: f # (None, H, W, N, T)
        """

        n, c, t, h, w = pytorch_utils.get_shape(phi)
        n_nodes, node_dim = pytorch_utils.get_shape(theta)

        assert node_dim == c

        # reshape phi
        phi = phi.permute(0, 2, 3, 4, 1)  # (None, T, H, W, C)
        phi = phi.contiguous().view(n * t * h * w, c)  # (None*T*H*W, C)

        # transpose for matrix multiplication
        theta = theta.permute(1, 0)  # (C, N)

        f = torch.matmul(phi, theta)  # (None*T*H*W, C) x (C, N) = (None*T*H*W, N)
        f = f.view(n, t, h, w, n_nodes)  # (None, T, H ,W, N)
        f = f.permute(0, 2, 3, 4, 1)  # (None, H, W, N, T)

        return f

    def __tensor_product_2(self, f, g):
        """
        Takes two input tensors and does matrix multiplication across node dimension (n).
        :param f: (None, H, W, N, T)
        :param g:  (N, C)
        :return: y # (None, C, T, H, W)
        """

        n, h, w, n_c, t = pytorch_utils.get_shape(f)
        n_nodes, node_dim = pytorch_utils.get_shape(g)

        assert n_nodes == n_c

        # reshape f
        f = f.permute(0, 1, 2, 4, 3)  # (None, H, W, T, N)
        f = f.contiguous().view(n * h * w * t, n_nodes)  # (None*H*W*T, N)

        y = torch.matmul(f, g)  # (None*H*W*T, C)
        y = y.view(n, h, w, t, node_dim)  # (None, H, W, T, C)
        y = y.permute(0, 4, 3, 1, 2)  # (None, C, T, H, W)

        return y

class NodeAttentionHardSigmoidSimple(nn.Module):
    def __init__(self, n_channels, n_nodes, n_timesteps, nodes):
        super(NodeAttentionHardSigmoidSimple, self).__init__()

        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.nodes = nodes

        self.bn_1 = nn.BatchNorm3d(n_channels)

        self.bn_2 = nn.BatchNorm1d(n_channels)
        self.bn_3 = pl.BatchNorm(n_nodes, dim=0)

        self.bn_4 = pl.BatchNorm3d(n_nodes, dim=3)
        self.bn_6 = nn.BatchNorm3d(n_channels)

        self.linear_2 = nn.Linear(n_channels, n_channels)
        self.conv3d_2 = nn.Conv3d(n_nodes, n_nodes, (1, 1, 1))

        self.relu_1 = nn.LeakyReLU(negative_slope=0.2)

        self.sigmoid_1 = nn.Sigmoid()
        self.threshold = pf.Threshold.apply

    def forward(self, input):
        # input is of shape (None, C, T, H, W)

        nodes = self.nodes
        x = input

        input_shape = pytorch_utils.get_shape(x)
        n, c, t, h, w = input_shape

        assert len(input_shape) == 5

        # features
        x = self.bn_1(x)  # (None, C, T, H, W)

        # nodes
        nodes = self.bn_2(nodes)  # (N, C)
        nodes = self.linear_2(nodes)  # (N, C)
        nodes = self.bn_3(nodes)  # (N, C)

        # f is the similarity between features and nodes
        f = self.__tensor_product_1(x, nodes)  # (None, C, T, H, W) * (N, C) => (None, H, W, N, T)

        # sigmoid + threshold to select few nodes for teach timestep
        alpha = self.sigmoid_1(f)
        alpha = self.threshold(alpha)

        # sum over selected nodes
        y = self.__tensor_product_2(alpha, nodes)  # (None, H, W, N, T) * # (N, C) => (None, C, T, H, W)
        y = self.bn_6(y)
        y = self.relu_1(y)

        return y

    def __tensor_product_1(self, phi, theta):
        """
        Takes two input tensors and does matrix multiplication across channel dimension (c).
        :param phi: # (None, C, T, H, W)
        :param theta: # (N, C)
        :return: f # (None, H, W, N, T)
        """

        n, c, t, h, w = pytorch_utils.get_shape(phi)
        n_nodes, node_dim = pytorch_utils.get_shape(theta)

        assert node_dim == c

        # reshape phi
        phi = phi.permute(0, 2, 3, 4, 1)  # (None, T, H, W, C)
        phi = phi.contiguous().view(n * t * h * w, c)  # (None*T*H*W, C)

        # transpose for matrix multiplication
        theta = theta.permute(1, 0)  # (C, N)

        f = torch.matmul(phi, theta)  # (None*T*H*W, C) x (C, N) = (None*T*H*W, N)
        f = f.view(n, t, h, w, n_nodes)  # (None, T, H ,W, N)
        f = f.permute(0, 2, 3, 4, 1)  # (None, H, W, N, T)

        return f

    def __tensor_product_2(self, f, g):
        """
        Takes two input tensors and does matrix multiplication across node dimension (n).
        :param f: (None, H, W, N, T)
        :param g:  (N, C)
        :return: y # (None, C, T, H, W)
        """

        n, h, w, n_c, t = pytorch_utils.get_shape(f)
        n_nodes, node_dim = pytorch_utils.get_shape(g)

        assert n_nodes == n_c

        # reshape f
        f = f.permute(0, 1, 2, 4, 3)  # (None, H, W, T, N)
        f = f.contiguous().view(n * h * w * t, n_nodes)  # (None*H*W*T, N)

        y = torch.matmul(f, g)  # (None*H*W*T, C)
        y = y.view(n, h, w, t, node_dim)  # (None, H, W, T, C)
        y = y.permute(0, 4, 3, 1, 2)  # (None, C, T, H, W)

        return y

class NodeAttentionVariational(nn.Module):
    def __init__(self, n_channels):
        super(NodeAttentionVariational, self).__init__()

        n_nodes = 128
        nodes_dim = 1024

        self.nodes_dim = nodes_dim
        self.nodes = torch.from_numpy(sobol.sobol_generate(nodes_dim, n_nodes)).cuda()

        self.n_nodes = n_nodes
        self.n_channels = n_channels

        # for learning nodes based on mean and std
        self.nodes_bn_1 = nn.BatchNorm1d(nodes_dim)
        self.nodes_mean_linear = nn.Linear(nodes_dim, nodes_dim)
        self.nodes_std_linear = nn.Linear(nodes_dim, nodes_dim)

        self.nodes_normal_sampler = distributions.normal.Normal(loc=1, scale=1)
        self.nodes_bn_2 = nn.BatchNorm1d(nodes_dim)

        # attention layers
        self.bn_1 = nn.BatchNorm3d(n_channels)
        self.bn_2 = nn.BatchNorm1d(n_channels)
        self.bn_3 = pl.BatchNorm(n_nodes, dim=0)
        self.bn_4 = pl.BatchNorm3d(n_nodes, dim=3)
        self.bn_5 = nn.BatchNorm1d(n_channels)
        self.bn_6 = nn.BatchNorm3d(n_channels)

        self.linear_1 = nn.Linear(n_channels, n_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input):
        # input is of shape (None, C, T, H, W)

        x = input
        input_shape = pytorch_utils.get_shape(x)
        batch_size = input_shape[0]
        assert len(input_shape) == 5

        # learn mean and std
        nodes = self.nodes
        nodes = self.nodes_bn_1(nodes)
        nodes_mean = self.nodes_mean_linear(nodes)
        nodes_std = self.nodes_std_linear(nodes)  # (None, C)

        # if self.training:
        #     nodes_eps = self.nodes_normal_sampler.sample((self.n_nodes, self.nodes_dim)).cuda()
        #     # nodes_eps = torch.ones((self.n_nodes, self.nodes_dim)).cuda()
        # else:
        #     nodes_eps = self.nodes_normal_sampler.sample((self.n_nodes, self.nodes_dim)).cuda()
        #     # nodes_eps = torch.ones((self.n_nodes, self.nodes_dim)).cuda()

        # re-parametrization trick
        nodes = torch.exp(0.5 * nodes_std) + nodes_mean
        nodes1 = self.nodes_bn_2(nodes)  # (N, C)

        x = self.bn_1(x)

        # f is the similarity between features and nodes
        f = self.__tensor_product_1(x, nodes1)  # (None, H, W, N, T)
        f = self.bn_4(f)

        # g path (V)
        nodes2 = self.bn_5(nodes)  # (N, C)

        y = self.__tensor_product_2(f, nodes2)  # (None, C, T, H, W)
        y = self.bn_6(y)
        y = self.relu(y)

        return y

    def __tensor_product_1(self, phi, theta):
        """
        Takes two input tensors and does matrix multiplication across channel dimension (c).
        :param phi: # (None, C, T, H, W)
        :param theta: # (N, C)
        :return: f # (None, H, W, N, T)
        """

        n, c, t, h, w = pytorch_utils.get_shape(phi)
        n_nodes, node_dim = pytorch_utils.get_shape(theta)

        assert node_dim == c

        # reshape phi
        phi = phi.permute(0, 2, 3, 4, 1)  # (None, T, H, W, C)
        phi = phi.contiguous().view(n * t * h * w, c)  # (None*T*H*W, C)

        # transpose for matrix multiplication
        theta = theta.permute(1, 0)  # (C, N)

        f = torch.matmul(phi, theta)  # (None*T*H*W, C) x (C, N) = (None*T*H*W, N)
        f = f.view(n, t, h, w, n_nodes)  # (None, T, H ,W, N)
        f = f.permute(0, 2, 3, 4, 1)  # (None, H, W, N, T)

        return f

    def __tensor_product_2(self, f, g):
        """
        Takes two input tensors and does matrix multiplication across node dimension (n).
        :param f: (None, H, W, N, T)
        :param g:  (N, C)
        :return: y # (None, C, T, H, W)
        """

        n, h, w, n_c, t = pytorch_utils.get_shape(f)
        n_nodes, node_dim = pytorch_utils.get_shape(g)

        assert n_nodes == n_c

        # reshape f
        f = f.permute(0, 1, 2, 4, 3)  # (None, H, W, T, N)
        f = f.contiguous().view(n * h * w * t, n_nodes)  # (None*H*W*T, N)

        y = torch.matmul(f, g)  # (None*H*W*T, C)
        y = y.view(n, h, w, t, node_dim)  # (None, H, W, T, C)
        y = y.permute(0, 4, 3, 1, 2)  # (None, C, T, H, W)

        return y

# endregion

# region Node Attention Experimental

class NodeAttentionHardExprimental(nn.Module):
    def __init__(self, n_channels, n_nodes, nodes):
        super(NodeAttentionHardExprimental, self).__init__()

        self.n_channels = n_channels
        self.n_nodes = n_nodes
        self.nodes = nodes

        self.bn_1 = nn.BatchNorm3d(n_channels)
        self.bn_2 = nn.BatchNorm1d(n_channels)
        self.bn_3 = pl.BatchNorm(n_nodes, dim=0)
        self.bn_4 = pl.BatchNorm3d(n_nodes, dim=3)
        self.bn_5 = nn.BatchNorm1d(n_channels)
        self.bn_6 = nn.BatchNorm3d(n_channels)

        self.linear_1 = nn.Linear(n_channels, n_channels)
        self.conv3d = nn.Conv3d(n_nodes, n_nodes, (1, 1, 1))

        self.relu_1 = nn.LeakyReLU(negative_slope=0.2)
        self.softmax_1 = nn.Softmax(dim=3)
        self.hardmax_1 = pf.Hardmax.apply

        self.sigmoid_1 = nn.Sigmoid()
        self.threshold = pf.Threshold.apply

    def forward(self, input):
        # input is of shape (None, C, T, H, W)

        input_shape = pytorch_utils.get_shape(input)
        n, c, t, h, w = input_shape

        assert len(input_shape) == 5

        # phi path (Q) or (x)
        x = self.bn_1(input)
        phi = x  # (None, C, T, H, W)

        # theta path (K) or (c)
        theta = self.bn_2(self.nodes)  # (N, C)
        theta = self.linear_1(theta)  # (N, C)
        theta = self.bn_3(theta)  # (N, C)

        # f is the similarity between (theta and phi) or (Q and K)
        f = self.__tensor_product_1(phi, theta)  # (None, H, W, N, T)
        # model nodes after product
        # f = f.permute(0, 3, 1, 2, 4)
        # f = self.conv3d(f)
        # f = f.permute(0, 2, 3, 1, 4)
        # batchnorm across the node dimension
        f = self.bn_4(f)  # (None, H, W, N, T)
        # softmax to select only one node
        # f = self.softmax_1(f)  # (None, H, W, N, T)
        # now, hardmax to select only one node for each timestep
        # f = self.hardmax_1(f) # (None, H, W, N, T)

        f = self.sigmoid_1(f)
        f = self.threshold(f)

        # g path (V)
        g = self.bn_5(self.nodes)  # (N, C)

        y = self.__tensor_product_2(f, g)  # (None, C, T, H, W)
        y = self.bn_6(y)
        y = self.relu_1(y)

        return y

    def __tensor_product_1(self, phi, theta):
        """
        Takes two input tensors and does matrix multiplication across channel dimension (c).
        :param phi: # (None, C, T, H, W)
        :param theta: # (N, C)
        :return: f # (None, H, W, N, T)
        """

        n, c, t, h, w = pytorch_utils.get_shape(phi)
        n_nodes, node_dim = pytorch_utils.get_shape(theta)

        assert node_dim == c

        # reshape phi
        phi = phi.permute(0, 2, 3, 4, 1)  # (None, T, H, W, C)
        phi = phi.contiguous().view(n * t * h * w, c)  # (None*T*H*W, C)

        # transpose for matrix multiplication
        theta = theta.permute(1, 0)  # (C, N)

        f = torch.matmul(phi, theta)  # (None*T*H*W, C) x (C, N) = (None*T*H*W, N)
        f = f.view(n, t, h, w, n_nodes)  # (None, T, H ,W, N)
        f = f.permute(0, 2, 3, 4, 1)  # (None, H, W, N, T)

        return f

    def __tensor_product_2(self, f, g):
        """
        Takes two input tensors and does matrix multiplication across node dimension (n).
        :param f: (None, H, W, N, T)
        :param g:  (N, C)
        :return: y # (None, C, T, H, W)
        """

        n, h, w, n_c, t = pytorch_utils.get_shape(f)
        n_nodes, node_dim = pytorch_utils.get_shape(g)

        assert n_nodes == n_c

        # reshape f
        f = f.permute(0, 1, 2, 4, 3)  # (None, H, W, T, N)
        f = f.contiguous().view(n * h * w * t, n_nodes)  # (None*H*W*T, N)

        y = torch.matmul(f, g)  # (None*H*W*T, C)
        y = y.view(n, h, w, t, node_dim)  # (None, H, W, T, C)
        y = y.permute(0, 4, 3, 1, 2)  # (None, C, T, H, W)

        return y

# endregion

# region Node Attention + Gumbel

class NodeAttentionGumbelHardSigmoidNoSparsity(nn.Module):
    def __init__(self, n_channels, n_nodes, nodes):
        super(NodeAttentionGumbelHardSigmoidNoSparsity, self).__init__()

        self.n_channels = n_channels
        self.n_nodes = n_nodes
        self.nodes = nodes

        self.bn_1 = nn.BatchNorm3d(n_channels)
        self.bn_2 = nn.BatchNorm1d(n_channels)
        self.bn_3 = pl.BatchNorm(n_nodes, dim=0)
        self.bn_5 = nn.BatchNorm1d(n_channels)
        self.bn_6 = nn.BatchNorm3d(n_channels)

        self.linear_1 = nn.Linear(n_channels, n_channels)
        self.conv3d = nn.Conv3d(n_nodes, n_nodes, (1, 1, 1))

        self.relu_1 = nn.LeakyReLU(negative_slope=0.2)

        self.gumbel_sigmoid_1 = pl.GumbelSigmoid()
        self.threshold = pf.Threshold.apply

    def forward(self, input):
        # input is of shape (None, C, T, H, W)

        input_shape = pytorch_utils.get_shape(input)
        n, c, t, h, w = input_shape

        assert len(input_shape) == 5

        # phi path (Q) or (x)
        x = self.bn_1(input)
        phi = x  # (None, C, T, H, W)

        # theta path (K) or (c)
        theta = self.bn_2(self.nodes)  # (N, C)
        theta = self.linear_1(theta)  # (N, C)
        theta = self.bn_3(theta)  # (N, C)

        # f is the similarity between (theta and phi) or (Q and K)
        f = self.__tensor_product_1(phi, theta)  # (None, H, W, N, T)

        # sigmoid + threshold to select few nodes for teach timestep
        f = self.gumbel_sigmoid_1(f)  # (None, H, W, N, T)
        # f = self.sigmoid_1(f)
        f = self.threshold(f)  # (None, H, W, N, T)

        # g path (V)
        g = self.bn_5(self.nodes)  # (N, C)

        # sum over selected nodes
        y = self.__tensor_product_2(f, g)  # (None, C, T, H, W)
        y = self.bn_6(y)
        y = self.relu_1(y)

        return y

    def __tensor_product_1(self, phi, theta):
        """
        Takes two input tensors and does matrix multiplication across channel dimension (c).
        :param phi: # (None, C, T, H, W)
        :param theta: # (N, C)
        :return: f # (None, H, W, N, T)
        """

        n, c, t, h, w = pytorch_utils.get_shape(phi)
        n_nodes, node_dim = pytorch_utils.get_shape(theta)

        assert node_dim == c

        # reshape phi
        phi = phi.permute(0, 2, 3, 4, 1)  # (None, T, H, W, C)
        phi = phi.contiguous().view(n * t * h * w, c)  # (None*T*H*W, C)

        # transpose for matrix multiplication
        theta = theta.permute(1, 0)  # (C, N)

        f = torch.matmul(phi, theta)  # (None*T*H*W, C) x (C, N) = (None*T*H*W, N)
        f = f.view(n, t, h, w, n_nodes)  # (None, T, H ,W, N)
        f = f.permute(0, 2, 3, 4, 1)  # (None, H, W, N, T)

        return f

    def __tensor_product_2(self, f, g):
        """
        Takes two input tensors and does matrix multiplication across node dimension (n).
        :param f: (None, H, W, N, T)
        :param g:  (N, C)
        :return: y # (None, C, T, H, W)
        """

        n, h, w, n_c, t = pytorch_utils.get_shape(f)
        n_nodes, node_dim = pytorch_utils.get_shape(g)

        assert n_nodes == n_c

        # reshape f
        f = f.permute(0, 1, 2, 4, 3)  # (None, H, W, T, N)
        f = f.contiguous().view(n * h * w * t, n_nodes)  # (None*H*W*T, N)

        y = torch.matmul(f, g)  # (None*H*W*T, C)
        y = y.view(n, h, w, t, node_dim)  # (None, H, W, T, C)
        y = y.permute(0, 4, 3, 1, 2)  # (None, C, T, H, W)

        return y

class NodeAttentionGumbelHardSigmoidV1(nn.Module):
    def __init__(self, n_channels, n_nodes, nodes):
        super(NodeAttentionGumbelHardSigmoidV1, self).__init__()

        self.n_channels = n_channels
        self.n_nodes = n_nodes
        self.nodes = nodes

        self.bn_1 = nn.BatchNorm3d(n_channels)
        self.bn_2 = nn.BatchNorm1d(n_channels)
        self.bn_3 = pl.BatchNorm(n_nodes, dim=0)

        self.bn_4 = pl.BatchNorm3d(n_nodes, dim=3)
        self.bn_6 = nn.BatchNorm3d(n_channels)

        self.linear_1 = nn.Linear(n_channels, n_channels)
        self.linear_2 = nn.Linear(n_channels, n_channels)
        self.conv3d = nn.Conv3d(n_nodes, n_nodes, (1, 1, 1))

        self.relu_1 = nn.LeakyReLU(negative_slope=0.2)

        self.gumbel_sigmoid_1 = pl.GumbelSigmoid()
        self.threshold = pf.Threshold.apply

    def forward(self, input):
        # input is of shape (None, C, T, H, W)

        input_shape = pytorch_utils.get_shape(input)
        n, c, t, h, w = input_shape

        assert len(input_shape) == 5

        # features
        x = self.bn_1(input)  # (None, C, T, H, W)

        # nodes
        nodes = self.bn_2(self.nodes)  # (N, C)
        nodes = self.linear_1(nodes)  # (N, C)
        nodes = self.bn_3(nodes)  # (N, C)

        # f is the similarity between features and nodes
        f = self.__tensor_product_1(x, nodes)  # (None, C, T, H, W) * (N, C) => (None, H, W, N, T)

        # sigmoid + threshold to select few nodes for teach timestep
        alpha = self.gumbel_sigmoid_1(f)  # (None, H, W, N, T)
        alpha = self.threshold(alpha)

        # sum over selected nodes
        y = self.__tensor_product_2(alpha, nodes)  # (None, H, W, N, T) * # (N, C) => (None, C, T, H, W)
        y = self.bn_6(y)
        y = self.relu_1(y)

        return y

class NodeAttentionGumbelHardSigmoid(nn.Module):
    def __init__(self, n_channels, n_nodes, n_timesteps, nodes):
        super(NodeAttentionGumbelHardSigmoid, self).__init__()

        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.nodes = nodes

        self.bn_1 = nn.BatchNorm3d(n_channels)

        self.bn_2 = nn.BatchNorm1d(n_channels)
        self.bn_3 = pl.BatchNorm(n_nodes, dim=0)

        self.bn_4 = pl.BatchNorm3d(n_nodes, dim=3)
        self.bn_5 = pl.BatchNorm3d(n_timesteps, dim=4)
        self.bn_6 = nn.BatchNorm3d(n_channels)

        self.linear_2 = nn.Linear(n_channels, n_channels)
        self.conv3d_2 = nn.Conv3d(n_nodes, n_nodes, (1, 1, 1))

        self.relu_1 = nn.LeakyReLU(negative_slope=0.2)

        self.gumbel_sigmoid_1 = pl.GumbelSigmoid()
        self.gumbel = pl.Gumbel(temperature=1)
        self.gumbel_noise = pl.GumbelNoise()
        self.sigmoid_1 = nn.Sigmoid()

        self.threshold = pf.Threshold.apply
        # self.threshold = pf.ThresholdDualBath.apply

        self.softmax_1 = nn.Softmax(dim=3)
        self.hardmax_1 = pf.Hardmax.apply

    def forward(self, input):
        # input is of shape (None, C, T, H, W)

        nodes = self.nodes
        x = input

        input_shape = pytorch_utils.get_shape(x)
        n, c, t, h, w = input_shape

        assert len(input_shape) == 5

        # features
        x = self.bn_1(x)  # (None, C, T, H, W)

        # nodes
        nodes = self.bn_2(nodes)  # (N, C)
        nodes = self.linear_2(nodes)  # (N, C)
        nodes = self.bn_3(nodes)  # (N, C)

        # f is the similarity between features and nodes
        f = self.__tensor_product_1(x, nodes)  # (None, C, T, H, W) * (N, C) => (None, H, W, N, T)

        alpha = f
        # alpha = self.softmax_1(alpha)
        # alpha = self.hardmax_1(alpha)

        alpha = self.sigmoid_1(alpha)
        alpha = self.threshold(alpha)

        self.attention_values = alpha.tolist()

        # f = self.bn_5(f)

        # batchnorm across the node dimension
        # f = self.bn_4(f)  # (None, H, W, N, T)

        # f = f.permute(0, 3, 1, 2, 4)
        # f = self.conv3d_1(f)
        # f = f.permute(0, 2, 3, 1, 4)

        # sigmoid + threshold to select few nodes for teach timestep
        # alpha = self.gumbel_sigmoid_1(f)  # (None, H, W, N, T)
        # alpha = self.sigmoid_1(f)
        # alpha = self.threshold(alpha)

        # temperature = 0.67
        # alpha = f
        # noise = self.gumbel_noise(alpha)
        # alpha = (alpha + noise) / temperature
        # alpha = self.sigmoid_1(alpha)
        # alpha = self.threshold(alpha)

        # sum over selected nodes
        y = self.__tensor_product_2(alpha, nodes)  # (None, H, W, N, T) * # (N, C) => (None, C, T, H, W)
        y = self.bn_6(y)
        y = self.relu_1(y)

        return y

    def __tensor_product_1(self, phi, theta):
        """
        Takes two input tensors and does matrix multiplication across channel dimension (c).
        :param phi: # (None, C, T, H, W)
        :param theta: # (N, C)
        :return: f # (None, H, W, N, T)
        """

        n, c, t, h, w = pytorch_utils.get_shape(phi)
        n_nodes, node_dim = pytorch_utils.get_shape(theta)

        assert node_dim == c

        # reshape phi
        phi = phi.permute(0, 2, 3, 4, 1)  # (None, T, H, W, C)
        phi = phi.contiguous().view(n * t * h * w, c)  # (None*T*H*W, C)

        # transpose for matrix multiplication
        theta = theta.permute(1, 0)  # (C, N)

        f = torch.matmul(phi, theta)  # (None*T*H*W, C) x (C, N) = (None*T*H*W, N)
        f = f.view(n, t, h, w, n_nodes)  # (None, T, H ,W, N)
        f = f.permute(0, 2, 3, 4, 1)  # (None, H, W, N, T)

        return f

    def __tensor_product_2(self, f, g):
        """
        Takes two input tensors and does matrix multiplication across node dimension (n).
        :param f: (None, H, W, N, T)
        :param g:  (N, C)
        :return: y # (None, C, T, H, W)
        """

        n, h, w, n_c, t = pytorch_utils.get_shape(f)
        n_nodes, node_dim = pytorch_utils.get_shape(g)

        assert n_nodes == n_c

        # reshape f
        f = f.permute(0, 1, 2, 4, 3)  # (None, H, W, T, N)
        f = f.contiguous().view(n * h * w * t, n_nodes)  # (None*H*W*T, N)

        y = torch.matmul(f, g)  # (None*H*W*T, C)
        y = y.view(n, h, w, t, node_dim)  # (None, H, W, T, C)
        y = y.permute(0, 4, 3, 1, 2)  # (None, C, T, H, W)

        return y

# endregion

# region Node Attention + Loss

class NodeAttentionLZeroNormV1(nn.Module):
    def __init__(self, n_channels, n_nodes, n_timesteps, nodes, n_samples, penalty_loc_mean=1, penalty_coef=1e-1):
        super(NodeAttentionLZeroNormV1, self).__init__()

        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.nodes = nodes

        self.bn_1 = nn.BatchNorm3d(n_channels)

        self.bn_2 = nn.BatchNorm1d(n_channels)
        self.bn_3 = pl.BatchNorm(n_nodes, dim=0)

        self.bn_4 = pl.BatchNorm3d(n_nodes, dim=3)
        self.bn_6 = nn.BatchNorm3d(n_channels)

        self.linear_1 = nn.Linear(n_channels, n_channels)

        self.sigmoid_1 = nn.Sigmoid()
        self.threshold = pf.Threshold.apply

        self.relu_1 = nn.LeakyReLU(negative_slope=0.2)

        # for l0_norm regularization
        self.l_zero_norm = pl.LZeroNorm(penalty_loc_mean, fix_temp=True, beta=2 / 3.0)
        self.penalty_coef = penalty_coef
        self.n_samples = n_samples

    def forward(self, input):
        # input is of shape (None, C, T, H, W)

        nodes = self.nodes
        x = input

        input_shape = pytorch_utils.get_shape(x)
        n, c, t, h, w = input_shape

        assert len(input_shape) == 5

        # features
        x = self.bn_1(x)  # (None, C, T, H, W)

        # nodes
        nodes = self.bn_2(nodes)  # (N, C)
        nodes = self.linear_1(nodes)  # (N, C)
        nodes = self.bn_3(nodes)  # (N, C)

        # f is the similarity between features and nodes
        f = self.__tensor_product_1(x, nodes)  # (None, C, T, H, W) * (N, C) => (None, H, W, N, T)
        alpha = f

        # for logging
        self.attention_values_before = alpha.tolist()

        # sparsify the attentions using l0_norm
        alpha = self.l_zero_norm(alpha)

        # for logging
        self.attention_values_after = alpha.tolist()
        self.gate_values = self.l_zero_norm.gate_values

        # update penalty
        self.penalty_coef = 0.1
        self.gating_loss = (self.penalty_coef / self.n_samples) * self.l_zero_norm.penalty

        # sum over selected nodes
        y = self.__tensor_product_2(alpha, nodes)  # (None, H, W, N, T) * # (N, C) => (None, C, T, H, W)
        y = self.bn_6(y)
        y = self.relu_1(y)

        return y

    def __tensor_product_1(self, phi, theta):
        """
        Takes two input tensors and does matrix multiplication across channel dimension (c).
        :param phi: # (None, C, T, H, W)
        :param theta: # (N, C)
        :return: f # (None, H, W, N, T)
        """

        n, c, t, h, w = pytorch_utils.get_shape(phi)
        n_nodes, node_dim = pytorch_utils.get_shape(theta)

        assert node_dim == c

        # reshape phi
        phi = phi.permute(0, 2, 3, 4, 1)  # (None, T, H, W, C)
        phi = phi.contiguous().view(n * t * h * w, c)  # (None*T*H*W, C)

        # transpose for matrix multiplication
        theta = theta.permute(1, 0)  # (C, N)

        f = torch.matmul(phi, theta)  # (None*T*H*W, C) x (C, N) = (None*T*H*W, N)
        f = f.view(n, t, h, w, n_nodes)  # (None, T, H ,W, N)
        f = f.permute(0, 2, 3, 4, 1)  # (None, H, W, N, T)

        return f

    def __tensor_product_2(self, f, g):
        """
        Takes two input tensors and does matrix multiplication across node dimension (n).
        :param f: (None, H, W, N, T)
        :param g:  (N, C)
        :return: y # (None, C, T, H, W)
        """

        n, h, w, n_c, t = pytorch_utils.get_shape(f)
        n_nodes, node_dim = pytorch_utils.get_shape(g)

        assert n_nodes == n_c

        # reshape f
        f = f.permute(0, 1, 2, 4, 3)  # (None, H, W, T, N)
        f = f.contiguous().view(n * h * w * t, n_nodes)  # (None*H*W*T, N)

        y = torch.matmul(f, g)  # (None*H*W*T, C)
        y = y.view(n, h, w, t, node_dim)  # (None, H, W, T, C)
        y = y.permute(0, 4, 3, 1, 2)  # (None, C, T, H, W)

        return y

class NodeAttentionLZeroNormV2(nn.Module):
    def __init__(self, n_channels, n_nodes, n_timesteps, nodes):
        super(NodeAttentionLZeroNormV2, self).__init__()

        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.nodes = nodes

        self.bn_1 = nn.BatchNorm3d(n_channels)
        self.bn_2 = nn.BatchNorm1d(n_channels)
        self.bn_3 = pl.BatchNorm(n_nodes, dim=0)
        self.bn_5 = nn.BatchNorm1d(n_channels)
        self.bn_4 = pl.BatchNorm3d(n_nodes, dim=3)
        self.bn_6 = nn.BatchNorm3d(n_channels)

        self.linear_1 = nn.Linear(n_channels, n_channels)

        self.sigmoid_1 = nn.Sigmoid()
        self.threshold = pf.Threshold.apply

        self.relu_1 = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input):
        # input is of shape (None, C, T, H, W)

        nodes = self.nodes
        x = input

        input_shape = pytorch_utils.get_shape(x)
        n, c, t, h, w = input_shape
        batch_size = input_shape[0]

        assert len(input_shape) == 5

        # features
        x = self.bn_1(x)  # (None, C, T, H, W)

        # nodes
        nodes = self.bn_2(nodes)  # (N, C)
        nodes = self.linear_1(nodes)  # (N, C)
        nodes = self.bn_3(nodes)  # (N, C)

        # f is the similarity between features and nodes
        f = self.__tensor_product_1(x, nodes)  # (None, C, T, H, W) * (N, C) => (None, H, W, N, T)

        # batchnorm across the node dimension
        f = self.bn_4(f)  # (None, H, W, N, T)

        # sigmoid + threshold to select few nodes for teach timestep
        f = self.sigmoid_1(f)  # (None, H, W, N, T)

        # threshold for gating after sigmoid
        alpha = self.threshold(f)  # (None, H, W, N, T)

        # gating loss
        self.gating_loss = torch.sum(f) / batch_size

        # for logging
        self.attention_values_before = f.tolist()
        self.attention_values_after = alpha.tolist()

        # sum over selected nodes
        nodes = self.bn_5(nodes)  # (N, C)
        y = self.__tensor_product_2(alpha, nodes)  # (None, H, W, N, T) * # (N, C) => (None, C, T, H, W)
        y = self.bn_6(y)
        y = self.relu_1(y)

        return y

    def __tensor_product_1(self, phi, theta):
        """
        Takes two input tensors and does matrix multiplication across channel dimension (c).
        :param phi: # (None, C, T, H, W)
        :param theta: # (N, C)
        :return: f # (None, H, W, N, T)
        """

        n, c, t, h, w = pytorch_utils.get_shape(phi)
        n_nodes, node_dim = pytorch_utils.get_shape(theta)

        assert node_dim == c

        # reshape phi
        phi = phi.permute(0, 2, 3, 4, 1)  # (None, T, H, W, C)
        phi = phi.contiguous().view(n * t * h * w, c)  # (None*T*H*W, C)

        # transpose for matrix multiplication
        theta = theta.permute(1, 0)  # (C, N)

        f = torch.matmul(phi, theta)  # (None*T*H*W, C) x (C, N) = (None*T*H*W, N)
        f = f.view(n, t, h, w, n_nodes)  # (None, T, H ,W, N)
        f = f.permute(0, 2, 3, 4, 1)  # (None, H, W, N, T)

        return f

    def __tensor_product_2(self, f, g):
        """
        Takes two input tensors and does matrix multiplication across node dimension (n).
        :param f: (None, H, W, N, T)
        :param g:  (N, C)
        :return: y # (None, C, T, H, W)
        """

        n, h, w, n_c, t = pytorch_utils.get_shape(f)
        n_nodes, node_dim = pytorch_utils.get_shape(g)

        assert n_nodes == n_c

        # reshape f
        f = f.permute(0, 1, 2, 4, 3)  # (None, H, W, T, N)
        f = f.contiguous().view(n * h * w * t, n_nodes)  # (None*H*W*T, N)

        y = torch.matmul(f, g)  # (None*H*W*T, C)
        y = y.view(n, h, w, t, node_dim)  # (None, H, W, T, C)
        y = y.permute(0, 4, 3, 1, 2)  # (None, C, T, H, W)

        return y

class NodeAttentionGumbelLZeroNorm(nn.Module):
    def __init__(self, n_channels, n_nodes, n_timesteps, nodes):
        super(NodeAttentionGumbelLZeroNorm, self).__init__()

        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.nodes = nodes

        self.bn_1 = nn.BatchNorm3d(n_channels)
        self.bn_2 = nn.BatchNorm1d(n_channels)
        self.bn_3 = pl.BatchNorm(n_nodes, dim=0)
        self.bn_5 = nn.BatchNorm1d(n_channels)
        self.bn_4 = pl.BatchNorm3d(n_nodes, dim=3)
        self.bn_6 = nn.BatchNorm3d(n_channels)

        self.linear_1 = nn.Linear(n_channels, n_channels)

        self.gumbel = pl.Gumbel(temperature=2 / 3.0)
        self.gumbel_sigmoid = pl.GumbelSigmoid(temperature=2 / 3.0)
        self.sigmoid_1 = nn.Sigmoid()
        self.threshold = pf.Threshold.apply

        self.relu_1 = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input):
        # input is of shape (None, C, T, H, W)

        nodes = self.nodes
        x = input

        input_shape = pytorch_utils.get_shape(x)
        n, c, t, h, w = input_shape
        batch_size = input_shape[0]

        assert len(input_shape) == 5

        # features
        x = self.bn_1(x)  # (None, C, T, H, W)

        # nodes
        nodes = self.bn_2(nodes)  # (N, C)
        nodes = self.linear_1(nodes)  # (N, C)
        nodes = self.bn_3(nodes)  # (N, C)

        # f is the similarity between features and nodes
        f = self.__tensor_product_1(x, nodes)  # (None, C, T, H, W) * (N, C) => (None, H, W, N, T)

        # batchnorm across the node dimension
        f = self.bn_4(f)  # (None, H, W, N, T)

        # sigmoid + threshold to select few nodes for teach timestep
        # f = self.sigmoid_1(f)  # (None, H, W, N, T)
        f = self.gumbel_sigmoid(f)

        # threshold for gating after sigmoid
        alpha = self.threshold(f)  # (None, H, W, N, T)

        # gating loss
        self.gating_loss = torch.sum(f) / batch_size

        # for logging
        self.attention_values_before = f.tolist()
        self.attention_values_after = alpha.tolist()

        # sum over selected nodes
        nodes = self.bn_5(nodes)  # (N, C)
        y = self.__tensor_product_2(alpha, nodes)  # (None, H, W, N, T) * # (N, C) => (None, C, T, H, W)
        y = self.bn_6(y)
        y = self.relu_1(y)

        return y

    def __tensor_product_1(self, phi, theta):
        """
        Takes two input tensors and does matrix multiplication across channel dimension (c).
        :param phi: # (None, C, T, H, W)
        :param theta: # (N, C)
        :return: f # (None, H, W, N, T)
        """

        n, c, t, h, w = pytorch_utils.get_shape(phi)
        n_nodes, node_dim = pytorch_utils.get_shape(theta)

        assert node_dim == c

        # reshape phi
        phi = phi.permute(0, 2, 3, 4, 1)  # (None, T, H, W, C)
        phi = phi.contiguous().view(n * t * h * w, c)  # (None*T*H*W, C)

        # transpose for matrix multiplication
        theta = theta.permute(1, 0)  # (C, N)

        f = torch.matmul(phi, theta)  # (None*T*H*W, C) x (C, N) = (None*T*H*W, N)
        f = f.view(n, t, h, w, n_nodes)  # (None, T, H ,W, N)
        f = f.permute(0, 2, 3, 4, 1)  # (None, H, W, N, T)

        return f

    def __tensor_product_2(self, f, g):
        """
        Takes two input tensors and does matrix multiplication across node dimension (n).
        :param f: (None, H, W, N, T)
        :param g:  (N, C)
        :return: y # (None, C, T, H, W)
        """

        n, h, w, n_c, t = pytorch_utils.get_shape(f)
        n_nodes, node_dim = pytorch_utils.get_shape(g)

        assert n_nodes == n_c

        # reshape f
        f = f.permute(0, 1, 2, 4, 3)  # (None, H, W, T, N)
        f = f.contiguous().view(n * h * w * t, n_nodes)  # (None*H*W*T, N)

        y = torch.matmul(f, g)  # (None*H*W*T, C)
        y = y.view(n, h, w, t, node_dim)  # (None, H, W, T, C)
        y = y.permute(0, 4, 3, 1, 2)  # (None, C, T, H, W)

        return y

# endregion

# region ConceptAttention

class ConceptAttentionVariational(nn.Module):
    def __init__(self, n_channels):
        super(ConceptAttentionVariational, self).__init__()

        n_nodes = 128
        nodes_dim = 1024

        self.nodes_dim = nodes_dim
        self.nodes = torch.from_numpy(sobol.sobol_generate(nodes_dim, n_nodes)).cuda()

        self.n_nodes = n_nodes
        self.n_channels = n_channels

        # for learning nodes based on mean and std
        self.nodes_bn_1 = nn.BatchNorm1d(nodes_dim)
        self.nodes_mean_linear = nn.Linear(nodes_dim, nodes_dim)
        self.nodes_std_linear = nn.Linear(nodes_dim, nodes_dim)

        self.nodes_normal_sampler = distributions.normal.Normal(loc=1, scale=1)
        self.nodes_bn_2 = nn.BatchNorm1d(nodes_dim)

        # attention layers
        self.bn_1 = nn.BatchNorm3d(n_channels)
        self.bn_2 = nn.BatchNorm1d(n_channels)
        self.bn_3 = pl.BatchNorm(n_nodes, dim=0)
        self.bn_4 = pl.BatchNorm3d(n_nodes, dim=3)
        self.bn_5 = nn.BatchNorm1d(n_channels)
        self.bn_6 = nn.BatchNorm3d(n_channels)

        self.linear_1 = nn.Linear(n_channels, n_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input):
        # input is of shape (None, C, T, H, W)

        x = input
        input_shape = pytorch_utils.get_shape(x)
        batch_size = input_shape[0]
        assert len(input_shape) == 5

        # learn mean and std
        nodes = self.nodes
        nodes = self.nodes_bn_1(nodes)
        nodes_mean = self.nodes_mean_linear(nodes)
        nodes_std = self.nodes_std_linear(nodes)  # (None, C)

        # if self.training:
        #     nodes_eps = self.nodes_normal_sampler.sample((self.n_nodes, self.nodes_dim)).cuda()
        #     # nodes_eps = torch.ones((self.n_nodes, self.nodes_dim)).cuda()
        # else:
        #     nodes_eps = self.nodes_normal_sampler.sample((self.n_nodes, self.nodes_dim)).cuda()
        #     # nodes_eps = torch.ones((self.n_nodes, self.nodes_dim)).cuda()

        # re-parametrization trick
        nodes = torch.exp(0.5 * nodes_std) + nodes_mean
        nodes1 = self.nodes_bn_2(nodes)  # (N, C)

        x = self.bn_1(x)

        # f is the similarity between features and nodes
        f = self.__tensor_product_1(x, nodes1)  # (None, H, W, N, T)
        f = self.bn_4(f)

        # g path (V)
        nodes2 = self.bn_5(nodes)  # (N, C)

        y = self.__tensor_product_2(f, nodes2)  # (None, C, T, H, W)
        y = self.bn_6(y)
        y = self.relu(y)

        return y

    def __tensor_product_1(self, phi, theta):
        """
        Takes two input tensors and does matrix multiplication across channel dimension (c).
        :param phi: # (None, C, T, H, W)
        :param theta: # (N, C)
        :return: f # (None, H, W, N, T)
        """

        n, c, t, h, w = pytorch_utils.get_shape(phi)
        n_nodes, node_dim = pytorch_utils.get_shape(theta)

        assert node_dim == c

        # reshape phi
        phi = phi.permute(0, 2, 3, 4, 1)  # (None, T, H, W, C)
        phi = phi.contiguous().view(n * t * h * w, c)  # (None*T*H*W, C)

        # transpose for matrix multiplication
        theta = theta.permute(1, 0)  # (C, N)

        f = torch.matmul(phi, theta)  # (None*T*H*W, C) x (C, N) = (None*T*H*W, N)
        f = f.view(n, t, h, w, n_nodes)  # (None, T, H ,W, N)
        f = f.permute(0, 2, 3, 4, 1)  # (None, H, W, N, T)

        return f

    def __tensor_product_2(self, f, g):
        """
        Takes two input tensors and does matrix multiplication across node dimension (n).
        :param f: (None, H, W, N, T)
        :param g:  (N, C)
        :return: y # (None, C, T, H, W)
        """

        n, h, w, n_c, t = pytorch_utils.get_shape(f)
        n_nodes, node_dim = pytorch_utils.get_shape(g)

        assert n_nodes == n_c

        # reshape f
        f = f.permute(0, 1, 2, 4, 3)  # (None, H, W, T, N)
        f = f.contiguous().view(n * h * w * t, n_nodes)  # (None*H*W*T, N)

        y = torch.matmul(f, g)  # (None*H*W*T, C)
        y = y.view(n, h, w, t, node_dim)  # (None, H, W, T, C)
        y = y.permute(0, 4, 3, 1, 2)  # (None, C, T, H, W)

        return y

# endregion
