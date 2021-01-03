#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from core import pytorch_utils
from modules import layers_pytorch as pl

# region Global Self-Attention

class GlobalSelfAttentionHead(nn.Module):
    def __init__(self, n_channels_in, n_channels_inter):
        """
        Initialize the module.
        """
        super(GlobalSelfAttentionHead, self).__init__()

        self.n_channels_in = n_channels_in
        self.n_channels_inter = n_channels_inter
        self.is_softmax_activation = True
        self.dropout_ratio = 0.25

        self.__define_layers()

    def __define_layers(self):

        dropout_ratio = self.dropout_ratio
        key_linear = pl.Linear3d(self.n_channels_in, self.n_channels_inter)
        query_linear = pl.Linear3d(self.n_channels_in, self.n_channels_inter)
        value_linear = pl.Linear3d(self.n_channels_in, self.n_channels_inter)
        output_linear = pl.Linear3d(self.n_channels_inter, self.n_channels_in)

        # value embedding
        self.value_embedding = nn.Sequential(nn.Dropout(dropout_ratio), value_linear)

        # key embedding
        self.key_embedding = nn.Sequential(nn.Dropout(dropout_ratio), key_linear)

        # query embedding
        self.query_embedding = nn.Sequential(nn.Dropout(dropout_ratio), query_linear)

        # output embedding
        self.output_embedding = nn.Sequential(nn.Dropout(dropout_ratio), output_linear)

    def forward(self, x):
        """
        :param x: (B, C, T, H, W)
        :return:
        """

        batch_size = x.size(0)
        x_shape = pytorch_utils.get_shape(x)
        B, C, T, H, W = x_shape

        # key embedding
        key = self.key_embedding(x)  # (B, C, T, H, W)
        key = key.view(batch_size, self.n_channels_inter, -1)  # (B, C, T*H*W)
        key = key.permute(0, 2, 1)  # (B, T*H*W, C)

        # query embedding
        query = self.query_embedding(x)  # (B, C, T, H, W)
        query = query.view(batch_size, self.n_channels_inter, -1)  # (B, C, T*H*W)

        # value embedding
        value = self.value_embedding(x)  # (B, C, T, H, W)
        value = value.view(batch_size, self.n_channels_inter, -1)  # (B, C, T*H*W)
        value = value.permute(0, 2, 1)  # (B, T*H*W, C)

        # attention
        alpha = torch.matmul(key, query)  # (B, T*H*W, T*H*W)

        # normalize over timesteps
        alpha = alpha / float(T)

        # use softmax or sigmoid
        if self.is_softmax_activation:
            alpha = F.softmax(alpha, dim=-1)  # (B, T*H*W, T*H*W)
        else:
            alpha = alpha / alpha.size(-1)  # (B, T*H*W, T*H*W)
            alpha = F.sigmoid(alpha)  # (B, T*H*W, T*H*W)

        # multiply alpha with values
        y = torch.matmul(alpha, value)  # (B, T*H*W, C)
        y = y.permute(0, 2, 1).contiguous()  # (B, C, T*H*W)
        y = y.view(batch_size, self.n_channels_inter, T, H, W)  # (B, C, T, H, W)

        # output embedding
        y = self.output_embedding(y)

        # residual connection
        y += x

        return y

    def __apply_mask_on_alpha(self, alpha, x_shape, mask=None):

        # alpha (B, T*H*W, T*H*W)
        # mask (B, C, T, H, W)
        B, C, T, H, W = x_shape

        N = T * H * W

        if mask is None:
            return alpha

        # reshape alpha
        alpha = alpha.view(B, N, T, H, W)  # (B, T*H*W, T, H, W)
        alpha_masked = torch.mul(alpha, mask)  # (B, T*H*W, T, H, W)

        # reshape back
        alpha_masked = alpha_masked.view(B, N, N)

        return alpha_masked

class GlobalSelfAttentionMultiHead(nn.Module):
    def __init__(self, input_shape, n_heads, reduction_factor=1):
        """
        Initialize the module.
        """
        super(GlobalSelfAttentionMultiHead, self).__init__()

        C, T, H, W = input_shape
        n_channels_in = C

        assert n_channels_in % n_heads == 0

        self.n_channels_in = n_channels_in
        self.n_heads = n_heads

        n_channels_inter = int(n_channels_in / (n_heads * reduction_factor))

        # we use n heads, each has inner dim
        for idx_head in range(n_heads):
            head_num = idx_head + 1
            attention_head_name = 'attention_head_%d' % (head_num)
            attention_head = GlobalSelfAttentionHead(n_channels_in, n_channels_inter)
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

        # pool over the head dimension
        y = torch.stack(y, dim=1)  # (B, N, C, T, H, W)
        y = torch.sum(y, dim=1)  # (B, C, T, H, W)

        return y

# endregion

# region Global Self-Attention With Dimension Reduction

class GlobalSelfAttentionWithReductionHead(nn.Module):
    def __init__(self, n_channels_in, n_channels_inter):
        """
        Initialize the module.
        """
        super(GlobalSelfAttentionWithReductionHead, self).__init__()

        self.n_channels_in = n_channels_in
        self.n_channels_inter = n_channels_inter
        self.is_softmax_activation = True
        self.dropout_ratio = 0.25

        self.__define_layers()

    def __define_layers(self):

        dropout_ratio = self.dropout_ratio
        key_linear = pl.Linear3d(self.n_channels_in, self.n_channels_inter)
        query_linear = pl.Linear3d(self.n_channels_in, self.n_channels_inter)
        value_linear = pl.Linear3d(self.n_channels_in, self.n_channels_inter)
        output_linear = pl.Linear3d(self.n_channels_inter, self.n_channels_inter)

        # value embedding
        self.value_embedding = nn.Sequential(nn.Dropout(dropout_ratio), value_linear)

        # key embedding
        self.key_embedding = nn.Sequential(nn.Dropout(dropout_ratio), key_linear)

        # query embedding
        self.query_embedding = nn.Sequential(nn.Dropout(dropout_ratio), query_linear)

        # output embedding
        self.output_embedding = nn.Sequential(nn.Dropout(dropout_ratio), output_linear)

    def forward(self, x):
        """
        :param x: (B, C, T, H, W)
        :return:
        """

        batch_size = x.size(0)
        x_shape = pytorch_utils.get_shape(x)
        B, C, T, H, W = x_shape

        # key embedding
        key = self.key_embedding(x)  # (B, C, T, H, W)
        key = key.view(batch_size, self.n_channels_inter, -1)  # (B, C, T*H*W)
        key = key.permute(0, 2, 1)  # (B, T*H*W, C)

        # query embedding
        query = self.query_embedding(x)  # (B, C, T, H, W)
        query = query.view(batch_size, self.n_channels_inter, -1)  # (B, C, T*H*W)

        # value embedding
        value = self.value_embedding(x)  # (B, C, T, H, W)
        value = value.view(batch_size, self.n_channels_inter, -1)  # (B, C, T*H*W)
        value = value.permute(0, 2, 1)  # (B, T*H*W, C)

        # attention
        alpha = torch.matmul(key, query)  # (B, T*H*W, T*H*W)

        # normalize over timesteps
        alpha = alpha / float(T)

        # use softmax or sigmoid
        if self.is_softmax_activation:
            alpha = F.softmax(alpha, dim=-1)  # (B, T*H*W, T*H*W)
        else:
            alpha = alpha / alpha.size(-1)  # (B, T*H*W, T*H*W)
            alpha = F.sigmoid(alpha)  # (B, T*H*W, T*H*W)

        # multiply alpha with values
        y = torch.matmul(alpha, value)  # (B, T*H*W, C)
        y = y.permute(0, 2, 1).contiguous()  # (B, C, T*H*W)
        y = y.view(batch_size, self.n_channels_inter, T, H, W)  # (B, C, T, H, W)

        # output embedding
        y = self.output_embedding(y)

        # residual connection
        # y += x

        return y

    def __apply_mask_on_alpha(self, alpha, x_shape, mask=None):

        # alpha (B, T*H*W, T*H*W)
        # mask (B, C, T, H, W)
        B, C, T, H, W = x_shape

        N = T * H * W

        if mask is None:
            return alpha

        # reshape alpha
        alpha = alpha.view(B, N, T, H, W)  # (B, T*H*W, T, H, W)
        alpha_masked = torch.mul(alpha, mask)  # (B, T*H*W, T, H, W)

        # reshape back
        alpha_masked = alpha_masked.view(B, N, N)

        return alpha_masked

class GlobalSelfAttentionWithReductionMultiHead(nn.Module):
    def __init__(self, input_shape, n_heads, reduction_factor):
        """
        Initialize the module.
        """
        super(GlobalSelfAttentionWithReductionMultiHead, self).__init__()

        C, T, H, W = input_shape
        n_channels_in = C

        assert n_channels_in % n_heads == 0

        self.n_channels_in = n_channels_in
        self.n_heads = n_heads

        # n_channels_inter = int(n_channels_in / float(n_heads))
        n_channels_inter = int(n_channels_in / float(reduction_factor))

        # we use n heads, each has inner dim
        for idx_head in range(n_heads):
            head_num = idx_head + 1
            attention_head_name = 'attention_head_%d' % (head_num)
            attention_head = GlobalSelfAttentionWithReductionHead(n_channels_in, n_channels_inter)
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

        # pool over the head dimension
        y = torch.stack(y, dim=1)  # (B, N, C, T, H, W)
        y = torch.sum(y, dim=1)  # (B, C, T, H, W)

        return y

# endregion

# region Local Self-Attention [Sum Heads]

class LocalSelfAttentionHeadSum(nn.Module):
    """
    Original Implementation on the Self-Attention Head.
    """

    def __init__(self, n_channels_in, n_channels_inter):
        """
        Initialize the module.
        """
        super(LocalSelfAttentionHeadSum, self).__init__()

        self.n_channels_in = n_channels_in
        self.n_channels_inter = n_channels_inter
        dropout_ratio = 0.25

        key_linear = pl.Linear3d(self.n_channels_in, self.n_channels_inter)
        query_linear = pl.Linear3d(self.n_channels_in, self.n_channels_inter)
        value_linear = pl.Linear3d(self.n_channels_in, self.n_channels_inter)
        output_linear = pl.Linear3d(self.n_channels_inter, self.n_channels_in)

        # key embedding
        self.key_embedding = nn.Sequential(nn.Dropout(dropout_ratio), key_linear)

        # query embedding
        self.query_embedding = nn.Sequential(nn.Dropout(dropout_ratio), query_linear)

        # value embedding
        self.value_embedding = nn.Sequential(nn.Dropout(dropout_ratio), value_linear)

        # output embedding
        self.output_embedding = nn.Sequential(nn.Dropout(dropout_ratio), output_linear)

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
        query = self.query_embedding(x_item)  # (B, C, 1, H, W)
        query = query.view(batch_size, self.n_channels_inter, -1)  # (B, C, 1*H*W)

        # key embedding
        key = self.key_embedding(x_window)  # (B, C, T, H, W)
        key = key.view(batch_size, self.n_channels_inter, -1)  # (B, C, T*H*W)
        key = key.permute(0, 2, 1)  # (B, T*H*W, C)

        # value embedding
        value = self.value_embedding(x_window)  # (B, C, T, H, W)
        value = value.view(batch_size, self.n_channels_inter, -1)  # (B, C, T*H*W)
        value = value.permute(0, 2, 1)  # (B, T*H*W, C)

        # attention
        alpha = torch.matmul(key, query)  # (B, T*H*W, 1*H*W)
        alpha = alpha.permute(0, 2, 1)  # (B, 1*H*W, T*H*W)
        alpha = F.softmax(alpha, dim=-1)  # (B, 1*H*W, T*H*W)

        # scale over channels or over the timesteps
        # alpha = alpha / np.sqrt(self.n_channels_inter)  # (B, 1*H*W, T*H*W)
        # alpha = alpha / alpha.size(-1)  # (B, 1*H*W, T*H*W)

        # use sigmoid instead of softmax
        # alpha = F.sigmoid(alpha)  # (B, 1*H*W, T*H*W)

        # multiply alpha with values
        y = torch.matmul(alpha, value)  # (B, 1*H*W, C)
        y = y.permute(0, 2, 1).contiguous()  # (B, C, 1*H*W)
        y = y.view(batch_size, self.n_channels_inter, 1, H, W)  # (B, C, 1, H, W)

        # output embedding
        y = self.output_embedding(y)

        return y

class LocalSelfAttentionMultiHeadSum(nn.Module):
    """
    MultiHead for Self-Attention.
    """

    def __init__(self, input_shape, window_size, n_heads):
        """
        Initialize the module.
        """
        super(LocalSelfAttentionMultiHeadSum, self).__init__()

        C, T, H, W = input_shape
        n_channels_in = C

        assert n_channels_in % n_heads == 0

        self.n_channels_in = n_channels_in
        self.n_heads = n_heads
        self.window_size = window_size

        n_channels_inter = int(n_channels_in / n_heads)

        self.input_bn = nn.BatchNorm3d(n_channels_in)
        self.input_padding = pl.Pad3d((window_size, 1, 1), T, H, W)

        # we use n heads, each has inner dim
        for idx_head in range(n_heads):
            head_num = idx_head + 1
            attention_head_name = 'attention_head_%d' % (head_num)
            attention_head = LocalSelfAttentionHeadSum(n_channels_in, n_channels_inter)
            setattr(self, attention_head_name, attention_head)

    def forward(self, x):
        """
        :param x: (B, C, T, H, W)
        :return:
        """

        K = self.window_size

        # requires batchnorm for input
        x = self.input_bn(x)  # (None, C, T, H, W)

        # padd the input
        x_padded = self.input_padding(x)  # (None, C, T, H, W)

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

# endregion

# region Local Self-Attention [Concat Heads]

class LocalSelfAttentionHeadConcat(nn.Module):
    """
    Original Implementation on the Self-Attention Head.
    """

    def __init__(self, n_channels_in, n_channels_inter):
        """
        Initialize the module.
        """
        super(LocalSelfAttentionHeadConcat, self).__init__()

        self.n_channels_in = n_channels_in
        self.n_channels_inter = n_channels_inter
        dropout_ratio = 0.25

        query_linear = pl.Linear3d(self.n_channels_in, self.n_channels_inter)
        key_linear = pl.Linear3d(self.n_channels_in, self.n_channels_inter)
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

        B, C, T, H, W = pytorch_utils.get_shape(x_window)
        batch_size = x_window.size(0)
        assert T % 2 == 1

        # get middle item of the window
        idx_item = int(T / 2.0)
        x_item = x_window[:, :, idx_item:idx_item + 1]  # (B, C, 1, H, W)

        # query embedding
        query = self.query_embedding(x_item)  # (B, C, 1, H, W)
        query = query.view(batch_size, self.n_channels_inter, -1)  # (B, C, 1*H*W)

        # key embedding
        key = self.key_embedding(x_window)  # (B, C, T, H, W)
        key = key.view(batch_size, self.n_channels_inter, -1)  # (B, C, T*H*W)
        key = key.permute(0, 2, 1)  # (B, T*H*W, C)

        # value embedding
        value = self.value_embedding(x_window)  # (B, C, T, H, W)
        value = value.view(batch_size, self.n_channels_inter, -1)  # (B, C, T*H*W)
        value = value.permute(0, 2, 1)  # (B, T*H*W, C)

        # attention
        alpha = torch.matmul(key, query)  # (B, T*H*W, 1*H*W)
        alpha = alpha.permute(0, 2, 1)  # (B, 1*H*W, T*H*W)

        # scale and softmax
        # alpha = alpha / np.sqrt(self.n_channels_inter)  # (B, 1*H*W, T*H*W)
        alpha = F.softmax(alpha, dim=-1)  # (B, 1*H*W, T*H*W)

        # scale and sigmoid
        # alpha = alpha / alpha.size(-1)  # (B, 1*H*W, T*H*W)
        # alpha = F.sigmoid(alpha)  # (B, 1*H*W, T*H*W)

        # multiply alpha with values
        y = torch.matmul(alpha, value)  # (B, 1*H*W, C)
        y = y.permute(0, 2, 1).contiguous()  # (B, C, 1*H*W)
        y = y.view(batch_size, self.n_channels_inter, H, W)  # (B, C, H, W)

        return y

class LocalSelfAttentionMultiHeadConcat(nn.Module):
    """
    MultiHead for Self-Attention.
    """

    def __init__(self, input_shape, window_size, n_heads):
        """
        Initialize the module.
        """
        super(LocalSelfAttentionMultiHeadConcat, self).__init__()

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
            attention_head = LocalSelfAttentionHeadConcat(n_channels_in, n_channels_inter)
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

# endregion
