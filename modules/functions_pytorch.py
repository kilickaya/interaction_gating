#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch

from core import pytorch_utils

class Threshold(torch.autograd.Function):
    THRESHOLD_VALUE = 0.5

    @staticmethod
    def forward(ctx, input):
        """
        Zero values below threshold.
        :param ctx:
        :param input:
        :return:
        """

        threshold_value = Threshold.THRESHOLD_VALUE

        y_hard = input.clone()
        idx = input < threshold_value
        ctx.idx = idx
        y_hard[idx] = 0.0

        return y_hard

    @staticmethod
    def backward(ctx, grad_input):
        """
        Zero gradients of thresholded neurons.
        :param ctx:
        :param grad_input:
        :return:
        """

        grad_output = grad_input.clone()
        idx = ctx.idx
        grad_output[idx] = 0.0
        return grad_output, None

class BinaryThreshold(torch.autograd.Function):
    THRESHOLD_VALUE = 0.5

    @staticmethod
    def forward(ctx, input):
        """
        Zero values below threshold.
        :param ctx:
        :param input:
        :return:
        """

        threshold_value = Threshold.THRESHOLD_VALUE

        y_hard = torch.ones_like(input)
        idx = input < threshold_value
        ctx.idx = idx
        y_hard[idx] = 0.0

        return y_hard

    @staticmethod
    def backward(ctx, grad_input):
        """
        Zero gradients of thresholded neurons, i.e. below threshold.
        :param ctx:
        :param grad_input:
        :return:
        """

        grad_output = grad_input.clone()
        idx = ctx.idx
        grad_output[idx] = 0.0
        return grad_output, None

class Hardmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """
        # input shape (B, T). Hardmax on the node dimension (dim=1)
        """

        input_shape = pytorch_utils.get_shape(input)
        B, T = input_shape
        rng = torch.arange(B)

        # find idx of max
        idx = torch.argmax(input, dim=1)

        # set all but max to zero, set max to 1
        mask = torch.zeros_like(input)  # (B, T)
        mask[rng, idx] = 1.0

        # save for backward pass
        ctx.mask = mask

        output = input.clone()  # copy input
        output = output * mask  # (B, T)

        return output

    @staticmethod
    def backward(ctx, grad_input):
        # just pass the gradients of positive values

        # copy gradients
        # grad_output = grad_input
        grad_output = grad_input.clone()

        # multiply by mask
        mask = ctx.mask
        grad_output = grad_output * mask

        return grad_output, None
