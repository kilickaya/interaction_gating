#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
PlacesCNN for scene classification
by Bolei Zhou
last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)
"""

import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
from torch import optim
from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1
import torchsummary

from core import utils, image_utils, pytorch_utils
from core.utils import Path as Pth

# region Const

RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]

# endregion

class ResNet50Hico(nn.Module):

    def __init__(self, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet50Hico, self).__init__()

        block = Bottleneck
        layers = [3, 4, 6, 3]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _prepare_for_finetuning(self):
        self.fc = nn.Linear(2048, 600)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        x = F.sigmoid(x)

        return x

    def forward_no_activation(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x

    def forward_till_conv3(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

    def forward_till_conv4(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def extract_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x1 = self.layer4(x)

        x2 = self.avgpool(x1)

        return x1, x2

def test_model_predictions_on_images():
    weight_path = Pth('Torch_Models/ResNet/resnet50_places365.pth.tar')
    category_list_path = Pth('Places365/annotation/categories_places365.txt')

    # load the class label
    category_list = utils.txt_load(category_list_path)

    # load the pre-trained weights
    model = __load_model_pretrained(weight_path)
    model = model.cuda()
    model.eval()

    image_names = ['01.jpg', '02.jpg', '03.jpg', '12.jpg']
    for image_name in image_names:
        image_path = '/home/nour/Pictures/scene_images/%s' % image_name

        img = __read_image_preprocessed(image_path)
        img = torch.from_numpy(np.array([img])).cuda()

        # forward pass
        logit = model.forward_no_activation(img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        print('\n prediction on {}'.format(image_name, ))
        # output the prediction
        for i in range(0, 5):
            print('{:.3f} -> {}'.format(probs[i], category_list[idx[i]]))

def __get_resne50_for_finetuning_on_hico():
    weight_path = Pth('Torch_Models/ResNet/resnet50_places365.pth.tar')

    # load the pre-trained weights
    model = __load_model_pretrained(weight_path)

    # freeze all but last block
    layer_names = ['bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']
    pytorch_utils.freeze_model_layers_recursive(model, layer_names)

    # prepare model for fine-tuning
    model._prepare_for_finetuning()
    model = model.cuda()

    loss_fn = F.binary_cross_entropy
    metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
    optimizer = optim.Adam(model.parameters(), lr=0.001, eps=1e-4)

    return model, optimizer, loss_fn, metric_fn

def __load_model_pretrained(weight_path):
    # load the pre-trained weights
    model = ResNet50Hico(num_classes=365)
    checkpoint = torch.load(weight_path, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    return model

def __read_image_preprocessed(image_path):
    # read image as rgb, channel first, cropped, and transformed
    img = cv2.imread(image_path)
    img = img[:, :, (2, 1, 0)]
    img = image_utils.resize_crop(img)
    img = img.astype(np.float32) / 255.0
    img[:, :] -= RGB_MEAN
    img[:, :] /= RGB_STD
    img = np.transpose(img, (2, 0, 1))
    return img
