#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import cv2
import importlib

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.models as models
import torch.nn.functional as F
import torchsummary
from torch import optim
from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1

from modules import layers_pytorch as pl

from core import utils, pytorch_utils, image_utils
from core.utils import Path as Pth
from core import const as c

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

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
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
        layers.append(block(inplanes = self.inplanes, planes= planes, stride=stride, downsample = downsample, groups= self.groups, base_width = self.base_width, dilation = previous_dilation, norm_layer = norm_layer))
        
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

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

        return x2

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



class ResiNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResiNet, self).__init__()
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
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
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
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(state_dict_path=None, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    if state_dict_path is not None:
        pytorch_utils.load_model_dict(model, state_dict_path)

    return model

def resnet34(state_dict_path=None, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    if state_dict_path is not None:
        pytorch_utils.load_model_dict(model, state_dict_path)

    return model

def resnet50(state_dict_path=None, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    if state_dict_path is not None:
        pytorch_utils.load_model_dict(model, state_dict_path)

    return model

def test_resnet():
    # testing these networks
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # load model and weights
    model_type = c.CNN_MODEL_TYPES.resnet50
    model_path = Pth('Torch_Models/ResNet/resnet50-19c8e357.pth')
    model = resnet50()

    class_names_path = Pth('ImageNet/class_names.txt')
    test_img1 = '/local/mnt/workspace/Pictures/test_img_car.jpg'
    test_img2 = '/local/mnt/workspace/Pictures/test_img_cat.jpg'
    test_img3 = '/local/mnt/workspace/Pictures/test_img_stove.jpg'
    test_img4 = '/local/mnt/workspace/Pictures/test_img_dog.jpg'

    test_imgs = [test_img1, test_img2, test_img3, test_img4]

    class_names = utils.txt_load(class_names_path)
    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict, strict=True)

    # flag the model as testing only
    model = model.cuda()
    model.eval()
    model.training = False

    # print summary
    input_size = (3, 224, 224)  # (B, C, H, W)
    torchsummary.summary(model, input_size)

    for test_img in test_imgs:
        # load test imag, and pre-process it
        img = cv2.imread(test_img)
        img = img[:, :, (2, 1, 0)]
        img = image_utils.resize_crop(img)
        img = img.astype(np.float32)

        # normalize image
        img /= 255.0
        img[:, :] -= mean
        img[:, :] /= std

        print(np.min(img))
        print(np.max(img))

        print(img.shape)
        img = np.transpose(img, (2, 0, 1))
        input = np.expand_dims(img, axis=0)
        print(input.shape)

        input = torch.from_numpy(input).cuda()
        predictions = model(input)
        predictions = F.softmax(predictions)
        predictions = predictions.tolist()
        predictions = np.array(predictions)
        predictions *= 100

        print(np.min(predictions))
        print(np.max(predictions))
        predictions = predictions[0]

        idx = np.argsort(predictions)[::-1][:5]
        class_name = ' # '.join([class_names[i] for i in idx])
        prob = ' # '.join(['%.02f' % predictions[i] for i in idx])
        print('#########################')
        print('')
        print(test_img)
        print(class_name)
        print(prob)
        print('')

def __get_resne50_for_finetuning_on_hico():
    # load model and weights
    # model_path = Pth('Torch_Models/ResNet/resnet50-19c8e357.pth')
    # model.fc = nn.Linear(2048, 10)

    model_path = Pth('Torch_Models/ResNet/resnet50-19c8e357.pth')
    model_dict = torch.load(model_path)

    # define model
    model = ResNet50Hico()

    # load weights
    model.load_state_dict(model_dict, strict=True)

    # freeze all but last block
    layer_names = ['bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']
    pytorch_utils.freeze_model_layers_recursive(model, layer_names)

    # prepare for fine-tuning
    model._prepare_for_finetuning()

    # as cuda
    model = model.cuda()

    loss_fn = F.binary_cross_entropy
    metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
    optimizer = optim.Adam(model.parameters(), lr=0.001, eps=1e-4)

    return model, optimizer, loss_fn, metric_fn

def __get_resnet18_for_feature_extraction():

    model_path = 'resnet18.pth'
    model_dict = torch.load(model_path)

    print('Constructing the model')

    model = resnet18()

    #model = resnet50()

    #model = ResNet50Hico(num_classes=10)

    #model = ResiNet(BasicBlock, [2, 2, 2, 2])

    print('Loading the model')

    # load weights
    model.load_state_dict(model_dict, strict=True)
    model = model.cuda()
    model.training = False
    model.eval()

    return model

def __get_resne50_breakfast_for_testing():
    # load model and weights
    model_path = Pth('Breakfast/models_featurenet/resnet50_breakfast_1.1/025.pt')
    model_dict = torch.load(model_path)

    # define model
    model = ResNet50Hico(num_classes=10)

    # load weights
    model.load_state_dict(model_dict, strict=True)
    model = model.cuda()
    model.training = False
    model.eval()

    return model
