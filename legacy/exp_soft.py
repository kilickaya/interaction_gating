

import sys
import time
import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics as sk_metrics

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torch.nn.modules
import torch.nn.functional as F
import torch.optim as optim
import torchsummary
import torchviz
from torch.autograd import Variable

from modules import node_attention, self_attention, context_fusion
from modules import layers_pytorch as pl
from modules import functions_pytorch as pf

from core import const as c
from core import utils, image_utils, plot_utils, configs, data_utils, pytorch_utils
from core.utils import Obj, Path as Pth

#from datasets import ds_breakfast
from nets import resnet_torch

N_CLASSES = 600

def train_human_object_multiple_context_gating(soft_flag = True, backbone = 'rcnn'):

    n_epochs = 100
    batch_size_tr = 32
    batch_size_te = 32
    n_classes = N_CLASSES

    if backbone == 'rcnn':
        print('Using backbone rcnn')
        feature_path_interaction = Pth('Hico/features/h5/features_base_subject_object.h5')
        n_channels, n_regions, channel_side_dim = 4096, 12,1
        (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])
        x_tr = np.swapaxes(x_tr, 1,2)
        x_te = np.swapaxes(x_te, 1,2)
    elif backbone == 'pairatt':
        print('Using backbone pairatt')
        feature_path_interaction = Pth('Hico/features/h5/features_pairattn.h5')
        n_channels, n_regions, channel_side_dim = 4096, 3,1
        (x_tr, x_te) = utils.h5_load_multi(feature_path_interaction, ['x_tr', 'x_te'])

    # Features of the pose: f_context
    feature_path_c3= Pth('Hico/features/h5/deformation.h5')
    x_cs_shape = [(512, 1, 1, 1)]

    # Features of the pose: f_context
    feature_path_c1 = Pth('Hico/features/h5/lvis.h5')
    x_cs_shape = [(1300, 1, 1, 1)]

    feature_path_c2 = Pth('Hico/features/h5/local_scene.h5')
    x_cs_shape = [(2048, 1, 1, 1)]

    feature_path_context = Pth('Hico/features/h5/stuff.h5')
    x_cs_shape = [(649, 1, 1, 1)]

    # Features of the pose: f_context
    feature_path_context = Pth('Hico/features/h5/part_states.h5')
    x_cs_shape = [(1032, 1, 1, 1)]

    feature_path_c4 = Pth('Hico/features/h5/local_pose.h5')
    x_cs_shape = [(4096, 1, 1, 1)]

    x_cs_shape = [(1300, 1, 1, 1), (2048, 1, 1, 1), (512, 1, 1, 1), (4096, 1, 1, 1)]

    # Annotation of the image
    annot_path = Pth('Hico/features/h5/anno_hico.pkl')
    model_name = 'classifier_%s' % (utils.timestamp())
    input_shape = (n_channels, n_regions, channel_side_dim, channel_side_dim)

    print('--- start time')
    print(datetime.datetime.now())

    print('... loading data')
    t1 = time.time()

    (img_names_tr, y_tr, y_tr_mask, img_names_te, y_te, y_te_mask) = utils.pkl_load(annot_path)
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)

    y_tr_mask = y_tr_mask.astype(np.float32)
    y_te_mask = y_te_mask.astype(np.float32)

    print('... context features')
    (x_tr_c1, x_te_c1) = utils.h5_load_multi(feature_path_c1, ['x_tr', 'x_te'])
    #x_tr_c1 = expand_feats(x_tr_c1)
    #x_te_c1 = expand_feats(x_te_c1)

    (x_tr_c2, x_te_c2) = utils.h5_load_multi(feature_path_c2, ['x_tr', 'x_te'])
    x_tr_c2 = expand_feats(x_tr_c2)
    x_te_c2 = expand_feats(x_te_c2)

    (x_tr_c3, x_te_c3) = utils.h5_load_multi(feature_path_c3, ['x_tr', 'x_te'])
    x_tr_c3 = expand_feats(x_tr_c3)
    x_te_c3 = expand_feats(x_te_c3)

    (x_tr_c4, x_te_c4) = utils.h5_load_multi(feature_path_c4, ['x_tr', 'x_te'])
    x_tr_c4 = expand_feats(x_tr_c4)
    x_te_c4 = expand_feats(x_te_c4)

    print('train_set_shape_interaction: ', x_tr.shape)
    print('test_set_shape_interaction: ', x_te.shape)

    print('train_set_shape_context-1: ', x_tr_c1.shape)
    print('test_set_shape_context-1: ',  x_te_c1.shape)

    print('train_set_shape_context-2: ', x_tr_c2.shape)
    print('test_set_shape_context-2: ',  x_te_c2.shape)

    print('train_set_shape_context-3: ', x_tr_c3.shape)
    print('test_set_shape_context-3: ',  x_te_c3.shape)

    print('train_set_shape_context-4: ', x_tr_c4.shape)
    print('test_set_shape_context-4: ',  x_te_c4.shape)

    t2 = time.time()
    duration = t2 - t1
    print('... loading data, duration (sec): %d' % (duration))

    # building the model
    print('... building model %s' % (model_name))
    t1 = time.time()
    if soft_flag == True:
        print('Training soft fusion model')
        model = ClassifierContextLateFusionMultiSoftGate(n_classes, input_shape, x_cs_shape) 


    t2 = time.time()
    duration = t2 - t1
    model = model.cuda()
    input_sizes = [input_shape] + list(x_cs_shape)
    #pytorch_utils.model_summary_multi_input(model, input_sizes=input_sizes, batch_size=-1, device='cuda')    
    print('... model built, duration (sec): %d' % (duration))

    # callbacks
    callbacks = []

    print('Interaction_feat: %s, Context_feat-1: %s, Context_feat-2: %s, Context_feat-3: %s\n' %(feature_path_interaction, feature_path_c1, feature_path_c2, feature_path_c3))

    # start training
    pytorch_utils.train_model_custom_metric_mask(model, model._optimizer, model._loss_fn, model._metric_fn, [x_tr, x_tr_c1,x_tr_c2, x_tr_c3, x_tr_c4], y_tr, y_tr_mask, [x_te, x_te_c1, x_te_c2, x_te_c3, x_te_c4], y_te, y_te_mask, n_epochs, batch_size_tr, batch_size_te, callbacks=callbacks)

    print('--- finish time')
    print(datetime.datetime.now())



class ClassifierContextLateFusionMultiSoftGate(nn.Module):
    def __init__(self, n_classes,  x_so_shape, x_cs_shape):
        super(ClassifierContextLateFusionMultiSoftGate, self).__init__()

        self.n_contexts = len(x_cs_shape)
        self.layer_name_context_emb   = 'dense_context_%d'
        self.layer_name_context_class = 'class_context_%d'
        self.layer_name_context_selection =   'imp_context'

        self.__init_layers(n_classes, x_so_shape, x_cs_shape)
        self.__init_optimizer()

    def __init_layers(self, n_classes, x_so_shape, x_cs_shape):
        """
        Define model layers.
        """

        self.n_classes = 600

        n_units = 600
        n_channels = 512

        self.n_channels = n_channels

        C_so, N, H, W = x_so_shape
        self.C_so = C_so
        self.N = N

        self.feature_selection = context_fusion.ContextGatingClassifierSoft(x_so_shape, x_cs_shape)

        self.softmax = nn.Softmax(dim = 0)

        # Map so features to a smaller size
        self.dense_so = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_so, n_channels), nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))

        # Loop over existing context features: Map them into interaction categories
        for idx_context in range(self.n_contexts):
            C_c = x_cs_shape[idx_context][0]

            # embedding of multi_ context
            layer_name = self.layer_name_context_emb % (idx_context + 1)
            layer = nn.Sequential(nn.Dropout(0.25), pl.Linear3d(C_c, n_channels), nn.BatchNorm3d(n_channels), nn.LeakyReLU(0.2))

            setattr(self, layer_name, layer)

        # spatial pooling
        self.spatial_pooling = pl.Max(dim=(3, 4))

        # layers for classification
        classifier_layers = []
        classifier_layers.append(nn.Dropout(0.25))
        classifier_layers.append(nn.Linear(2*n_channels, n_channels))
        classifier_layers.append(nn.BatchNorm1d(n_channels))
        classifier_layers.append(nn.LeakyReLU(0.2))
        classifier_layers.append(nn.Linear(n_channels, n_units))
        self.classifier_layers = nn.Sequential(*classifier_layers)

    def __init_optimizer(self):
        """
        Define loss, metric and optimizer.
        """
        self._loss_fn = F.binary_cross_entropy
        self._metric_fn = pytorch_utils.METRIC_FUNCTIONS.ap_hico
        #self._optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-4)
        self._optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9)

    def get_context_embeddings(self, x_cs, B):

        x_cs_embed = []

        # loop on multi_contexts
        for idx_context in range(self.n_contexts):
            # embedding of context
            x_c = x_cs[idx_context] # (B, C, 1,1,1)
            x_c = x_c.repeat(1, 1, self.N,1,1) # (B, C, N, 1, 1)

            layer = getattr(self, self.layer_name_context_emb % (idx_context + 1))
            x_c = layer(x_c)

            # append to list of context embeddings
            x_cs_embed.append(x_c.view(1, B, self.n_channels, self.N)) # (n_context, B, C, N)

        # process context features to get context embedding from x_cs features
        x_cs_embed = torch.stack(x_cs_embed, dim=0).view(-1, B, self.n_channels, self.N) # (n_context, B, C, N)
        return x_cs_embed

    def get_context_class(self, x_cs, x_so, B):

        x_cs_class = []

        # loop on multi_contexts
        for idx_context in range(self.n_contexts):
            # embedding of context
            x_c = x_cs[idx_context] # (B, C, N)

            x_c = x_c.permute(0, 2, 1)  # (B, N, C)

            # hide N dimension
            B, N, C = pytorch_utils.get_shape(x_c)
            x_c = x_c.contiguous().view(B * N, C)  # (B*N, C)

            x_c = torch.cat((x_so, x_c), dim=1)

            layer = self.classifier_layers
            x_c = layer(x_c)

            _, C = pytorch_utils.get_shape(x_c)
            x_c = x_c.view(B, N, C)  # (B, N, C)

            # append to list of context class predictions
            x_cs_class.append(x_c.view(1, B, self.N, self.n_classes)) # (1, B,N, C)

        # Process context features to get context category from x_cs features
        x_cs_class = torch.stack(x_cs_class, dim=0).view(-1, B, self.N, self.n_classes) # (n_context, B, N, C)
        return x_cs_class

    def get_context_relevance(self, x_so, x_cs):

        x_cs_value = []
        B, C, N, _,_ = pytorch_utils.get_shape(x_so)

        # loop on multi_contexts
        for idx_context in range(self.n_contexts):
            # embedding of context
            x_c = x_cs[idx_context]
            x_c = x_c.view(B,C, N,1,1) 

            x_c = self.feature_selection(x_so, x_c) # (B, N)
            x_cs_value.append(x_c.view(1, B, N)) # (1, B, C)

        x_cs_value = torch.stack(x_cs_value, dim=0).view(self.n_contexts, B, N) # (num_context, B, N)
        return x_cs_value

    def modulate_context_classifier(self, x_so, x_cs, x_cs_classes, B):

        # return context importance per-category
        x_cs_relevance = self.get_context_relevance(x_so, x_cs) # (num_context, B, N)

        # Reweigh class predictions with activated relevance scores
        x_cs_relevance = x_cs_relevance.view(self.n_contexts, B, self.N, 1) # 

        #x_cs_relevance = self.softmax(x_cs_relevance) # which context to use? over dim=0
        x_cs_relevance = torch.sigmoid(x_cs_relevance)

        # Modulate context classifiers with relevance scores
        x_cs_classes = x_cs_classes * x_cs_relevance # (nco, B, 12, 600)
        x_cs = torch.sum(x_cs_classes,dim=0) # (B, N, 600)

        return x_cs, x_cs_relevance

    def forward(self, *input):
        """
        input is two features: subject-object feature and context feature
        :param x_so: pairattn feature (B, C, N, H, W)
        :param x_c: scene feature (B, C, N, H, W)
        :return:
        """

        # return x_so embeddings
        x_so = input[0]
        x_so = self.dense_so(x_so)

        B, C, N, _,_ = pytorch_utils.get_shape(x_so)

        x_cs = input[1:]

        # return context embeddings
        x_c = self.get_context_embeddings(x_cs, B)

        x = x_so
        # spatial pooling
        x = self.spatial_pooling(x)  # (B, C, N)
        x = x.permute(0, 2, 1)  # (B, N, C)

        # hide N dimension
        B, N, C = pytorch_utils.get_shape(x)
        x_action = x.contiguous().view(B * N, C)  # (B*N, C)

        # return context categories
        x_cs_classes = self.get_context_class(x_c, x_action, B) # (nco, B, N, C)

        x, _  = self.modulate_context_classifier(x_so, x_c, x_cs_classes, B) # (B, N, 600)
 
        # Add modulated response to human-object classifier and max-pool over N

        x,_ = torch.max(x, dim=1)  # (B, C)
        x = torch.sigmoid(x)

        return x 




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
