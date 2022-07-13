#!/usr/bin/env python
# coding: utf-8

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import math
import sklearn.preprocessing as sk
import seaborn as sns
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import random
from random import randint
from sklearn.model_selection import StratifiedKFold

import tqdm
import os
from captum.attr import IntegratedGradients
import pickle

from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW, Adam, Optimizer

def get_binary(x):
    if x=='R': return 0
    if x=='S': return 1

def optimal_params(_drug, _type, args):
    if _drug.lower() == 'docetaxel': args.drop_conv, args.drop_fc1, args.drop_fc2, args.lr, args.layers, args.n_epoch, args.DB = 0.4, 0.65, 0.65, 0.000001, [16], 31, 'TCGA'
    if _drug.lower() == 'cisplatin': args.drop_conv, args.drop_fc1, args.drop_fc2, args.lr, args.layers, args.n_epoch, args.DB = 0.3, 0.15, 0.15, 0.00001,[8,8,8], 61, 'TCGA'

    if _drug.lower() == 'gemcitabine': 
        if _type =='TCGA':          args.drop_conv, args.drop_fc1, args.drop_fc2, args.lr, args.layers, args.n_epoch = 0.2, 0.5, 0.5, 0.00001, [16],      31
        else:                       args.drop_conv, args.drop_fc1, args.drop_fc2, args.lr, args.layers, args.n_epoch = 0.1, 0.2, 0.2, 0.00001, [16],     151

    if _drug.lower() == 'paclitaxel': args.drop_conv, args.drop_fc1, args.drop_fc2, args.lr, args.layers, args.n_epoch, args.DB = 0.2, 0.5, 0.5, 0.0001, [32],    31, 'PDX'
    if _drug.lower() == 'cetuximab': args.drop_conv, args.drop_fc1, args.drop_fc2, args.lr, args.layers, args.n_epoch, args.DB = 0.2, 0.65, 0.65, 0.00001,[8, 16], 81, 'PDX'
    if _drug.lower() == 'erlotinib': args.drop_conv, args.drop_fc1, args.drop_fc2, args.lr, args.layers, args.n_epoch, args.DB = 0.5, 0.5, 0.5, 0.00005, [16],    151, 'PDX'


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, 
    num_cycles: float = 0.5, last_epoch: int = -1, min_lr: float = 1e-7
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_act(ACT_func):
    if str(ACT_func.lower())   == 'leakyrelu': return nn.LeakyReLU()
    elif str(ACT_func.lower()) == 'elu' : return nn.ELU()
    elif str(ACT_func.lower()) == 'silu': return nn.SiLU()
    elif str(ACT_func.lower()) == 'celu': return nn.CELU()
    elif str(ACT_func.lower()) == 'relu': return nn.ReLU()
    elif str(ACT_func.lower()) == 'shrink': return nn.Softshrink()
    elif str(ACT_func.lower()) == 'rrelu': return nn.RReLU()
    elif str(ACT_func.lower()) == 'prelu': return nn.PReLU()
    else: return None

class build_cnn(nn.Module):
    def __init__(self, args):
        super(build_cnn, self).__init__()
        dim_in, hidden_dims, ACT_func, drop2d = args.n_channel, args.layers, args.ACT_func, args.drop_conv
        layers = []
        n_layers = len(hidden_dims)+1
        hidden_dims = [dim_in]+hidden_dims+[1]
        ACT_func = get_act(ACT_func)

        for idx, i in enumerate( range(n_layers)):
            layers += [nn.Conv2d( hidden_dims[i], hidden_dims[i+1], kernel_size=[1,1], stride=1) ]
            layers += [ACT_func]
            if drop2d>0:
                layers += [nn.Dropout(p = drop2d)]
        layers = layers[:-1] # delete last dropout
        self.model_type = args.model_type
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

def proj_model(args):
    if args.model_type == 'conv':
        proj_model = build_cnn(args)
    elif args.model_type == 'flat':
        proj_model = build_flat(args)
    return proj_model

class build_flat(nn.Module):
    def __init__(self, args):
        super(build_flat, self).__init__()
        self.dim_in = args.n_genes*args.n_channel
        self.dim_out = 256
        self.layers = nn.ModuleList([ nn.Linear(self.dim_in, self.dim_out),
                                      get_act(args.ACT_func)
                                    ])
        self.model_type = args.model_type

    def forward(self, x):
        x = torch.flatten(x, 1)
        for l in self.layers:
            x = l(x)
        return x


class CNN_model(nn.Module):
    def __init__(self, proj_model, n_genes, drop_fc1, drop_fc2):
        super(CNN_model, self).__init__()        
        self.proj_model = proj_model
        self.fc1 = nn.Linear(n_genes, 256)
        self.fc2 = nn.Linear(256, 256*3)
        self.fc_last = nn.Linear(256*3, 1)
        self.drop1 = nn.Dropout(drop_fc1)
        self.drop2 = nn.Dropout(drop_fc2)

    def forward(self, x):
        x = self.proj_model(x)

        x = torch.flatten(x, 1) if not self.proj_model.model_type=='flat' else x

        x = self.drop1(x)
        x = self.fc1(x)
        x = self.drop2(x)
        x = self.fc2(x)
        x = self.fc_last(x)

        return torch.sigmoid(x)

def get_3d( e, m, c, single_omics):
    feat_list = [e, m, c ] if not single_omics else [e, np.zeros_like(m), np.zeros_like(c)]
    concat_list = []
    for _feat in feat_list:
        concat_list.append( _feat[None, :,:])

    np_cell = np.concatenate(concat_list)
    np_cell = np.swapaxes(np_cell, 0, 1)
    np_cell = np_cell[:,:, None, :]
    return np_cell

def get_ig(Clas, dataloader, n_epoch):
    ig_list = []
    ig = IntegratedGradients(Clas)
    for data in dataloader:
#         X, y, cell_names = data[0], data[1], data[2]
        X, y = data[0], data[1]
        n_bsz = data[0].shape[0]
        if n_bsz==1:
            continue

        att_list = ig.attribute(X, 
                              # target = y.long(),
                              internal_batch_size=n_bsz,
                              n_steps = n_epoch,
                              return_convergence_delta=False
                              ).squeeze().cpu().detach().numpy()
        ig_list.append( att_list)
    return np.vstack(ig_list)

def get_ig_test(Clas, X, n_epoch):
    ig_list = []
    ig = IntegratedGradients(Clas)
#     for data in dataloader:
#         X, y, cell_names = data[0], data[1], data[2]
#         X, y = data[0], data[1]
    n_bsz = X.shape[0]
#     if n_bsz==1:
#         continue

    att_list = ig.attribute(X, 
                          # target = y.long(),
                          internal_batch_size=n_bsz,
                          n_steps = n_epoch,
                          return_convergence_delta=False
                          ).squeeze().cpu().detach().numpy()
#     ig_list.append( att_list)
#     return np.vstack(ig_list)
    return att_list

























