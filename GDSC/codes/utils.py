#!/usr/bin/env python
# coding: utf-8

import os
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW, Adam, Optimizer
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import tqdm
import math

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

import argparse
import pickle
import random
import sys
from captum.attr import IntegratedGradients

def _normalize_attr( attr, outlier_perc, has_channel):
    attr_combined = np.sum(attr, axis=2) if has_channel else attr

    attr_combined = (attr_combined > 0) * attr_combined
    threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
    return _normalize_scale(attr_combined, threshold)

def _normalize_scale(attr, scale_factor):
    assert scale_factor != 0, "Cannot normalize by scale factor = 0"
    attr_norm = attr / scale_factor
    return np.clip(attr_norm, -1, 1)

def _cumulative_sum_threshold(values, percentile):
    # given values should be non-negative
    assert percentile >= 0 and percentile <= 100, (
        "Percentile for thresholding must be " "between 0 and 100 inclusive."
    )
    sorted_vals = np.sort(values.flatten())
    cum_sums = np.cumsum(sorted_vals)
    threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
    return sorted_vals[threshold_id]

def cell_list(args):
    cell_list = [ args.rma, args.cnv, args.mut]
    name_list = ['RMA', 'CNV',  'MUT']
    cell_name = np.array( name_list)[cell_list]
    ret = '-'.join(cell_name)
    args.n_feat = sum(cell_list)
    return ret


def idx_sort(path):
    df = pd.read_csv(path, index_col=0)
    idx = df.index.values.tolist()
    idx.sort()
    return idx

def get_raw():
    idx_depmapID  = idx_sort('../data/cell_line_info.csv')
    return pd.read_csv('../data/ic50bi.csv', index_col=0).loc[idx_depmapID]

def get_cv(tr_te, i_drug_y):
    tr, te = tr_te
    return i_drug_y.iloc[tr].index, i_drug_y.iloc[te].index

def get_cell(cell_feat, args):
    idx_depmapID  = idx_sort('../data/cell_line_info.csv')
    
    name_list = ['RMA', 'CNV', 'MUT']
    path_cell= '../data/{}_631x14070.csv'

    dict_feats = {_name:path_cell.format(_name) for _name in name_list}
    cell_tables, path_list =[], []

    if len(cell_feat.split('-'))>0:
        for _feat in cell_feat.split('-'):
            path_list.append(dict_feats[_feat])

    # Load CSV Tables
    for i, _path in enumerate(path_list):
        df = pd.read_csv(_path, index_col=0).loc[idx_depmapID].fillna(0) 
        if args.single_omics:
            df = pd.DataFrame(np.zeros_like(df.values), index = df.index, columns=df.columns) if i>0 else df
        cell_tables.append(df)
    return cell_tables

def scaling_cell(_feat, scaler):
    np_feat = _feat.values
    if isNone(scaler):
        scaler = np.mean(np_feat), np.std(np_feat)
    _mean, _std = scaler
    scaled_df = pd.DataFrame((np_feat-_mean)/_std, index=_feat.index, columns=_feat.columns)
    
    return scaled_df, scaler

def scaled_cell_tensor(cell_feat_list, cell_name, scaler, args):
    feat_list = []
    df_zeros = df = pd.DataFrame(0.0, index=cell_name, columns=range(14070))

    for _feat in cell_feat_list:
        scaled_df, scaler = scaling_cell(_feat.loc[cell_name], scaler) 
        scaled_np = scaled_df.loc[cell_name].values
        feat_list.append(scaled_np)

    concat_list = []
    for _feat in feat_list:
        concat_list.append( _feat[None, :,:])
    np_cell = np.concatenate(concat_list)
    np_cell = np.swapaxes(np_cell, 0, 1)
    np_cell = np_cell[:,:, None, :]
    return torch.Tensor(np_cell).float(), scaler

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def init_layer(args, x_shape):
    args.layers = convert_layers(args.layers)
    args.n_channel = x_shape[1]
    args.n_genes = x_shape[-1]


def save_model(model, path, epoch):
    os.makedirs(path, exist_ok=True)
    path_save = path+'/'+f'model_{epoch}.pt'
    torch.save(model.state_dict(), path_save)
    return path_save

def get_ic50_single(ic50, cell_name, scaler):
    ic50 = ic50.loc[cell_name]
    return ic50, scaler

def dropNaN(_df):
    ret_df = _df.dropna()
    return ret_df, ret_df.index


def convert_layers(_layers):
    if type(_layers[0]) is not int and _layers[0].find(' ')>0:
        _layers = _layers[0].split()
    return list(map(int, _layers))


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


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class _Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y, cell_name, DEVICE ):
        self.X = X.to(DEVICE)
        self.y = torch.from_numpy(y.values).float().to(DEVICE)
        self.cell_name = cell_name
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.cell_name[idx]


def get_loader( X, y, cell_name, n_batch, DEVICE, is_train):
    dataset = _Dataset(X, y, cell_name, DEVICE)
    return DataLoader( dataset, batch_size= n_batch,  shuffle=True if is_train else False)

def scaling(_df, scaler=None):
    if isNone(scaler):
        scaler = StandardScaler()
        scaler.fit(_df)
    scaled_df = pd.DataFrame(scaler.transform(_df), index = _df.index, columns=_df.columns)
    return scaled_df, scaler

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

def notNone(x):
    if type(x) is str:
        return x.lower() != 'none'
    else:
        return x is not None

def isNone(x):
    if type(x) is str:
        return x.lower() == 'none'
    else:
        return x is None

class build_cnn(nn.Module):
    def __init__(self, args):
        super(build_cnn, self).__init__()
        dim_in, hidden_dims, ACT_func, drop2d = args.n_channel, args.layers, args.ACT_func, args.drop_conv

        layers = []
        n_layers = len(hidden_dims)+1
        hidden_dims = [dim_in]+hidden_dims+[1]
        # hidden_dims = [4] + [16 8] + [1]
        ACT_func = get_act(ACT_func)

        for idx, i in enumerate( range(n_layers)):
            layers += [nn.Conv2d( hidden_dims[i], hidden_dims[i+1], kernel_size=[1,1], stride=1) ]
            layers += [ACT_func]
            if drop2d>0.0:
                layers += [nn.Dropout(p = drop2d)]
        layers = layers[:-1] # delete last dropout
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return torch.flatten(x, 1)
        return x

def proj_model(args):
    if args.model_type == 'conv':
        proj_model = build_cnn(args)
    elif args.model_type == 'early':
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
    def forward(self, x):
        x = torch.flatten(x, 1)
        for l in self.layers:
            x = l(x)
        return x

class CNN_model(nn.Module):
    def __init__(self, args):
        super(CNN_model, self).__init__()        
        self.cnn = proj_model(args)
        self.n_h = 768
        self.fc1 = nn.Linear(args.n_genes, self.n_h)
        self.fc_last = nn.Linear(self.n_h, 1)
        self.drop1 = nn.Dropout(args.drop_fc1)
        self.drop2 = nn.Dropout(args.drop_fc2)

    def forward(self, x):
        x = self.cnn(x)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.drop2(x)
        x = self.fc_last(x)
        # return x
        return x.squeeze()

    def extract_feat(self, x):
        x = self.cnn(x)
        x = self.drop1(x)
        x = self.fc1(x)
        # return x
        return x.squeeze()

class Trainer:
    def __init__(self, model, dataloader, args, criterion, n_cv, i_drug,
                 _cuda=False, _cuda_device=-1, lr=1e-4, 
                  log_freq_epoch=1, 
                 _not_tqdm=True, is_verbose=True ):
        is_cuda = True if torch.cuda.is_available() and _cuda else False    
        self.device = torch.device(f'cuda:{_cuda_device}' if is_cuda else 'cpu')
        self.model = model.to(self.device)
        
        self.dataloader = dataloader
        steps_total = len(dataloader)*args.n_epoch
        steps_warmup = math.ceil(steps_total * args.warm_ratio)

        self.optim = AdamW(model.parameters(), lr=lr) if args.AdamW else Adam(model.parameters(), lr=lr)
        if args.cosine:
            self.scheduler = get_cosine_schedule_with_warmup(self.optim, 
                                num_warmup_steps=steps_warmup, 
                                num_training_steps=steps_total, min_lr=args.min_lr
                                )

        self.criterion = criterion

        self.log_freq_epoch = log_freq_epoch
        self.is_verbose = is_verbose
        self._not_tqdm = _not_tqdm
        self.args = args
        self.n_cv = n_cv
    
    def train(self, epoch ):
        self.model.train()
        ret_dict = self.iteration(dataloader = self.dataloader, epoch= epoch, is_train=True )
        return ret_dict
        
    def test(self, dataloader_test, epoch=0):
        self.model.eval()
        ret_dict = self.iteration(dataloader= dataloader_test, epoch=epoch, is_train=False)
        return ret_dict

    def get_ig(self, dataloader):
        ig_list = []
        ig = IntegratedGradients(self.model)
        for data in dataloader:
            X, y, cell_names = data[0], data[1], data[2]
            n_bsz = data[0].shape[0]
            if n_bsz==1:
                continue
            att_list = ig.attribute(X, 
                                  n_steps = self.args.n_epoch,
                                  return_convergence_delta=False
                                  ).squeeze().cpu().detach().numpy()
            att_list = _normalize_attr( att_list, 0.01, False)
            ig_list.append( att_list)
        return np.vstack(ig_list)

    def get_feat(self, dataloader, drug_name):
        self.model.eval()
        feat_list = []
        target_list = []
        for data in dataloader:
            X, y, cell_names = data[0], data[1], data[2]
            feats = self.model.extract_feat(X).detach().cpu().detach().view(-1, self.model.n_h).numpy()
            feat_list.append(pd.DataFrame(feats, index=cell_names) )
            y = y.detach().cpu().detach().numpy()
            target_list.append( pd.DataFrame(y, index=cell_names, columns = drug_name) )

        return pd.concat( feat_list), pd.concat( target_list)

    def iteration(self, dataloader, is_train, epoch=0 ):
        status = 'Train' if is_train else 'Test '
        data_iter = tqdm.tqdm(enumerate(dataloader),
                              total=len(dataloader),
                              bar_format="{l_bar}{r_bar}",
                              leave=True,
                              disable=self._not_tqdm
                              )
        avg_loss = 0.0
        y_true_list, y_pred_list = [], []
        feat_list = []

        for i, data in data_iter:            
            X = data[0]
            y_true = data[1]
            cell_names = data[2]
            with torch.set_grad_enabled(is_train):
                logit = self.model(X).view_as(y_true)
                loss = self.criterion(logit, y_true)
                if is_train:
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    if self.args.cosine: self.scheduler.step()

            #######################
            # Foward & Backward are Done
            # Manipulate Pred and True

            y_pred = torch.sigmoid(logit)

            np_pred = tf2np(y_pred)
            np_true = tf2np(y_true)
            NaN_check(np_pred)
            y_true_list.append(np_true)
            y_pred_list.append(np_pred)

            avg_loss += loss.item()

        #######################################
        ## Epoch is done
        ## Get Metric
        _ROC, _auPR = stack_N_metric_bi_single( y_pred_list, y_true_list, is_stack=True)
        ret_dict = {'loss':avg_loss / (i + 1), 'ROC':_ROC, 'auPR':_auPR,
                    }        
        return ret_dict


def tf2np(tf):
    return np.squeeze(tf.detach().cpu().numpy()).astype(np.float32)

def get_dict(suffix, _dict, values=None):
    _keys = list(_dict.keys())
    _values = list(_dict.values() ) if values is None else values
    return dict([[k+suffix, v] for k, v in zip(_keys, _values)])

def isZero(ic50_te_df):
    Not_zeros = np.sum(np.zeros_like(ic50_te_df.values) != ic50_te_df.values)
    return Not_zeros == 0
    

def stack_N_metric_bi_single(y_pred, y_true, is_stack=True):
    y_true = np.hstack( y_true)
    y_pred = np.hstack( y_pred)
    _ROC = roc_auc_score(y_true, y_pred)
    _auPR = average_precision_score(y_true, y_pred)
    return _ROC, _auPR


def NaN_check(y_pred):
    y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
    try:
        np.asarray_chkfinite(y_pred)
    except:
        error_msg = '\n'+'-'*20
        error_msg += '\n'+'ValueError: Inf or NaN\n{}'.format(y_pred)
        sys.exit(error_msg)

def save_feats( ret, path_feat, drug_name, is_train):
    os.makedirs(path_feat, exist_ok=True)
    feat, y = ret
    path_X = path_feat+'feat_Drug[{}]_X_{}.csv'.format(drug_name, 'Train' if is_train else 'Test')
    path_y = path_feat+'feat_Drug[{}]_y_{}.csv'.format(drug_name, 'Train' if is_train else 'Test')
    feat.to_csv(path_X)
    y.to_csv(path_y)

