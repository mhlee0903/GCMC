#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd

import tqdm
import os


def get_rank_top(att_cell, n_top):
    cell_rank_sum   = pd.DataFrame(0.0, index=[f'cell{i}' for i in range(len( att_cell))], columns=['mut', 'exp', 'cn_'])
    _idx = [ f'exp{i}' for i in range(14070)]+[ f'mut{i}' for i in range(14070)]+[ f'cn_{i}' for i in range(14070)]

    for cell_i, att_i in enumerate(att_cell):
        df_1dim = attr_1dim(att_i, _idx=_idx)
        _rank_sum = get_rank(df_1dim, n_top)

        for _col in ['mut', 'exp', 'cn_']:
            try: 
                cell_rank_sum.loc[f'cell{cell_i}', _col] = _rank_sum.loc[_col]/_rank_sum.sum()
            except: pass
    return cell_rank_sum # M cells lines

def attr_1dim(norm_attr, _idx):
    # norm_attr = _normalize_image_attr(att_i, outlier_perc, has_channel)
    np_1dim = norm_attr.reshape([norm_attr.shape[1]*3])
    df_1dim = pd.DataFrame( np_1dim, index=_idx, columns=['val'] )
    df_1dim['feat'] = [_idx[:3] for _idx in df_1dim.index]
    return df_1dim

def get_rank(df_rank, n_top):
    df_rank['top'] = df_rank['val'].rank(pct=False, ascending=False)

    top100 = df_rank[df_rank['top']<=n_top].sort_values(by=['top'])
    top100_sum = top100.groupby(['feat']).sum()['val']
    return top100_sum


def save_rank(rank_tr, rank_te, drug_list, path_ret):
    df_rank_tr = pd.concat(rank_tr, 1).T
    df_rank_te = pd.concat(rank_te, 1).T
    
    df_rank_tr.index = drug_list
    df_rank_te.index = drug_list
    df_rank_tr.to_csv(path_ret+'Omics-ratio_Train.csv')
    df_rank_te.to_csv(path_ret+'Omics-ratio_Test.csv')


def rank_df(ig_drugs, drug_id):
    rank_sum = pd.DataFrame( np.array(ig_drugs), index= drug_id, columns=['mut', 'exp', 'cn_'])
    return rank_sum
