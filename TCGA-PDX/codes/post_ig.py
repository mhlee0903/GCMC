#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd

import tqdm
import os

def get_rank(df_rank, n_top):
    df_rank['top'] = df_rank['val'].rank(pct=False, ascending=False)

    top100 = df_rank[df_rank['top']<=n_top].sort_values(by=['top'])
    top100_sum = top100.groupby(['feat']).sum()['val']
    return top100_sum

def attr_1dim(att_i, _idx):
    norm_attr = _normalize_attr(att_i, outlier_perc=0.01, has_channel=False)
    np_1dim = norm_attr.reshape([norm_attr.shape[1]*3])
    df_1dim = pd.DataFrame( np_1dim, index=_idx, columns=['val'] )
    df_1dim['feat'] = [_idx[:3] for _idx in df_1dim.index]
    return df_1dim

def get_rank_top(att_cell, n_top):
    n_genes = att_cell.shape[-1]
    cell_rank_sum   = pd.DataFrame(0.0, index=[f'cell{i}' for i in range(len( att_cell))], columns=['mut', 'exp', 'cn_'])
    _idx = [ f'exp{i}' for i in range(n_genes)]+[ f'mut{i}' for i in range(n_genes)]+[ f'cn_{i}' for i in range(n_genes)]

    for cell_i, att_i in enumerate(att_cell):
        df_1dim = attr_1dim(att_i, _idx=_idx)
        _rank_sum = get_rank(df_1dim, n_top)

        for _col in ['mut', 'exp', 'cn_']:
            try: 
                cell_rank_sum.loc[f'cell{cell_i}', _col] = _rank_sum.loc[_col]/_rank_sum.sum()
            except: pass
    return cell_rank_sum

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

def rank_df(ig_drugs, drug_id):
    rank_sum = pd.DataFrame( np.array(ig_drugs), index= drug_id, columns=['mut', 'exp', 'cn_'])
    return rank_sum

def mean_rank(_ig, drug_name):
    cell_rank_sum = get_rank_top( _ig, 100)
    mean_rank_sum = pd.DataFrame(cell_rank_sum.mean(0), columns=[drug_name]).T
    return mean_rank_sum



