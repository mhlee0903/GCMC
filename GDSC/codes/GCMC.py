#!/usr/bin/env python
# coding: utf-8

import torch
from torch.nn import BCEWithLogitsLoss, MSELoss
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

import argparse
from utils import *
from post_ig import *

def main(args):
    # 1. Args setting
    args.cell_feat = cell_list(args)
    DEVICE = torch.device(f'cuda:{args.cuda_id}' if args.cuda else 'cpu')

    # ## 2. Data
    cell_feat_list = get_cell(args.cell_feat, args)
    ic50_raw = get_raw()

    # ## 4. Train Pred model
    # ### 4.2  Iteration N-times with CV
    LossFunc = BCEWithLogitsLoss() 
    epoch_global = 0
    model_global = 0
    metric_byCV = []
    rank_tr, rank_te = [], []
    metric_byDrug = []
    tqdm_drug = tqdm.tqdm(enumerate(ic50_raw.columns[args.drug_start:]), 
                          total=len(ic50_raw.columns[args.drug_start:]),
                          bar_format="{l_bar}{r_bar}", leave=True)

    for i_drug, drug_name in tqdm_drug:
        i_drug += args.drug_start
        i_drug_y = ic50_raw.iloc[:, i_drug].dropna()
        
        kf = StratifiedKFold(shuffle=True, random_state=args.seed)
        rank_cv_tr, rank_cv_te = [], []
        for n_cv, tr_te in enumerate(kf.split(i_drug_y.index, i_drug_y.values)): 
            if args.cv_start > n_cv:
                continue
            if args.cv_end < n_cv:
                continue
            set_seeds(0)

            train_cell, test_cell = get_cv(tr_te, i_drug_y)
            ic50_tr = i_drug_y.loc[train_cell]
            ic50_te = i_drug_y.loc[test_cell]


            if len(test_cell)==0:
                continue
            if  isZero(ic50_te) or isZero(ic50_tr):
                continue

            cell_tensor_tr, scaler_cell = scaled_cell_tensor(cell_feat_list, train_cell, None, args)
            cell_tensor_te, scaler_cell = scaled_cell_tensor(cell_feat_list, test_cell,  scaler_cell, args)
            init_layer(args, cell_tensor_tr.shape)

            loader_pred_tr = get_loader(X=cell_tensor_tr, y=ic50_tr, cell_name=train_cell, n_batch=args.n_batch, DEVICE=DEVICE, is_train = True )
            loader_pred_te = get_loader(X=cell_tensor_te, y=ic50_te, cell_name=test_cell,  n_batch=args.n_batch, DEVICE=DEVICE, is_train = False )

            _model = CNN_model(args=args)
            _trainer = Trainer(model=_model, dataloader=loader_pred_tr, criterion=LossFunc, args=args, 
                               _cuda=args.cuda, _cuda_device=args.cuda_id, lr=args.lr, 
                               log_freq_epoch = args.ckpt_epoch, n_cv=n_cv, i_drug=i_drug,
                               _not_tqdm=True, is_verbose=args.is_verbose)

            for epoch in range(args.n_epoch):
                epoch_global += 1
                dict_tr = _trainer.train(epoch=epoch )

                if epoch % args.ckpt_epoch ==0 or epoch+1 == args.n_epoch:
                    dict_te = _trainer.test(dataloader_test = loader_pred_te, epoch=epoch)
                    if args.save or args.load_test: 
                        save_model(_model, args.save_dir.format(args.model_name, n_cv), epoch)

                    dict_te_log, dict_tr_log = get_dict('_te', dict_te), get_dict('_tr', dict_tr)

            metric_byDrug.append( [drug_name]+list(dict_te_log.values()) +list(dict_tr_log.values()) )
            
            ############################################
            ## [i/5] CV for 265 Drugs is Done.      
            ## 0. set paths
            _path = f'./out_GCMC/' if args.model_type =='conv' else './out_Early/'
            path_feat_ig = _path + f'[IG-Feat]CV{n_cv+1}/'

            ### 1. Extract Feats
            if args.get_feat:
                os.makedirs(path_feat_ig, exist_ok=True)
                save_feats( _trainer.get_feat(loader_pred_tr, [drug_name]), path_feat_ig, drug_name, is_train=True)
                save_feats( _trainer.get_feat(loader_pred_te, [drug_name]), path_feat_ig, drug_name, is_train=False)

            ### 2. IG
            if args.get_ig:
                os.makedirs(path_feat_ig, exist_ok=True)
                rank_drug_i_tr = get_rank_top(_trainer.get_ig(loader_pred_tr), 100)
                rank_drug_i_te = get_rank_top(_trainer.get_ig(loader_pred_te), 100)
                rank_drug_i_tr.to_csv(path_feat_ig+f'IG_omics-ratio_drug[{drug_name}]Train.csv')
                rank_drug_i_tr.to_csv(path_feat_ig+f'IG_omics-ratio_drug[{drug_name}]Test.csv')

                rank_cv_tr.append(rank_drug_i_tr.mean())
                rank_cv_te.append(rank_drug_i_te.mean())
                if n_cv+1==5:
                    rank_tr.append( pd.concat(rank_cv_tr, 1).T.mean())
                    rank_te.append( pd.concat(rank_cv_te, 1).T.mean())

            ### 3. Save model
            if args.save or args.load_test: 
                model_last = save_model(_model, args.save_dir.format(args.model_name, n_cv), epoch)
            
            ### 4. Test model at the Last epoch .. for i-th drug
            dict_te = _trainer.test(dataloader_test = loader_pred_te, epoch=epoch)

        ############################################
        ## [i/265] Drug is Done. 

        ####################
        ## 1. Save Metric by Drugs
        ## 1.2 Save metric_byDrug
        path_ret = _path
        os.makedirs(path_ret, exist_ok=True)

        df_cols = ['drug_name']+list(dict_te_log.keys()) +list(dict_tr_log.keys()) 
        df_metric = pd.DataFrame(metric_byDrug, columns=df_cols)

        df_avg = df_metric.groupby(by=['drug_name']).mean()
        # df_avg.to_csv(path_ret+f'Results_drug[{drug_name}]CV_mean.csv')
        df_avg.to_csv(path_ret+f'Results_mean.csv')

    # save df_rank
    save_rank(rank_tr, rank_te, ic50_raw.columns[args.drug_start:], path_ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--get_ig', type=str2bool, nargs='?',const=True, default=True)
    parser.add_argument('--get_feat', type=str2bool, nargs='?',const=True, default=False)
    parser.add_argument('--single_omics', type=str2bool, nargs='?',const=True, default=False)

    parser.add_argument('--n_epoch', type=int, default= 151)
    parser.add_argument('--ckpt_epoch', type=int, default = 5)
    parser.add_argument('--drug_start', type=int, default = 0)
    parser.add_argument('--cv_start', type=int, default = 0)
    parser.add_argument('--cv_end', type=int, default = 4)
    parser.add_argument('--min_lr', type=float, default= 0.2)

    parser.add_argument('--drop_conv', type=float, default= 0.1)
    parser.add_argument('--drop_fc1', type=float, default= 0.3)
    parser.add_argument('--drop_fc2', type=float, default= 0.3)
    parser.add_argument('--lr', type=float, default= 0.001)
    parser.add_argument('--warm_ratio', type=float, default= 0.05)

    parser.add_argument('--rma', type=str2bool, nargs='?',const=True, default=True)
    parser.add_argument('--cnv', type=str2bool, nargs='?',const=True, default=True)
    parser.add_argument('--mut', type=str2bool, nargs='?',const=True, default=True)

    parser.add_argument('--seed', type=int, default = 0)
    parser.add_argument('--save', type=str2bool, nargs='?',const=True, default=False)

    parser.add_argument('--save_dir', type=str, default = './save_model/{}/cv{}')
    parser.add_argument('--model_name', type=str, default = '')
    parser.add_argument('--model_type', type=str, default = 'conv', help='conv, early')
    parser.add_argument('--load_test', type=str2bool, nargs='?',const=True, default=False)

    # model
    parser.add_argument('--ACT_func', type=str, default='prelu')
    parser.add_argument('--layers', nargs="+", default=[8])
    # train args
    
    ## optim
    parser.add_argument('--AdamW', type=str2bool, nargs='?',const=True, default=True)
    parser.add_argument('--cosine', type=str2bool, nargs='?',const=True, default=True)

    ## Batch & Epoch
    parser.add_argument('--n_batch', type=int, default=128)
    
    parser.add_argument('--is_verbose', type=str2bool, nargs='?',const=True, default=True)
    # Gpu Env
    parser.add_argument('--cuda', type=str2bool, nargs='?',const=True, default=True)
    parser.add_argument('--cuda_id', type=int, default= 0)

    # Data path
    
    args = parser.parse_args()
    main(args)