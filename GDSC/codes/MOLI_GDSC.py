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
from metrics import AverageNonzeroTripletsMetric
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import roc_auc_score, average_precision_score
import random
from random import randint
from sklearn.model_selection import StratifiedKFold, KFold

from utils_moli import *
from post_ig import *
import pickle

import tqdm
import os

class _Dataset(torch.utils.data.Dataset):
    def __init__(self, cell_name, y_trainE, _exp, _mut, _cn , DEVICE):
        self.cell_name = list(cell_name)
        self.Y = torch.Tensor(y_trainE).to( DEVICE)
        self.exp = torch.FloatTensor(_exp).to(DEVICE)
        self.mut = torch.FloatTensor(_mut).to(DEVICE)
        self.cn = torch.FloatTensor(_cn).to(DEVICE)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.exp[idx], self.mut[idx], self.cn[idx], self.Y[idx], self.cell_name[idx], idx


################################################
path_ZT = './out_MOLI/ZT/'
path_IG = './out_MOLI/IG/'
path_metric = './out_MOLI/'
model_name = 'results'
_cuda_device = 0

torch.manual_seed(0)

###########################
idx_depmapID  = pd.read_csv('../data/cell_line_info.csv', index_col=0)
GDSCR = pd.read_csv('../data/ic50bi.csv', index_col=0).loc[idx_depmapID.index]
GDSCE_raw = pd.read_csv('../data/RMA_631x14070.csv',  index_col=0)
GDSCC_raw = pd.read_csv('../data/CNV_631x14070.csv', index_col=0)
GDSCM_raw = pd.read_csv('../data/MUT_631x14070.csv', index_col=0)
GDSCM_raw[GDSCM_raw != 0.0] = 1
feat_list = [GDSCE_raw, GDSCM_raw, GDSCC_raw]

selector = VarianceThreshold(0.05)
selector.fit_transform(GDSCE_raw)
GDSCE_raw = GDSCE_raw[GDSCE_raw.columns[selector.get_support(indices=True)]]

iters = 0
metric_all_drug_avg = []
metric_all_drug_CVs = []

DEVICE = torch.device(f'cuda:{_cuda_device}' if _cuda_device >-1 else 'cpu')
is_Weight_sampler = False

drug_idx = 0
drug_list = GDSCR.columns[drug_idx:]
tqdm_drug = tqdm.tqdm( drug_list, total=len(drug_list),
                      bar_format="{l_bar}{r_bar}", leave=True)
for id_drug in tqdm_drug:
    Zeros_All = 0
    metric_cv_ith_drug = []
    i_drug_y = GDSCR.iloc[:, drug_idx].dropna()

    drug_idx += 1
    kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    for num_cv, cv_idx in enumerate(kf.split(i_drug_y.index, i_drug_y.values)): 
        num_cv = num_cv + 1
        Zeros_CV = 0

        n_batch = 96
        
        hdm1 = 256
        hdm2 = 256
        hdm3 = 256
        mrg = 2
        lre = .001
        lrm = .001
        lrc = .001
        lrCL = .0005
        epch = 25
        lr_exp = .5
        lr_mut = .5
        lr_cn = .5
        rate4 = .6
        wd = .001
        lam = .3


        ###########################################
        ## Build Train-Test set
        train_cell, test_cell = get_cv(cv_idx, i_drug_y)

        [X_trainE, X_trainM, X_trainC], y_trainE, cells_tr = get_X_y(feat_list, i_drug_y, train_cell)
        [X_testE,  X_testM,  X_testC],  y_testE,  cells_te = get_X_y(feat_list, i_drug_y, test_cell)

        scalerGDSC = sk.StandardScaler()
        X_trainE = scalerGDSC.fit_transform(X_trainE)
        X_testE  = scalerGDSC.transform(X_testE)

        X_trainM, X_trainC, X_testM, X_testC = [np.nan_to_num(X) for X in [X_trainM, X_trainC, X_testM,  X_testC]]
        TX_testE, TX_testM, TX_testC = [torch.FloatTensor(X).to(DEVICE) for X in [X_testE,  X_testM,  X_testC]]
        ty_testE = torch.FloatTensor(y_testE).to(DEVICE)

        #######################################
        ## Data Loader
        #Train
        if is_Weight_sampler:
            y_trainE = y_trainE.astype(int)
            class_sample_count = np.array([len(np.where(y_trainE==t)[0]) for t in np.unique(y_trainE)])
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[t] for t in y_trainE])

            samples_weight = torch.from_numpy(samples_weight)
            sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)


        trainDataset = _Dataset(cells_tr, y_trainE, X_trainE, X_trainM, X_trainC, DEVICE)
        trainLoader  = torch.utils.data.DataLoader(dataset = trainDataset, batch_size=n_batch, shuffle=False if is_Weight_sampler else True, 
                                                  num_workers=0, sampler = sampler if is_Weight_sampler else None)



        ##############################
        ### Build model

        n_sampE, IE_dim = X_trainE.shape
        n_sampM, IM_dim = X_trainM.shape
        n_sampC, IC_dim = X_trainC.shape

        h_dim1 = hdm1
        h_dim2 = hdm2
        h_dim3 = hdm3        
        Z_in = h_dim1 + h_dim2 + h_dim3
        marg = mrg
        lrE = lre
        lrM = lrm
        lrC = lrc
        epoch = epch

        costtr = []
        auc_tr_list = []
        costts = []
        auc_te_list = []
        auPR_tr_list = []
        auPR_te_list = []

        triplet_selector = RandomNegativeTripletSelector(marg)
        triplet_selector2 = AllTripletSelector()

        class AEE(nn.Module):
            def __init__(self):
                super(AEE, self).__init__()
                self.EnE = torch.nn.Sequential(
                    nn.Linear(IE_dim, h_dim1),
                    nn.BatchNorm1d(h_dim1),
                    nn.ReLU(),
                    nn.Dropout(lr_exp))
            def forward(self, x):
                output = self.EnE(x)
                return output

        class AEM(nn.Module):
            def __init__(self):
                super(AEM, self).__init__()
                self.EnM = torch.nn.Sequential(
                    nn.Linear(IM_dim, h_dim2),
                    nn.BatchNorm1d(h_dim2),
                    nn.ReLU(),
                    nn.Dropout(lr_mut))
            def forward(self, x):
                output = self.EnM(x)
                return output    


        class AEC(nn.Module):
            def __init__(self):
                super(AEC, self).__init__()
                self.EnC = torch.nn.Sequential(
                    nn.Linear(IM_dim, h_dim3),
                    nn.BatchNorm1d(h_dim3),
                    nn.ReLU(),
                    nn.Dropout(lr_cn))
            def forward(self, x):
                output = self.EnC(x)
                return output    

        class OnlineTriplet(nn.Module):
            def __init__(self, marg, triplet_selector):
                super(OnlineTriplet, self).__init__()
                self.marg = marg
                self.triplet_selector = triplet_selector
            def forward(self, embeddings, target):
                triplets = self.triplet_selector.get_triplets(embeddings, target)
                return triplets

        class OnlineTestTriplet(nn.Module):
            def __init__(self, marg, triplet_selector):
                super(OnlineTestTriplet, self).__init__()
                self.marg = marg
                self.triplet_selector = triplet_selector
            def forward(self, embeddings, target):
                triplets = self.triplet_selector.get_triplets(embeddings, target)
                return triplets    

        class Classifier(nn.Module):
            def __init__(self):
                super(Classifier, self).__init__()

                self.FC = torch.nn.Sequential(
                    nn.Linear(Z_in, 1),
                    nn.Dropout(rate4),
                    nn.Sigmoid())
            
            def forward(self, x):
                return self.FC(x)

        class MOLI_model(nn.Module):
            def __init__(self, aeE, aeM, aeC, Clas):
                super(MOLI_model, self).__init__()
                self.aeE = aeE
                self.aeM = aeM
                self.aeC = aeC
                self.Clas = Clas

            def forward(self, dataE, dataM, dataC):
                self.ZE = self.aeE(dataE)
                self.ZM = self.aeM(dataM)
                self.ZC = self.aeC(dataC)

                ZT = torch.cat((self.ZE, self.ZM, self.ZC), 1)
                self.ZT = F.normalize(ZT, p=2, dim=0)

                return self.Clas(self.ZT)

        ####################
        ### Set Model
        torch.cuda.manual_seed_all(42)

        trip_criterion = torch.nn.TripletMarginLoss(margin=marg, p=2)
        TripSel = OnlineTriplet(marg, triplet_selector)
        TripSel2 = OnlineTestTriplet(marg, triplet_selector2)

        _MOLI = MOLI_model(AEE(), AEM(), AEC(), Classifier()).to(DEVICE)


        solverE = optim.Adagrad(_MOLI.aeE.parameters(), lr=lrE)
        solverM = optim.Adagrad(_MOLI.aeM.parameters(), lr=lrM)
        solverC = optim.Adagrad(_MOLI.aeC.parameters(), lr=lrC)
        SolverClass = optim.Adagrad(_MOLI.Clas.parameters(), lr=lrCL, weight_decay = wd)
        C_loss = torch.nn.BCELoss()
        
        #########################
        ## Train
        target_list = []
        for num_epoch in range(epoch):
            epoch_cost4 = 0
            epoch_cost_auc = []
            epoch_cost_aupr = []
            
        
            # num_minibatches = int(n_sampE / mb_size) 

            for i, (dataE, dataM, dataC, target, cell_name_it, idx) in enumerate(trainLoader):

                if torch.mean(target)==0. or torch.mean(target)==1.:
                    # print(f'CV{num_cv}[Drug:{_idx}/265] has no positive target')
                    Zeros_CV += 1
                    Zeros_All += 1
                    continue

                _MOLI.train()

                # if torch.mean(target)!=0. and torch.mean(target)!=1.: 
                Pred = _MOLI(dataE, dataM, dataC)
                ZT = _MOLI.ZT

                Triplets = TripSel2(ZT, target).to(DEVICE)
                if len(Triplets.shape) <2 or len(ZT.shape)<2:
                    continue
                _c_loss = C_loss(Pred, target.view(-1,1))
                loss = lam * trip_criterion(ZT[Triplets[:,0],:],ZT[Triplets[:,1],:],ZT[Triplets[:,2],:]) + _c_loss

                y_true = target.view(-1,1)
                y_pred = Pred

                solverE.zero_grad()
                solverM.zero_grad()
                solverC.zero_grad()
                SolverClass.zero_grad()

                loss.backward()

                solverE.step()
                solverM.step()
                solverC.step()
                SolverClass.step()
                # epoch_cost4 = epoch_cost4 + (loss / num_minibatches)

                AUC = roc_auc_score(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
                auPR = average_precision_score(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
                epoch_cost_auc.append(AUC)
                epoch_cost_aupr.append(auPR)


            #########
            ## 1Epoch training is Done.
            ## Metric Training results
            # if flat == 1:
            # costtr.append(torch.mean(epoch_cost4))
            auc_tr_list.append(np.mean(epoch_cost_auc))
            auPR_tr_list.append(np.mean(epoch_cost_aupr))

            ## Eval
            if torch.mean(ty_testE)==0. or torch.mean(ty_testE)==1.:
                continue

            with torch.no_grad():
                _MOLI.eval()
                PredT = _MOLI(TX_testE, TX_testM, TX_testC)

            ################
            ## Get Metric: ith Drug
            y_truet = ty_testE.view(-1,1)
            y_predt = PredT

            AUC_test = roc_auc_score(y_truet.detach().cpu().numpy(),  y_predt.detach().cpu().numpy())        
            auPR_test = average_precision_score(y_truet.detach().cpu().numpy(),  y_predt.detach().cpu().numpy())        

            # costts.append(lossT)
            auc_te_list.append(AUC_test)
            auPR_te_list.append(auPR_test)

        ###########
        ## N Epochs is Done @ith Drug.
        #############
        #############
        ## 2. ZTT
        zt_list = []
        # for i, (dataE, dataM, dataC, target, cell_name_it, idx) in enumerate(trainLoader):
        #     zt_list.append(getZT(_MOLI, (dataE, dataM, dataC), cell_name_it)) 

        # if torch.mean(ty_testE)!=0. and torch.mean(ty_testE)!=1.:
        #     zt_df_te = getZT(_MOLI, (TX_testE, TX_testM, TX_testC), cells_te)

        # ## Build Test df & save
        # zt_path = path_ZT+f'/CV{num_cv}/'
        # os.makedirs(zt_path, exist_ok=True)

        # zt_df_te.to_csv(zt_path+f'ZT_drug[{id_drug}]Test.csv')
        # pd.concat( zt_list).to_csv(zt_path+f'ZT_drug[{id_drug}]Train.csv')


        ############
        ## 3. Integrated Gradients @ith Drug.
        ig_tr = []
        ## 3.1 Train Cells
        ig_norm_drug_i=[]
        for i, (dataE, dataM, dataC, target, cell_name_it, idx) in enumerate(trainLoader):
            ig_norm_drug_i.append(getIG( _MOLI, (dataE, dataM, dataC), target))
        rank_ij_tr = get_rank_top(np.vstack(ig_norm_drug_i), 100)

        ## 3.2 Test Cells
        if torch.mean(ty_testE)!=0. and torch.mean(ty_testE)!=1.:
            rank_ij_te = get_rank_top(getIG( _MOLI, (TX_testE, TX_testM, TX_testC), ty_testE), 100)

        ## 3.3 Save them
        ig_path = path_IG+f'/CV{num_cv}/'
        os.makedirs(ig_path, exist_ok=True)
        rank_ij_tr.to_csv( ig_path+f'omics_ratio_drug[{id_drug}]Train.csv')
        rank_ij_te.to_csv( ig_path+f'omics_ratio_drug[{id_drug}]Test.csv')

        rank_cv_tr.append(rank_ij_tr.mean())
        rank_cv_te.append(rank_ij_te.mean())
        if n_cv+1==5:
            rank_tr.append( pd.concat(rank_cv_tr, 1).T.mean())
            rank_te.append( pd.concat(rank_cv_te, 1).T.mean())
        ###########
        ## N Epochs is Done @ith Drug.
        if torch.mean(ty_testE)!=0. and torch.mean(ty_testE)!=1.:
            metric_cv_ith_drug.append([ num_cv, id_drug, 
                                        auc_te_list[-1], auPR_te_list[-1],
                                        auc_tr_list[-1], auPR_tr_list[-1]
                                      ])

    ##########
    ## 5-Fold CV is Done @ith Drug.
    df_cols = ['num_cv', 'id_drug', 'AUC_te', 'AUPR_te', 'AUC_tr', 'AUPR_tr']
    os.makedirs(path_metric, exist_ok=True)
    ith_drug = pd.DataFrame(metric_cv_ith_drug, index=[f'CV{i+1}' for i in range(len(metric_cv_ith_drug))], 
                                             columns= df_cols )
    ith_drug.to_csv(path_metric+f'drug[{id_drug}].csv')

    # metric_all_drug_avg.append([id_drug, ith_drug['AUC'].mean(), ith_drug['AUPR'].mean() ])
    metric_all_drug_CVs.append(ith_drug)


    ############
    ### All Drug is Done.
    df_all = pd.concat( metric_all_drug_CVs)
    df_all.to_csv(path_metric+f'All_Drugs_CVs.csv')

    df_avg = df_all.groupby(by=['id_drug']).mean()
    df_avg.to_csv(path_metric+f'All_Drugs_avg.csv')

save_rank(rank_tr, rank_te, GDSCR.columns, path_metric)