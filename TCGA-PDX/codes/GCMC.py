#!/usr/bin/env python
# coding: utf-8
import argparse
from utils import *

path_data = '../data/'
set_seeds(42)


parser = argparse.ArgumentParser()

parser.add_argument('--model_type', type=str, default='conv', help='conv, flat')
parser.add_argument('--drug_name', type=str, default='Gemcitabine')
parser.add_argument('--DB', type=str, default='TCGA')
parser.add_argument('--single_omics', type=str2bool, nargs='?',const=True, default=False)

parser.add_argument('--drop_conv', type=float, default= 0.5)
parser.add_argument('--drop_fc1', type=float, default= 0.5)
parser.add_argument('--drop_fc2', type=float, default= 0.5)
parser.add_argument('--lr', type=float, default= 0.00005)
parser.add_argument('--warm_ratio', type=float, default= 0.05)

# model
parser.add_argument('--ACT_func', type=str, default='prelu')
parser.add_argument('--layers', nargs="+", default=[8])

## optim
parser.add_argument('--min_lr', type=float, default= 0.1)
parser.add_argument('--AdamW', type=str2bool, nargs='?',const=True, default=True)
parser.add_argument('--cosine', type=str2bool, nargs='?',const=True, default=True)
parser.add_argument('--verbose', type=str2bool, nargs='?',const=True, default=False)

## Batch & Epoch
parser.add_argument('--n_batch', type=int, default=196)
parser.add_argument('--n_epoch', type=int, default=301)
parser.add_argument('--n_channel', type=int, default=3)

parser.add_argument('--is_verbose', type=str2bool, nargs='?',const=True, default=True)
# Gpu Env
parser.add_argument('--cuda', type=str2bool, nargs='?',const=True, default=True)
parser.add_argument('--cuda_id', type=int, default= 2)
args = parser.parse_args()


# ## 1. Get Data

# ### 1.1 Feat Table

optimal_params(args.drug_name, args.DB, args)
print(f'[{args.DB}-{args.drug_name}] Performance Evaluation')


GDSCR = pd.read_csv(path_data+f"response/GDSC_response.{args.drug_name}.tsv", 
                    sep = "\t", index_col=0, decimal = ",")

PDXER = pd.read_csv(path_data+f"response/{args.DB}_response.{args.drug_name}.tsv", 
                    sep = "\t", index_col=0, decimal = ",")

GDSCE = pd.read_csv(path_data+f"exprs_homogenized/GDSC_exprs.{args.drug_name}.eb_with.{args.DB}_exprs.{args.drug_name}.tsv", 
                    sep = "\t", index_col=0, decimal = ",")
GDSCE = pd.DataFrame.transpose(GDSCE)

PDXE = pd.read_csv(path_data+f"exprs_homogenized/{args.DB}_exprs.{args.drug_name}.eb_with.GDSC_exprs.{args.drug_name}.tsv", 
                   sep = "\t", index_col=0, decimal = ",")
PDXE = pd.DataFrame.transpose(PDXE)

PDXM = pd.read_csv(path_data+f"SNA_binary/{args.DB}_mutations.{args.drug_name}.tsv", 
                   sep = "\t", index_col=0, decimal = ".")
PDXM = pd.DataFrame.transpose(PDXM)

PDXC = pd.read_csv(path_data+f"CNA_binary/{args.DB}_CNA.{args.drug_name}.tsv", 
                   sep = "\t", index_col=0, decimal = ".")
PDXC.drop_duplicates(keep='last')
PDXC = pd.DataFrame.transpose(PDXC)

GDSCM = pd.read_csv(path_data+f"SNA_binary/GDSC_mutations.{args.drug_name}.tsv", 
                    sep = "\t", index_col=0, decimal = ".")
GDSCM = pd.DataFrame.transpose(GDSCM)


GDSCC = pd.read_csv(path_data+f"CNA_binary/GDSC_CNA.{args.drug_name}.tsv", 
                    sep = "\t", index_col=0, decimal = ".")
GDSCC.drop_duplicates(keep='last')
GDSCC = pd.DataFrame.transpose(GDSCC)


selector = VarianceThreshold(0.05)
selector.fit_transform(GDSCE)
GDSCE = GDSCE[GDSCE.columns[selector.get_support(indices=True)]]

GDSCM = GDSCM.fillna(0)
GDSCM[GDSCM != 0.0] = 1
PDXM = PDXM.fillna(0)
PDXM[PDXM != 0.0] = 1

GDSCC = GDSCC.fillna(0)
GDSCC[GDSCC != 0.0] = 1
PDXC = PDXC.fillna(0)
PDXC[PDXC != 0.0] = 1

ls = GDSCE.columns.intersection(GDSCM.columns)
ls = ls.intersection(GDSCC.columns)
ls = ls.intersection(PDXE.columns)
ls = ls.intersection(PDXM.columns)
ls = ls.intersection(PDXC.columns)
ls = pd.unique(ls)

ls2 = GDSCE.index.intersection(GDSCM.index)
ls2 = ls2.intersection(GDSCC.index)

ls3 = PDXE.index.intersection(PDXM.index)
ls3 = ls3.intersection(PDXC.index)

PDXE = PDXE.loc[ls3,ls]
PDXM = PDXM.loc[ls3,ls]
PDXC = PDXC.loc[ls3,ls]
PDXC = PDXC.loc[:, ~PDXC.columns.duplicated()] if args.DB =='TCGA' else PDXC

GDSCE = GDSCE.loc[ls2,ls]
GDSCM = GDSCM.loc[ls2,ls]
GDSCC = GDSCC.loc[ls2,ls]

GDSCR['targets'] = GDSCR['response'].apply(get_binary)
GDSCR = GDSCR.loc[ls2.astype(int),:]

PDXER['targets'] = PDXER['response'].apply(get_binary)
PDXER = PDXER.loc[ls3,:]


Y = GDSCR['targets'].values



DEVICE = torch.device(f'cuda:{args.cuda_id}' if args.cuda_id >-1 else 'cpu')



X_trainE = GDSCE.values[:,:]
X_trainM = GDSCM.values[:,:]
X_trainC = GDSCC.values[:,:]
y_trainE = Y[:].astype(int)

scalerGDSC = sk.StandardScaler()
scalerGDSC.fit(X_trainE)
X_trainE = scalerGDSC.transform(X_trainE)

X_trainM = np.nan_to_num(X_trainM)
X_trainC = np.nan_to_num(X_trainC)


X_tr = get_3d( X_trainE, X_trainM, X_trainC, single_omics)


PDXE_scaled = scalerGDSC.transform(PDXE.values)
PDX_X = get_3d( PDXE_scaled, PDXM.values, PDXC.values, single_omics)

PDX_X = torch.FloatTensor(PDX_X).to(DEVICE)
PDX_y = torch.FloatTensor(PDXER['targets'].values.astype(int)).to(DEVICE)


# ### 2.3 Sampler 

is_Weight_sampler = False if args.DB == 'PDX' else True
# #Train
class_sample_count = np.array([len(np.where(y_trainE==t)[0]) for t in np.unique(y_trainE)])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in y_trainE])

samples_weight = torch.from_numpy(samples_weight)
sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)

mb_size = args.n_batch

trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_tr).to(DEVICE), torch.FloatTensor(y_trainE.astype(int)).to(DEVICE))

trainLoader = torch.utils.data.DataLoader(dataset = trainDataset, batch_size=mb_size, shuffle=False if is_Weight_sampler else True, 
                                          num_workers=0, sampler = sampler if is_Weight_sampler else None)


foo = y_trainE[ list(WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True))]
if args.verbose:
    print('W-sampler\'s Pos Ratio: {:.3}%'.format(foo.sum()/len(foo)*100))
    print('Train set\'s Pos Ratio: {:.3}%'.format(y_trainE.sum()/len(y_trainE)*100))
    print('PDX \'s Pos Ratio: {:.3}%'.format(PDX_y.sum()/len(PDX_y)*100))


n_sampE, IE_dim = X_trainE.shape
n_sampM, IM_dim = X_trainM.shape
n_sampC, IC_dim = X_trainC.shape


args.n_genes = len(ls)
_proj_model = proj_model(args)
_model = CNN_model(_proj_model, args.n_genes, args.drop_fc1, args.drop_fc2)

torch.cuda.manual_seed_all(42)

Clas = _model.to(DEVICE)
SolverClass = optim.AdamW(_model.parameters(), lr=args.lr)
C_loss = torch.nn.BCELoss()


# ### Epoch Start

auctr = []
aucts = []
auPR_tr = []
auPR_te = []

steps_total =len(trainLoader)*args.n_epoch
steps_warmup = math.ceil(len(trainLoader)*args.n_epoch * args.warm_ratio)
scheduler = get_cosine_schedule_with_warmup(SolverClass, 
                        num_warmup_steps=steps_warmup, 
                        num_training_steps=steps_total, min_lr=args.min_lr
                        )


epoch_tqdm = tqdm.tqdm(range(args.n_epoch),
              total=args.n_epoch,
              bar_format="{l_bar}{r_bar}",
              leave=True)

for it in epoch_tqdm:

    epoch_cost4 = 0
    epoch_cost_auc = []
    epoch_cost_aupr = []

    num_minibatches = int(n_sampE / mb_size) 

    for i, (X, target) in enumerate(trainLoader):
        flag = 0
        Clas.train()

        if torch.mean(target)!=0. and torch.mean(target)!=1.: 
            with torch.set_grad_enabled(True):
                Pred = Clas(X)

                y_true = target.view(-1,1)
                y_pred = Pred
                loss = C_loss(y_pred, y_true)

                AUC = roc_auc_score(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
                auPR = average_precision_score(y_true.detach().cpu().numpy(),y_pred.detach().cpu().numpy())

                SolverClass.zero_grad()

                loss.backward()

                SolverClass.step()
                scheduler.step()

            epoch_cost_auc.append(AUC)
            epoch_cost_aupr.append(auPR)

            flag = 1
    if flag == 1:
        auctr.append(np.mean(epoch_cost_auc))
        auPR_tr.append(np.mean(epoch_cost_aupr))

    with torch.no_grad():

        Clas.eval()

        PredT = Clas(PDX_X)

        y_truet = PDX_y.view(-1,1)
        y_predt = PredT
        AUC_test = roc_auc_score(y_truet.detach().cpu().numpy(),  y_predt.detach().cpu().numpy())        
        auPR_test = average_precision_score(y_truet.detach().cpu().numpy(),  y_predt.detach().cpu().numpy())        

        aucts.append(AUC_test)
        auPR_te.append(auPR_test)

ret = 'Result\tAUC:{:.2}  AUPR:{:.2}'
print(ret.format( aucts[-1], auPR_te[-1]))


from post_ig import *
print('Extracting contribution ratios of omics types')

path_ig = f'./[IG]/'
os.makedirs(path_ig, exist_ok=True)

ig_tr = get_ig(Clas, trainLoader, args.n_epoch )
ig_te = get_ig_test(Clas, PDX_X, args.n_epoch )
mean_tr = mean_rank(ig_tr, args.drug_name)
mean_te = mean_rank(ig_te, args.drug_name)

mean_tr.to_csv(path_ig+f'{args.DB}-{args.drug_name}_omics_ratio_Train.csv')
mean_tr.to_csv(path_ig+f'{args.DB}-{args.drug_name}_omics_ratio_Test.csv')
print('Done')
