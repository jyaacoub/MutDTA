#%%
# test gvpl model to make sure it worked correctly!
from src.utils.loader import Loader

m = Loader.init_model('GVPL', 'nomsa', 'binary')

#%%
import torch
smi, lig = next(iter(torch.load('../data/DavisKibaDataset/davis/nomsa_binary_gvp_binary/test/data_mol.pt').items()))
pid, pro = next(iter(torch.load('../data/DavisKibaDataset/davis/nomsa_binary_gvp_binary/test/data_pro.pt').items()))

m(pro,lig)


#%%
train_genes = ["CDC2L1", "ABL1(E255K)", "IRAK1", "AAK1", "RET(M918T)", "RSK1(KinDom.2-C-terminal)", "HIPK1", "CSNK1G3,"
    "CHEK2", "PDPK1", "EGFR(S752I759del)", "CDC2L2", "ERN1", "CDK4-cyclinD1", "PLK3", "TIE1", "TIE2", "CDKL5,"
    "RSK4(KinDom.2-C-terminal)", "WEE1", "RIPK4", "TNK2", "SGK3", "MRCKA", "SLK", "MAK", "GCN2(KinDom2S808G),"
    "p38-beta", "PIK4CB", "CLK2", "FLT3(N841I)", "TEC", "AMPK-alpha2", "MLK3", "DAPK1", "SYK", "GRK1", "MARK2,"
    "PRKCD", "PRKCQ", "TBK1", "PRKCE", "ERK5", "CAMKK2", "JAK3(JH1domain-catalytic)", "INSR", "PKN2", "ADCK3,"
    "ABL1(F317I)", "ADCK4", "NEK2", "S6K1", "EGFR(L747E749del)", "EPHA4", "EPHB3", "SgK110", "JAK1(JH1domain-catalytic),"
    "PLK2", "PRKG2", "ERBB4", "PHKG2", "SRPK2", "CHEK1", "DCAMKL2", "NEK9", "ACVR2A", "AKT1", "CSK", "STK35,"
    "IKK-alpha", "CSF1R", "MST1", "TYK2(JH2domain-pseudokinase)", "PCTK3", "YANK3", "ACVRL1", "DAPK3", "EGFR(L747S752del),"
    "EPHA6", "LIMK1", "CDKL3", "HIPK2", "PKAC-alpha", "MST2", "CSNK1D", "OSR1", "EPHA5", "CDKL2", "MEK3,"
    "PHKG1", "PIP5K2B", "TLK2", "CAMKK1", "MINK", "EGFR", "MKNK2", "PRKD3", "INSRR", "BRK", "EIF2AK1", "AURKA,"
    "ERK3", "HCK", "JAK2(JH1domain-catalytic)", "TRKB", "TNK1", "MAP3K1", "MAP3K4", "LIMK2", "GAK", "ERBB3,"
    "FLT3(R834Q)", "MYO3A", "MARK3", "LATS2", "IRAK4", "MEK4", "PRKR", "STK39", "YES", "RIPK1", "FLT3", "BIKE,"
    "CSNK1E", "LYN", "PKN1", "RET", "ABL1(F317L)p", "RSK1(KinDom.1-N-terminal)", "ROS1", "CAMK1", "VEGFR2,"
    "MERTK", "BRSK1", "IKK-beta", "RSK4(KinDom.1-N-terminal)", "p38-gamma", "YSK4", "PFPK5(Pfalciparum),"
    "TTK", "MYO3B", "CLK4", "PRKG1", "MAP4K2", "LZK", "RIOK1", "EGFR(L858RT790M)", "LCK", "FRK", "PLK1,"
    "DYRK1A", "TSSK1B", "MST3", "CSNK1A1L", "EGFR(L861Q)", "FAK", "ABL1(T315I)p", "MET(Y1235D)", "PIM2,"
    "TRKC", "RPS6KA4(KinDom.2-C-terminal)", "TNIK", "FLT3(D835Y)", "PAK7", "PAK3", "QSK", "MKNK1", "VRK2,"
    "PIM1", "MKK7", "CSNK1A1", "ROCK2", "RET(V804L)", "MEK5", "ARK5", "FER", "CDK5", "ERK8", "RIPK5", "NLK,"
    "PIP5K1C", "PKAC-beta", "ABL1", "CAMK2G", "MEK6", "RIOK2", "ABL1(M351T)", "CSNK2A1", "ZAP70", "RSK2(KinDom.1-N-terminal),"
    "TESK1", "STK36", "CDK9", "CAMK2B", "ABL1(F317I)p", "HUNK", "NEK1", "TAOK3", "MST1R", "YSK1", "CTK,"
    "MYLK2", "PIM3", "PIK3CG", "FLT4", "HPK1", "AURKB", "PKNB(Mtuberculosis)", "SRMS", "ICK", "TLK1", "CSNK1G1,"
    "FLT1", "PAK1", "NEK4", "RPS6KA4(KinDom.1-N-terminal)", "MYLK", "DYRK2", "CDK11", "GSK3B", "CDC2L5,"
    "MAPKAPK5", "DAPK2", "MLK1", "WEE2", "DCAMKL3", "TRPM6", "FYN", "ROCK1", "MELK", "FGFR1", "ULK1", "SNARK,"
    "FES", "PLK4", "TAOK2", "MAP3K15", "EPHB2", "CAMK1D", "RSK3(KinDom.2-C-terminal)", "EPHA8", "TYK2(JH1domain-catalytic),"
    "TYRO3", "HIPK3", "BMPR1B", "CDK2", "ZAK", "LATS1", "ABL1(Q252H)", "RSK3(KinDom.1-N-terminal)", "FLT3(ITD),"
    "ABL1(F317L)", "MAP4K4", "LTK", "PYK2", "TAOK1", "SIK", "RIPK2", "PAK4", "MTOR", "EPHB4", "ANKK1", "MAP3K3,"
    "JAK1(JH2domain-pseudokinase)", "CSNK1G2", "MUSK", "ULK2", "ABL1(H396P)", "ABL1(Y253F)", "STK16", "ABL2,"
    "FLT3(D835H)", "MAP4K5", "TGFBR2", "PRP4", "PIK3CB", "ALK", "ABL1(Q252H)p", "CDKL1", "EGFR(G719C)", "AKT2,"
    "EPHA7", "ULK3", "TRKA", "ABL1(T315I)", "MEK2", "SBK1", "RET(V804M)", "HIPK4", "CAMK2A", "ASK1", "CLK1,"
    "PFTK1", "JNK1", "YANK2", "DMPK", "MARK1", "p38-alpha", "MLCK", "PRKD1", "MARK4", "ASK2", "DYRK1B", "FGR,"
    "EPHB6", "ITK", "PFTAIRE2", "SRPK1", "ABL1(H396P)p", "ERK1", "ABL1p", "DDR2", "DMPK2", "SRC", "JNK3,"
    "YANK1", "CDK4-cyclinD3", "MET", "PIK3C2G", "GRK4", "PKMYT1", "NEK6", "STK33", "ERK4", "MRCKB", "CDK8,"
    "NEK11", "ACVR1B", "TNNI3K", "DRAK2", "EPHA1", "EGFR(L747T751del)", "ERK2", "DLK", "PDGFRB", "TGFBR1,"
    "CAMK2D", "EGFR(T790M)", "GSK3A", "PAK6", "BMX", "LKB1", "IGF1R", "MYLK4", "AKT3", "BLK", "EPHB1", "CDK7,"
    "MAPKAPK2", "PCTK2", "FGFR4", "EGFR(L858R)", "NIM1", "DDR1", "PIK3CD", "CASK", "MAP3K2", "CDK3", "IRAK3,"
    "MST4", "EGFR(G719S)", "SNRK", "BMPR1A", "AURKC", "PRKCI", "EGFR(E746A750del)", "CAMK4", "PFCDPK1(Pfalciparum),"
    "PAK2", "AXL", "MAST1", "PRKCH", "CLK3", "NDR1", "GRK7", "MET(M1250T)", "DRAK1", "EPHA2", "PRKX", "AMPK-alpha1,"
    "TXK", "SRPK3", "RIOK3", "FLT3(K663Q)", "CSNK2A2", "CIT", "DCAMKL1", "LRRK2(G2019S)", "PRKD2", "EPHA3,"
    "BTK", "p38-delta", "ACVR1", "CAMK1G", "LRRK2", "PCTK1", "BRSK2", "JNK2", "MAP4K3"]
#%%
import pandas as pd
train_df = pd.read_csv('/cluster/home/t122995uhn/projects/data/DavisKibaDataset/davis/nomsa_binary_original_binary/full/XY.csv')
test_df = pd.read_csv('/cluster/home/t122995uhn/projects/MutDTA/splits/davis/test.csv')
train_df = train_df[~train_df.prot_id.isin(set(test_df.prot_id))]

# %%
import pandas as pd

davis_test_df = pd.read_csv(f"/home/jean/projects/MutDTA/splits/davis/test.csv")
davis_test_df['gene'] = davis_test_df['prot_id'].str.split('(').str[0]

#%% ONCO KB MERGE
onco_df = pd.read_csv("../data/oncoKB_DrugGenePairList.csv")
davis_join_onco = davis_test_df.merge(onco_df.drop_duplicates("gene"), on="gene", how="inner")

# %%
onco_df = pd.read_csv("../data/oncoKB_DrugGenePairList.csv")
onco_df.merge(davis_test_df.drop_duplicates("gene"), on="gene", how="inner").value_counts("gene")









# %%
from src.train_test.splitting import resplit
from src import cfg

db_p = lambda x: f'{cfg.DATA_ROOT}/DavisKibaDataset/davis/nomsa_{x}_gvp_binary'

db = resplit(dataset=db_p('binary'), split_files=db_p('aflow'), use_train_set=True)



# %%
########################################################################
########################## VIOLIN PLOTTING #############################
########################################################################
import logging
from matplotlib import pyplot as plt

from src.analysis.figures import prepare_df, custom_fig, fig_combined

models = {
    'DG': ('nomsa', 'binary', 'original', 'binary'),
    'esm': ('ESM', 'binary', 'original', 'binary'), # esm model
    'aflow': ('nomsa', 'aflow', 'original', 'binary'),
    # 'gvpP': ('gvp', 'binary', 'original', 'binary'),
    'gvpL': ('nomsa', 'binary', 'gvp', 'binary'),
    # 'aflow_ring3': ('nomsa', 'aflow_ring3', 'original', 'binary'),
    'gvpL_aflow': ('nomsa', 'aflow', 'gvp', 'binary'),
    # 'gvpL_aflow_rng3': ('nomsa', 'aflow_ring3', 'gvp', 'binary'),
    #GVPL_ESMM_davis3D_nomsaF_aflowE_48B_0.00010636872718329864LR_0.23282479481785903D_2000E_gvpLF_binaryLE
    # 'gvpl_esm_aflow': ('ESM', 'aflow', 'gvp', 'binary'),
}

df = prepare_df('/cluster/home/t122995uhn/projects/MutDTA/results/v113/model_media/model_stats.csv')
fig, axes = fig_combined(df, datasets=['davis'], fig_callable=custom_fig,
             models=models, metrics=['cindex', 'mse'],
             fig_scale=(10,5), add_stats=True, title_postfix=" test set performance")
plt.xticks(rotation=45)

df = prepare_df('/cluster/home/t122995uhn/projects/MutDTA/results/v113/model_media/model_stats_val.csv')
fig, axes = fig_combined(df, datasets=['davis'], fig_callable=custom_fig,
             models=models, metrics=['cindex', 'mse'],
             fig_scale=(10,5), add_stats=True, title_postfix=" validation set performance")
plt.xticks(rotation=45)

#%%
from src.data_prep.init_dataset import create_datasets
from src import cfg
import logging
cfg.logger.setLevel(logging.DEBUG)

splits = '/cluster/home/t122995uhn/projects/MutDTA/splits/kiba/'
create_datasets(cfg.DATA_OPT.kiba, 
                feat_opt=cfg.PRO_FEAT_OPT.nomsa, 
                edge_opt=cfg.PRO_EDGE_OPT.aflow,
                ligand_features=cfg.LIG_FEAT_OPT.gvp, 
                ligand_edges=cfg.LIG_EDGE_OPT.binary,
                k_folds=5, 
                test_prots_csv=f'{splits}/test.csv',
                val_prots_csv=[f'{splits}/val{i}.csv' for i in range(5)])

# %%
from src.data_prep.init_dataset import create_datasets
from src import cfg
import logging
cfg.logger.setLevel(logging.DEBUG)

splits = '/cluster/home/t122995uhn/projects/MutDTA/splits/kiba/'
create_datasets(cfg.DATA_OPT.kiba, 
                feat_opt=cfg.PRO_FEAT_OPT.nomsa, 
                edge_opt=[cfg.PRO_EDGE_OPT.binary, cfg.PRO_EDGE_OPT.aflow],
                ligand_features=[cfg.LIG_FEAT_OPT.original, cfg.LIG_FEAT_OPT.gvp], 
                ligand_edges=cfg.LIG_EDGE_OPT.binary,
                k_folds=5, 
                test_prots_csv=f'{splits}/test.csv',
                val_prots_csv=[f'{splits}/val{i}.csv' for i in range(5)])

# %%
from src.utils.loader import Loader

db_aflow = Loader.load_dataset('../data/DavisKibaDataset/davis/nomsa_aflow_original_binary/full')
db = Loader.load_dataset('../data/DavisKibaDataset/davis/nomsa_binary_original_binary/full')

# %%
# 5-fold cross validation + test set
import pandas as pd
from src import cfg
from src.train_test.splitting import balanced_kfold_split
from src.utils.loader import Loader
test_df = pd.read_csv('/cluster/home/t122995uhn/projects/MutDTA/splits/pdbbind_test.csv')
test_prots = set(test_df.prot_id)
db = Loader.load_dataset(f'{cfg.DATA_ROOT}/PDBbindDataset/nomsa_binary_original_binary/full/')

train, val, test = balanced_kfold_split(db,
                k_folds=5, test_split=0.1, val_split=0.1, 
                test_prots=test_prots, random_seed=0, verbose=True
                )

#%%
db.save_subset_folds(train, 'train')
db.save_subset_folds(val, 'val')
db.save_subset(test, 'test')

#%%
import shutil, os

src = "/cluster/home/t122995uhn/projects/data/PDBbindDataset/nomsa_binary_original_binary/"
dst = "/cluster/home/t122995uhn/projects/MutDTA/splits/pdbbind"
os.makedirs(dst, exist_ok=True)

for i in range(5):
    sfile = f"{src}/val{i}/XY.csv"
    dfile = f"{dst}/val{i}.csv"
    shutil.copyfile(sfile, dfile)

# %%
