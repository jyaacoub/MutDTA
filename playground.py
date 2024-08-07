# # %%
# import numpy as np
# import torch

# d = torch.load("/cluster/home/t122995uhn/projects/data/v131/DavisKibaDataset/davis/nomsa_aflow_original_binary/full/data_pro.pt")
# np.array(list(d['ABL1(F317I)p'].pro_seq))[d['ABL1(F317I)p'].pocket_mask].shape



# %%
# building pocket datasets:
from src.utils.pocket_alignment import pocket_dataset_full
import shutil
import os

data_dir = '/cluster/home/t122995uhn/projects/data/'
db_type = ['kiba', 'davis']
db_feat = ['nomsa_binary_original_binary', 'nomsa_aflow_original_binary', 
           'nomsa_binary_gvp_binary',      'nomsa_aflow_gvp_binary']

for t in db_type:
    for f in db_feat:
        print(f'\n---{t}-{f}---\n')
        dataset_dir= f"{data_dir}/DavisKibaDataset/{t}/{f}/full"
        save_dir   = f"{data_dir}/v131/DavisKibaDataset/{t}/{f}/full"
        
        pocket_dataset_full(
            dataset_dir= dataset_dir,
            pocket_dir = f"{data_dir}/{t}/",
            save_dir   = save_dir,
            skip_download=True
        )
        

#%%
import pandas as pd

def get_test_oncokbs(train_df=pd.read_csv('/cluster/home/t122995uhn/projects/data/test/PDBbindDataset/nomsa_binary_original_binary/full/cleaned_XY.csv'),
                     oncokb_fp='/cluster/home/t122995uhn/projects/data/tcga/mart_export.tsv', 
                     biomart='/cluster/home/t122995uhn/projects/downloads/oncoKB_DrugGenePairList.csv'):
    #Get gene names for PDBbind
    dfbm = pd.read_csv(oncokb_fp, sep='\t')
    dfbm['PDB ID'] = dfbm['PDB ID'].str.lower()
    train_df.reset_index(names='idx',inplace=True)

    df_uni = train_df.merge(dfbm, how='inner', left_on='prot_id', right_on='UniProtKB/Swiss-Prot ID')
    df_pdb = train_df.merge(dfbm, how='inner', left_on='code', right_on='PDB ID')

    # identifying ovelap with oncokb
    # df_all will have duplicate entries for entries with multiple gene names...
    df_all = pd.concat([df_uni, df_pdb]).drop_duplicates(['idx', 'Gene name'])[['idx', 'code', 'Gene name']]

    dfkb = pd.read_csv(biomart)
    df_all_kb = df_all.merge(dfkb.drop_duplicates('gene'), left_on='Gene name', right_on='gene', how='inner')

    trained_genes = set(df_all_kb.gene)

    #Identify non-trained genes
    return dfkb[~dfkb['gene'].isin(trained_genes)], dfkb[dfkb['gene'].isin(trained_genes)], dfkb


train_df = pd.read_csv('/cluster/home/t122995uhn/projects/data/test/PDBbindDataset/nomsa_binary_original_binary/train0/cleaned_XY.csv')
val_df = pd.read_csv('/cluster/home/t122995uhn/projects/data/test/PDBbindDataset/nomsa_binary_original_binary/val0/cleaned_XY.csv')

train_df = pd.concat([train_df, val_df])

get_test_oncokbs(train_df=train_df)





#%%
##############################################################################
########################## BUILD/SPLIT DATASETS ##############################
##############################################################################
import os
from src.data_prep.init_dataset import create_datasets
from src import cfg
import logging
cfg.logger.setLevel(logging.DEBUG)

dbs = [cfg.DATA_OPT.davis, cfg.DATA_OPT.kiba]
splits = ['davis', 'kiba']
splits = ['/cluster/home/t122995uhn/projects/MutDTA/splits/' + s for s in splits]
print(splits)

#%%
for split, db in zip(splits, dbs):
    print('\n',split, db)
    create_datasets(db, 
                feat_opt=cfg.PRO_FEAT_OPT.nomsa, 
                edge_opt=[cfg.PRO_EDGE_OPT.binary, cfg.PRO_EDGE_OPT.aflow],
                ligand_features=[cfg.LIG_FEAT_OPT.original, cfg.LIG_FEAT_OPT.gvp], 
                ligand_edges=cfg.LIG_EDGE_OPT.binary, overwrite=False,
                k_folds=5,
                test_prots_csv=f'{split}/test.csv',
                val_prots_csv=[f'{split}/val{i}.csv' for i in range(5)])

#%% TEST INFERENCE
from src import cfg
from src.utils.loader import Loader

# db2 = Loader.load_dataset(cfg.DATA_OPT.davis, 
#                          cfg.PRO_FEAT_OPT.nomsa, cfg.PRO_EDGE_OPT.aflow,
#                          path='/cluster/home/t122995uhn/projects/data/',
#                          subset="full")

db2 = Loader.load_DataLoaders(cfg.DATA_OPT.davis, 
                         cfg.PRO_FEAT_OPT.nomsa, cfg.PRO_EDGE_OPT.aflow,
                         path='/cluster/home/t122995uhn/projects/data/v131',
                         training_fold=0,
                         batch_train=2)
for b2 in db2['test']: break


# %%
m = Loader.init_model(cfg.MODEL_OPT.DG, cfg.PRO_FEAT_OPT.nomsa, cfg.PRO_EDGE_OPT.aflow,
                  dropout=0.3480, output_dim=256,
                  )

#%%
# m(b['protein'], b['ligand'])
m(b2['protein'], b2['ligand'])
#%%
model = m
loaders = db2
device = 'cpu'
NUM_EPOCHS = 1
LEARNING_RATE = 0.001
from src.train_test.training import train

logs = train(model, loaders['train'], loaders['val'], device, 
            epochs=NUM_EPOCHS, lr_0=LEARNING_RATE)
# %%
