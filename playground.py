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
########################################################################
########################## BUILD DATASETS ##############################
########################################################################
import os
from src.data_prep.init_dataset import create_datasets
from src import cfg
import logging
cfg.logger.setLevel(logging.DEBUG)

splits = '/cluster/home/t122995uhn/projects/MutDTA/splits/davis/'
create_datasets([cfg.DATA_OPT.PDBbind, cfg.DATA_OPT.davis, cfg.DATA_OPT.kiba], 
                feat_opt=cfg.PRO_FEAT_OPT.nomsa, 
                edge_opt=[cfg.PRO_EDGE_OPT.binary, cfg.PRO_EDGE_OPT.aflow],
                ligand_features=[cfg.LIG_FEAT_OPT.original, cfg.LIG_FEAT_OPT.gvp], 
                ligand_edges=cfg.LIG_EDGE_OPT.binary, overwrite=False,
                k_folds=5,
                test_prots_csv=f'{splits}/test.csv',
                val_prots_csv=[f'{splits}/val{i}.csv' for i in range(5)],)
                # data_root=os.path.abspath('../data/test/'))

# %% Copy splits to commit them:
#from to:
import shutil
from_dir_p = '/cluster/home/t122995uhn/projects/data/v131/'
to_dir_p = '/cluster/home/t122995uhn/projects/MutDTA/splits/'
from_db = ['PDBbindDataset', 'DavisKibaDataset/kiba', 'DavisKibaDataset/davis']
to_db = ['pdbbind', 'kiba', 'davis']

from_db = [f'{from_dir_p}/{f}/nomsa_binary_original_binary/' for f in from_db]
to_db = [f'{to_dir_p}/{f}' for f in to_db]

for src, dst in zip(from_db, to_db):
    for x in ['train', 'val']:
        for i in range(5):
            print(f"{src}/{x}{i}/XY.csv", f"{dst}/{x}{i}.csv")
            shutil.copy(f"{src}/{x}{i}/XY.csv", f"{dst}/{x}{i}.csv")
    
    print(f"{src}/test/XY.csv", f"{dst}/test.csv")
    shutil.copy(f"{src}/test/XY.csv", f"{dst}/test.csv")
    
    
# %%
