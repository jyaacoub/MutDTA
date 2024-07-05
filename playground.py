# %%
import pandas as pd
import logging
DATA_ROOT = '../data'
biom_df = pd.read_csv(f'{DATA_ROOT}/tcga/mart_export.tsv', sep='\t')
biom_df.rename({'Gene name': 'gene'}, axis=1, inplace=True)

# %% Specific to kiba:
kiba_df = pd.read_csv(f'{DATA_ROOT}/DavisKibaDataset/kiba/nomsa_binary_original_binary/full/XY.csv')
kiba_df = kiba_df.merge(biom_df.drop_duplicates('UniProtKB/Swiss-Prot ID'), 
              left_on='prot_id', right_on="UniProtKB/Swiss-Prot ID", how='left')
kiba_df.drop(['PDB ID', 'UniProtKB/Swiss-Prot ID'], axis=1, inplace=True)

if kiba_df.gene.isna().sum() != 0: logging.warning("Some proteins failed to get their gene names!")

# %% making sure to add any matching davis prots to the kiba test set
davis_df = pd.read_csv('/cluster/home/t122995uhn/projects/MutDTA/splits/davis_test.csv')
davis_test_prots = set(davis_df.prot_id.str.split('(').str[0])
kiba_davis_gene_overlap = kiba_df[kiba_df.gene.isin(davis_test_prots)].gene.value_counts()
print("Total # of gene overlap with davis TEST set:", len(kiba_davis_gene_overlap))
print("                       # of entries in kiba:", kiba_davis_gene_overlap.sum())

# Starting off with davis test set as the initial test set:
kiba_test_df = kiba_df[kiba_df.gene.isin(davis_test_prots)]

# %% using previous kiba test db:
kiba_test_old_df = pd.read_csv('/cluster/home/t122995uhn/projects/downloads/test_prots_gene_names.csv')
kiba_test_old_df = kiba_test_old_df[kiba_test_old_df['db'] == 'kiba']
kiba_test_old_prots = set(kiba_test_old_df.gene_name)

kiba_test_df = pd.concat([kiba_test_df, kiba_df[kiba_df.gene.isin(kiba_test_old_prots)]], axis=0).drop_duplicates(['prot_id', 'lig_id'])
print("Combined kiba test set with davis matching genes size:", len(kiba_test_df))

#%% NEXT STEP IS TO ADD MORE PROTS FROM ONCOKB IF AVAILABLE.
onco_df = pd.read_csv("/cluster/home/t122995uhn/projects/downloads/oncoKB_DrugGenePairList.csv")

kiba_join_onco = set(kiba_test_df.merge(onco_df.drop_duplicates("gene"), on="gene", how="left")['gene'])

#%%
remaining_onco = onco_df[~onco_df.gene.isin(kiba_join_onco)].drop_duplicates('gene')

# match with remaining kiba:
remaining_onco_kiba_df = kiba_df.merge(remaining_onco, on='gene', how="inner")
counts = remaining_onco_kiba_df.value_counts('gene')
print(counts)
# this gives us 3680 which still falls short of our 11,808 goal for the test set size
print("total entries in kiba with remaining (not already in test set) onco genes", counts.sum()) 


#%%
# drop_duplicates is redundant but just in case.
kiba_test_df = pd.concat([kiba_test_df, remaining_onco_kiba_df], axis=0).drop_duplicates(['prot_id', 'lig_id']) 
print("Combined kiba test set with remaining OncoKB genes:", len(kiba_test_df))

# %% For the remaining 2100 entries we will just choose those randomly until we reach our target of 11808 entries
# code is from balanced_kfold_split function
from collections import Counter
import numpy as np

# Get size for each dataset and indices
dataset_size = len(kiba_df)
test_size = int(0.1 * dataset_size) # 11808
indices = list(range(dataset_size))

# getting counts for each unique protein
prot_counts = kiba_df['prot_id'].value_counts().to_dict()
prots = list(prot_counts.keys())
np.random.shuffle(prots)

# manually selected prots:
test_prots = set(kiba_test_df.prot_id)
# increment count by number of samples in test_prots
count = sum([prot_counts[p] for p in test_prots])

#%%
## Sampling remaining proteins for test set (if we are under the test_size) 
for p in prots: # O(k); k = number of proteins
    if count + prot_counts[p] < test_size:
        test_prots.add(p)
        count += prot_counts[p]

additional_prots = test_prots - set(kiba_test_df.prot_id)
print('additional prot_ids to add:', len(additional_prots))
print('                     count:', count)

#%% ADDING FINAL PROTS
rand_sample_df = kiba_df[kiba_df.prot_id.isin(additional_prots)]
kiba_test_df = pd.concat([kiba_test_df, rand_sample_df], axis=0).drop_duplicates(['prot_id', 'lig_id'])

kiba_test_df.drop(['cancerType', 'drug'], axis=1, inplace=True)
print('final test dataset for kiba:')
kiba_test_df

#%% saving
kiba_test_df.to_csv('/cluster/home/t122995uhn/projects/MutDTA/splits/kiba_test.csv', index=False)

# %%
########################################################################
########################## RESPLITTING DBS #############################
########################################################################
import os
from src.train_test.splitting import resplit
from src import cfg

csv_files = {}
for split in ['test'] + [f'val{i}' for i in range(5)]:
    csv_files[split] = f'./splits/davis_{split}.csv'
    assert os.path.exists(csv_files[split]), csv_files[split]

print(csv_files)

#%%
for d in os.listdir(f'{cfg.DATA_ROOT}/DavisKibaDataset/davis/'):
    if len(d.split('_')) < 4 or d !='nomsa_aflow_original_binary':
        print('skipping:', d)
        continue
    print('resplitting:', d)
    resplit(f'{cfg.DATA_ROOT}/DavisKibaDataset/davis/{d}', split_files=csv_files)


#%% VALIDATION OF SPLITS - Checking for overlap
import pandas as pd


for d in os.listdir(f'{cfg.DATA_ROOT}/DavisKibaDataset/davis/'):
    if len(d.split('_')) < 4:
        print('skipping:', d)
        continue
    # Define file paths
    file_paths = {
        'test': 'test/cleaned_XY.csv',
        'val0': 'val0/cleaned_XY.csv',
        'train0': 'train0/cleaned_XY.csv',
    }
    file_paths = {name: f'{cfg.DATA_ROOT}/DavisKibaDataset/davis/{d}/{path}' for name, path in file_paths.items()}    
    count = 0
    print(f"\n{'-'*10}{d}{'-'*10}")
    for k, v in file_paths.items():
        df = pd.read_csv(v)
        print(f"{k:>12}: {len(df):>6d}")
        count += len(df)
        
    print(f'            = {count:>6d}')
    
    df = f'{cfg.DATA_ROOT}/DavisKibaDataset/davis/{d}/full/cleaned_XY.csv'
    df = pd.read_csv(df)
    # print(f'            = {count:>6d}')
    print(f'Dataset Size: {len(df):>6d}')
    
    


# %%
########################################################################
########################## VIOLIN PLOTTING #############################
########################################################################
import logging
from typing import OrderedDict

import seaborn as sns
from matplotlib import pyplot as plt
from statannotations.Annotator import Annotator

from src.analysis.figures import prepare_df, custom_fig, fig_combined

models = {
    'DG': ('nomsa', 'binary', 'original', 'binary'),
    'esm': ('ESM', 'binary', 'original', 'binary'), # esm model
    'aflow': ('nomsa', 'aflow', 'original', 'binary'),
    # 'gvpP': ('gvp', 'binary', 'original', 'binary'),
    # 'gvpL': ('nomsa', 'binary', 'gvp', 'binary'),
    # 'gvpL': ('nomsa', 'binary', 'gvp', 'binary'),
    # 'aflow_ring3': ('nomsa', 'aflow_ring3', 'original', 'binary'),
    'gvpL_aflow': ('nomsa', 'aflow', 'gvp', 'binary'),
    # 'gvpL_aflow_rng3': ('nomsa', 'aflow_ring3', 'gvp', 'binary'),
    #GVPL_ESMM_davis3D_nomsaF_aflowE_48B_0.00010636872718329864LR_0.23282479481785903D_2000E_gvpLF_binaryLE
    'gvpl_esm_aflow': ('ESM', 'aflow', 'gvp', 'binary'),
}

df = prepare_df()
fig, axes = fig_combined(df, datasets=['davis'], fig_callable=custom_fig,
             models=models, metrics=['cindex', 'mse'],
             fig_scale=(10,5), add_stats=True)
plt.xticks(rotation=45)


# %%
########################################################################
########################## PLATINUM ANALYSIS ###########################
########################################################################
import torch, os
import pandas as pd

from src import cfg
from src import TUNED_MODEL_CONFIGS
from src.utils.loader import Loader
from src.train_test.training import test
from src.analysis.figures import predictive_performance, tbl_stratified_dpkd_metrics, tbl_dpkd_metrics_overlap, tbl_dpkd_metrics_in_binding

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

INFERENCE = True
VERBOSE = True
out_dir = f'{cfg.MEDIA_SAVE_DIR}/test_set_pred/'
os.makedirs(out_dir, exist_ok=True)
cp_dir = cfg.CHECKPOINT_SAVE_DIR
RAW_PLT_CSV=f"{cfg.DATA_ROOT}/PlatinumDataset/raw/platinum_flat_file.csv"

#%% load up model:
for KEY, CONFIG in TUNED_MODEL_CONFIGS.items():
    MODEL_KEY = lambda fold: Loader.get_model_key(CONFIG['model'], CONFIG['dataset'], CONFIG['feature_opt'], CONFIG['edge_opt'], 
                                    CONFIG['batch_size'], CONFIG['lr'], CONFIG['architecture_kwargs']['dropout'],
                                    n_epochs=2000, fold=fold, 
                                    ligand_feature=CONFIG['lig_feat_opt'], ligand_edge=CONFIG['lig_edge_opt'])
    print('\n\n'+ '## ' + KEY)
    OUT_PLT = lambda i: f'{out_dir}/{MODEL_KEY(i)}_PLATINUM.csv'
    db_p = f"{CONFIG['feature_opt']}_{CONFIG['edge_opt']}_{CONFIG['lig_feat_opt']}_{CONFIG['lig_edge_opt']}"
    
    if CONFIG['dataset'] in ['kiba', 'davis']:
        db_p = f"DavisKibaDataset/{CONFIG['dataset']}/{db_p}"
    else:
        db_p = f"{CONFIG['dataset']}Dataset/{db_p}"
        
    train_p = lambda set: f"{cfg.DATA_ROOT}/{db_p}/{set}0/cleaned_XY.csv"
    
    if not os.path.exists(OUT_PLT(0)) and INFERENCE:
        print('running inference!')
        cp = lambda fold: f"{cp_dir}/{MODEL_KEY(fold)}.model"
        
        model = Loader.init_model(model=CONFIG["model"], pro_feature=CONFIG["feature_opt"],
                                    pro_edge=CONFIG["edge_opt"],**CONFIG['architecture_kwargs'])

        # load up platinum test db
        loaders = Loader.load_DataLoaders(cfg.DATA_OPT.platinum,
                                    pro_feature    = CONFIG['feature_opt'], 
                                    edge_opt       = CONFIG['edge_opt'],
                                    ligand_feature = CONFIG['lig_feat_opt'], 
                                    ligand_edge    = CONFIG['lig_edge_opt'],
                                    datasets=['test'])

        for i in range(5):
            model.safe_load_state_dict(torch.load(cp(i), map_location=device))
            model.to(device)
            model.eval()

            loss, pred, actual = test(model, loaders['test'], device, verbose=True)
            
            # saving as csv with columns code, pred, actual
            # get codes from test loader
            codes, pid = [b['code'][0] for b in loaders['test']], [b['prot_id'][0] for b in loaders['test']]
            df = pd.DataFrame({'prot_id': pid, 'pred': pred, 'actual': actual}, index=codes)
            df.index.name = 'code'
            df.to_csv(OUT_PLT(i))

    # run platinum eval:
    print('\n### 1. predictive performance')
    mkdown = predictive_performance(OUT_PLT, train_p, verbose=VERBOSE, plot=False)
    print('\n### 2 Mutation impact analysis')
    print('\n#### 2.1 $\Delta pkd$ predictive performance')
    mkdn = tbl_dpkd_metrics_overlap(OUT_PLT, train_p, verbose=VERBOSE, plot=False)
    print('\n#### 2.2 Stratified by location of mutation (binding pocket vs not in binding pocket)')
    m = tbl_dpkd_metrics_in_binding(OUT_PLT, RAW_PLT_CSV, verbose=VERBOSE, plot=False)
    
# %%
dfr = pd.read_csv(RAW_PLT_CSV, index_col=0)

# add in_binding info to df
def get_in_binding(df, dfr):
    """
    df is the predicted csv with index as <raw_idx>_wt (or *_mt) where raw_idx 
    corresponds to an index in dfr which contains the raw data for platinum including 
    ('mut.in_binding_site')
        - 0: wildtype rows
        - 1: close (<8 Ang)
        - 2: Far (>8 Ang)
    """
    pocket = dfr[dfr['mut.in_binding_site'] == 'YES'].index   
    pclass = []
    for code in df.index:
        if '_wt' in code:
            pclass.append(0)
        elif int(code.split('_')[0]) in pocket:
            pclass.append(1)
        else:
            pclass.append(2)
    df['pocket'] = pclass
    return df

df = get_in_binding(pd.read_csv(OUT_PLT(0), index_col=0), dfr)
if VERBOSE: 
    cnts = df.pocket.value_counts()
    cnts.index = ['wt', 'in pocket', 'not in pocket']
    cnts.name = "counts"
    print(cnts.to_markdown(), end="\n\n")

tbl_stratified_dpkd_metrics(OUT_PLT, NORMALIZE=True, n_models=5, df_transform=get_in_binding,
                            conditions=['(pocket == 0) | (pocket == 1)', '(pocket == 0) | (pocket == 2)'], 
                            names=['in pocket', 'not in pocket'], 
                            verbose=VERBOSE, plot=True, dfr=dfr)

