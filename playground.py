# %%
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

from src.analysis.figures import prepare_df, fig_combined, custom_fig

dft = prepare_df('./results/v115/model_media/model_stats.csv')
dfv = prepare_df('./results/v115/model_media/model_stats_val.csv')

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

fig, axes = fig_combined(dft, datasets=['davis'], fig_callable=custom_fig,
             models=models, metrics=['cindex', 'mse'],
             fig_scale=(10,5), add_stats=True, title_postfix=" test set performance", box=True, fold_labels=True)
plt.xticks(rotation=45)

fig, axes = fig_combined(dfv, datasets=['davis'], fig_callable=custom_fig,
             models=models, metrics=['cindex', 'mse'],
             fig_scale=(10,5), add_stats=True, title_postfix=" validation set performance", box=True, fold_labels=True)
plt.xticks(rotation=45)

#%%
########################################################################
########################## BUILD DATASETS ##############################
########################################################################
from src.data_prep.init_dataset import create_datasets
from src import cfg
import logging
cfg.logger.setLevel(logging.DEBUG)

splits = '/cluster/home/t122995uhn/projects/MutDTA/splits/pdbbind/'
create_datasets(cfg.DATA_OPT.PDBbind, 
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
