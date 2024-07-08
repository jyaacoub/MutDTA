# %%
from src.data_prep.init_dataset import create_datasets
from src import cfg

splits = '/cluster/home/t122995uhn/projects/MutDTA/splits/davis/'
create_datasets(cfg.DATA_OPT.davis, 
                feat_opt=cfg.PRO_FEAT_OPT.nomsa, 
                edge_opt=cfg.PRO_EDGE_OPT.aflow,
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
