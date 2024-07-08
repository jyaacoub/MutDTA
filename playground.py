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
