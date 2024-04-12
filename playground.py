# %% Create and test on platinum dataset:
from src.data_prep.init_dataset import create_datasets
from src import config as cfg

create_datasets(cfg.DATA_OPT.platinum,
                cfg.PRO_FEAT_OPT.nomsa,
                cfg.PRO_EDGE_OPT.binary,
                ligand_features=cfg.LIG_FEAT_OPT.original,
                ligand_edges=cfg.LIG_EDGE_OPT.binary,
                k_folds=None,
                train_split=0.0, val_split=0.0) # just a test dataset, no training can be done on this

#%%
