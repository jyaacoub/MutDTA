#%%
import logging
logging.getLogger().setLevel(logging.DEBUG)

from src import config as cfg
from src.data_prep.init_dataset import create_datasets
from src.utils.loader import Loader

# create_datasets(cfg.DATA_OPT.PDBbind, cfg.PRO_FEAT_OPT.nomsa, cfg.PRO_EDGE_OPT.aflow_ring3,
#                 ligand_features=cfg.LIG_FEAT_OPT.gvp, ligand_edges=cfg.LIG_EDGE_OPT.binary,
#                 k_folds=5, overwrite=False)

l = Loader.load_DataLoaders(cfg.DATA_OPT.PDBbind, cfg.PRO_FEAT_OPT.nomsa, cfg.PRO_EDGE_OPT.aflow_ring3,
                            training_fold=0, ligand_feature=cfg.LIG_FEAT_OPT.gvp, batch_train=2)

# %%

m = Loader.init_model(cfg.MODEL_OPT.GVPL_RNG, cfg.PRO_FEAT_OPT.nomsa, cfg.PRO_EDGE_OPT.aflow_ring3,
                  dropout=0.2)
# %%
sample = next(iter(l['train']))

m(sample['protein'], sample['ligand'])
# %%
