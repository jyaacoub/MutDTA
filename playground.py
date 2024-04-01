# # %%
# from src.data_prep.init_dataset import create_datasets
# from src import config as cfg

# create_datasets(cfg.DATA_OPT.PDBbind, 
#                 cfg.PRO_FEAT_OPT.nomsa, 
#                 cfg.PRO_EDGE_OPT.binary,
#                 ligand_features=cfg.LIG_FEAT_OPT.gvp,
#                 k_folds=5)

#%%
from src.utils.loader import Loader
from src import config as cfg

l = Loader.load_DataLoaders(cfg.DATA_OPT.PDBbind, cfg.PRO_FEAT_OPT.nomsa,
                        cfg.PRO_EDGE_OPT.binary,
                        ligand_feature=cfg.LIG_FEAT_OPT.gvp,
                        training_fold=0, datasets=['train', 'test'],
                        batch_train=2)

# %%
sample = next(iter(l['test']))
# %%
from src.models.gvp_models import GVPLigand_DGPro
m = GVPLigand_DGPro()

# %%
sample = next(iter(l['train']))

out = m(sample['protein'], sample['ligand'])
out.shape
# %%
outl = m.gvp_ligand(sample['ligand'])
outl.shape

outp = m.forward_pro(sample['protein'])

# %%
import torch
xc = torch.cat((outl, outp), 1)

# %%
m.dense_combined_out(xc)
# %%
