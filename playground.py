# #%%
# from src.data_prep.init_dataset import create_datasets
# from src import config as cfg

# create_datasets([cfg.DATA_OPT.PDBbind], [cfg.PRO_FEAT_OPT.nomsa],
#                 [cfg.PRO_EDGE_OPT.aflow], k_folds=5)

# %% test aflow dataset on DG model
from src import config as cfg
from src.utils.loader import Loader
from src.models.prior_work import DGraphDTA



mdl: DGraphDTA = Loader.init_model("DG", "nomsa", cfg.PRO_EDGE_OPT.aflow, 0.5)
db = Loader.load_DataLoaders(cfg.DATA_OPT.PDBbind, cfg.PRO_FEAT_OPT.nomsa, cfg.PRO_EDGE_OPT.aflow,
                             datasets=['test'], batch_train=1)
sample = next(iter(db['test']))

# %%
p = sample['protein']
out = mdl.pro_conv1(p.x, p.edge_index, p.edge_weight[:,1]) # edge weight is incorrect shape! must be [E,1] and not [E,6]


# %%
from tqdm import tqdm
mdl.train()

for sample in tqdm(db['test']):
    mdl.forward(sample['protein'], sample['ligand'])

# %%
import numpy as np

v0 = np.load('../data/pdbbind/v2020-other-PL/edge_weights/aflow/1eub.npy')
v1 = np.load('../data/pdbbind/v2020-other-PL/edge_weights/aflow_ring3/1eub.npy')

# %%
(v0 == v1).all()
# %%
