
# %%
from pathlib import Path
from glob import glob
from src.utils.residue import Chain
from src.data_prep.feature_extraction.protein_edges import get_target_edge_weights
from src import config as cfg
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv
import torch

af_conf_dir = f'{Path.home()}/projects/data/misc/'
af_confs_EGFR = glob(f'{af_conf_dir}/EGFR*/EGFR_unrelaxed_rank_*00.pdb')

target = Chain(af_confs_EGFR[0])
L = len(target)
x = torch.rand(L, 2, dtype=torch.float32) # [N, feat_dim]

# %% get edge information
dist_cmap = target.get_contact_map() < 8.0
ei = torch.tensor(np.tril(dist_cmap)).nonzero().T # [2, E]

ea = get_target_edge_weights('', target.sequence, 
                             edge_opt=cfg.EDGE_OPT.ring3.value,
                             af_confs=af_confs_EGFR)
# using only the first cmap to determine which edge values to use
ea = torch.Tensor(ea[ei[0], ei[1], :]) # [E, 6]

sample_data = Data(x=x, edge_index=ei,edge_attr=ea)

#%%
model = TransformerConv(in_channels=2, out_channels=2, heads=1, 
                        edge_dim=6) # 6 edge attributes
model(sample_data.x, sample_data.edge_index, sample_data.edge_attr).shape
# %%
