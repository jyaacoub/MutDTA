#%%
from glob import glob
from pathlib import Path
from src.utils.residue import Ring3Runner
import os
import logging
logging.getLogger().setLevel(logging.INFO)

# %%
pdb_7lqt = f'{Path.home()}/projects/data/misc/7LQT.pdb'

af_conf_dir = f'{Path.home()}/projects/data/misc/'
af_confs_EGFR = glob(f'{af_conf_dir}/EGFR*/EGFR_unrelaxed_rank_*.pdb')


#%%
from src.utils.residue import Chain
import matplotlib.pyplot as plt
import numpy as np
opt = af_confs_EGFR
opt = pdb_7lqt
thr = 8.0

for opt in [af_confs_EGFR]:
    # get distance contact map
    if opt is af_confs_EGFR:
        chains = [Chain(p) for p in opt]
        M = np.array([c.get_contact_map() for c in chains]) < thr
        dist_cmap = np.sum(M, axis=0) / len(M)
    else:
        dist_cmap = Chain(opt).get_contact_map() < thr

    # ring3 edge attribute extraction
    # Note: this will create a "combined" pdb file in the same directory as the confirmaions
    input_pdb, files = Ring3Runner.run(opt, overwrite=True)
    seq_len = len(Chain(input_pdb))

    # Converts output files into LxLx6 matrix for the 6 ring3 edge attributes
    r3_cmaps = []
    for k, fp in files.items():
        cmap = Ring3Runner.build_cmap(fp, seq_len)
        r3_cmaps.append(cmap)

    # COMBINE convert to numpy array
    # plot all 6 cmaps
    fig, axs = plt.subplots(2,3, figsize=(15,10))

    ks = list(files.keys()) + ['dist']
    for i, cmap in enumerate(r3_cmaps + [dist_cmap]):
        ax = axs[i//3, i%3]
        ax.matshow(cmap)
        ax.set_title(ks[i])

    plt.suptitle(f'Ring3 Edge Attributes for {"EGFR" if opt is af_confs_EGFR else "7LQT"}')
    plt.show()
    
#%%
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data
import torch

model = TransformerConv(in_channels=2, out_channels=2, heads=1, 
                        edge_dim=6) # 6 edge attributes
# %% Real data:
L = seq_len
x = torch.rand(L, 2, dtype=torch.float32) # [N, feat_dim]


ei = torch.tensor(np.tril(dist_cmap)).nonzero().T # [2, E]

all_cmaps = np.array(r3_cmaps + [dist_cmap], dtype=np.float32) # [6, L, L]
all_cmaps = torch.tensor(all_cmaps, dtype=torch.float32).permute(1,2,0) # [L, L, 6]

# using only the first cmap to determine which edge values to use
ea = all_cmaps[ei[0], ei[1], :] # [E, 6]

sample_data = Data(x=x, edge_index=ei,edge_attr=ea)

# %%
model(sample_data.x, sample_data.edge_index, sample_data.edge_attr).shape

# %%
# from src.data_prep.feature_extraction.protein_edges import get_target_edge_weights
# from src import config as cfg

# seq = Chain(pdb_7lqt).sequence
# ea = get_target_edge_weights(pdb_7lqt, seq, edge_opt=cfg.EDGE_OPT.ring3,
#                              af_confs=)