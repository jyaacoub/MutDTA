#%%
from glob import glob
from pathlib import Path
from src.utils.residue import Ring3Runner
import os
import logging
logging.getLogger().setLevel(logging.DEBUG)

# %%
pdb_7lqt = f'{Path.home()}/projects/data/misc/7LQT.pdb'

af_conf_dir = f'{Path.home()}/projects/data/misc/'
af_confs_EGFR = glob(f'{af_conf_dir}/EGFR*/EGFR_unrelaxed_rank_*.pdb')


#%%
from src.utils.residue import Chain
import matplotlib.pyplot as plt
import numpy as np
af_confs = af_confs_EGFR

#%% get contact maps for distance matrix
chains = [Chain(p) for p in af_confs]
M = np.array([c.get_contact_map() for c in chains]) < 8.0

dist_cmap = np.sum(M, axis=0) / len(M)

# %% ring3 edge attribute extraction
# Note: this will create a "combined" pdb file in the same directory as the confirmaions
input_pdb, files = Ring3Runner.run(pdb_7lqt, overwrite=False)
seq_len = len(Chain(input_pdb))

# %% Converts output files into LxLx6 matrix for the 6 ring3 edge attributes
r3_cmaps = []
for k, fp in files.items():
    cmap = Ring3Runner.build_cmap(fp, seq_len)
    r3_cmaps.append(cmap)

# %% COMBINE convert to numpy array
# plot all 6 cmaps
fig, axs = plt.subplots(2,3, figsize=(15,10))
dist_cmap = np.zeros_like(r3_cmaps[0])

ks = list(files.keys()) + ['dist']
for i, cmap in enumerate(r3_cmaps + [dist_cmap]):
    ax = axs[i//3, i%3]
    ax.matshow(cmap)
    ax.set_title(ks[i])

#%% Convert to numpy array of shape (6, L, L)
r3_cmaps = np.array(r3_cmaps + [dist_cmap])

#%%
# deletes all intermediate output files, since the main LxLx6 matrix should be saved at the end
Ring3Runner.cleanup(input_pdb, all=True)