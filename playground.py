#%%
from pathlib import Path
from src.utils.residue import Ring3Runner
import os
import logging
logging.getLogger().setLevel(logging.DEBUG)

# %%
sample_pdb = f'{Path.home()}/projects/data/misc/7LQT.pdb'
out_dir = f'{Path.home()}/projects/data/misc/ring3_out'

# %%
Ring3Runner.cleanup(sample_pdb, out_dir, all=True)
files = Ring3Runner.run(sample_pdb, out_dir)
# %%
import pandas as pd
df = pd.read_csv(edge_fp, sep='\t')
# %%
df[df.Interaction.str.contains('HBOND')]

# %%
