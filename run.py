#%% testing download of sequences from FASTA files
import pandas as pd

y_path = 'data/PDBbind/kd_ki/Y.csv'
vina_out = 'results/PDBbind/vina_out/run1.csv'

#%%
vina_pred = pd.read_csv(vina_out)
actual = pd.read_csv(y_path)

# %%
mrgd = actual.merge(vina_pred, on='PDBCode')
# %%
