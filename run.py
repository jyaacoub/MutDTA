#%%
import pandas as pd

#%% Filtering out database to get only PDBs that didnt have errors
err = pd.read_csv('/cluster/home/t122995uhn/projects/MutDTA/src/docking/bash_scripts/pdb_error.txt',names=['PDBCode'], index_col=0)
full = pd.read_csv('/cluster/home/t122995uhn/projects/MutDTA/data/PDBbind/kd_ki/info.csv', index_col=0)

full_m = full.merge(err, how='left', on='PDBCode', indicator=True) # indicator gives _merge col
full_m[full_m._merge == 'left_only']