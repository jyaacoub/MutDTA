#%% get metrics across different quality pdbs
from collections import OrderedDict
from typing import Tuple
import random, os

import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from lifelines.utils import concordance_index
from tqdm import tqdm

from src.data_analysis import get_metrics
#%% Function to get missing residue counts:
pdb_path = lambda x: f'/home/jyaacoub/projects/data/refined-set/{x}/{x}_protein.pdb'
def count_missing_res(pdb_file: str) -> Tuple[int,int]:
    """
    Returns the number of missing residues

    Parameters
    ----------
    `pdb_file` : str
        The path to the PDB file

    Returns
    -------
    Tuple[int,int]
        Gap instances, number of missing residues
    """

    chains = OrderedDict() # chain dict of dicts
    # read and filter
    with open(pdb_file, 'r') as f:
        lines = f.readlines()

        ter = 0 # chain terminator
        chains[0] = [] # res counts for first chain
        for line in lines:
            if (line[:6].strip() == 'TER'): # TER indicates new chain "terminator"
                ter += 1
                chains[ter] = [] # res count
            
            if (line[:6].strip() != 'ATOM'): continue # skip non-atom lines

            # appending res count to list
            curr_res = int(line[22:26])
            chains[ter].append(curr_res)
    
    # counting number of missing residues
    num_missing = 0
    num_gaps = 0
    for c in chains.values():
        curr, prev = None, None
        for res_num in sorted(c):
            prev = curr
            curr = res_num
            if (prev is not None) and \
                (curr != prev and curr != prev+1):
                num_missing += curr - prev
                num_gaps +=1
    
    return num_gaps, num_missing


#%%
run_num=9
y_path = 'data/PDBbind/kd_ki/Y.csv'
vina_out = f'results/PDBbind/vina_out/run{run_num}.csv'
save_path = 'results/PDBbind/media/kd_ki'

##%%
vina_pred = pd.read_csv(vina_out) # vina_deltaG(kcal/mol),vina_kd(uM)
vina_pred['vina_pkd'] = -np.log(vina_pred['vina_kd(uM)']*1e-6)
vina_pred.drop(['vina_deltaG(kcal/mol)', 'vina_kd(uM)'], axis=1, inplace=True)

actual = pd.read_csv(y_path)      # affinity (in uM)
actual['actual_pkd'] = -np.log(actual['affinity']*1e-6)
actual.drop('affinity', axis=1, inplace=True)

mrgd = actual.merge(vina_pred, on='PDBCode')

#%% 
col = 'missing_res'
# col='res'
# res_file = '/home/jyaacoub/projects/data/refined-set/index/INDEX_refined_set.2020'
# with open(res_file, 'r') as f:
#     lines = f.readlines()
#     print(len(lines))
#     res_dict = {}
#     for l in lines:
#         if l[0] == '#': continue
#         code = l[0:5].strip()
#         res = float(l[5:10])
#         res_dict[code] = res

# df_ = pd.DataFrame.from_dict(res_dict, orient='index', columns=['res'])
missing = [count_missing_res(pdb_path(code))[1] for code in tqdm(mrgd['PDBCode'], 'Getting missing count')]

mrgd[col] = missing

#%%
df_ = mrgd.sort_values(by=col)

num_bins=10
bin_size = int(len(df_)/num_bins)
print(f'bin size: {bin_size}')
bins = {} # dic of pd dfs and avg for col
for i in range(num_bins):
    df_bin = df_.iloc[i*bin_size:(i+1)*bin_size]
    avg = df_bin[col].max()
    bins[i] = (avg, df_bin)

# %% 
metrics = []
for i in range(num_bins):
    df_b = bins[i][1]
    pkd_y, pkd_z = df_b['actual_pkd'].to_numpy(), df_b['vina_pkd'].to_numpy()
    print(f'\nBin {i}, size: {len(df_b)}, {col}: {bins[i][0]}')
    metrics.append(get_metrics(pkd_y, pkd_z, save_results=False, show=False))
print("sample metrics:", *metrics[0])

# %%
options = ['c_index', 'p_corr', 's_corr', 'mse', 'mae', 'rmse']
choice = 0
# cindex scatter by col
bin_ = [str(v[0]) for v in bins.values()]
cindex = [m[0] for m in metrics]
plt.bar(bin_, cindex)
plt.ylim((0.5,0.9))
plt.xlabel('Avg missing res')
if num_bins > 20: plt.xticks(rotation=30)
plt.ylabel('C-index')
plt.title(f'Vina cindex vs {col}')

# %%
