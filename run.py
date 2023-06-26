#%%TODO: get metrics across different quality pdbs
import random, os
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from lifelines.utils import concordance_index

from src.data_analysis import get_metrics
#%%

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
res_file = '/home/jyaacoub/projects/data/refined-set/index/INDEX_refined_set.2020'
with open(res_file, 'r') as f:
    lines = f.readlines()
    print(len(lines))
    res_dict = {}
    for l in lines:
        if l[0] == '#': continue
        code = l[0:5].strip()
        res = float(l[5:10])
        res_dict[code] = res

df_res = pd.DataFrame.from_dict(res_dict, orient='index', columns=['res'])
df_res.index.name = 'PDBCode'
df_res = mrgd.merge(df_res, on='PDBCode')

df_res.sort_values(by='res', inplace=True)

#%%
num_bins=10
bin_size = int(len(df_res)/num_bins)
print(f'bin size: {bin_size}')
bins = {} # dic of pd dfs and avg res
for i in range(num_bins):
    df_res_bin = df_res.iloc[i*bin_size:(i+1)*bin_size]
    avg_len = int(df_res_bin['res'].mean()*100)/100
    bins[i] = (avg_len, df_res_bin)

# %% 
metrics = []
for i in range(num_bins):
    df_b = bins[i][1]
    pkd_y, pkd_z = df_b['actual_pkd'].to_numpy(), df_b['vina_pkd'].to_numpy()
    print(f'\nBin {i}, size: {len(df_b)}, res: {bins[i][0]}')
    metrics.append(get_metrics(pkd_y, pkd_z, save_results=False, show=False))
print("sample metrics:", *metrics[0])

# %%
options = ['c_index', 'p_corr', 's_corr', 'mse', 'mae', 'rmse']
choice = 0
# cindex scatter by res
bin_res = [str(v[0]) for v in bins.values()]
cindex = [m[0] for m in metrics]
plt.bar(bin_res, cindex)
plt.ylim((0.5,0.9))
plt.xlabel('Resolution (A)')
plt.ylabel('C-index')
plt.title('Vina cindex vs resolution')

# %%
