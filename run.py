#%%TODO: get metrics across different quality pdbs
import random, os
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from lifelines.utils import concordance_index

from src.data_analysis import get_metrics

#%% plot bar graph of results by row
# cols are: run,cindex,pearson,spearman,mse,mae,rmse
# df_res = pd.read_csv('results/model_media/DGraphDTA_stats.csv')[2:]
# df_res.sort_values(by='run', inplace=True)

# df_res.loc[-1] = ['vina', 0.68,0.508,0.520,17.812,3.427,4.220]

# for col in df_res.columns[1:]:
#     plt.figure()
#     bars = plt.bar(df_res['run'],df_res[col])
#     bars[0].set_color('green')
#     bars[2].set_color('green')
#     bars[-1].set_color('red')
#     plt.title(col)
#     plt.xlabel('run')
#     plt.xticks(rotation=30)
#     # plt.ylim((0.2, 0.8))
#     plt.show()
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
df_seq = pd.read_csv('data/PDBbind/kd_ki/pdb_seq.csv')
df_seq['seq_len'] = df_seq['seq'].str.len()
df_seq.drop('seq', axis=1, inplace=True)
df_seq = mrgd.merge(df_seq, on='PDBCode')

df_seq.sort_values(by='seq_len', inplace=True)

#%%
num_bins=15
bin_size = int(len(df_seq)/num_bins)
bins = {} # dic of pd dfs and avg seq len
for i in range(num_bins):
    df_seq_bin = df_seq.iloc[i*bin_size:(i+1)*bin_size]
    avg_len = int(df_seq_bin['seq_len'].mean())
    bins[i] = (avg_len, df_seq_bin)
    

# %% 
for i in range(num_bins):
    len = bins[i][0]
    df_b = bins[i][1]
    pkd_y, pkd_z = df_b['actual_pkd'].to_numpy(), df_b['vina_pkd'].to_numpy()
    print(f'\nBin {i}, {len}')
    get_metrics(pkd_y, pkd_z, save_results=False, show=False)

# %%
