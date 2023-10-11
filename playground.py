#%%
from src.data_processing.datasets import DavisKibaDataset

data='kiba'
FEATURE = 'nomsa'
EDGE = 'anm'
DATA_ROOT = f'/cluster/home/t122995uhn/projects/data/{data}/'

dataset = DavisKibaDataset(
        save_root=f'../data/DavisKibaDataset/{data}/',
        data_root=DATA_ROOT,
        aln_dir=f'{DATA_ROOT}/aln/', 
        cmap_threshold=-0.5, 
        feature_opt=FEATURE,
        af_conf_dir=f'../colabfold/{data}_af2_out/',
        edge_opt=EDGE
)



















#%%
import pandas as pd
import os
from glob import glob

data = 'davis'
fp = f'/cluster/home/t122995uhn/projects/data/DavisKibaDataset/{data}/nomsa_binary/full/XY.csv'
done_dir = f'/cluster/home/t122995uhn/projects/colabfold/{data}_af2_out/highQ/*.done.txt'

df = pd.read_csv(fp)

#%%
# replace '(' with '_' in prot_id col
df['pid_fix'] = df['prot_id'].str.replace(r'[()]', '_', regex=True)
unique_pids = df['pid_fix'].unique()
print(f'{len(unique_pids)} unique proteins')

def getID(fn):
    return os.path.splitext(os.path.basename(fn))[0].split('.done')[0]

done_IDs =  {getID(filename) for filename in glob(done_dir)}

remaining = [id for id in unique_pids if id not in done_IDs]

print(len(remaining), 'protiens remaining')

done_rows = df.iloc[df[~df['pid_fix'].isin(remaining)][['pid_fix']].drop_duplicates().index]
remaining_rows = df.iloc[df[df['pid_fix'].isin(remaining)][['pid_fix']].drop_duplicates().index]
remaining_rows['len'] = remaining_rows['prot_seq'].str.len()

max_prot_len_done = done_rows['prot_seq'].str.len().max()
max_prot_len = remaining_rows['len'].max()
print(f'{max_prot_len} is the max protein length for remaining structures.\n{max_prot_len_done} for done structures')
remaining_rows[['pid_fix', 'len']]

#%% get pids of those above 2149
files_to_remove = remaining_rows[remaining_rows['len'] > 2000]['prot_id']

for i in range(1,4):
    print('\nPART', i)
    p3p = f'/cluster/home/t122995uhn/projects/colabfold/{data}_a3m/part{i}'

    for f in files_to_remove:
        try:
            os.remove(os.path.join(p3p, f'{f}.a3m'))
            print(f'{f} REMOVED')
        except FileNotFoundError:
            pass
            # print(f'{f} doesnt exist')


# %% Rename old models so that they match with new format
import os
from os import path as osp
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.data_analysis.figures import fig1_pro_overlap, fig2_pro_feat, fig3_edge_feat

csv = 'results/model_media/model_stats.csv'

#%%
df = pd.read_csv(csv)
df = pd.concat([df, pd.read_csv('results/model_media/old_model_stats.csv')]) # concat with old model results since we get the max value anyways...

# create data, feat, and overlap columns for easier filtering.
df['data'] = df['run'].str.extract(r'_(davis|kiba|PDBbind)', expand=False)
df['feat'] = df['run'].str.extract(r'_(nomsa|msa|shannon)F_', expand=False)
df['edge'] = df['run'].str.extract(r'_(binary|simple|anm|af2)E_', expand=False)
df['ddp'] = df['run'].str.contains('DDP-')
df['improved'] = df['run'].str.contains('IM_') # trail of model name will include I if "improved"
df['batch_size'] = df['run'].str.extract(r'_(\d+)B_', expand=False)

df.loc[df['run'].str.contains('EDIM') & df['run'].str.contains('nomsaF'), 'feat'] = 'ESM'
df.loc[df['run'].str.contains('EDAIM'), 'feat'] += '-ESM'

df['overlap'] = df['run'].str.contains('overlap')

df[['run', 'data', 'feat', 'edge', 'batch_size', 'overlap']]

#%%
fig2_pro_feat(df, sel_col='cindex', show=False)
plt.savefig('fig2_pro_feat_jitter.png', dpi=300)
plt.show()

#%% Fig 4 - DDP vs non DDP
df_new = df[df['data'] == 'PDBbind']
fig1_pro_overlap(df_new, verbose=False, sel_col='mse', show=True)

fig2_pro_feat(df_new, sel_col='mse', show=False)

fig3_edge_feat(df_new, sel_col='mse')

df_new = df[~(df['data'] == 'PDBbind')]
fig1_pro_overlap(df_new, verbose=False, sel_col='mse')
fig2_pro_feat(df_new, sel_col='mse')
fig3_edge_feat(df_new, sel_col='mse')

# %%
grouped_df = df[(df['feat'] == 'nomsa') 
                & (df['batch_size'] == '64') 
                & (df['edge'] == 'binary')
                & (~df['ddp'])              
                & (~df['improved'])].groupby(['data'])

# each group is a dataset with 2 bars (overlap and no overlap)
for group_name, group_data in grouped_df:
    print(f"\nGroup Name: {group_name}")
    print(group_data[['cindex', 'mse', 'overlap']])

# these groups are spaced by the data type, physically grouping bars of the same dataset together.
# Initialize lists to store cindex values for each dataset type
t_overlap = []
f_overlap = []
dataset_types = []
# %%
for dataset, group in grouped_df: break

# %%
