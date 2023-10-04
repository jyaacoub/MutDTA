


# %% Rename old models so that they match with new format
import os
from os import path as osp
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

csv = 'results/model_media/model_stats.csv'

df = pd.read_csv('/cluster/home/t122995uhn/projects/data/PDBbindDataset/nomsa_binary/full/XY.csv', index_col=0)

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


#%% Fig 4 - DDP vs non DDP
from src.data_analysis.figures import fig1_pro_overlap, fig2_pro_feat, fig3_edge_feat
df_new = df[df['data'] == 'PDBbind']
fig1_pro_overlap(df_new, verbose=False, sel_col='mse')
fig2_pro_feat(df_new, sel_col='mse')
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
