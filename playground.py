#%% Checking max length of each sequence
import pandas as pd
csv = '/cluster/home/t122995uhn/projects/data/PDBbindDataset/nomsa_binary/full/XY.csv'
df = pd.read_csv(csv, index_col=0)
csvk = '/cluster/home/t122995uhn/projects/data/DavisKibaDataset/kiba/nomsa_binary/full/XY.csv'
dfk = pd.read_csv(csvk, index_col=0)


# %% Rename old models so that they match with new format
import os
from os import path as osp
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

csv = 'results/model_media/model_stats.csv'

#%%
df = pd.read_csv(csv)

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

fig1_pro_overlap(df)
fig2_pro_feat(df)
fig3_edge_feat(df)

