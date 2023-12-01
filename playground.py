#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.data_analysis.stratify_protein import map_davis_to_kinbase

# Get kinbase data to map to family names
kin_df = pd.read_csv('../data/misc/kinase_base_updated.csv', index_col='name')
sns.set_style('darkgrid')

# %%
# get count of every subgroup
# cols are code,SMILE,prot_seq,pkd,prot_id
df_full = pd.read_csv('../data/DavisKibaDataset/davis/nomsa_binary_original_binary/full/XY.csv',
                      index_col='code')

prot_counts = df_full.index.value_counts()
kb_dict = map_davis_to_kinbase(df_full.index.unique(), 
                               df=kin_df) # should be the same for all folds (same test set)
kb_df = pd.DataFrame.from_dict(kb_dict, orient='index', 
                               columns=['kinbase', 'main_family', 'subgroup'])

subgroup_counts = {} # {subgroup: count, ...}
for idx in df_full.index:
    subgrp = kb_df.loc[idx, 'subgroup']
    subgroup_counts[subgrp] = subgroup_counts.get(subgrp, 0) + 1

# %% get subgroup mse
model_type = 'DG'
for model_type in ['EDI', 'DG']:
    if model_type == 'EDI':
        model_path = lambda x: f'results/model_media/test_set_pred/EDIM_davis{x}D_nomsaF_binaryE_48B_0.0001LR_0.4D_2000E_testPred.csv'
    elif model_type == 'DG':
        model_path = lambda x: f'results/model_media/test_set_pred/DGM_davis{x}D_nomsaF_binaryE_64B_0.0001LR_0.4D_2000E_testPred.csv'


    # Do the same but this time with error bars by using cross validation
    # data will be a dict of {main_family: [mse1, mse2, ...], ...}
    data_main = {}
    data_subgroups = {} # {main_family: {subgroup: [mse1, mse2, ...], ...}, ...}

    for fold in range(5):
        pred = pd.read_csv(model_path(fold), index_col='name')
        
        # returns a dict of {davis_name: (kinbase_name, main_family, subgroup)}
        kb_dict = map_davis_to_kinbase(pred.index.unique(), df=kin_df) # should be the same for all folds (same test set)
        
        # update pred to have kinbase info
        pred['kinbase_name'] = pred.index.map(lambda x: kb_dict[x][0])
        pred['main_family'] = pred.index.map(lambda x: kb_dict[x][1])
        pred['subgroup'] = pred.index.map(lambda x: kb_dict[x][2])
        
        for f in pred.main_family.unique():
            matched = pred[pred.main_family == f]
            mse = ((matched.pred - matched.actual)**2).mean()
            
            # add main family mse to dict
            data_main[f] = data_main.get(f, []) + [mse]
            
            # add main_family subgroup mse to dict
            data_subgroups[f] = data_subgroups.get(f, {})
            
            for g in matched.subgroup.unique():
                g_matched = matched[matched.subgroup == g]
                mse = ((g_matched.pred - g_matched.actual)**2).mean()
                data_subgroups[f][g] = data_subgroups[f].get(g, []) + [mse]

    subgroup_mse = {} # {subgroup: [mse1, mse2, ...], ...}
    for k in data_subgroups.keys():
        for k2 in data_subgroups[k].keys():
            subgroup_mse[k2] = subgroup_mse.get(k2, []) + data_subgroups[k][k2]

    # merge counts with mse
    x = []
    y = []
    z = []
    for k in subgroup_mse.keys():
        # check for nan
        if np.isnan(subgroup_mse[k]).any():
            continue
        x += [subgroup_counts[k]] * len(subgroup_mse[k])
        y += subgroup_mse[k]
        z += [k] * len(subgroup_mse[k])

    # scatter plot with x axis as count and y axis as mse
    plt.figure(figsize=(10, 5))
    ax = sns.scatterplot(x=x, y=y, hue=z)
    # line of best fit
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*np.array(x) + b, color='black', linestyle='dotted', label=f'y={m*10000:.2f}e-4x+{b:.2f}', linewidth=2)

    plt.xlabel('Number of Proteins in Subgroup')
    plt.ylabel('MSE')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(f'Subgroup Size vs Test MSE ({model_type} Model)')
    plt.show()
    plt.clf()


    # %%
