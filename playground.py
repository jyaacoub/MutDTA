#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.data_analysis.stratify_protein import map_davis_to_kinbase

# Get kinbase data to map to family names
kin_df = pd.read_csv('../data/misc/kinase_base_updated.csv', index_col='name')


#%%
subgroups_to_plot = ['TK', 'STE', 'Other', 'CAMK', 'AGC']
models_to_plot = ['EDI', 'DG']

fig, axes = plt.subplots(len(subgroups_to_plot)+1, len(models_to_plot), 
                       figsize=(5*len(models_to_plot), 4*(len(subgroups_to_plot)+1)))
for i, model_type in enumerate(models_to_plot):
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
        pred_kb = map_davis_to_kinbase(pred.index.unique(), df=kin_df) # should be the same for all folds (same test set)
        
        # update pred to have kinbase info
        pred['kinbase_name'] = pred.index.map(lambda x: pred_kb[x][0])
        pred['main_family'] = pred.index.map(lambda x: pred_kb[x][1])
        pred['subgroup'] = pred.index.map(lambda x: pred_kb[x][2])
        
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
        

    # plot mse as bar chart
    plot_df = pd.DataFrame(data_main)
    curr_ax = axes[0, i]
    sns.barplot(data=plot_df, ax=curr_ax)
    curr_ax.set_ylabel(f'MSE')
    curr_ax.set_xlabel('Main Family')
    curr_ax.set_title(f'MSE loss for {model_type}M by Protein Family')
    curr_ax.set_ylim(0, 1.4)
    
    
    for j, f in enumerate(subgroups_to_plot):
        curr_ax = axes[j+1, i]
        
        sns.barplot(data=pd.DataFrame(data_subgroups[f]), ax=curr_ax)
        if i == 0:
            curr_ax.set_ylabel('MSE')
        curr_ax.set_xlabel(f'{f} Subgroups')
        curr_ax.set_ylim(0, 1.4)
    
plt.tight_layout()

# print counts for each subgroup for a specific main family
# pred[pred.main_family == f].subgroup.value_counts()


# #%% getting full list of proteins and their binding scores from the entire dataset
# # cols is code, SMILE, prot_seq, pkd, prot_id
# df = pd.read_csv('../data/DavisKibaDataset/davis/nomsa_binary_original_binary/full/XY.csv', 
#                  index_col='code')

# df_kb = map_davis_to_kinbase(df.index.unique(), df=kin_df) # gets kinbase info for each protein in dataset

# df['kinbase_name'] = df.index.map(lambda x: df_kb[x][0])
# df['main_family'] = df.index.map(lambda x: df_kb[x][1])
# df['subgroup'] = df.index.map(lambda x: df_kb[x][2])

# plot_df = df[['main_family', 'pkd']]

# print(plot_df.groupby('main_family').mean())
# print(plot_df.groupby('main_family').median())
# print(plot_df.groupby('main_family').std())
# sns.boxenplot(data=plot_df, x='main_family', y='pkd')
# plt.title('Binding Scores by Main Family')


# %%
