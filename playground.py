# %%
from src.utils.mmseq2 import MMseq2Runner
import pandas as pd 

# tsvp = '../data/misc/davis_clustering//tsvs/davisDB_4sens_198clust.tsv'
tsvp = '../data/misc/davis_clustering/tsvs/davisDB_4sens_5cov_9c.tsv'
# read tsv
df = pd.read_csv(tsvp, sep='\t', header=None)
# rename cols
df.columns = ['rep', 'member']

clusters = df.groupby('rep')['member'].apply(list).to_dict()
len(clusters)

# %% group clusters with less than 5 members into one cluster (cluster 0)
clusters_new = {}
for k in clusters.keys():
    if len(clusters[k]) < 1: # remove outlier clusters
        continue
        #clusters_new[0] = clusters_new.get(0, []) + clusters[k]
    else:
        clusters_new[k] = clusters[k]
        

# for k in clusters_new.keys():
#     print(len(clusters_new[k]))
    
clusters = clusters_new

#%% Rename clusters keys to be ints 
clusters = {i: set(clusters[k]) for i, k in enumerate(clusters)}


#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style('darkgrid')

# %%
# get count of every subgroup
# cols are code,SMILE,prot_seq,pkd,prot_id
df_full = pd.read_csv('../data/DavisKibaDataset/davis/nomsa_binary_original_binary/full/XY.csv',
                      index_col='code')

prot_counts = df_full.index.value_counts()


cluster_counts = {} # {cluster: count, ...}
for idx in df_full.index:
    for k in clusters.keys():
        if idx in clusters[k]:
            cluster_counts[k] = cluster_counts.get(k, 0) + 1
            break
    

# %% get subgroup mse
subset = 'test' #'test'
for model_type in ['DG', 'EDI']:
    if model_type == 'EDI':
        model_path = lambda x: f'results/model_media/{subset}_set_pred/EDIM_davis{x}D_nomsaF_binaryE_48B_0.0001LR_0.4D_2000E_{subset}Pred.csv'
    elif model_type == 'DG':
        model_path = lambda x: f'results/model_media/{subset}_set_pred/DGM_davis{x}D_nomsaF_binaryE_64B_0.0001LR_0.4D_2000E_{subset}Pred.csv'

    data_clust = {} # {cluster: [mse1, mse2, ...], ...}
    for fold in range(5):
        pred = pd.read_csv(model_path(fold), index_col='name')
        
        for k in clusters.keys():
            # get mse for cluster
            matched = pred[pred.index.isin(clusters[k])]
            if matched.empty:
                continue
            mse = ((matched.pred - matched.actual)**2).mean()
        
            # add main mse to dict
            data_clust[k] = data_clust.get(k, []) + [mse]

    # merge counts with mse
    x = []
    y = []
    z = []
    for k in data_clust.keys():
        x += [cluster_counts[k]] * len(data_clust[k])
        y += data_clust[k]
        z += [k] * len(data_clust[k])

    # scatter plot with x axis as count and y axis as mse
    plt.figure(figsize=(10, 5))
    ax = sns.scatterplot(x=x, y=y, hue=z)
    # line of best fit
    m, b = np.polyfit(x, y, 1)
    lines = plt.plot(x, m*np.array(x) + b, color='black', linestyle='dotted', 
                     label=f'y={m*10000:.2f}e-4x+{b:.2f}', linewidth=2)

    # correlation
    corr = np.corrcoef(x, y)[0, 1]
    plt.xlabel('Number of Proteins in cluster')
    plt.ylabel('MSE')
    plt.legend(handles=[lines[0]], loc='upper left', title=f'Correlation: {corr:.3f}')
    plt.title(f'Subgroup Size vs {subset} MSE ({model_type} Model)')
    plt.show()
    plt.clf()





#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# %% plot mse vs protein sequence length
# df_full has prot_seq columns
df_full.reset_index(inplace=True)
#%%
df_prolen = df_full.iloc[df_full.code.drop_duplicates().index]
df_prolen = df_prolen[['code', 'prot_seq']]
df_prolen['prot_len'] = df_prolen.prot_seq.str.len()

#%%
subset = 'train' #'test'
prolen_mse = None
for model_type in ['EDI']:
    if model_type == 'EDI':
        model_path = lambda x: f'results/model_media/{subset}_set_pred/EDIM_{dataset}{x}D_nomsaF_binaryE_48B_0.0001LR_0.4D_2000E_{subset}Pred.csv'
    elif model_type == 'DG':
        model_path = lambda x: f'results/model_media/{subset}_set_pred/DGM_{dataset}{x}D_nomsaF_binaryE_64B_0.0001LR_0.4D_2000E_{subset}Pred.csv'


    data = [] # [(mse, prot_len), ...]
    for fold in range(5):
        pred = pd.read_csv(model_path(fold), index_col='name')
        pred.index.name = 'code'
        pred = pd.merge(pred, df_prolen, on='code', how='left')
        
        prolen_mse = pred if prolen_mse is None else pd.concat([prolen_mse, pred], axis=0)
        


#%% group by prot_len
plot_data = prolen_mse.groupby('prot_len').apply(lambda x: ((x.pred - x.actual)**2).mean())

sns.scatterplot(x=plot_data.index, y=plot_data.values)
m, b = np.polyfit(plot_data.index, plot_data.values, 1)
plt.plot(plot_data.index, m*np.array(plot_data.index) + b, color='black', linestyle='dotted', label=f'y={m*10000:.2f}e-4x+{b:.2f}', linewidth=2)

# correlation
corr = np.corrcoef(plot_data.index, plot_data.values)[0, 1]
print(f'Correlation: {corr}')

plt.title('Protein Length vs MSE')
plt.xlabel('Protein Length')
plt.ylabel('MSE')
plt.legend()
plt.show()
# %%
