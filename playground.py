# %%
# filter out training data PDBs for pdbbind:

# /cluster/home/t122995uhn/projects/data/PDBbindDataset/nomsa_aflow_gvp_binary/train0/cleaned_XY.csv
# /cluster/home/t122995uhn/projects/data/PDBbindDataset/nomsa_aflow_gvp_binary/val0/cleaned_XY.csv
# /cluster/home/t122995uhn/projects/data/PDBbindDataset/nomsa_aflow_gvp_binary/test/cleaned_XY.csv
# /cluster/home/t122995uhn/projects/data/PDBbindDataset/nomsa_aflow_gvp_binary/full/cleaned_XY.csv
#%%
from src.analysis.figures import fig_dpkd_dist, fig_sig_mutations_conf_matrix, generate_roc_curve
from src.analysis.metrics import get_metrics
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns

MODEL = lambda i: f"results/model_media/test_set_pred/GVPLM_PDBbind{i}D_nomsaF_aflowE_128B_0.00022659LR_0.02414D_2000E_gvpLF_binaryLE_PLATINUM.csv" 

data_p = lambda set: f'/cluster/home/t122995uhn/projects/data/PDBbindDataset/nomsa_aflow_gvp_binary/{set}0/cleaned_XY.csv'
df_t = pd.Index.append(pd.read_csv(data_p('train'), index_col=0).index, 
                       pd.read_csv(data_p('val'), index_col=0).index)
df_t = df_t.str.upper()
NORMALIZE = True

results_with_overlap = []
results_without_overlap = []

import functools

dpkd_dist = functools.partial(fig_dpkd_dist, verbose=False, normalize=NORMALIZE, show_plot=False)
fig = plt.figure(figsize=(14,10))
axes = fig.subplots(2,1)
ax_placeholder = [[None]]*2
for i in range(5):
    df = pd.read_csv(MODEL(i), index_col=0).dropna()
    df['pdb'] = df['prot_id'].str.split('_').str[0]
    
    true_dpkd_w = dpkd_dist(df, pkd_col='actual')
    pred_dpkd_w = dpkd_dist(df, pkd_col='pred')
    if i==0:
        axes[0].set_title(f"{'Normalized 'if NORMALIZE else ''}Δpkd distribution")
        sns.histplot(true_dpkd_w, kde=True, ax=axes[0], alpha=0.5, label='True Δpkd', color='blue')
        sns.histplot(pred_dpkd_w, kde=True, ax=axes[0], alpha=0.5, label='Predicted Δpkd', color='orange')
        axes[0].legend()
    
    df = df[~(df['pdb'].isin(df_t))].dropna()
    true_dpkd = dpkd_dist(df, pkd_col='actual')
    pred_dpkd = dpkd_dist(df, pkd_col='pred')
    if i==0:
        axes[1].set_title(f"{'Normalized 'if NORMALIZE else ''}Δpkd distribution without overlap")
        sns.histplot(true_dpkd, kde=True, ax=axes[1], alpha=0.5, label='True Δpkd', color='blue')
        sns.histplot(pred_dpkd, kde=True, ax=axes[1], alpha=0.5, label='Predicted Δpkd', color='orange')
        axes[1].legend()
    
    if i==0: plt.show()

    _, p_corr, s_corr, mse, mae, rmse = get_metrics(true_dpkd_w, pred_dpkd_w)
    results_with_overlap.append([p_corr[0], s_corr[0], mse, mae, rmse])
    
    _, p_corr, s_corr, mse, mae, rmse = get_metrics(true_dpkd, pred_dpkd)
    results_without_overlap.append([p_corr[0], s_corr[0], mse, mae, rmse])
    

# Convert results to DataFrame
results_with_df = pd.DataFrame(results_with_overlap, columns=['pcorr', 'scorr', 'mse', 'mae', 'rmse'])
results_without_df = pd.DataFrame(results_without_overlap, columns=['pcorr', 'scorr', 'mse', 'mae', 'rmse'])

# Statistics computation
mean_with = results_with_df.mean()
std_with = results_with_df.std()
mean_without = results_without_df.mean()
std_without = results_without_df.std()

# T-tests for significance
ttests = {col: ttest_ind(results_with_df[col], results_without_df[col]) for col in results_with_df.columns}

# Combine mean and std
combined_with = mean_with.map(lambda x: f"{x:.3f}") + " $\pm$ " + std_with.map(lambda x: f"{x:.3f}")
combined_without = mean_without.map(lambda x: f"{x:.3f}") + " $\pm$ " + std_without.map(lambda x: f"{x:.3f}")

# Add significance marks
significance = pd.Series({col: '*' if ttests[col].pvalue < 0.05 else '' for col in results_with_df.columns})

# Prepare markdown table
md_table = pd.concat([combined_with, combined_without, significance], axis=1)
md_table.columns = ['With Overlap', 'Without Overlap', 'Significance']
md_output = md_table.to_markdown()
print(md_output)


#%%
conf1, tpr1, tnr1 = fig_sig_mutations_conf_matrix(true_dpkd, pred_dpkd, std=0.3)
conf2, tpr2, tnr2 = fig_sig_mutations_conf_matrix(true_dpkd, pred_dpkd, std=1)
# %%
# ROC 
generate_roc_curve(true_dpkd, pred_dpkd, thres_range=(0,5), step=0.1)
# %%