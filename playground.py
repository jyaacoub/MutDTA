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

MODEL = lambda i: f"results/model_media/test_set_pred/GVPLM_PDBbind{i}D_nomsaF_aflowE_128B_0.00022659LR_0.02414D_2000E_gvpLF_binaryLE_PLATINUM.csv" 

data_p = lambda set: f'/cluster/home/t122995uhn/projects/data/PDBbindDataset/nomsa_aflow_gvp_binary/{set}0/cleaned_XY.csv'
df_t = pd.Index.append(pd.read_csv(data_p('train'), index_col=0).index, 
                       pd.read_csv(data_p('val'), index_col=0).index)
df_t = df_t.str.upper()

results_with_overlap = []
results_without_overlap = []

for i in range(5):
    df = pd.read_csv(MODEL(i), index_col=0).dropna()
    df['pdb'] = df['prot_id'].str.split('_').str[0]

    # with overlap
    metrics_with = list(get_metrics(np.array(df['actual']), np.array(df['pred']), 
                               save_figs=False, save_data=False, show=False))
    # dont return p vals:
    metrics_with[1] = metrics_with[1][0]
    metrics_with[2] = metrics_with[2][0]
    results_with_overlap.append(metrics_with)

    # without overlap
    df_no_overlap = df[~(df['pdb'].isin(df_t))]
    metrics_without = list(get_metrics(np.array(df_no_overlap['actual']), np.array(df_no_overlap['pred']), 
                                  save_figs=False, save_data=False, show=False))
    metrics_without[1] = metrics_without[1][0]
    metrics_without[2] = metrics_without[2][0]
    results_without_overlap.append(metrics_without)

# Convert results to DataFrame
results_with_df = pd.DataFrame(results_with_overlap, columns=['cindex', 'pcorr', 'scorr', 'mse', 'mae', 'rmse'])
results_without_df = pd.DataFrame(results_without_overlap, columns=['cindex', 'pcorr', 'scorr', 'mse', 'mae', 'rmse'])

# Statistics computation
mean_with = results_with_df.mean()
std_with = results_with_df.std()
mean_without = results_without_df.mean()
std_without = results_without_df.std()

# T-tests for significance
ttests = {col: ttest_ind(results_with_df[col], results_without_df[col]) for col in results_with_df.columns}

# Combine mean and std
combined_with = mean_with.map(lambda x: f"{x:.3f}") + " +- " + std_with.map(lambda x: f"{x:.3f}")
combined_without = mean_without.map(lambda x: f"{x:.3f}") + " +- " + std_without.map(lambda x: f"{x:.3f}")

# Add significance marks
significance = pd.Series({col: '*' if ttests[col].pvalue < 0.05 else '' for col in results_with_df.columns})

# Prepare markdown table
md_table = pd.concat([combined_with, combined_without, significance], axis=1)
md_table.columns = ['With Overlap', 'Without Overlap', 'Significance']
md_output = md_table.to_markdown()
print(md_output)

# %% treat mutated and wt proteins as the same 
wt_df = df[df.index.str.contains("_wt")]
mt_df = df[df.index.str.contains("_mt")]

missing_wt = delta_pkds = 0
for m in mt_df.index:
    i_wt = m.split('_')[0] + '_wt'
    if i_wt not in wt_df.index:
        missing_wt += 1
    else:
        delta_pkds += 1

print("missing wt:", missing_wt)
print("delta_pkds:", delta_pkds)

#%%
true_dpkd = fig_dpkd_dist(df, pkd_col='actual', verbose=True)
pred_dpkd = fig_dpkd_dist(df, pkd_col='pred', verbose=True)

#%%
conf1, tpr1, tnr1 = fig_sig_mutations_conf_matrix(true_dpkd, pred_dpkd, std=0.3)
conf2, tpr2, tnr2 = fig_sig_mutations_conf_matrix(true_dpkd, pred_dpkd, std=1)
# %%
# ROC 
generate_roc_curve(true_dpkd, pred_dpkd, thres_range=(0,5), step=0.1)
# %%
