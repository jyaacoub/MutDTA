#%%
from src.analysis.figures import get_dpkd, fig_sig_mutations_conf_matrix, generate_roc_curve
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
i=0

df = pd.read_csv(MODEL(i), index_col=0).dropna()
df['pdb'] = df['prot_id'].str.split('_').str[0]
df_no = df[~(df['pdb'].isin(df_t))]

#%%
true_dpkd = get_dpkd(df, pkd_col='actual')
pred_dpkd = get_dpkd(df, pkd_col='pred')
true_dpkd_no = get_dpkd(df_no, pkd_col='actual')
pred_dpkd_no = get_dpkd(df_no, pkd_col='pred')

# %%
# ROC 
_, _, _, best_threshold = generate_roc_curve(true_dpkd, pred_dpkd, thres_range=(0,5), step=0.1)
_ = fig_sig_mutations_conf_matrix(true_dpkd, pred_dpkd, std=round(best_threshold, 3))

# %%
_, _, _, best_threshold = generate_roc_curve(true_dpkd_no, pred_dpkd_no, thres_range=(0,5), step=0.1)
_ = fig_sig_mutations_conf_matrix(true_dpkd_no, pred_dpkd_no, std=round(best_threshold, 3))

#%%