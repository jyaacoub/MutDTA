# %%
# filter out training data PDBs for pdbbind:

# /cluster/home/t122995uhn/projects/data/PDBbindDataset/nomsa_aflow_gvp_binary/train0/cleaned_XY.csv
# /cluster/home/t122995uhn/projects/data/PDBbindDataset/nomsa_aflow_gvp_binary/val0/cleaned_XY.csv
# /cluster/home/t122995uhn/projects/data/PDBbindDataset/nomsa_aflow_gvp_binary/test/cleaned_XY.csv
# /cluster/home/t122995uhn/projects/data/PDBbindDataset/nomsa_aflow_gvp_binary/full/cleaned_XY.csv
#%%
from src.analysis.figures import get_dpkd, fig_sig_mutations_conf_matrix, generate_roc_curve, tbl_dpkd_metrics
NORMALIZE=True
MODEL=lambda i: f"results/model_media/test_set_pred/GVPLM_PDBbind{i}D_nomsaF_aflowE_128B_0.00022659LR_0.02414D_2000E_gvpLF_binaryLE_PLATINUM.csv" ,
TRAIN_DATA_P=lambda set: f'/cluster/home/t122995uhn/projects/data/PDBbindDataset/nomsa_aflow_gvp_binary/{set}0/cleaned_XY.csv', 

#%%
# 2. Mutation impact analysis
md_table = tbl_dpkd_metrics(MODEL, TRAIN_DATA_P, NORMALIZE, verbose=True, plot=True)


#%%
# 3. significant mutation impact analysis
import pandas as pd
df = pd.read_csv(MODEL(0), index_col=0).dropna()
true_dpkd = get_dpkd(df, 'actual', NORMALIZE)
pred_dpkd = get_dpkd(df, 'pred', NORMALIZE)
conf1, tpr1, tnr1 = fig_sig_mutations_conf_matrix(true_dpkd, pred_dpkd, std=0.3)
conf2, tpr2, tnr2 = fig_sig_mutations_conf_matrix(true_dpkd, pred_dpkd, std=1)

generate_roc_curve(true_dpkd, pred_dpkd, thres_range=(0,5), step=0.1)

# %%