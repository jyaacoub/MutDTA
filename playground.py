# %%
from src.analysis.figures import tbl_dpkd_metrics_overlap, tbl_dpkd_metrics_n_mut
MODEL = lambda i: f"results/model_media/test_set_pred/GVPLM_PDBbind{i}D_nomsaF_aflowE_128B_0.00022659LR_0.02414D_2000E_gvpLF_binaryLE_PLATINUM.csv"
TRAIN_DATA_P = lambda set: f'/cluster/home/t122995uhn/projects/data/PDBbindDataset/nomsa_aflow_gvp_binary/{set}0/cleaned_XY.csv'
NORMALIZE = True

print('NUM MUTATIONS:')
mkdnm = tbl_dpkd_metrics_n_mut(MODEL, NORMALIZE, conditions=[1,2], plot=True)
