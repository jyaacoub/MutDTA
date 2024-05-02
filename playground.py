# %%
from src.analysis.figures import tbl_dpkd_metrics_overlap, tbl_stratified_dpkd_metrics
from src.analysis.utils import get_mut_count
# %%
MODEL = lambda i: f"results/model_media/test_set_pred/GVPLM_PDBbind{i}D_nomsaF_aflowE_128B_0.00022659LR_0.02414D_2000E_gvpLF_binaryLE_PLATINUM.csv"
TRAIN_DATA_P = lambda set: f'/cluster/home/t122995uhn/projects/data/PDBbindDataset/nomsa_aflow_gvp_binary/{set}0/cleaned_XY.csv'
NORMALIZE = True
verbose=True
plot=False
n_models=5

conditions = ["(n_mut == 1) | (n_mut == 0)", "(n_mut > 1) | (n_mut == 0)"]
names = ['single mutation', '2+ mutations']

mkd = tbl_stratified_dpkd_metrics(MODEL, NORMALIZE, df_transform=get_mut_count, conditions=conditions, names=names)

_=tbl_dpkd_metrics_overlap(MODEL, TRAIN_DATA_P, NORMALIZE, plot=False)
# %%
