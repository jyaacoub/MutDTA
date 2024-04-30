#%%
from src.data_prep.init_dataset import create_datasets
from src import config as cfg
from src.utils.loader import Loader
from src.train_test.training import test
import torch, os
import pandas as pd
import tqdm
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#%%
# create_datasets(cfg.DATA_OPT.platinum, 
#                 cfg.PRO_FEAT_OPT.nomsa, cfg.PRO_EDGE_OPT.aflow, 
#                 ligand_features=cfg.LIG_FEAT_OPT.gvp, ligand_edges=cfg.LIG_EDGE_OPT.binary,
#                 k_folds=None, train_split=0, val_split=0)
#%%
loaders = Loader.load_DataLoaders(cfg.DATA_OPT.platinum,
                               cfg.PRO_FEAT_OPT.nomsa, cfg.PRO_EDGE_OPT.aflow,
                               ligand_feature=cfg.LIG_FEAT_OPT.gvp, ligand_edge=cfg.LIG_EDGE_OPT.binary,
                               datasets=['test'])

#%%
model = Loader.init_model(cfg.MODEL_OPT.GVPL, cfg.PRO_FEAT_OPT.nomsa, cfg.PRO_EDGE_OPT.aflow,
                          dropout=0.02414, output_dim=256)

#%%
cp_dir = "/cluster/home/t122995uhn/projects/MutDTA/results/model_checkpoints/ours"
MODEL_KEY = lambda fold: f"GVPLM_PDBbind{fold}D_nomsaF_aflowE_128B_0.00022659LR_0.02414D_2000E_gvpLF_binaryLE"
cp = lambda fold: f"{cp_dir}/{MODEL_KEY(fold)}.model"

out_dir = f'{cfg.MEDIA_SAVE_DIR}/test_set_pred/'
os.makedirs(out_dir, exist_ok=True)

for i in range(5):
    model.safe_load_state_dict(torch.load(cp(i), map_location=device))
    model.to(device)
    model.eval()

    loss, pred, actual = test(model, loaders['test'], device, verbose=True)
    
    # saving as csv with columns code, pred, actual
    # get codes from test loader
    codes, pid = [b['code'][0] for b in loaders['test']], [b['prot_id'][0] for b in loaders['test']]
    df = pd.DataFrame({'prot_id': pid, 'pred': pred, 'actual': actual}, index=codes)
    df.index.name = 'code'
    df.to_csv(f'{out_dir}/{MODEL_KEY(i)}_PLATINUM.csv')

# %%
# filter out training data PDBs for pdbbind:

# /cluster/home/t122995uhn/projects/data/PDBbindDataset/nomsa_aflow_gvp_binary/train0/cleaned_XY.csv
# /cluster/home/t122995uhn/projects/data/PDBbindDataset/nomsa_aflow_gvp_binary/val0/cleaned_XY.csv
# /cluster/home/t122995uhn/projects/data/PDBbindDataset/nomsa_aflow_gvp_binary/test/cleaned_XY.csv
# /cluster/home/t122995uhn/projects/data/PDBbindDataset/nomsa_aflow_gvp_binary/full/cleaned_XY.csv
#%%
from src.analysis.figures import fig_dpkd_dist, fig_sig_mutations_conf_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("results/model_media/test_set_pred/GVPLM_PDBbind0D_nomsaF_aflowE_128B_0.00022659LR_0.02414D_2000E_gvpLF_binaryLE_PLATINUM.csv",
                 index_col=0)

true_dpkd = fig_dpkd_dist(df, pkd_col='actual')
pred_dpkd = fig_dpkd_dist(df, pkd_col='pred')

fig_sig_mutations_conf_matrix(true_dpkd, pred_dpkd)
# %%
