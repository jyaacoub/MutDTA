#%%
from typing import Any, Callable, Optional
from torch_geometric import data as geo_data
import torch

import numpy as np
import pandas as pd
import networkx as nx

from rdkit import Chem

import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from lifelines.utils import concordance_index
from tqdm import tqdm
from rdkit import RDLogger

from src.models.prior_work import DGraphDTA
from src.models.helpers.contact_map import get_contact, get_sequence, create_save_cmaps
from src.models.helpers.feature_extraction import smile_to_graph, target_to_graph
from src.models.helpers.dataset_creation import create_dataset_for_test, collate

PDBBIND_STRC = '/home/jyaacoub/projects/data/refined-set'
DATA = 'davis'
PDB_CODE = '1b38' 
# codes with MSA available (uniprot = P24941)
#           1b38
#           1e1v
#           1e1x
#           1jsv
#           1pxn
#           1pxo
#           1pxp
#           2exm
#           2fvd
#           2xmy
#           2xnb
#           5jq5
#           6guh
#           6guk
#           6q4g
#           6q4e

#%%
device = 'cpu' # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = DGraphDTA()
model.to(device)

# loading checkpoint
model_file_name = f'results/model_checkpoints/prior_work/DGraphDTA_{DATA}_t2.model'
model.load_state_dict(torch.load(model_file_name, map_location=device))

# %% preparing data for inference
# col: PDBCode,protID,lig_name,prot_seq,SMILE
df_x = pd.read_csv('data/PDBbind/kd_ki/X.csv', index_col=0) 
df_y = pd.read_csv('data/PDBbind/kd_ki/Y.csv', index_col=0) # col: PDBCode,affinity
df_seq = pd.read_csv('data/PDBbind/kd_ki/pdb_seq_lrgst.csv', index_col=0) # col: PDBCode,seq


# %% Getting protein graph
path = lambda c: f'{PDBBIND_STRC}/{c}/{c}_protein.pdb'
cmap_p = lambda c: f'{PDBBIND_STRC}/{c}/{c}_contact_CB_lone.npy'
msa_p = lambda up: f'../data/msa/{up}_clean.aln'  # no longer needed due to issue with DGraphDTA\

#%%
actual = []
pred = []
model.eval()
errors = []
RDLogger.DisableLog('rdApp.*') # supress rdkit warnings
for code in tqdm(df_x.index): # 10gs 5j41 1a4k fails
    cmap = np.load(cmap_p(code))
    pro_seq = df_seq.loc[code]['seq']
    lig_seq = df_x.loc[code]['SMILE']
    label = -np.log(df_y.loc[code]['affinity'] * 1e-6 )
    
    pro_size, pro_feat, pro_edge = target_to_graph(pro_seq, cmap, threshold=8.0)
    try:
        mol_size, mol_feat, mol_edge = smile_to_graph(lig_seq)
    except ValueError:
        errors.append(code)
        continue
    # Loading into tensors
    pro = geo_data.Data(x=torch.Tensor(pro_feat), # node feature matrix
                        edge_index=torch.LongTensor(pro_edge).transpose(1, 0),
                        y=label).to(device)
    lig = geo_data.Data(x=torch.Tensor(mol_feat), # node feature matrix
                        edge_index=torch.LongTensor(mol_edge).transpose(1, 0),
                        y=label).to(device)
    
    try:
        p = model(lig, pro)
    except RuntimeError as e:
        print(f'{code}\n{pro}\n{lig}')
        raise e
    
    pred.append(p.item())
    actual.append(label)
    
print(f'{len(errors)} errors out of {len(df_x)}')
assert len(actual) == len(pred), 'actual and pred are not the same length'
# enable rdkit warnings
RDLogger.EnableLog('rdApp.*')


# %% Stats
log_y, log_z = np.array(actual), np.array(pred)
# calc concordance index 
c_index = concordance_index(log_y, log_z)
print(f"Concordance index: {c_index:.3f}")

# pearson correlation
p_corr = pearsonr(log_y, log_z)
print(f"Pearson correlation: {p_corr[0]:.3f}")
print(f"Pearson p-value: {p_corr[1]:.3f}")

# spearman correlation
s_corr = spearmanr(log_y, log_z)
print(f"Spearman correlation: {s_corr[0]:.3f}")
print(f"Spearman p-value: {s_corr[1]:.3f}")

# error
mse = np.mean((log_y-log_z)**2)
mae = np.mean(np.abs(log_y-log_z))
rmse = np.sqrt(mse)
print(f"MSE: {mse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")

# %%
plt.hist(log_y, bins=10, alpha=0.5)
plt.hist(log_z, bins=10, alpha=0.5)
plt.legend(['Experimental', 'Vina'])
plt.title(f'Histogram of affinity values (-log(Kd))')
# plt.savefig(f'{save_path}/vina_{run_num}_hist.png')
plt.show()

# scatter plot of affinity values
# fitting a line
m, b = np.polyfit(log_y, log_z, 1)
plt.scatter(log_y, log_z, alpha=0.5)
plt.plot(log_y, m*log_y + b, color='black', alpha=0.8)
plt.xlabel('Experimental affinity value')
plt.ylabel('DGraphDTA prediction')
plt.title(f'Scatter plot of affinity values (-log(Kd))')

# plt.savefig(f'{save_path}/vina_{run_num}_scatter.png')
plt.show()

# %%
