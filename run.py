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

from src.models.prior_work import DGraphDTA
from src.models.helpers.contact_map import get_contact, create_save_cmaps
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

# # %% visualizing weights for inputs
# print(model.state_dict()['pro_conv1.lin.weight'].shape)
# plt.matshow(model.state_dict()['pro_conv1.lin.weight'].cpu().detach().numpy())
# # displaying colorbar
# plt.colorbar()
# plt.show()

# print(model.state_dict()['mol_conv1.lin.weight'].shape)
# plt.matshow(model.state_dict()['mol_conv1.lin.weight'].cpu().detach().numpy())
# # displaying colorbar
# plt.colorbar()
# plt.show()

# %% preparing data for inference
# col: PDBCode,protID,lig_name,prot_seq,SMILE
df_x = pd.read_csv('data/PDBbind/kd_ki/X.csv', index_col=0) 
df_y = pd.read_csv('data/PDBbind/kd_ki/Y.csv', index_col=0) # col: PDBCode,affinity

up_id = df_x.loc[PDB_CODE]['protID'] # uniprot id
pro_seq = df_x.loc[PDB_CODE]['prot_seq']
lig_seq = df_x.loc[PDB_CODE]['SMILE']
label = df_y.loc[PDB_CODE]['affinity'] # kd in uM
label = -np.log(label * 1e-6) # convert to M and take -log (pKd)


# %% Getting protein graph
path = lambda c: f'{PDBBIND_STRC}/{c}/{c}_protein.pdb'
cmap_p = lambda c: f'{PDBBIND_STRC}/{c}/{c}_contact_CB_lone.npy'
msa_p = lambda up: f'../data/msa/{up}_clean.aln'  # no longer needed due to issue with DGraphDTA

create_save_cmaps(df_x.index, path, cmap_p) # create and save contact maps
#%%
pred = []
model.eval()
for code in tqdm(['10gs', '5j41', '1a4k']): # 10gs 5j41 1a4k fails
    cmap = np.load(cmap_p(code))
    pro_seq = df_x.loc[code]['prot_seq']
    lig_seq = df_x.loc[code]['SMILE']
    label = -np.log(df_y.loc[code]['affinity'] * 1e-6 )
    
    pro_size, pro_feat, pro_edge = target_to_graph(pro_seq, cmap, threshold=10.5)
    mol_size, mol_feat, mol_edge = smile_to_graph(lig_seq)
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

# %%
# merge with affinity convert to pKd
# df_x['affinity'] = -np.log(df_y['affinity']*1e-6)
# test_data = create_dataset_for_test(df_x[:100], cmap_p=cmap_p)

# %%
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False,
                                            collate_fn=collate)
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            print('exe')
            print(data)
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            # data = data.to(device)
            output = model(data_mol, data_pro)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            print(data_mol.y.view(-1, 1).cpu())
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
    print(total_labels)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

# %%
log_y, log_z = predicting(model, device, test_loader)

# %% Stats
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
