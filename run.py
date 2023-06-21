#%%
from typing import Any, Callable, Optional
import torch
from torch_geometric import data as geo_data
import numpy as np
import pandas as pd
from rdkit import Chem
import networkx as nx
import matplotlib.pyplot as plt

from src.models.prior_work import DGraphDTA
from src.models.helpers.contact_map import get_contact
from src.models.helpers.feature_extraction import smile_to_graph, target_to_graph

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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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

# %%
mol_size, mol_feat, mol_edge = smile_to_graph(lig_seq)

# %% Getting protein graph
path = lambda c: f'{PDBBIND_STRC}/{c}/{c}_protein.pdb'
cmap_p = lambda c: f'{PDBBIND_STRC}/{c}/{c}_contact_CB.npy'
msa_p = lambda up: f'../data/msa/{up}_clean.aln'  # no longer needed due to issue with DGraphDTA

cmap = get_contact(path(PDB_CODE), 
                   CA_only=False, # CB is needed by DGraphDTA
                   check_missing=False) 
# np.save(cmap_p(PDB_CODE), cmap)
pro_size, pro_feat, pro_edge = target_to_graph(pro_seq, cmap, threshold=0)

# Loading into tensors
pro = geo_data.Data(x=torch.Tensor(pro_feat), # node feature matrix
                    edge_index=torch.LongTensor([[0,0]]).transpose(1, 0),
                    y=None).to(device)
lig = geo_data.Data(x=torch.Tensor(mol_feat), # node feature matrix
                    edge_index=torch.LongTensor(mol_edge).transpose(1, 0),
                    y=None).to(device)

model.eval()
pred = model(lig, pro) # output of model is pKd
print('pred:', pred.item())
print('label:', label)
# %%
# %%
class TestDataset(geo_data.InMemoryDataset):
    def __init__(self, root: str | None = None,  dataset: str | None = 'kd_ki', xd=None, y=None,
                 smile_graph=None, target_key=None, target_graph=None,
                 transform: Callable[..., Any] | None = None, 
                 pre_transform: Callable[..., Any] | None = None, 
                 pre_filter: Callable[..., Any] | None = None, 
                 log: bool = True):
        super().__init__(root, transform, pre_transform, pre_filter, log)


def create_test_data():
    # col: PDBCode,protID,lig_name,prot_seq,SMILE
    df_x = pd.read_csv('data/PDBbind/kd_ki/X.csv', index_col=0)
    # col: PDBCode,affinity 
    df_y = pd.read_csv('data/PDBbind/kd_ki/Y.csv', index_col=0)
    
    #
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                              collate_fn=collate)
    