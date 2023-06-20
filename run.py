#%%
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
import networkx as nx
import matplotlib.pyplot as plt

from src.models.prior_work import DGraphDTA
from src.models.helpers.contact_map import get_contact
from src.models.helpers.feature_extraction import smile_to_graph

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
df_info = pd.read_csv('data/PDBbind/kd_ki/info.csv', index_col=0)
df_SMILES = pd.read_csv('data/PDBbind/kd_ki/unique_lig.csv', index_col=0)

#%%
lig_name = df_info.loc[PDB_CODE]['lig_name']
lig_seq = df_SMILES.loc[lig_name]['SMILE']

# %%
mol_size, mol_feat, mol_edge = smile_to_graph(lig_seq)

# %% Getting protein graph
path = lambda c: f'{PDBBIND_STRC}/{c}/{c}_protein.pdb'
cmap_p = lambda c: f'{PDBBIND_STRC}/{c}/{c}_contact_CB.npy'
msa_p = lambda up: f'../data/msa/{up}_clean.aln' 

up_id = df_info.loc[PDB_CODE]['protID']

cmap = get_contact(path(PDB_CODE), CA_only=False) # CB is needed by DGraphDTA
#TODO: load target features!
# np.save(cmap_p(PDB_CODE), cmap)

# %%

