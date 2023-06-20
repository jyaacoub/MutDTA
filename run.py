#%%
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
import networkx as nx

from src.models.prior_work import DGraphDTA
from src.models.helpers.contact_map import get_contact
from src.models.helpers.feature_extraction import smile_to_graph

PDBBIND_STRC = '/home/jyaacoub/projects/data/refined-set'
DATA = 'davis'
PDB_CODE = '1a1e'

#%%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = DGraphDTA()
model.to(device)

# loading checkpoint
model_file_name = f'results/model_checkpoints/prior_work/DGraphDTA_{DATA}_t2.model'
model.load_state_dict(torch.load(model_file_name, map_location=device))


# %% preparing data for inference
df_info = pd.read_csv('data/PDBbind/kd_ki/info.csv', index_col=0)
df_SMILES = pd.read_csv('data/PDBbind/kd_ki/unique_lig.csv', index_col=0)

#%%
lig_name = df_info.loc[PDB_CODE]['lig_name']
lig_seq = df_SMILES.loc[lig_name]['SMILE']

# %%
mol_size, mol_feat, mol_edge = smile_to_graph(smile_seq)
# %% Getting protein graph
path = lambda c: f'{PDBBIND_STRC}/{c}/{c}_protein.pdb'
cmap_p = lambda c: f'{PDBBIND_STRC}/{c}/{c}_contact_CB.npy'

cmap = get_contact(path(PDB_CODE), CA_only=False) # CB is needed by DGraphDTA
# np.save(cmap_p(PDB_CODE), cmap)

# %%

