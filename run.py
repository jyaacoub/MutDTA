#%%
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
import networkx as nx

from src.models.helpers.contact_map import get_contact
from src.models.prior_work import DGraphDTA

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
smile_seq = df_SMILES.loc[lig_name]['SMILE']

# %%
# one hot encoding
def one_hot(x, allowable_set, cap=False):
    if x not in allowable_set:
        # print(x)
        if not cap:
            raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
        else:
            x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

# mol atom feature for mol graph
def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_hot(atom.GetSymbol(),
                        ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                        'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                        'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                        'Pt', 'Hg', 'Pb', 'X'],                                           cap=True) +
                    one_hot(atom.GetDegree(),         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]          ) +
                    one_hot(atom.GetTotalNumHs(),     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], cap=True) +
                    one_hot(atom.GetImplicitValence(),[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], cap=True) +
                    [atom.GetIsAromatic()])
    
# mol smile to mol graph edge index
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed() # to_directed() is important because the model is trained on directed graph
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1

    
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    # print('smile_to_graph')
    # print(np.array(features).shape)
    return c_size, features, edge_index

# %%
mol_size, mol_feat, mol_edge = smile_to_graph(smile_seq)
# %% Getting protein graph
path = lambda c: f'{PDBBIND_STRC}/{c}/{c}_protein.pdb'
cmap_p = lambda c: f'{PDBBIND_STRC}/{c}/{c}_contact_CB.npy'

cmap = get_contact(path(PDB_CODE), CA_only=False) # CB is needed by DGraphDTA
# np.save(cmap_p(PDB_CODE), cmap)

# %%

