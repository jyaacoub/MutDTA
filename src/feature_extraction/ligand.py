import os
import pandas as pd
import numpy as np
from rdkit import Chem
import networkx as nx
from src.feature_extraction import one_hot

########################################################################
###################### Ligand Feature Extraction #######################
########################################################################
# mol atom feature for mol graph
def atom_features(atom):
    # 44 +11 +11 +11 +1 = 78
    return np.concatenate(
        (one_hot(atom.GetSymbol(),
                ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                'Pt', 'Hg', 'Pb', 'X'],                                       cap=True),
        one_hot(atom.GetDegree(),         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]          ), # WARNING: why use one hot here instead of just the number?
        one_hot(atom.GetTotalNumHs(),     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], cap=True),
        one_hot(atom.GetImplicitValence(),[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], cap=True),
        [atom.GetIsAromatic()]))

# mol smile to mol graph edge index
def smile_to_graph(smile):
    try:
        mol = Chem.MolFromSmiles(smile)
        c_size = mol.GetNumAtoms()
    except AttributeError as e:
        # adding to stack trace
        raise ValueError(f'rdkit failed to convert SMILE: {smile}') from e

    # getting node features
    atoms = mol.GetAtoms()
    features = np.zeros((len(atoms), 78))
    for i, atom in enumerate(atoms):
        feature = atom_features(atom) # 78 features
        features[i] = feature / sum(feature) # why / sum(feature)? #WARNING: this doesnt make sense since all the values are 0 or 1 and sum(feature) always equals 5 (number of features)

    # getting bonds using rdKit
    edges = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
    
    # to_directed() is important because the model is trained on directed graph
    g = nx.Graph(edges).to_directed()
    mol_adj = nx.to_numpy_array(g) # adjacency matrix

    # adding self loop
    mol_adj += np.eye(mol_adj.shape[0])
    # converting edge matrix to edge index for pytorch geometric
    index_row, index_col = np.where(mol_adj >= 0.5)
    edge_index = np.array([[i,j] for i,j in zip(index_row, index_col)])
    # print('smile_to_graph')
    # print(np.array(features).shape)
    return c_size, features, edge_index
