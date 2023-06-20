import os
import pandas as pd
import numpy as np
from rdkit import Chem
import networkx as nx

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
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    # getting node features
    atoms = mol.GetAtoms()
    features = np.zeros((len(atoms), 78))
    for i, atom in enumerate(atoms):
        feature = atom_features(atom) # 78 features
        features[i] = feature / sum(feature) # why / sum(feature)? #WARNING: this doesnt make sense

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


########################################################################
###################### Protein Feature Extraction ######################
########################################################################
# pconsc4 predicted contact map save in data/dataset/pconsc4
def target_to_graph(target_sequence, contact_map, threshold=10.5):
    """
    Feature extraction for protein sequence using contact map to generate 
    edge index and node features

    Args:
        target_sequence (str): sequence of target protein
        contact_map (str or np.array): file path for contact map or the 
            actual map itself.
        threshold (float, optional): Threshold for what defines an edge 
            (DGraphDTA used prediced contact map by pcons4 and so 0.5 was 
            used as the threshold). Defaults to 10.5 (Angstroms for real 
            contact maps)

    Returns:
        Tuple(np.array): truple of (target_size, target_feature, target_edge_index)
    """
    # loading up contact map if it is a file path
    if type(contact_map) == str: contact_map = np.load(contact_map)
    
    
    target_size = len(target_sequence)
    # Applying threshold to contact map and adding self loop
    index_row, index_col = np.where(contact_map >= threshold)
    contact_map += np.matrix(np.eye(contact_map.shape[0]))
    
    # converting edge matrix to edge index for pytorch geometric
    target_edge_index = np.array([[i,j] for i, j in zip(index_row, index_col)])
    
    # getting node features
    target_feature = target_to_feature(target_sequence)
    
    return target_size, target_feature, target_edge_index

# target aln file save in data/dataset/aln
def target_to_feature(target_seq):
    # aln_dir = 'data/' + dataset + '/aln'
    pssm = np.zeros((len(pro_res_table), len(target_seq))) #NOTE: DGraphDTA never uses pssm due to logic error (see: https://github.com/jyaacoub/DGraphDTA/commit/ba06f7ece847ba0c806c6a9034748b62cfd2c09a)
    other_feature = seq_feature(target_seq)
    return np.concatenate((np.transpose(pssm, (1, 0)), other_feature), axis=1)

def seq_feature(pro_seq):
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        # if 'X' in pro_seq:
        #     print(pro_seq)
        pro_hot[i,] = one_hot(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)

def residue_features(residue):
    feats = [residue in pro_res_aliphatic_table, residue in pro_res_aromatic_table,
            residue in pro_res_polar_neutral_table, residue in pro_res_acidic_charged_table,
            residue in pro_res_basic_charged_table,
            
            res_weight_table[residue], res_pka_table[residue], 
            res_pkb_table[residue], res_pkx_table[residue],
            res_pl_table[residue], res_hydrophobic_ph2_table[residue], 
            res_hydrophobic_ph7_table[residue]]
    return np.array(feats)


########################################################################
###################### Misc and Tables #################################
########################################################################
# one hot encoding
def one_hot(x, allowable_set, cap=False):
    """Return the one-hot encoding of x as a numpy array."""
    if x not in allowable_set:
        if not cap:
            raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
        else:
            x = allowable_set[-1]
    return np.eye(len(allowable_set))[allowable_set.index(x)]

def dic_normalize(dic):
    # print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic

pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)