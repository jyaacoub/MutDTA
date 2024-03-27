# %%
from prody import fetchPDB

fetchPDB('10gs', compressed=False)

# %%
from src.utils.residue import Chain
c = Chain('10gs.pdb', grep_atoms={'CA', 'N', 'C'})
# %%
import logging
logging.getLogger().setLevel(logging.DEBUG)

c.getCoords(get_all=True).shape # (N, 3)

# %%
from src.data_prep.feature_extraction.gvp_feats import GVPFeatures

gvp_f = GVPFeatures()

# %%
f = gvp_f.featurize_as_graph('10gs', c.getCoords(get_all=True), c.sequence)

# %%
from src.models.gvp import GVP_Protein

m = GVP_Protein(node_in_dim=(6, 3), node_h_dim=(6, 3),
                edge_in_dim=(32, 1), edge_h_dim=(32, 1),
                seq_in=False, num_layers=3, drop_rate=0.0)

# %%
# N=number of nodes, E=number of edges

# input must be (h_V, edge_index, h_E)
#   where h_V = ([N,6], [N,3,3]), 
#        edge_index = [2, E], 
#        h_E = ([E, 32], [E, 1, 3])

# for batch processing, 

import torch
def randn(n, dims, device="cpu"):
    '''
    Returns random tuples (s, V) drawn elementwise from a normal distribution.
    
    :param n: number of data points
    :param dims: tuple of dimensions (n_scalar, n_vector)
    
    :return: (s, V) with s.shape = (n, n_scalar) and
             V.shape = (n, n_vector, 3)
    '''
    return torch.randn(n, dims[0], device=device), \
            torch.randn(n, dims[1], 3, device=device)
            
N,E, B = 10, 4, 2
nodes = randn(N, (6, 3))
edges = randn(E, (32, 1))

edge_index = torch.randint(0, N, (2, E)) # (min, max, shape)

batch_idx = torch.randint(0, 2, (N,)) # batch size of 2

# %%
p, out, out1, h_V = m(nodes, edge_index, edges, batch=None)

# %%
