#%%
from src.utils.loader import Loader
from src.train_test.simple import simple_train, simple_eval
from src.utils import config as cfg
config = {
    ## constants:
    "epochs": 20,
    "model": cfg.MODEL_OPT.GVP,
    "dataset": cfg.DATA_OPT.PDBbind,
    "feature_opt": cfg.PRO_FEAT_OPT.gvp, # NOTE: SPD requires foldseek features!!!
    "edge_opt": cfg.PRO_EDGE_OPT.binary,
    "fold_selection": 0,
    "save_checkpoint": False,
            
    ## hyperparameters to tune:
    "lr": 1e-5,
    "batch_size": 8,        # batch size is per GPU! #NOTE: multiply this by num_workers
    
    # model architecture hyperparams
    "architecture_kwargs":{
        "dropout": 0.0, # for fc layers
        "dropout_prot":0.0,
        "pro_emb_dim": 128, # input from SaProt is 480 dims
    }
}


# ============ Init Model ==============
model = Loader.init_model(model=config["model"], pro_feature=config["feature_opt"],
                        pro_edge=config["edge_opt"],
                        # additional kwargs send to model class to handle
                        **config['architecture_kwargs']
                        )

# ============ Load dataset ==============
print("Loading Dataset")
loaders = Loader.load_DataLoaders(data=config['dataset'], pro_feature=config['feature_opt'], 
                                    edge_opt=config['edge_opt'], 
                                    path=cfg.DATA_ROOT, 
                                    batch_train=config['batch_size'],
                                    datasets=['train', 'val'],
                                    training_fold=config['fold_selection'])

#%%
from src.train_test.simple import simple_train
import torch
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
device = torch.device('cpu')
simple_train(model, optimizer, loaders['val'], epochs=1, device=device)


#%%
from src.utils.loader import Loader
from src import config as cfg
dl = Loader.load_DataLoaders(cfg.DATA_OPT.PDBbind, cfg.PRO_FEAT_OPT.gvp, cfg.PRO_EDGE_OPT.binary, training_fold=0, batch_train=2)
sample = next(iter(dl['train']))

#%%
from src.models.gvp_models import GVPModel
m = GVPModel()
#%%
m.pro_branch(sample['protein'])

# %%
o = m(sample['protein'], sample['ligand'])


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
from src.models.gvp_branch import GVPBranchProt
# MutDTA/src/models/gvp_branch.py
m = GVPBranchProt(node_in_dim=(6, 3), node_h_dim=(6, 3),
                edge_in_dim=(32, 1), edge_h_dim=(32, 1),
                seq_in=False, num_layers=3, drop_rate=0.0)

#%%
p, out, out1, h_V  = m((f.node_s, f.node_v), f.edge_index, (f.edge_s, f.edge_v))

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


