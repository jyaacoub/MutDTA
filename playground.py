# %% Minimal LIME example
from lime.lime_tabular import LimeTabularExplainer
import torch_geometric as torchg
import pandas as pd
import numpy as np
import torch
import logging
logging.getLogger().setLevel(logging.DEBUG)

from src.utils.loader import Loader
from src import config as cfg


ckpt_dir = "/cluster/home/t122995uhn/projects/MutDTA/results/model_checkpoints/"
ckpt = f"{ckpt_dir}/ours/DGM_davis0D_nomsaF_binaryE_64B_0.0001LR_0.4D_2000E.model"

#%%
loaders = Loader.load_DataLoaders(cfg.DATA_OPT.davis,
                            cfg.PRO_FEAT_OPT.nomsa, 
                            cfg.PRO_EDGE_OPT.binary,
                            datasets=['test'])

# note the distribution of pkd values since we will be using this for min-max normalization
# this is important since LimeTextExplainer only works for classification...
# So we can adjust this to be a classification by applying a threshold for what is considered to be "binding"
xy_p = '/cluster/home/t122995uhn/projects/data/DavisKibaDataset/davis/nomsa_binary_original_binary/full/XY.csv'
df = pd.read_csv(xy_p, index_col=0)
df.pkd.describe()

# %% Loading model and checkpoint
mdl = Loader.init_model(cfg.MODEL_OPT.DG, 
                  cfg.PRO_FEAT_OPT.nomsa,
                  cfg.PRO_EDGE_OPT.binary,
                  dropout=0.4)

mdl.load_state_dict(torch.load(ckpt,
                               map_location='cpu'))

# %% Loading full data for explainer "train data"
datasets = Loader.load_datasets(data=cfg.DATA_OPT.davis,
                     pro_feature=cfg.PRO_FEAT_OPT.nomsa,
                     edge_opt=cfg.PRO_EDGE_OPT.binary,
                     subsets=['full'],
                     ligand_edge=cfg.LIG_EDGE_OPT.binary,
                     ligand_feature=cfg.LIG_FEAT_OPT.original)


#%% creating "train data" for Explainer model
for i, INSTANCE in enumerate(loaders['test']): 
    if INSTANCE['protein'].x.shape[0] < 1000: break
    
pnode_INST = INSTANCE['protein'].x
inst_size = pnode_INST.shape[0] * pnode_INST.shape[1]


train_data = None
max_instances = 442 # 442 is the max number of unique pids
seen_pids = set()
for d in datasets['full']._data_pro.values():
    if d.prot_id in seen_pids: continue
    else:
        seen_pids.add(d.prot_id)
        
    # d.x has the shape of Lx54
    # we will reshape it so that we end up with rows where each row is 
    # the flattened protein node features
    x_flat = d.x.flatten()
    
    # and cut off any residues past 
    # the length of our instance
    if len(x_flat) < inst_size: continue
    x_flat = x_flat[:inst_size]
    
    x_flat = x_flat.reshape((1,-1)) # [L*54] -> [1, L*54]
    
    train_data = torch.cat([train_data, x_flat]) if train_data is not None else x_flat
    
    if len(seen_pids) >= max_instances:
        break

# train data must be as a simple numpy 2d array
train_data = train_data.numpy()

# %%
exp = LimeTabularExplainer(training_data=train_data, mode="regression")

# %%
from tqdm import tqdm
def predict_fn_tabular(node_feats: list[np.ndarray]):
    print(f'input {type(node_feats)} of len {len(node_feats)}')
    mdl.eval()
        
    outs = None
    for node_feat in tqdm(node_feats, desc="Running pertubations through model"):
        # Unflattening node_feat
        node_feat = torch.Tensor(node_feat.reshape(pnode_INST.shape))
        
        # Build graph for model
        protein_graph = torchg.data.Data(x=node_feat,
                                    edge_index=INSTANCE['protein'].edge_index, # reuse same contacts
                                    edge_weight=None) # no edge weights
        
        ################### Run through model and return output ######################
        # Using CONSTANT ligand graph.
        ligand_graph = INSTANCE['ligand']
        
        out = mdl(protein_graph, ligand_graph)
        outs = out if outs is None else torch.cat([outs, out])
    
    print(f'outs {type(outs)} of len {len(outs)}')
    return outs.flatten().detach().numpy()

#%%
exp_out = exp.explain_instance(pnode_INST.flatten().numpy(),
                                predict_fn_tabular,
                                num_features=inst_size,
                                num_samples=10000)

# %%
vals = exp_out.as_map()[1] # returned as (feature_id, weight)
# feature_id is just the index position for the instance
vals = sorted(vals, key=lambda x: x[0])
# reshape into the original Lx54 node features
vals = np.array([x[1] for x in vals]).reshape(pnode_INST.shape)

#%% 
avg_vals = np.mean(vals, axis=1)
max_vals = np.max(vals, axis=1)
min_vals = np.min(vals, axis=1)

#%% Moving average
def moving_average_with_indices(data, window_size):
    if window_size <= 1:
        return data, list(range(len(data)))  # No smoothing needed for window_size 1 or less
    
    smoothed_data = []
    indices = []
    half_window = window_size // 2
    
    for i in range(len(data)):
        start = max(i - half_window, 0)
        end = min(i + half_window + 1, len(data))
        
        window_average = sum(data[start:end]) / (end - start)
        smoothed_data.append(window_average)
        # For indices, use the current index as the center of the window
        indices.append(i)
    
    return smoothed_data, indices

savg_vals, savg_idxs = moving_average_with_indices(avg_vals, 10)
smax_vals, smax_idxs = moving_average_with_indices(max_vals, 10)
smin_vals, smin_idxs = moving_average_with_indices(min_vals, 10)

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(15,5))
idx = list(range(len(avg_vals)))
plt.plot(idx, avg_vals, label='mean')
plt.plot(idx, max_vals, label='max')
plt.plot(idx, min_vals, label='min')
plt.plot(savg_idxs, savg_vals)
plt.plot(smax_idxs, smax_vals)
plt.plot(smin_idxs, smin_vals)
plt.legend()
plt.title("Explained values (smoothed=10)")
plt.xlabel('Amino acid index')

# %%
plt.figure(figsize=(15,5))
abs_vals = np.max(np.abs(vals), axis=1)
sabs_vals, sabs_idxs = moving_average_with_indices(abs_vals, 10)

plt.plot(idx, abs_vals)
plt.plot(sabs_idxs, sabs_vals)

plt.title("max values after np.absolute (smoothed=10)")
plt.xlabel('Amino acid index')


