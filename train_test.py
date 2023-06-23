#%%
import os
from typing import Any, Callable, Optional
from torch_geometric import data as geo_data
import torch

import numpy as np
import pandas as pd
import networkx as nx
import random

from rdkit import Chem

import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from lifelines.utils import concordance_index
from tqdm import tqdm
from rdkit import RDLogger

from src.models.prior_work import DGraphDTA
from src.models.helpers.contact_map import get_contact, get_sequence, create_save_cmaps
from src.models.helpers.feature_extraction import smile_to_graph, target_to_graph
from src.models.helpers.dataset_creation import create_dataset_for_test, collate

PDBBIND_STRC = '/home/jyaacoub/projects/data/refined-set'
DATA = 'davis'
SET = 'test'
TRAIN = False
MODEL_KEY = f'trained_{DATA}_{SET}' if TRAIN else f'pretrained_{DATA}_{SET}'
SAVE_RESULTS = True

save_mdl_path = f'results/model_checkpoints/ours/{MODEL_KEY}.mdl'
np.random.seed(0)
random.seed(0)

path = lambda c: f'{PDBBIND_STRC}/{c}/{c}_protein.pdb'
cmap_p = lambda c: f'{PDBBIND_STRC}/{c}/{c}_contact_CB_lone.npy'

media_save_p = 'results/model_media/'
csv_file = f'{media_save_p}/DGraphDTA_stats.csv'

#%% col: PDBCode,protID,lig_name,prot_seq,SMILE
df_x = pd.read_csv('data/PDBbind/kd_ki/X.csv', index_col=0) 
df_y = pd.read_csv('data/PDBbind/kd_ki/Y.csv', index_col=0) # col: PDBCode,affinity
df_seq = pd.read_csv('data/PDBbind/kd_ki/pdb_seq_lrgst.csv', index_col=0) # col: PDBCode,seq

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

#%% loading model checkpoint
model = DGraphDTA()
model.to(device)

model_file_name = f'results/model_checkpoints/prior_work/DGraphDTA_{DATA}_t2.model'
model.load_state_dict(torch.load(model_file_name, map_location=device))

#%% randomly splitting data into train, val, test
pdbcodes = np.array(df_x.index)
random.shuffle(pdbcodes)
pdb_train, pdb_test = np.split(df_x.index, [int(.8*len(df_x))])

#%% vina stats


#%% nonbatched training
# training:
if TRAIN:

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    train_set = pdb_train

    model.train()
    errors = []
    RDLogger.DisableLog('rdApp.*') # supress rdkit warnings
    for code in tqdm(train_set, 'training'):
        cmap = np.load(cmap_p(code))
        pro_seq = df_seq.loc[code]['seq']
        lig_seq = df_x.loc[code]['SMILE']
        label = torch.Tensor([[-np.log(df_y.loc[code]['affinity'] * 1e-6)]]).to(device)
        
        pro_size, pro_feat, pro_edge = target_to_graph(pro_seq, cmap, threshold=8.0)
        try:
            mol_size, mol_feat, mol_edge = smile_to_graph(lig_seq)
        except ValueError:
            errors.append(code)
            continue
        
        pro = geo_data.Data(x=torch.Tensor(pro_feat),
                            edge_index=torch.LongTensor(pro_edge).transpose(1, 0),
                            y=None).to(device)
        lig = geo_data.Data(x=torch.Tensor(mol_feat),
                            edge_index=torch.LongTensor(mol_edge).transpose(1, 0),
                            y=None).to(device)
        
        optimizer.zero_grad()
        output = model(lig, pro)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
            
        # Update tqdm description with the current loss value
        # tqdm.set_postfix(Loss=loss.item())

    # enable rdkit warnings
    RDLogger.EnableLog('rdApp.*')
    print(f'Final MSE loss: {loss}')
    print(f'{len(errors)} errors out of {len(train_set)}')

    # saving model as checkpoint
    if SAVE_RESULTS:
        torch.save(model.state_dict(), save_mdl_path)
        print(f'Model saved to: {save_mdl_path}')

#%% Testing
test_set = pdb_test if SET == "test" else df_x.index
actual = []
pred = []
model.eval()
errors = []
RDLogger.DisableLog('rdApp.*') # supress rdkit warnings
for code in tqdm(test_set, 'testing'): #TODO: batch the data for faster inference using GPU
    cmap = np.load(cmap_p(code))
    pro_seq = df_seq.loc[code]['seq']
    lig_seq = df_x.loc[code]['SMILE']
    label = -np.log(df_y.loc[code]['affinity'] * 1e-6 )
    
    pro_size, pro_feat, pro_edge = target_to_graph(pro_seq, cmap, threshold=8.0)
    try:
        mol_size, mol_feat, mol_edge = smile_to_graph(lig_seq)
    except ValueError:
        errors.append(code)
        continue
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
    actual.append(label)
    
print(f'{len(errors)} errors out of {len(test_set)}')
assert len(actual) == len(pred), 'actual and pred are not the same length'
# enable rdkit warnings
RDLogger.EnableLog('rdApp.*')
log_y, log_z = np.array(actual), np.array(pred)

# %%
plt.hist(log_y, bins=10, alpha=0.5)
plt.hist(log_z, bins=10, alpha=0.5)
plt.legend(['Experimental', MODEL_KEY])
plt.title(f'Histogram of affinity values (pkd)')
if SAVE_RESULTS: plt.savefig(f'{media_save_p}/{MODEL_KEY}_his.png')
plt.show()

# scatter plot of affinity values
# fitting a line
m, b = np.polyfit(log_y, log_z, 1)
plt.scatter(log_y, log_z, alpha=0.5)
plt.plot(log_y, m*log_y + b, color='black', alpha=0.8)
plt.xlabel('Experimental affinity value')
plt.ylabel(f'{MODEL_KEY} prediction')
plt.title(f'Scatter plot of affinity values (pkd)')

if SAVE_RESULTS: plt.savefig(f'{media_save_p}/{MODEL_KEY}_scatter.png')
plt.show()

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

#%% saving to csv file
# creating stats csv if it doesnt exist
if not os.path.exists(f'{media_save_p}/DGraphDTA_stats.csv'): 
    stats = pd.DataFrame(columns=['run', 'cindex', 'pearson', 'spearman', 'mse', 'mae', 'rmse'])
    stats.set_index('run', inplace=True)
    stats.to_csv(csv_file)

#%% replacing existing record if run_num already exists
if SAVE_RESULTS:
    stats = pd.read_csv(csv_file, index_col=0)
    stats.loc[MODEL_KEY] = [c_index, p_corr[0], s_corr[0], mse, mae, rmse]
    stats.to_csv(csv_file)
#%%