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

from tqdm import tqdm
from rdkit import RDLogger

from src.data_analysis import get_metrics
from src.models import DGraphDTA

from src.feature_extraction import smile_to_graph, target_to_graph
from src.feature_extraction.protein import get_contact, get_sequence, create_save_cmaps

from data_processing.datasets import create_dataset_for_test, collate

PDBBIND_STRC = '/home/jyaacoub/projects/data/v2020-other-PL'
DATA = 'kiba'
SET = 'test'
TRAIN = True
MODEL_KEY = f'generalTrained_{DATA}_{SET}' if TRAIN else f'pretrained_{DATA}_{SET}'
SAVE_RESULTS = True

save_mdl_path = f'results/model_checkpoints/ours/{MODEL_KEY}.mdl'
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

path = lambda c: f'{PDBBIND_STRC}/{c}/{c}_protein.pdb'
cmap_p = lambda c: f'{PDBBIND_STRC}/{c}/{c}_cmap_CB_lone.npy'

media_save_p = 'results/model_media/'
csv_file = f'{media_save_p}/DGraphDTA_stats.csv'

#%% col: PDBCode,protID,lig_name,prot_seq,SMILE
df_XY = pd.read_csv('data/PDBbind/general_XY.csv', index_col=0) # col: PDBCode,resolution,release_year,pkd,lig_name, SMILE, prot_seq
df_x = pd.read_csv('data/PDBbind/kd_ki/X.csv', index_col=0)
df_y = pd.read_csv('data/PDBbind/kd_ki/Y.csv', index_col=0) # col: PDBCode,affinity
df_seq = pd.read_csv('data/PDBbind/kd_ki/pdb_seq_lrgst.csv', index_col=0) # col: PDBCode,seq


#%%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

#%% loading model checkpoint
model = DGraphDTA()
model.to(device)

if DATA != 'random':
    model_file_name = f'results/model_checkpoints/prior_work/DGraphDTA_{DATA}_t2.model'
    model.load_state_dict(torch.load(model_file_name, map_location=device))

#%% randomly splitting data into train, val, test
pdbcodes = np.array(df_x.index)
random.shuffle(pdbcodes)
_, pdb_test = np.split(df_x.index, [int(.8*len(df_x.index))])
pdb_train = df_XY.index.difference(pdb_test)

#%% vina stats


#%% nonbatched training
# training:
if TRAIN:

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss()

    train_set = pdb_train

    model.train()
    errors = []
    RDLogger.DisableLog('rdApp.*') # supress rdkit warnings
    for code in tqdm(train_set, 'training'):
        cmap = np.load(cmap_p(code))
        pro_seq = df_XY.loc[code]['prot_seq'] if code in df_XY.index else df_seq.loc[code]['seq']
        lig_seq = df_XY.loc[code]['SMILE'] if code in df_XY.index else df_x.loc[code]['SMILE']
        label = df_XY.loc[code]['pkd'] if code in df_XY.index else -np.log(df_y.loc[code]['affinity']*1e-6)
        label = torch.Tensor([[label]]).to(device)
        
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
test_set = pdb_test if SET == "test" else df_XY.index
actual = []
pred = []
model.eval()
errors = []
RDLogger.DisableLog('rdApp.*') # supress rdkit warnings
for code in tqdm(test_set, 'testing'): #TODO: batch the data for faster inference using GPU
    cmap = np.load(cmap_p(code))
    pro_seq = df_XY.loc[code]['prot_seq'] if code in df_XY.index else df_seq.loc[code]['seq']
    lig_seq = df_XY.loc[code]['SMILE'] if code in df_XY.index else df_x.loc[code]['SMILE']
    label = df_XY.loc[code]['pkd'] if code in df_XY.index else -np.log(df_y.loc[code]['affinity']*1e-6)
    
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

#%%
get_metrics(np.array(actual), np.array(pred),
            save_results=SAVE_RESULTS,
            save_path=media_save_p,
            model_key=MODEL_KEY,
            csv_file=csv_file
            )
# %%
