#%%
from src.data_processing.datasets import PlatinumDataset
PlatinumDataset('./data/plat', './data/plat')


#%%
import os, random, itertools, math, pickle, json, time
import config


from tqdm import tqdm
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.models.prior_work import DGraphDTA, DGraphDTAImproved
from src.models.mut_dta import EsmDTA, EsmAttentionDTA
from src.data_processing.datasets import PDBbindDataset, DavisKibaDataset
from src.data_processing.utils import train_val_test_split
from src.models import train, test, CheckpointSaver
from src.models.utils import print_device_info
from src.data_analysis import get_metrics
from src.feature_extraction.protein import multi_save_cmaps
import torch
from src.models import debug

device = torch.device('cuda:0')

#%% 
DATA = 'davis'
FEATURE = 'nomsa'
DATA_ROOT = f'../data/{DATA}/' # where to get data from
TRAIN_SPLIT, VAL_SPLIT, RAND_SEED, BATCH_SIZE = 0.8, 0.1, 0, 4
dataset = DavisKibaDataset(
        save_root=f'../data/DavisKibaDataset/{DATA}_{FEATURE}/',
        data_root=DATA_ROOT,
        aln_dir=f'{DATA_ROOT}/aln/',
        cmap_threshold=-0.5, 
        feature_opt=FEATURE
        )
#%%
train_loader, val_loader, test_loader = train_val_test_split(dataset, 
        train_split=TRAIN_SPLIT, val_split=VAL_SPLIT,
        shuffle_dataset=True, random_seed=RAND_SEED, 
        batch_train=BATCH_SIZE, use_refined=False,
        split_by_prot=True
        )

# %%
mdl = EsmAttentionDTA().to(device)
debug(mdl, train_loader, device)

# %%
self = mdl
for data in train_loader: break

data = data['protein'].to(device)

#%%
# cls and sep tokens are added to the sequence by the tokenizer
seq_tok = self.esm_tok(data.pro_seq, 
                        return_tensors='pt', 
                        padding=True) # [B, L_max+2]
seq_tok['input_ids'] = seq_tok['input_ids'].to(data.x.device)
seq_tok['attention_mask'] = seq_tok['attention_mask'].to(data.x.device)

esm_emb = self.esm_mdl(**seq_tok).last_hidden_state # [B, L_max+2, emb_dim]

#%%
x1 = self.pro_encode(esm_emb, 
                    src_key_padding_mask=~seq_tok['attention_mask'].bool().T)

# [B, L_max+2, emb_dim]
# pool data -> [B, emb_dim]
x2 = torch.mean(x1, dim=1)
print(x2.shape)

# %%
