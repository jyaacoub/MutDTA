#%%
import os, random, itertools, math, pickle, json

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, EsmConfig, EsmForMaskedLM, EsmModel, EsmTokenizer

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.models.prior_work import DGraphDTA, DGraphDTAImproved
from src.models.mut_dta import EsmDTA
from src.data_processing import PDBbindDataset, DavisKibaDataset, train_val_test_split
from src.models import train, test, CheckpointSaver
from src.models.utils import print_device_info
from src.data_analysis import get_metrics
from src.feature_extraction.protein import multi_save_cmaps

#%% 
pdb_codes = os.listdir('../data/v2020-other-PL/')
# filter out readme and index folders
pdb_codes = [p for p in pdb_codes if p != 'index' and p != 'readme']

assert len(pdb_codes) > 0, 'Too few PDBCodes, need at least 1...'

# creating contact maps:
seqs = multi_save_cmaps(pdb_codes,
                  pdb_p=lambda x: f'../data/v2020-other-PL/{x}/{x}_protein.pdb',
                  cmap_p=lambda x: f'../data/v2020-other-PL/{x}/{x}.npy',
                  processes=2)
        
print('DONE!')
exit()
#%%
dataset = PDBbindDataset(save_root='../data/PDBbindDataset/nomsa',
                 data_root='../data/v2020-other-PL/',
                 aln_dir=None, 
                 cmap_threshold=8.0,
                 shannon=False
                 )

# # %%

# train_loader, val_loader, test_loader = train_val_test_split(dataset, 
#                   train_split=0.8, val_split=.1,
#                   shuffle_dataset=True, random_seed=0, 
#                   batch_size=64, use_refined=False)

#%% Testing esm model 
# based on https://github.com/facebookresearch/esm#main-models-you-should-use-
# I should be using ESM-MSA-1b (esm1b_t33_650M_UR50S)
# from https://github.com/facebookresearch/esm/tree/main#pre-trained-models-
# Luke used esm2_t6_8M_UR50D for his experiments

# https://huggingface.co/facebook/esm2_t33_650M_UR50D is <10GB
# https://huggingface.co/facebook/esm2_t36_3B_UR50D is 11GB
df = pd.read_csv('../data/DavisKibaDataset/davis_msa/processed/XY.csv', index_col=0)
config = EsmConfig.from_pretrained('facebook/esm2_t6_8M_UR50D')
esm_tok = AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
# this will raise a warning since lm head is missing but that is okay since we are not using it:
esm_mdl = EsmModel.from_pretrained('facebook/esm2_t6_8M_UR50D')
prot_seqs = list(df['prot_seq'].unique())
tok = esm_tok(prot_seqs, return_tensors='pt', padding=True)

# %%
out = esm_mdl(**tok)
pro_feat = out.last_hidden_state.squeeze() # L x emb_dim




# %% test:
sample_seq = 'MMSMA'
sample_tok = tok(sample_seq, return_tensors='pt')['input_ids']
sample_tok = sample_tok[0][1:-1]# remove <cls> and <sep> tokens
sample_tok = sample_tok.unsqueeze(0) # add batch dimension
sample_mask = torch.ones_like(sample_tok, dtype=torch.int8)
# %% NOTE: tokenizer adds special tokens to the beginning and end of the sequence
# the first token is <cls> and the last token is <sep>
# these are for classification tasks and are not needed for our purposes
sample_out = esm2(sample_tok, sample_mask)
sample_emb = sample_out.last_hidden_state
print(sample_seq)
print(sample_tok.shape)
print(list(sample_seq[0:5]), 'corresponds to', sample_tok[0][1:6])


#%%# print(sample_out)
print('Hidden state shape:', sample_emb.shape)
print('+ features shape:', feat.shape)
print('==',
      torch.cat((sample_emb, feat.reshape((1,*feat.shape))), axis=2).shape)

# %%
