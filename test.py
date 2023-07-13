#%%
import os, random, itertools, math

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.feature_extraction.protein import get_pssm
from src.models.prior_work import DGraphDTAImproved
from src.data_processing import PDBbindDataset, train_val_test_split
from src.models import train, test, CheckpointSaver
from src.data_analysis import get_metrics

# code = '5klt' # 10gs
# aln_p = f'../data/msa/outputs/{code}_cleaned.a3m'
# seq = open(aln_p, 'r').readline().strip()
# pssm, lc = get_pssm(aln_p, seq)

# pssm_norm = pssm / lc
# exp = np.exp(pssm_norm)
# shan1 = np.apply_along_axis(calculate_shannon, axis=0, arr=exp / np.sum(exp, axis=0))
# print(shan1[0])
##%% alternate shannon calc
# def entropy2(col):
#     ent = 0.0
#     for base in np.where(col > 0)[0]: # all bases being used
#         n_i = col[base]
#         P_i = n_i/float(lc) # number of res of type i/ total res in col
#         ent -= P_i*(math.log(P_i,2))
#     return ent
    
# shan2 = np.apply_along_axis(entropy2, axis=0, arr=pssm)
# print(shan2[0])

# #%% alt shannon
# def shannon_entropy(list_input):
#     """Calculate Shannon's Entropy per column of the alignment (H=-\sum_{i=1}^{M} P_i\,log_2\,P_i)"""

#     import math
#     unique_base = set(list_input)
#     sh_entropy = 0.0
#     # Number of residues in column
#     for base in unique_base:
#         if base not in ResInfo.amino_acids: continue
#         n_i = list_input.count(base) # Number of residues of type i
#         P_i = n_i/float(lc) # n_i(Number of residues of type i) / M(Number of residues in column)
#         sh_entropy -= P_i*(math.log(P_i,2))

#     return sh_entropy

# from Bio import AlignIO
# alignment = AlignIO.read(f'/home/jyaacoub/projects/data/msa/{code}_filtered.msa.fas', 'fasta')
# shan3 = np.zeros(len(list(alignment[0])))
# for col_no in range(len(list(alignment[0]))):
#     list_input = list(alignment[:, col_no])
#     shan3[col_no] = shannon_entropy(list_input)
# print(shan3[0])
# plt.plot(range(len(shan1)), shan1, label='shan1')
# plt.plot(range(len(shan2)), shan2, label='shan2')
# plt.plot(range(len(shan3)), shan3, label='shan3')
# plt.legend()
# plt.show()

#%%
PDB_RAW_DIR = '../data/v2020-other-PL/'
PDB_PROCESSED_DIR = '../data/PDBbindDataset/shannon/' #NOTE: type of dataset specified here
ALN_DIR = '../data/msa/outputs/'
MODEL_STATS_CSV = 'results/model_media/model_stats.csv'
#loading data and splitting into train, val, test
pdb_dataset = PDBbindDataset(PDB_PROCESSED_DIR, PDB_RAW_DIR, ALN_DIR,
                             cmap_threshold=8.0,
                             shannon=True)

# Dataset Hyperparameters
TRAIN_SPLIT= .8 # 80% of data for training
VAL_SPLIT = .1 # 10% for val and remaining is for testing (10%)
SHUFFLE_DATA = True
RAND_SEED=0

random.seed(RAND_SEED)
np.random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)

# Tune Hyperparameters after grid search
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
DROPOUT = 0.4
NUM_EPOCHS = 2000

SAVE_RESULTS = True
media_save_p = 'results/model_media/figures/'

metrics = {}
# {BATCH_SIZE}B_{LEARNING_RATE}LR_{DROPOUT}DO are fixed so not included in model key


# %% load data
train_loader, val_loader, test_loader = train_val_test_split(pdb_dataset, 
                    train_split=TRAIN_SPLIT, val_split=VAL_SPLIT,
                    shuffle_dataset=True, random_seed=RAND_SEED, 
                    batch_size=BATCH_SIZE, use_refined=True)

# %% loading model:
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
MODEL_KEY = f'randomW_{BATCH_SIZE}B_{LEARNING_RATE}LR_{DROPOUT}D_{NUM_EPOCHS}E_shannonExtra'
print(f'\n{MODEL_KEY}')

model = DGraphDTAImproved(num_features_pro=34, output_dim=512,
                          dropout=DROPOUT)
model.to(device)
saver = CheckpointSaver(model, 
                        save_path=f'results/model_checkpoints/ours/DGraphDTA_{MODEL_KEY}.model', 
                        train_all=True,
                        patience=5, min_delta=0.05)

# %% training
logs = train(model, train_loader, val_loader, device, 
        epochs=NUM_EPOCHS, lr=LEARNING_RATE, saver=saver)
saver.save()

ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
index = np.arange(1, NUM_EPOCHS+1)
plt.plot(index, logs['train_loss'], label='train')
plt.plot(index, logs['val_loss'], label='val')
plt.legend()
plt.title(f'{MODEL_KEY} Loss')
plt.xlabel('Epoch')
# plt.xticks(range(0,NUM_EPOCHS+1, 2))
plt.xlim(0, NUM_EPOCHS)
plt.ylabel('Loss')
if SAVE_RESULTS: plt.savefig(f'results/model_media/figures/{MODEL_KEY}_loss.png')
plt.show()

# %% testing
loss, pred, actual = test(model, test_loader, device)
get_metrics(pred, actual,
            save_results=SAVE_RESULTS,
            save_path=media_save_p,
            model_key=MODEL_KEY,
            csv_file=MODEL_STATS_CSV
            )
metrics[MODEL_KEY] = {'test_loss': loss,
                        'logs': logs}

# %%
