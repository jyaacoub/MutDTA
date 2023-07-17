#%%
import os, random, itertools, math

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.models.prior_work import DGraphDTA, DGraphDTAImproved
from src.data_processing import PDBbindDataset, DavisKibaDataset, train_val_test_split
from src.models import train, test, CheckpointSaver
from src.data_analysis import get_metrics

#/randomW_kiba_32B_0.001LR_0.2D_200E_msaF_DGraphDTAImproved.model
#randomW_davis_64B_0.0001LR_0.4D_200E_msaF_DGraphDTA.model
data_opt = ['kiba', 'davis']
FEATURE_opt = ['msa', 'shannon']
OG_MODEL_opt = [True, False]
data = data_opt[1]
FEATURE = FEATURE_opt[0]
OG_MODEL = OG_MODEL_opt[0]


MODEL_STATS_CSV = 'results/model_media/model_stats.csv'

# Dataset Hyperparameters
TRAIN_SPLIT= .8 # 80% of data for training
VAL_SPLIT = .1 # 10% for val and remaining is for testing (10%)
SHUFFLE_DATA = True
RAND_SEED=0

random.seed(RAND_SEED)
np.random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)

# Tune Hyperparameters after grid search
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
DROPOUT = 0.4
NUM_EPOCHS = 200

SAVE_RESULTS = True
SHOW_PLOTS = False
media_save_p = 'results/model_media/davis_kiba/'
# loading data
DATA_ROOT = f'../data/davis_kiba/{data}/' # where to get data from
dataset = DavisKibaDataset(
        save_root=f'../data/DavisKibaDataset/{data}_{FEATURE}/',
        data_root=DATA_ROOT,
        aln_dir=f'{DATA_ROOT}/aln/',
        cmap_threshold=-0.5, shannon=FEATURE=='shannon')
print(f'Number of samples: {len(dataset)}')

train_loader, val_loader, test_loader = train_val_test_split(dataset, 
                    train_split=TRAIN_SPLIT, val_split=VAL_SPLIT,
                    shuffle_dataset=True, random_seed=RAND_SEED, 
                    batch_size=BATCH_SIZE, use_refined=False)

# loading model:
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

num_feat_pro = 54 if FEATURE == 'msa' else 34
if OG_MODEL:
    model = DGraphDTA(num_features_pro=num_feat_pro,dropout=DROPOUT)
else:
    model = DGraphDTAImproved(num_features_pro=num_feat_pro, output_dim=128, # 128 is the same as the original model
                        dropout=DROPOUT)
    
MODEL_KEY = f'randomW_{data}_{BATCH_SIZE}B_{LEARNING_RATE}LR_{DROPOUT}D_{NUM_EPOCHS}E_{FEATURE}F_{model.__class__.__name__}'
print(f'\n{MODEL_KEY}')
model.to(device)

# %% loading checkpoint
cp = torch.load(f'results/model_checkpoints/ours/{MODEL_KEY}.model')
model.load_state_dict(cp)


# %% testing
loss, pred, actual = test(model, test_loader, device)
get_metrics(actual, pred,
            save_results=SAVE_RESULTS,
            save_path=media_save_p,
            model_key=MODEL_KEY,
            csv_file=MODEL_STATS_CSV,
            show=True,
            )
# %%
