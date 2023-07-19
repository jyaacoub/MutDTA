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
from src.models.utils import print_device_info
from src.data_analysis import get_metrics

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print_device_info(device)

#%% # Dataset Hyperparameters
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
# %% testing
MODEL_KEY = "randomW_davis_64B_0.0001LR_0.4D_2000E_nomsaF_DGraphDTAImproved"
model = DGraphDTAImproved(num_features_pro=54, output_dim=128, dropout=0.4)
cp = torch.load(f'results/model_checkpoints/ours/{MODEL_KEY}.model_tmp', 
                map_location=device)
model.load_state_dict(cp)
model.to(device)

dataset = DavisKibaDataset('/home/jyaacoub/projects/data/DavisKibaDataset/davis/')

train_loader, val_loader, test_loader = train_val_test_split(dataset, 
                    train_split=TRAIN_SPLIT, val_split=VAL_SPLIT,
                    shuffle_dataset=True, random_seed=RAND_SEED, 
                    batch_size=BATCH_SIZE, use_refined=False)

#%%
loss, pred, actual = test(model, test_loader, device)
get_metrics(actual, pred,
            save_results=False,
            save_path='',
            model_key=MODEL_KEY,
            csv_file='',
            show=True,
            )


