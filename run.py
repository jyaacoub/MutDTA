#%%
import os, random, itertools

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.models.prior_work import DGraphDTA
from src.data_processing import PDBbindDataset, train_val_test_split
from src.models import train, test
from src.data_analysis import get_metrics


PDB_RAW_DIR = '../data/v2020-other-PL/'
PDB_PROCESSED_DIR = '../data/pytorch_PDBbind/'

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
LEARNING_RATE = 0.001
DROPOUT = 0.2
NUM_EPOCHS = 50
weight_opt = ['kiba', 'davis', 'random']
WEIGHTS = 'davis'
#%% training and testing
metrics = {}
for WEIGHTS in weight_opt:
    SAVE_RESULTS = True
    media_save_p = 'results/model_media/figures/'
    csv_file = 'results/model_media/DGraphDTA_stats.csv'
    MODEL_KEY = f'{WEIGHTS}W_{NUM_EPOCHS}E' 
    # {BATCH_SIZE}B_{LEARNING_RATE}LR_{DROPOUT}DO are fixed so not included in model key

    #loading data and splitting into train, val, test
    pdb_dataset = PDBbindDataset(PDB_PROCESSED_DIR, PDB_RAW_DIR)

    train_loader, val_loader, test_loader = train_val_test_split(pdb_dataset, 
                        train_split=TRAIN_SPLIT, val_split=VAL_SPLIT,
                        shuffle_dataset=True, random_seed=RAND_SEED, 
                        batch_size=BATCH_SIZE)


    # loading model:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'\n{MODEL_KEY}')
        
    model = DGraphDTA(dropout=DROPOUT)
    model.to(device)
    assert WEIGHTS in weight_opt, 'WEIGHTS must be one of: kiba, davis, random'
    if WEIGHTS != 'random':
        model_file_name = f'results/model_checkpoints/prior_work/DGraphDTA_{WEIGHTS}_t2.model'
        model.load_state_dict(torch.load(model_file_name, map_location=device))

    # training
    logs = train(model, train_loader, val_loader, device, 
            epochs=NUM_EPOCHS, lr=LEARNING_RATE)

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
    
    #  testing
    loss, pred, actual = test(model, test_loader, device)
    get_metrics(pred, actual,
                save_results=SAVE_RESULTS,
                save_path=media_save_p,
                model_key=MODEL_KEY,
                csv_file=csv_file
                )
    
    metrics[MODEL_KEY] = {'test_loss': loss,
                          'logs': logs}

# %%
#plotting all model train and val loss together
index = np.arange(1, NUM_EPOCHS+1)
c = plt.plot(index, metrics['kibaW_50E']['logs']['train_loss'], label='kiba train')[0].get_color()
plt.plot(index, metrics['kibaW_50E']['logs']['val_loss'], label='kiba val', color=c, alpha=0.5)

c = plt.plot(index, metrics['davisW_50E']['logs']['train_loss'], label='davis train')[0].get_color()
plt.plot(index, metrics['davisW_50E']['logs']['val_loss'], label='davis val', color=c, alpha=0.5)

c = plt.plot(index, metrics['randomW_50E']['logs']['train_loss'], label='random train')[0].get_color()
plt.plot(index, metrics['randomW_50E']['logs']['val_loss'], label='random val', color=c, alpha=0.5)

plt.legend()
plt.title('DGraphDTA Weight Initialization Impact on Loss')
plt.xlabel('Epoch')
plt.xlim(0, NUM_EPOCHS)
plt.ylabel('Loss')
plt.savefig(f'results/model_media/figures/all_loss.png')
plt.show()

# %%
