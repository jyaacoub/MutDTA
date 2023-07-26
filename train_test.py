import argparse

# Define the options for data_opt and FEATURE_opt
data_opt_choices = ['davis', 'kiba']
feature_opt_choices = ['nomsa', 'msa', 'shannon']
model_opt_choices = ['DG', 'DGI', 'ED', 'EDA', 'EDI', 'EDAI']

# Create the argument parser
parser = argparse.ArgumentParser(description="Argument parser for selecting options.")

# Add the argument for data_opt
parser.add_argument('-d',
    '--data_opt',
    choices=data_opt_choices,
    nargs='+',  # Allows accepting multiple arguments
    # default=data_opt_choices[0],
    required=True,
    help=f'Select one of {data_opt_choices} (default: {data_opt_choices[0]}).'
)

# Add the argument for FEATURE_opt
parser.add_argument('-f',
    '--feature_opt',
    choices=feature_opt_choices,
    nargs='+',  # Allows accepting multiple arguments for FEATURE_opt
    required=True,
    help=f'Select one or more from {feature_opt_choices}.'
)

# Add the argument for FEATURE_opt
parser.add_argument('-m',
    '--model_opt',
    choices=model_opt_choices,
    nargs='+',  # Allows accepting multiple arguments for FEATURE_opt
    default=model_opt_choices,
    required=False,
    help=f'Select one or more from {model_opt_choices} where I = "improved" and '+\
        'A = "all features". For example: DG is DGraphDTA ' + \
        'DGI is DGraphDTAImproved, ED is EsmDTA with esm_only set to true, '+\
        'and EDA is the same but with esm_only set to False.'
)
# Parse the arguments from the command line
args = parser.parse_args()

#%%
import os, random, itertools, math, json, argparse
import config


from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.feature_extraction.protein import get_pfm
from src.models.mut_dta import EsmDTA
from src.models.prior_work import DGraphDTA, DGraphDTAImproved
from src.data_processing import PDBbindDataset, DavisKibaDataset, train_val_test_split
from src.models import train, test, CheckpointSaver
from src.models.utils import print_device_info
from src.data_analysis import get_metrics


# %%
# Access the selected options
data_opt = args.data_opt
feature_opt = args.feature_opt
model_opt = args.model_opt

# Now you can use the selected options in your code as needed
print(f"Selected data_opt: {data_opt}")
print(f"Selected feature_opt list: {feature_opt}")
print(f"Selected og_model_opt: {model_opt}")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print_device_info(device)

MODEL_STATS_CSV = 'results/model_media/model_stats.csv'

# Dataset Hyperparameters
TRAIN_SPLIT= .8 # 80% of data for training
VAL_SPLIT = .1 # 10% for val and remaining is for testing (10%)
SHUFFLE_DATA = True
RAND_SEED = 0

random.seed(RAND_SEED)
np.random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)

# Tune Hyperparameters after grid search
BATCH_SIZE = 10
LEARNING_RATE = 0.0001
DROPOUT = 0.4
NUM_EPOCHS = 2000

SAVE_RESULTS = True
SHOW_PLOTS = False
media_save_p = 'results/model_media/davis_kiba/'
model_save_p = 'results/model_checkpoints/ours/'
cp_saver = CheckpointSaver(model=None, 
                            save_path=None, 
                            train_all=False,
                            patience=10, min_delta=0.1)


# %% Training loop
metrics = {}

for DATA, FEATURE, MODEL in itertools.product(data_opt, feature_opt, model_opt):
    print(f'({DATA}, {FEATURE}, {MODEL}):')
    # loading data
    DATA_ROOT = f'../data/davis_kiba/{DATA}/' # where to get data from
    dataset = DavisKibaDataset(
            save_root=f'../data/DavisKibaDataset/{DATA}_{FEATURE}/',
            data_root=DATA_ROOT,
            aln_dir=f'{DATA_ROOT}/aln/',
            cmap_threshold=-0.5, shannon=FEATURE=='shannon')
    print(f'Number of samples: {len(dataset)}')

    train_loader, val_loader, test_loader = train_val_test_split(dataset, 
                        train_split=TRAIN_SPLIT, val_split=VAL_SPLIT,
                        shuffle_dataset=True, random_seed=RAND_SEED, 
                        batch_train=BATCH_SIZE, use_refined=False)

    # loading model:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    num_feat_pro = 54 if 'msa' in FEATURE else 34
    if MODEL == 'DG':
        model = DGraphDTA(num_features_pro=num_feat_pro,dropout=DROPOUT)
    elif MODEL == 'DGI':
        model = DGraphDTAImproved(num_features_pro=num_feat_pro, output_dim=128, # 128 is the same as the original model
                            dropout=DROPOUT)
    elif MODEL == 'ED':
        model = EsmDTA(esm_head='facebook/esm2_t6_8M_UR50D',
                       num_features_pro=320, # only esm features
                       pro_emb_dim=54, # inital embedding size after first GCN layer
                       dropout=DROPOUT,
                       esm_only=True)
    elif MODEL == 'EDA':
        model = EsmDTA(esm_head='facebook/esm2_t6_8M_UR50D',
                       num_features_pro=320+num_feat_pro, # esm features + other features
                       pro_emb_dim=54, # inital embedding size after first GCN layer
                       dropout=DROPOUT,
                       esm_only=False)
    elif MODEL == 'EDI':
        model = EsmDTA(esm_head='facebook/esm2_t6_8M_UR50D',
                       num_features_pro=320,
                       pro_emb_dim=512, # increase embedding size
                       dropout=DROPOUT,
                       esm_only=True)
    elif MODEL == 'EDAI':
        model = EsmDTA(esm_head='facebook/esm2_t6_8M_UR50D',
                       num_features_pro=320 + num_feat_pro,
                       pro_emb_dim=512,
                       dropout=DROPOUT,
                       esm_only=False)
        
    MODEL_KEY = f'randW_{DATA}_{BATCH_SIZE}B_{LEARNING_RATE}LR_{DROPOUT}D_{NUM_EPOCHS}E_{FEATURE}F_{model.__class__.__name__}'
    logs_out_p = f'{media_save_p}/train_log/{MODEL_KEY}.json'
    print(f'\n{MODEL_KEY}')
    
    cp_saver.new_model(model, save_path=f'{model_save_p}/{MODEL_KEY}.model')
    model.to(device)
    
    # check if model has already been trained:
    logs = None
    if os.path.exists(cp_saver.save_path):
        print('Model already trained')
        # load ckpnt for testing
        model.load_state_dict(torch.load(cp_saver.save_path, 
                                         map_location=device))
        # loading logs for plotting 
        if os.path.exists(logs_out_p):
            with open(logs_out_p, 'r') as f:
                logs = json.load(f)
    else:
        # training
        logs = train(model, train_loader, val_loader, device, 
                    epochs=NUM_EPOCHS, lr=LEARNING_RATE, saver=cp_saver)
        cp_saver.save()
        # load best model for testing
        model.load_state_dict(cp_saver.best_model_dict) 
        # save training logs for plotting later
        with open(logs_out_p, 'w') as f:
            json.dump(logs, f, indent=4)
        
            
    # testing
    loss, pred, actual = test(model, test_loader, device)
    get_metrics(actual, pred,
                save_results=SAVE_RESULTS,
                save_path=media_save_p,
                model_key=MODEL_KEY,
                csv_file=MODEL_STATS_CSV,
                show=SHOW_PLOTS,
                )
    plt.clf()

    # display train val plot
    if logs is not None: 
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        index = np.arange(1, len(logs['train_loss'])+1)
        plt.plot(index, logs['train_loss'], label='train')
        plt.plot(index, logs['val_loss'], label='val')
        plt.legend()
        plt.title(f'{MODEL_KEY} Loss')
        plt.xlabel('Epoch')
        # plt.xticks(range(0,NUM_EPOCHS+1, 2))
        plt.xlim(0, NUM_EPOCHS)
        plt.ylabel('Loss')
        if SAVE_RESULTS: plt.savefig(f'{media_save_p}/{MODEL_KEY}_loss.png')
        if SHOW_PLOTS: plt.show()
    plt.clf()
    

# %%
