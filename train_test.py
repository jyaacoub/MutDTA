import argparse

# Define the options for data_opt and FEATURE_opt
data_opt_choices = ['davis', 'kiba']
feature_opt_choices = ['nomsa', 'msa', 'shannon']
og_model_opt_choices = [True, False]

# Create the argument parser
parser = argparse.ArgumentParser(description="Argument parser for selecting options.")

# Add the argument for data_opt
parser.add_argument('-d'
    '--data_opt',
    choices=data_opt_choices,
    nargs='+',  # Allows accepting multiple arguments
    # default=data_opt_choices[0],
    required=True,
    help=f'Select one of {data_opt_choices} (default: {data_opt_choices[0]}).'
)

# Add the argument for FEATURE_opt
parser.add_argument('-f'
    '--feature_opt',
    choices=feature_opt_choices,
    nargs='+',  # Allows accepting multiple arguments for FEATURE_opt
    required=True,
    help=f'Select one or more from {feature_opt_choices}.'
)

# Add the argument for FEATURE_opt
parser.add_argument('-m'
    '--og_model_opt',
    choices=og_model_opt_choices,
    nargs='+',  # Allows accepting multiple arguments for FEATURE_opt
    default=og_model_opt_choices,
    required=False,
    help=f'Select one or more from {og_model_opt_choices}.'
)

# Parse the arguments from the command line
args = parser.parse_args()
# %%
# Access the selected options
data_opt = args.d
feature_opt = args.f
og_model_opt = args.m

# Now you can use the selected options in your code as needed
print(f"Selected data_opt: {data_opt}")
print(f"Selected feature_opt list: {feature_opt}")
print(f"Selected og_model_opt: {og_model_opt}")

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
from src.models.prior_work import DGraphDTA, DGraphDTAImproved
from src.data_processing import PDBbindDataset, DavisKibaDataset, train_val_test_split
from src.models import train, test, CheckpointSaver
from src.models.utils import print_device_info
from src.data_analysis import get_metrics


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print_device_info(device)

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
NUM_EPOCHS = 2000

SAVE_RESULTS = True
SHOW_PLOTS = False
media_save_p = 'results/model_media/davis_kiba/'
model_save_p = 'results/model_checkpoints/ours/'
cp_saver = CheckpointSaver(model=None, 
                            save_path=None, 
                            train_all=False,
                            patience=5, min_delta=0.05)


# %% Training loop
metrics = {}

for data, FEATURE, OG_MODEL in itertools.product(data_opt, feature_opt, og_model_opt):
    print(f'({data}, {FEATURE}, {OG_MODEL}):')
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

    num_feat_pro = 54 if 'msa' in FEATURE else 34
    if OG_MODEL:
        model = DGraphDTA(num_features_pro=num_feat_pro,dropout=DROPOUT)
    else:
        model = DGraphDTAImproved(num_features_pro=num_feat_pro, output_dim=128, # 128 is the same as the original model
                            dropout=DROPOUT)
    MODEL_KEY = f'randW_{data}_{BATCH_SIZE}B_{LEARNING_RATE}LR_{DROPOUT}D_{NUM_EPOCHS}E_{FEATURE}F_{model.__class__.__name__}'
    print(f'\n{MODEL_KEY}')
    
    cp_saver.new_model(model, save_path=f'{model_save_p}/{MODEL_KEY}.model')
    
    # check if model has already been trained:
    if os.path.exists(cp_saver.save_path):
        print('Model already trained')
        continue
    
    # training
    model.to(device)
    logs = train(model, train_loader, val_loader, device, 
                epochs=NUM_EPOCHS, lr=LEARNING_RATE, saver=cp_saver)
    cp_saver.save()

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
    if SAVE_RESULTS: plt.savefig(f'{media_save_p}/{MODEL_KEY}_loss.png')
    if SHOW_PLOTS: plt.show()
    plt.clf()
    
    # testing
    model.load_state_dict(cp_saver.best_model_dict)
    
    loss, pred, actual = test(model, test_loader, device)
    get_metrics(actual, pred,
                save_results=SAVE_RESULTS,
                save_path=media_save_p,
                model_key=MODEL_KEY,
                csv_file=MODEL_STATS_CSV,
                show=SHOW_PLOTS,
                )
    metrics[MODEL_KEY] = {'test_loss': loss, 
                          'best_epoch': cp_saver.best_epoch,
                            'logs': logs}
    

# save metrics
with open(f'{media_save_p}/metrics.json', 'a') as f:
    json.dump(metrics, f, indent=4)
