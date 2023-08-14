#%%
import argparse

# Define the options for data_opt and FEATURE_opt
model_opt_choices = ['DG', 'DGI', 'ED', 'EDA', 'EDI', 'EDAI', 'EAT']
data_opt_choices = ['davis', 'kiba', 'PDBbind']
feature_opt_choices = ['nomsa', 'msa', 'shannon']
edge_opt_choices = ['simple', 'binary']

# Create the argument parser
parser = argparse.ArgumentParser(description="Argument parser for selecting options.")

# Add the argument for model_opt
parser.add_argument('-m',
    '--model_opt',
    choices=model_opt_choices,
    nargs='+',  # Allows accepting multiple arguments for FEATURE_opt
    required=True,
    help=f'Select one or more from {model_opt_choices} where I = "improved" and '+ \
        'A = "all features". For example: DG is DGraphDTA ' + \
        'DGI is DGraphDTAImproved, ED is EsmDTA with esm_only set to true, '+ \
        'and EDA is the same but with esm_only set to False. Additional options:' + \
        '\n\t- EAT: EsmAttentionDTA (no graph for protein rep)' 
)

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
parser.add_argument('-e',
    '--edge_opt',
    choices=edge_opt_choices,
    nargs='+',  # Allows accepting multiple arguments for FEATURE_opt
    default=edge_opt_choices[0:1],
    required=False,
    help=f'Select one or more from {edge_opt_choices}. "simple" is just taking ' + \
        'the normalized values from the protein cmap, "binary" means no edge weights'
)

parser.add_argument('-t',
    '--train',
    action='store_true',
    help='Forces training, if already trained it will load up the state dict'
)

parser.add_argument('-D',
    '--debug',
    action='store_true',
    help='Enters debug mode, no training is done, just model initialization.'
)


# hyperparameter options
parser.add_argument('-bs',
    '--batch_size',
    action='store',
    type=int,
    default=64,
    help='Batch size for training (default: 64)'
)

parser.add_argument('-lr',
    '--learning_rate',
    action='store',
    type=float,
    default=0.0001,
    help='Learning rate for training (default: 0.0001)'
)

parser.add_argument('-do',
    '--dropout',
    action='store',
    type=float,
    default=0.4,
    help='Dropout rate for training (default: 0.4)'
)

parser.add_argument('-ne',
    '--num_epochs',
    action='store',
    type=int,
    default=2000,
    help='Number of epochs for training (default: 2000)'
)


#%% Parse the arguments from the command line
try:
    if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
        # Jupyter notebook
        in_args = '-m EAT -d davis -f nomsa -e simple -D'.split()
        args = parser.parse_args(args=in_args)
    else:  
        args = parser.parse_args()
except NameError:
    # Python script
    args = parser.parse_args()

# Access the selected options
model_opt = args.model_opt
data_opt = args.data_opt
feature_opt = args.feature_opt
edge_opt = args.edge_opt
FORCE_TRAINING = args.train
DEBUG = args.debug

# Now you can use the selected options in your code as needed
if DEBUG: print(f'---------------- DEBUG MODE ----------------')
print(f"     Selected og_model_opt: {model_opt}")
print(f"         Selected data_opt: {data_opt}")
print(f" Selected feature_opt list: {feature_opt}")
print(f"    Selected edge_opt list: {edge_opt}")
print(f"           forced training: {FORCE_TRAINING}\n")

# Dataset Hyperparameters
TRAIN_SPLIT= .8 # 80% of data for training
VAL_SPLIT = .1 # 10% for val and remaining is for testing (10%)
SHUFFLE_DATA = True
RAND_SEED = 0

# Model Hyperparameters
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
DROPOUT = args.dropout
NUM_EPOCHS = args.num_epochs

print(f"----------------- HYPERPARAMETERS -----------------")
print(f"               Batch size: {BATCH_SIZE}")
print(f"            Learning rate: {LEARNING_RATE}")
print(f"                  Dropout: {DROPOUT}")
print(f"               Num epochs: {NUM_EPOCHS}\n")

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
from src.models.mut_dta import EsmDTA, EsmAttentionDTA
from src.models.prior_work import DGraphDTA, DGraphDTAImproved
from src.data_processing.datasets import PDBbindDataset, DavisKibaDataset
from src.data_processing import train_val_test_split
from src.models import train, test, CheckpointSaver, print_device_info, debug
from src.data_analysis import get_metrics

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print_device_info(device)

MODEL_STATS_CSV = 'results/model_media/model_stats.csv'

random.seed(RAND_SEED)
np.random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)

SAVE_RESULTS = True
SHOW_PLOTS = False
media_save_dir = 'results/model_media/'
model_save_dir = 'results/model_checkpoints/ours/'
cp_saver = CheckpointSaver(model=None, 
                            save_path=None, 
                            train_all=False,
                            patience=10, min_delta=0.1,
                            save_freq=10)


# %% Training loop
metrics = {}

for DATA, FEATURE, EDGEW, MODEL in itertools.product(data_opt, feature_opt, edge_opt, model_opt):
    print(f'\n{"-"*40}\n({MODEL}, {DATA}, {FEATURE}, {EDGEW})')
    if MODEL in ['EAT']: # no edgew or features for this model type
        print('WARNING: edge weight and feature opt is not supported with the specified model.')
        MODEL_KEY = f'{MODEL}M_{DATA}D_{BATCH_SIZE}B_{LEARNING_RATE}LR_{DROPOUT}D_{NUM_EPOCHS}E'
    else:
        MODEL_KEY = f'{MODEL}M_{DATA}D_{FEATURE}F_{EDGEW}E_{BATCH_SIZE}B_{LEARNING_RATE}LR_{DROPOUT}D_{NUM_EPOCHS}E'
    
    print(f'# {MODEL_KEY} \n')
    
    #TODO: change this to split up davis and kiba dataset folders
    media_save_p = f'{media_save_dir}/davis_kiba/' if DATA in ['davis', 'kiba'] else f'{media_save_dir}/{DATA}/'
    print(f'    Saving media to: {media_save_p}')
    logs_out_p = f'{media_save_p}/train_log/{MODEL_KEY}.json'
    model_save_p = f'{model_save_dir}/{MODEL_KEY}.model'

    # loading data
    if DATA == 'PDBbind':
        dataset = PDBbindDataset(save_root=f'../data/PDBbindDataset/{FEATURE}',
                 data_root='../data/v2020-other-PL/',
                 aln_dir='../data/PDBbind_aln', 
                 cmap_threshold=8.0,
                 feature_opt=FEATURE
                 )
    else:
        DATA_ROOT = f'../data/{DATA}/' # where to get data from
        dataset = DavisKibaDataset(
                save_root=f'../data/DavisKibaDataset/{DATA}_{FEATURE}/',
                data_root=DATA_ROOT,
                aln_dir=f'{DATA_ROOT}/aln/',
                cmap_threshold=-0.5, 
                feature_opt=FEATURE
                )
    print(f'# Number of samples: {len(dataset)}')

    train_loader, val_loader, test_loader = train_val_test_split(dataset, 
                        train_split=TRAIN_SPLIT, val_split=VAL_SPLIT,
                        shuffle_dataset=True, random_seed=RAND_SEED, 
                        batch_train=BATCH_SIZE, use_refined=False,
                        split_by_prot=True)

    # loading model:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'#Device: {device}')

    num_feat_pro = 54 if 'msa' in FEATURE else 34
    if MODEL == 'DG':
        model = DGraphDTA(num_features_pro=num_feat_pro, 
                          dropout=DROPOUT, edge_weight_opt=EDGEW)
    elif MODEL == 'DGI':
        model = DGraphDTAImproved(num_features_pro=num_feat_pro, output_dim=128, # 128 is the same as the original model
                                  dropout=DROPOUT, edge_weight_opt=EDGEW)
    elif MODEL == 'ED':
        model = EsmDTA(esm_head='facebook/esm2_t6_8M_UR50D',
                       num_features_pro=320, # only esm features
                       pro_emb_dim=54, # inital embedding size after first GCN layer
                       dropout=DROPOUT,
                       esm_only=True,
                       edge_weight_opt=EDGEW)
    elif MODEL == 'EDA':
        model = EsmDTA(esm_head='facebook/esm2_t6_8M_UR50D',
                       num_features_pro=320+num_feat_pro, # esm features + other features
                       pro_emb_dim=54, # inital embedding size after first GCN layer
                       dropout=DROPOUT,
                       esm_only=False, # false to include all feats
                       edge_weight_opt=EDGEW)
    elif MODEL == 'EDI':
        model = EsmDTA(esm_head='facebook/esm2_t6_8M_UR50D',
                       num_features_pro=320,
                       pro_emb_dim=512, # increase embedding size
                       dropout=DROPOUT,
                       esm_only=True,
                       edge_weight_opt=EDGEW)
    elif MODEL == 'EDAI':
        model = EsmDTA(esm_head='facebook/esm2_t6_8M_UR50D',
                       num_features_pro=320 + num_feat_pro,
                       pro_emb_dim=512,
                       dropout=DROPOUT,
                       esm_only=False,
                       edge_weight_opt=EDGEW)
    elif MODEL == 'EAT':
        # this model only needs protein sequence, no additional features.
        model = EsmAttentionDTA(esm_head='facebook/esm2_t6_8M_UR50D',
                                dropout=DROPOUT)
    
    cp_saver.new_model(model, save_path=model_save_p)
    model.to(device)
    
    if DEBUG: 
        # run single batch through model
        debug(model, train_loader, device)
        continue # skip training
    
    # check if model has already been trained:
    logs = None
    if os.path.exists(cp_saver.save_path):
        print('# Model already trained')
        # load ckpnt
        model.load_state_dict(torch.load(cp_saver.save_path, 
                                         map_location=device))
        # loading logs for plotting
        if os.path.exists(logs_out_p):
            with open(logs_out_p, 'r') as f:
                logs = json.load(f)
    
    if not os.path.exists(cp_saver.save_path) or FORCE_TRAINING:
        # training
        logs = train(model, train_loader, val_loader, device, 
                    epochs=NUM_EPOCHS, lr=LEARNING_RATE, saver=cp_saver)
        cp_saver.save()
        # load best model for testing
        model.load_state_dict(cp_saver.best_model_dict) 
        # save training logs for plotting later
        os.makedirs(os.path.dirname(logs_out_p), exist_ok=True)
        with open(logs_out_p, 'w') as f:
            json.dump(logs, f, indent=4)
        
            
    # testing
    loss, pred, actual = test(model, test_loader, device)
    print(f'# Test loss: {loss}')
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
