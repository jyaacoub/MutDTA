# %%
from src.utils.arg_parse import parse_train_test_args
args = parse_train_test_args(verbose=True)
FORCE_TRAINING = args.train
DEBUG = args.debug

# Model Hyperparameters
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
DROPOUT = args.dropout
NUM_EPOCHS = args.num_epochs

# Dataset args
TRAIN_SPLIT= args.train_split # 80% for training (80%)
VAL_SPLIT = args.val_split # 10% for validation (10%)
SHUFFLE_DATA = not args.no_shuffle

SAVE_RESULTS = True
SHOW_PLOTS = False
MODEL_STATS_CSV = 'results/model_media/model_stats.csv'
media_save_dir = 'results/model_media/'
model_save_dir = 'results/model_checkpoints/ours/'

#%%
import os, random, itertools, json

import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.utils import config # sets up env vars
from src.models import train, test, CheckpointSaver, print_device_info, debug
from src.models.mut_dta import EsmDTA, EsmAttentionDTA
from src.models.prior_work import DGraphDTA, DGraphDTAImproved
from src.data_processing import train_val_test_split
from src.data_processing.datasets import PDBbindDataset, DavisKibaDataset
from src.data_analysis import get_metrics

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print_device_info(device)

random.seed(args.rand_seed)
np.random.seed(args.rand_seed)
torch.manual_seed(args.rand_seed)

cp_saver = CheckpointSaver(model=None, 
                            save_path=None, 
                            train_all=False,
                            patience=10, min_delta=0.1,
                            save_freq=10)

# %% Training loop
metrics = {}
for DATA, FEATURE, EDGEW, MODEL in itertools.product(args.model_opt, args.data_opt, 
                                                     args.feature_opt, args.edge_opt):
    print(f'\n{"-"*40}\n({MODEL}, {DATA}, {FEATURE}, {EDGEW})')
    if MODEL in ['EAT']: # no edgew or features for this model type
        print('WARNING: edge weight and feature opt is not supported with the specified model.')
        MODEL_KEY = f'{MODEL}M_{DATA}D_{BATCH_SIZE}B_{LEARNING_RATE}LR_{DROPOUT}D_{NUM_EPOCHS}E'
    else:
        MODEL_KEY = f'{MODEL}M_{DATA}D_{FEATURE}F_{EDGEW}E_{BATCH_SIZE}B_{LEARNING_RATE}LR_{DROPOUT}D_{NUM_EPOCHS}E'
    
    print(f'# {MODEL_KEY} \n')
    
    media_save_p = f'{media_save_dir}/{DATA}/'
    print(f'    Saving media to: {media_save_p}')
    logs_out_p = f'{media_save_p}/train_log/{MODEL_KEY}.json'
    model_save_p = f'{model_save_dir}/{MODEL_KEY}.model'
    # create paths if they dont exist already:
    os.makedirs(media_save_p, exist_ok=True)
    os.makedirs(f'{media_save_p}/train_log/', exist_ok=True)

    # loading data
    if DATA == 'PDBbind':
        dataset = PDBbindDataset(save_root=f'../data/PDBbindDataset/{FEATURE}',
                 data_root='../data/v2020-other-PL/',
                 aln_dir='../data/PDBbind_aln', 
                 cmap_threshold=8.0,
                 feature_opt=FEATURE
                 )
    else:
        dataset = DavisKibaDataset(
                save_root=f'../data/DavisKibaDataset/{DATA}_{FEATURE}/',
                data_root=f'../data/{DATA}/',
                aln_dir  =f'../data/{DATA}/aln/',
                cmap_threshold=-0.5, 
                feature_opt=FEATURE
                )
    print(f'# Number of samples: {len(dataset)}')

    train_loader, val_loader, test_loader = train_val_test_split(dataset, 
                        train_split=TRAIN_SPLIT, val_split=VAL_SPLIT,
                        shuffle_dataset=True, random_seed=args.rand_seed, 
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
    
    model.to(device)
    cp_saver.new_model(model, save_path=model_save_p)
    
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
