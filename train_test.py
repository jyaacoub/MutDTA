# %%
from src.utils.arg_parse import parse_train_test_args
args = parse_train_test_args(verbose=True,
                             jyp_args='-m DG -d PDBbind -f nomsa -e binary -bs 64')
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
from src.train_test.training import train, test
from src.train_test.utils import  CheckpointSaver, print_device_info, debug, train_val_test_split
from src.data_analysis import get_metrics
from src.utils.loader import Loader

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print_device_info(device)

random.seed(args.rand_seed)
np.random.seed(args.rand_seed)
torch.manual_seed(args.rand_seed)

cp_saver = CheckpointSaver(model=None, 
                            save_path=None, 
                            train_all=True,
                            patience=10, min_delta=0.1,
                            save_freq=10)

# %% Training loop
metrics = {}
for MODEL, DATA, FEATURE, EDGEW in itertools.product(args.model_opt, args.data_opt, 
                                                     args.feature_opt, args.edge_opt):
    print(f'\n{"-"*40}\n({MODEL}, {DATA}, {FEATURE}, {EDGEW})')
    MODEL_KEY = Loader.get_model_key(MODEL,DATA,FEATURE,EDGEW,
                                     BATCH_SIZE,LEARNING_RATE,DROPOUT,NUM_EPOCHS,
                                     pro_overlap=args.protein_overlap)
    print(f'# {MODEL_KEY} \n')
    
    # init paths for media and model checkpoints
    media_save_p = f'{media_save_dir}/{DATA}/'
    logs_out_p = f'{media_save_p}/train_log/{MODEL_KEY}.json'
    model_save_p = f'{model_save_dir}/{MODEL_KEY}.model'
    
    # create paths if they dont exist already:
    os.makedirs(media_save_p, exist_ok=True)
    os.makedirs(f'{media_save_p}/train_log/', exist_ok=True)


    # ==== LOAD DATA ====
    # WARNING: Deprecating use of split to ensure all models train on same dataset splits.
    # dataset = Loader.load_dataset(DATA, FEATURE, EDGEW, subset='full')
    # print(f'# Number of samples: {len(dataset)}')
    # train_loader, val_loader, test_loader = train_val_test_split(dataset, 
    #                     train_split=TRAIN_SPLIT, val_split=VAL_SPLIT,
    #                     shuffle_dataset=True, random_seed=args.rand_seed, 
    #                     batch_train=BATCH_SIZE, use_refined=False,
    #                     split_by_prot=not args.protein_overlap)
    
    loaders = Loader.load_DataLoaders(DATA, FEATURE, EDGEW, path='../', 
                                        batch_train=BATCH_SIZE,
                                        datasets=['train', 'test', 'val'],
                                        protein_overlap=args.protein_overlap)


    # ==== LOAD MODEL ====
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'#Device: {device}')
    model = Loader.load_model(MODEL, FEATURE, EDGEW, DROPOUT).to(device)
    cp_saver.new_model(model, save_path=model_save_p)
    
    if DEBUG: 
        # run single batch through model
        debug(model, loaders['train'], device)
        continue # skip training
    
    
    # ==== TRAINING ====
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
        logs = train(model, loaders['train'], loaders['val'], device, 
                    epochs=NUM_EPOCHS, lr_0=LEARNING_RATE, saver=cp_saver)
        cp_saver.save()
        # load best model for testing
        model.load_state_dict(cp_saver.best_model_dict) 
        # save training logs for plotting later
        os.makedirs(os.path.dirname(logs_out_p), exist_ok=True)
        with open(logs_out_p, 'w') as f:
            json.dump(logs, f, indent=4)
        
            
    # ==== EVALUATE ====
    # testing
    loss, pred, actual = test(model, loaders['test'], device)
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
