# %%
from src.utils.arg_parse import parse_train_test_args

args, unknown_args = parse_train_test_args(verbose=True,
                             jyp_args='--model_opt DG \
                     --data_opt davis \
                     \
                     --feature_opt nomsa \
                     --edge_opt binary \
                     --ligand_feature_opt original \
                     --ligand_edge_opt binary \
                     \
                     --learning_rate 0.00012 \
                     --batch_size 128 \
                     --dropout 0.24 \
                     --output_dim 128 \
                     \
                     --train \
                     --fold_selection 0 \
                     --num_epochs 2000')
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

SAVE_FIGS = False
SHOW_PLOTS = False

#%%
import os, random, itertools, json

import numpy as np
import torch

import matplotlib.pyplot as plt

from src.utils import config as cfg # sets up env vars

from src.train_test.training import train, test, CheckpointSaver
from src.train_test.utils import  print_device_info, debug
from src.analysis import get_save_metrics
from src.utils.loader import Loader

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print_device_info(device)

random.seed(args.rand_seed)
np.random.seed(args.rand_seed)
torch.manual_seed(args.rand_seed)

cp_saver = CheckpointSaver(model=None, 
                            save_path=None, 
                            train_all=False, # forces full training
                            patience=100)

# %% Training loop
metrics = {}
for (MODEL, DATA, 
     FEATURE, EDGEW, 
     ligand_feature, ligand_edge) in itertools.product(
                                                    args.model_opt, args.data_opt, 
                                                    args.feature_opt, args.edge_opt, 
                                                    args.ligand_feature_opt, args.ligand_edge_opt):
    print(f'\n{"-"*40}\n({MODEL}, {DATA}, {FEATURE}, {EDGEW})')
    MODEL_KEY = Loader.get_model_key(model=MODEL,data=DATA,pro_feature=FEATURE,edge=EDGEW,
                                     ligand_feature=ligand_feature, ligand_edge=ligand_edge,
                                     batch_size=BATCH_SIZE,lr=LEARNING_RATE,dropout=DROPOUT,n_epochs=NUM_EPOCHS,
                                     pro_overlap=args.protein_overlap, fold=args.fold_selection)
    print(f'# {MODEL_KEY} \n')
    
    # init paths for media and model checkpoints
    media_save_p = f'{cfg.MEDIA_SAVE_DIR}/{DATA}/'
    logs_out_p = f'{media_save_p}/train_log/{MODEL_KEY}.json'
    model_save_p = f'{cfg.MODEL_SAVE_DIR}/{MODEL_KEY}.model'
    
    # create paths if they dont exist already:
    os.makedirs(media_save_p, exist_ok=True)
    os.makedirs(f'{media_save_p}/train_log/', exist_ok=True)


    # ==== LOAD DATA ====
    loaders = Loader.load_DataLoaders(data=DATA, pro_feature=FEATURE, edge_opt=EDGEW, path=cfg.DATA_ROOT, 
                                        ligand_feature=ligand_feature, ligand_edge=ligand_edge,
                                        batch_train=BATCH_SIZE,
                                        datasets=['train', 'test', 'val'],
                                        training_fold=args.fold_selection, # default is None from arg_parse
                                        protein_overlap=args.protein_overlap)


    # ==== LOAD MODEL ====
    print(f'#Device: {device}')
    model = Loader.init_model(model=MODEL, pro_feature=FEATURE, pro_edge=EDGEW, dropout=DROPOUT,
                                ligand_feature=ligand_feature, ligand_edge=ligand_edge,
                                **unknown_args).to(device)
    cp_saver.new_model(model, save_path=model_save_p)
    cp_saver.min_delta = 0.2 if DATA == cfg.DATA_OPT.PDBbind else 0.05
    
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
    get_save_metrics(actual, pred,
                save_figs=SAVE_FIGS,
                save_path=media_save_p,
                model_key=MODEL_KEY,
                csv_file=cfg.MODEL_STATS_CSV,
                show=SHOW_PLOTS,
                logs=logs
                )
    plt.clf()
    
    # validation
    loss, pred, actual = test(model, loaders['val'], device)
    print(f'# Val loss: {loss}')
    get_save_metrics(actual, pred,
                save_figs=SAVE_FIGS,
                save_path=media_save_p,
                model_key=MODEL_KEY,
                csv_file=cfg.MODEL_STATS_CSV_VAL,
                show=False,
                )

    

# %%
