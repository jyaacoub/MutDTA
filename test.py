#%%
from src.utils import config as cfg
import os
import torch
import pandas as pd

from src.analysis.metrics import get_save_metrics
from src.train_test.training import test
from src.utils.loader import Loader
from src.utils.arg_parse import parse_train_test_args
args, unknown_args = parse_train_test_args(verbose=True,
                             jyp_args=' -m EDI -d PDBbind -f nomsa -e anm -lr 0.0001 -bs 20 -do 0.4 -ne 2000')
# %%
MODEL = args.model_opt[0]
DATA = args.data_opt[0]
FEATURE = args.feature_opt[0]
EDGE = args.edge_opt[0]
LIG_FEATURE = args.ligand_feature_opt[0]
LIG_EDGE = args.ligand_edge_opt[0]

# Model Hyperparameters
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
DROPOUT = args.dropout
EPOCHS = args.num_epochs

media_save_p = f'{cfg.MEDIA_SAVE_DIR}/{DATA}/'

MODEL_KEY = Loader.get_model_key(model=MODEL,data=DATA,pro_feature=FEATURE,edge=EDGE,
                                batch_size=BATCH_SIZE,lr=LEARNING_RATE,
                                dropout=DROPOUT,n_epochs=EPOCHS,
                                pro_overlap=args.protein_overlap,
                                fold=args.fold_selection,
                                ligand_feature=LIG_FEATURE,
                                ligand_edge=LIG_EDGE)

model_p_tmp = f'{cfg.MODEL_SAVE_DIR}/{MODEL_KEY}.model_tmp'
model_p = f'{cfg.MODEL_SAVE_DIR}/{MODEL_KEY}.model'

# MODEL_KEY = 'DDP-' + MODEL_KEY # distributed model
model_p = model_p if os.path.isfile(model_p) else model_p_tmp
assert os.path.isfile(model_p), f"MISSING MODEL CHECKPOINT {model_p}"

print(model_p)



# %% Initialize model and load checkpoint
model = Loader.init_model(model=MODEL, pro_feature=FEATURE, pro_edge=EDGE, dropout=DROPOUT, **unknown_args)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mdl_dict = torch.load(model_p, map_location=device)
model.safe_load_state_dict(mdl_dict)
    
model.to(device)

# %% load test and val data
subsets = ['test', 'val', 'train'] if args.save_pred_train else ['test', 'val']
loaders = Loader.load_DataLoaders(DATA, FEATURE, EDGE,
                                  datasets=subsets,
                                  path=cfg.DATA_ROOT,
                                  batch_train=1, # MUST BE 1 FOR test SET
                                  protein_overlap=args.protein_overlap,
                                  training_fold=args.fold_selection,
                                  ligand_feature=LIG_FEATURE,
                                  ligand_edge=LIG_EDGE)

#%% Run model on test set
loss, pred, actual = test(model, loaders['test'], device)
if args.save_pred_test:
    # saving as csv with columns code, pred, actual
    # get codes from test loader
    codes = [b['code'][0] for b in loaders['test']] # NOTE: batch size is 1
    df = pd.DataFrame({'pred': pred, 'actual': actual}, index=codes)
    out_dir = f'{cfg.MEDIA_SAVE_DIR}/test_set_pred/'
    os.makedirs(out_dir, exist_ok=True)
    df.index.name = 'name'
    df.to_csv(f'{out_dir}/{MODEL_KEY}_testPred.csv')

print(f'# Test loss: {loss}')
get_save_metrics(actual, pred,
            save_figs=False,
            save_path=media_save_p,
            model_key=MODEL_KEY,
            csv_file=cfg.MODEL_STATS_CSV,
            show=False,
            )

#%% Run model on val set
loss, pred, actual = test(model, loaders['val'], device)
print(f'# Val loss: {loss}')
get_save_metrics(actual, pred,
            save_figs=False,
            save_path=media_save_p,
            model_key=MODEL_KEY,
            csv_file=cfg.MODEL_STATS_CSV_VAL,
            show=False,
            )

#%% Run model on train set
if args.save_pred_train:
    loss, pred, actual = test(model, loaders['train'], device)
    print(f'# Train loss: {loss}')
    # save as csv with columns code, pred, actual
    # get codes from test loader
    codes = [b['code'][0] for b in loaders['train']] # NOTE: batch size is 1
    df = pd.DataFrame({'pred': pred, 'actual': actual}, index=codes)
    out_dir = f'{cfg.MEDIA_SAVE_DIR}/train_set_pred/'
    os.makedirs(out_dir, exist_ok=True)
    df.index.name = 'name'
    df.to_csv(f'{out_dir}/{MODEL_KEY}_trainPred.csv')
    
    # getting metrics is not needed for training set...

# %% renaming checkpoint to remove _tmp specification
if (not args.no_rename and           # Only rename if not specified, default is to rename
    not os.path.isfile(model_p) and  # ensuring no overwrite of existing model
    os.path.isfile(model_p_tmp)):    # ensuring _tmp model exists
    os.rename(model_p_tmp, model_p)
