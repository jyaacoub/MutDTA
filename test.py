#%%
from src.utils import config as cfg
import torch
import os
from torch_geometric.loader import DataLoader

from src.data_analysis.metrics import get_metrics
from src.train_test.training import test
from src.utils.loader import Loader
from src.utils.arg_parse import parse_train_test_args
args = parse_train_test_args(verbose=True,
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
model = Loader.init_model(model=MODEL, pro_feature=FEATURE, pro_edge=EDGE, dropout=DROPOUT)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mdl_dict = torch.load(model_p, map_location=device)
model.safe_load_state_dict(mdl_dict)
    
model.to(device)

# %% load test and val data
loaders = Loader.load_DataLoaders(DATA, FEATURE, EDGE,
                                  datasets=['test', 'val'],
                                  path=cfg.DATA_ROOT,
                                  batch_train=1, # batch size is irrelevant for eval
                                  protein_overlap=args.protein_overlap,
                                  training_fold=args.fold_selection,
                                  ligand_feature=LIG_FEATURE,
                                  ligand_edge=LIG_EDGE)

#%% Run model on test set
loss, pred, actual = test(model, loaders['test'], device)
print(f'# Test loss: {loss}')
get_metrics(actual, pred,
            save_figs=False,
            save_path=media_save_p,
            model_key=MODEL_KEY,
            csv_file=cfg.MODEL_STATS_CSV,
            show=False,
            )

#%% Run model on val set
loss, pred, actual = test(model, loaders['val'], device)
print(f'# Val loss: {loss}')
get_metrics(actual, pred,
            save_figs=False,
            save_path=media_save_p,
            model_key=MODEL_KEY,
            csv_file=cfg.MODEL_STATS_CSV_VAL,
            show=False,
            )


# %% renaming checkpoint to remove _tmp specification
if (not os.path.isfile(model_p) and  # ensuring no overwrite
    os.path.isfile(model_p_tmp)):
    os.rename(model_p_tmp, model_p)