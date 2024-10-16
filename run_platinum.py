import argparse
parser = argparse.ArgumentParser(description='Runs model on platinum dataset to evaluate it.')
parser.add_argument('--model_opt', type=str, default='davis_DG', 
                    choices=['davis_DG',    'davis_gvpl',   'davis_esm', 
                             'kiba_DG',     'kiba_esm',     'kiba_gvpl',
                             'PDBbind_DG',  'PDBbind_esm',  'PDBbind_gvpl'],
                    help='Model option. See MutDTA/src/__init__.py for details.')
parser.add_argument('--fold', type=int, default=1, 
                    help='Which model fold to use (there are 5 models for each option due to 5-fold CV).')
parser.add_argument('--out_dir', type=str, default='./', 
                    help='Output directory path to save csv file for prediction results.')

args = parser.parse_args()
MODEL_OPT = args.model_opt
FOLD = args.fold
OUT_DIR = args.out_dir

import logging
logging.getLogger().setLevel(logging.DEBUG)
logging.debug("#"*50)
logging.debug(f"MODEL_OPT: {MODEL_OPT}")
logging.debug(f"FOLD: {FOLD}")
logging.debug(f"OUT_DIR: {OUT_DIR}")
logging.debug("#"*50)

import torch, os
import pandas as pd

from src import cfg
from src import TUNED_MODEL_CONFIGS
from src.utils.loader import Loader
from src.train_test.training import test

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PARAMS = TUNED_MODEL_CONFIGS[MODEL_OPT]

### Loading the model
logging.debug(f"Loading the model {MODEL_OPT}")
model, model_kwargs = Loader.load_tuned_model(MODEL_OPT, fold=FOLD)
MODEL_KEY = Loader.get_model_key(**model_kwargs)
model.to(DEVICE)
model.eval()

### Loading the data and Test
logging.debug("Loading platinum test dataloader for model")
loaders = Loader.load_DataLoaders(cfg.DATA_OPT.platinum,
                               pro_feature    = MODEL_PARAMS['feature_opt'], 
                               edge_opt       = MODEL_PARAMS['edge_opt'],
                               ligand_feature = MODEL_PARAMS['lig_feat_opt'], 
                               ligand_edge    = MODEL_PARAMS['lig_edge_opt'],
                               datasets=['test'])

logging.debug("Running inference on test loader")
loss, pred, actual = test(model, loaders['test'], DEVICE, verbose=True)

# save as a CSV with cols: code, prot_id, pred, actual
logging.debug(f"Saving output to '{OUT_DIR}/{MODEL_KEY}_PLATINUM.csv'")
df = pd.DataFrame({
    'prot_id': [b['prot_id'][0] for b in loaders['test']], 
    'pred': pred, 
    'actual': actual
    },
    index=[b['code'][0] for b in loaders['test']])

df.index.name = 'code'
df.to_csv(f'{OUT_DIR}/{MODEL_KEY}_PLATINUM.csv')