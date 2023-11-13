import os
# for huggingface models:
os.environ['TRANSFORMERS_CACHE'] = '../hf_models/'

from prody import confProDy
confProDy(verbosity='none') # stop printouts from prody

# model and data options
MODEL_OPT = ['DG', 'DGI', 'ED', 'EDA', 'EDI', 'EDAI', 'EAT', 'CD', 'CED']

STRUCT_EDGE_OPT = ['anm', 'af2', 'af2-anm'] # edge options that require structural info (pdbs)
EDGE_OPT = ['simple', 'binary'] + STRUCT_EDGE_OPT
PRO_FEAT_OPT = ['nomsa', 'msa', 'shannon', 'foldseek']

LIG_FEAT_OPT = [None, 'original']
LIG_EDGE_OPT = [None, 'binary']

DATA_OPT = ['davis', 'kiba', 'PDBbind']

# data save paths
DATA_ROOT = '../data/'

# Model save paths
MEDIA_SAVE_DIR = 'results/model_media/'
MODEL_STATS_CSV = 'results/model_media/model_stats.csv'
MODEL_STATS_CSV_VAL = 'results/model_media/model_stats_val.csv'
MODEL_SAVE_DIR = 'results/model_checkpoints/ours'
CHECKPOINT_SAVE_DIR = MODEL_SAVE_DIR # alias for clarity

# cluster based configs:
import socket
DOMAIN_NAME = socket.getfqdn().split('.')
CLUSTER = DOMAIN_NAME[1]

SLURM_CONSTRAINT = None
SLURM_PARTITION = None
SLURM_ACCOUNT = None
SLURM_GPU_NAME = 'v100'

if 'uhnh4h' in DOMAIN_NAME:
    CLUSTER = 'h4h'
    SLURM_PARTITION = 'gpu'
    SLURM_CONSTRAINT = 'gpu32g'
    SLURM_ACCOUNT = 'kumargroup_gpu'
elif 'graham' in DOMAIN_NAME:
    CLUSTER = 'graham'
    SLURM_CONSTRAINT = 'cascade,v100'
elif 'cedar' in DOMAIN_NAME:
    CLUSTER = 'cedar'
    SLURM_GPU_NAME = 'v100l'
    
# bin paths
from pathlib import Path
FOLDSEEK_BIN = f'{Path.home()}/lib/foldseek/bin/foldseek'