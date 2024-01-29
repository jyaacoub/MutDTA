import os
# for huggingface models:
os.environ['TRANSFORMERS_CACHE'] = os.path.abspath('../hf_models/')

from prody import confProDy
confProDy(verbosity='none') # stop printouts from prody

from src.utils.enum import CustomEnum
#############################
# Model and data options
#############################
class MODEL_OPT(CustomEnum):
    DG = 'DG'
    DGI = 'DGI'
    
    ED = 'ED'
    EDA = 'EDA'
    EDI = 'EDI'
    EDAI = 'EDAI'
    EAT = 'EAT'
    
    CD = 'CD'
    CED = 'CED'
    SPD = 'SPD'

# protein options
class EDGE_OPT(CustomEnum):
    simple = 'simple'
    binary = 'binary'
    
    anm = 'anm'
    af2 = 'af2'
    af2_anm = 'af2-anm'
    
class PRO_FEAT_OPT(CustomEnum):
    nomsa = 'nomsa'
    msa = 'msa'
    shannon = 'shannon'
    
    foldseek = 'foldseek'
    
# Protein options that require PDB structure files to work
STRUCT_EDGE_OPT = CustomEnum('struct_edge_opt', ['anm', 'af2', 'af2-anm'])
STRUCT_PRO_FEAT_OPT = CustomEnum('struct_pro_feat_opt', ['foldseek'])

# ligand options
class LIG_EDGE_OPT(CustomEnum):
    binary = 'binary'

class LIG_FEAT_OPT(CustomEnum):
    original = 'original'

# Datasets
class DATA_OPT(CustomEnum):
    davis = 'davis'
    kiba = 'kiba'
    PDBbind = 'PDBbind'

#############################
# save paths
#############################
DATA_ROOT = os.path.abspath('../data/')

# Model save paths
MEDIA_SAVE_DIR      = os.path.abspath('results/model_media/')
MODEL_STATS_CSV     = os.path.abspath('results/model_media/model_stats.csv')
MODEL_STATS_CSV_VAL = os.path.abspath('results/model_media/model_stats_val.csv')
MODEL_SAVE_DIR      = os.path.abspath('results/model_checkpoints/ours')
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
elif 'narval' in DOMAIN_NAME:
    CLUSTER = 'narval'
    SLURM_GPU_NAME = 'a100'
    
# bin paths
from pathlib import Path
FOLDSEEK_BIN = f'{Path.home()}/lib/foldseek/bin/foldseek'
MMSEQ2_BIN = f'{Path.home()}/lib/mmseqs/bin/mmseqs'


###########################
# LOGGING STUFF:
# Adapted from - https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
###########################

import logging 

class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    blue = "\x1b[36;20m"
    reset = "\x1b[0m"
    format = "%(asctime)s|%(name)s:%(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: blue + '%(message)s' + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)
