import os
# for huggingface models:
os.environ['TRANSFORMERS_CACHE'] = '../hf_models/'

from prody import confProDy
confProDy(verbosity='none') # stop printouts from prody


MODEL_OPT = ['DG', 'DGI', 'ED', 'EDA', 'EDI', 'EDAI', 'EAT']

STRUCT_EDGE_OPT = ['anm', 'af2', 'af2-anm'] # edge options that require structural info (pdbs)
EDGE_OPT = ['simple', 'binary'] + STRUCT_EDGE_OPT

DATA_OPT = ['davis', 'kiba', 'PDBbind']
PRO_FEAT_OPT = ['nomsa', 'msa', 'shannon']

MODEL_STATS_CSV = 'results/model_media/model_stats.csv'
MEDIA_SAVE_DIR = 'results/model_media/'
MODEL_SAVE_DIR = 'results/model_checkpoints/ours'