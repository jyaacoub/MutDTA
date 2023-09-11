import os
# for huggingface models:
os.environ['TRANSFORMERS_CACHE'] = '../hf_models/'

from prody import confProDy
confProDy(verbosity='none') # stop printouts from prody


MODEL_OPT = ['DG', 'DGI', 'ED', 'EDA', 'EDI', 'EDAI', 'EAT']
EDGE_OPT = ['simple', 'binary', 'anm']
DATA_OPT = ['davis', 'kiba', 'PDBbind']
PRO_FEAT_OPT = ['nomsa', 'msa', 'shannon']