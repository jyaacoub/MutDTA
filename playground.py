###################################################
# %% Missing PDBbind confirmations
# full csv list is at data/PDBbindDataset/nomsa_ring3_original_binary/full/XY.csv
from src.data_prep.init_dataset import create_datasets
from src import config as cfg
import logging
logging.getLogger().setLevel(logging.DEBUG)

create_datasets([cfg.DATA_OPT.PDBbind], 
                [cfg.PRO_FEAT_OPT.nomsa], 
                [cfg.PRO_EDGE_OPT.ring3],
                k_folds=5,
                overwrite=True)

# %%
from src.data_prep.processors import PDBbindProcessor
import os, re
from glob import glob
from src.data_prep.feature_extraction.protein import multi_save_cmaps, get_sequences
import pandas as pd

DATA_ROOT = '/cluster/home/t122995uhn/projects/data//v2020-other-PL/'
DATA_P = f'{DATA_ROOT}/index/INDEX_general_PL_data.2020'
NAME_P = f'{DATA_ROOT}/index/INDEX_general_PL_name.2020'
OVERWRITE = False

def pdb_p(code):
    return os.path.join(DATA_ROOT, code, f'{code}_protein.pdb')

def get_unique_prots(df, verbose=True) -> pd.DataFrame:
    """Gets the unique proteins from a dataframe by their protein id"""
    # sorting by sequence length before dropping so that we keep the longest protein sequence instead of just the first.
    df['seq_len'] = df['prot_seq'].str.len()
    df = df.sort_values(by='seq_len', ascending=False)
    
    # create new numerated index col for ensuring the first unique uniprotID is fetched properly 
    df.reset_index(drop=False, inplace=True)
    unique_pro = df[['prot_id']].drop_duplicates(keep='first')
    # reverting index to code-based index
    df.set_index('code', inplace=True)
    unique_df = df.iloc[unique_pro.index]
    
    if verbose: logging.info(f'{len(unique_df)} unique proteins')
    return unique_df


# Get prot ids data:n 
df_pid = PDBbindProcessor.get_name_data(NAME_P) # _name.2020
df_pid.drop(columns=['release_year','prot_name'], inplace=True)
# contains col: prot_id
# some might not have prot_ids available so we need to use PDBcode as id instead
missing_pid = df_pid.prot_id == '------'
df_pid[missing_pid] = df_pid[missing_pid].assign(prot_id = df_pid[missing_pid].index)

# Get binding data:
df_binding = PDBbindProcessor.get_binding_data(DATA_P) # _data.2020
df_binding.drop(columns=['resolution', 'release_year', 'lig_name'], inplace=True)
pdb_codes = df_binding.index # pdbcodes

############# validating codes #############
valid_codes = [c for c in pdb_codes if os.path.isfile(pdb_p(c))]
    
pdb_codes = valid_codes
assert len(pdb_codes) > 0, 'Too few PDBCodes, need at least 1...'

# merge with binding data to get unique protids that are validated:
df = df_pid.merge(df_binding, on='PDBCode') # pids + binding
def cmap_p(pid, df=df):
    # cmap is saved in seperate directory under /v2020-other-PL/cmaps/
    # file names are unique protein ids...
    # check to make sure arg is a pid
    if df is not None and pid in df.index:
        pid = df.loc[pid]['prot_id']
    return os.path.join(DATA_ROOT, 'cmaps', f'{pid}.npy')

#%% ############ Getting protein seq: #############
df_seqs = get_sequences(pdb_codes, pdb_p)

#%% ############ Getting contact maps: #############
# Getting unique proteins to create list of tuples -> [(code, pid)]
# WARNING: CANNOT GET UNIQUE PROTS AT START SINCE WE DO NOT KNOW SEQUENCES
# Seperate out getting the sequences from getting the cmaps!
df_unique = get_unique_prots(df, verbose=True)
os.makedirs(os.path.dirname(cmap_p('')), exist_ok=True)
seqs = multi_save_cmaps([(code, pid) for code, pid in df_unique.items()],
                    pdb_p=pdb_p,
                    cmap_p=cmap_p,
                    overwrite=OVERWRITE)

assert len(seqs) == len(df_unique), 'Some codes failed to create contact maps'

df_seq = pd.DataFrame.from_dict(seqs, orient='index', columns=['prot_seq'])
df_seq.index.name = 'prot_id'

############## Get ligand info #############
# Extracting SMILE strings:
dict_smi = PDBbindProcessor.get_SMILE(pdb_codes,
                                        dir=lambda x: f'{DATA_ROOT}/{x}/{x}_ligand.sdf')
df_smi = pd.DataFrame.from_dict(dict_smi, orient='index', columns=['SMILE'])
df_smi.index.name = 'PDBCode'

df_smi = df_smi[df_smi.SMILE.notna()]
num_missing = len(pdb_codes) - len(df_smi)
if  num_missing > 0:
    print(f'\t{num_missing} ligands failed to get SMILEs')
    pdb_codes = list(df_smi.index)

#%% ############ FINAL MERGES #############
df_new = df.merge(df_smi, on='PDBCode') # + smiles
idx = df_new.index
#%%
df_new = df_new.merge(df_seq, on='prot_id', how='left') # + prot_seq

# maintaining same index with pdbcodes
df_new.set_index(idx, inplace=True)
# %%







# %%
from src.utils.residue import Chain
fp = "/cluster/home/t122995uhn/projects/data/v2020-other-PL/1i6v/1i6v_protein.pdb"

c = Chain(fp)

# %%
