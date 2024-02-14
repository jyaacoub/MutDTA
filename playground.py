# ###################################################
# # %% Missing PDBbind confirmations
# # full csv list is at data/PDBbindDataset/nomsa_ring3_original_binary/full/XY.csv
# from src.data_prep.init_dataset import create_datasets
# from src import config as cfg
# import logging
# logging.getLogger().setLevel(logging.DEBUG)

# create_datasets([cfg.DATA_OPT.PDBbind], 
#                 [cfg.PRO_FEAT_OPT.nomsa], 
#                 [cfg.PRO_EDGE_OPT.binary],
#                 k_folds=5)


# %%
from src.data_prep.processors import PDBbindProcessor
import os 
from src.data_prep.feature_extraction.protein import multi_save_cmaps
import pandas as pd

data_root = '/cluster/home/t122995uhn/projects/data/v2020-other-PL'

data_fp = f'{data_root}/index/INDEX_general_PL_data.2020'
name_fp = f'{data_root}/index/INDEX_general_PL_name.2020'
def pdb_p(code):
    return os.path.join(data_root, code, f'{code}_protein.pdb')

def cmap_p(code):
    # cmap is saved in seperate directory under /v2020-other-PL/cmaps/
    # file names are unique protein ids...
    return os.path.join(data_root, 'cmaps', f'{code}.npy')

df_pid = PDBbindProcessor.get_name_data(name_fp)
df_pid.drop(columns=['release_year','prot_name'], inplace=True)
missing_pid = df_pid.prot_id == '------'
df_pid[missing_pid] = df_pid[missing_pid].assign(prot_id=df_pid[missing_pid].index)

# Get binding data:
df_binding = PDBbindProcessor.get_binding_data(data_fp) # _data.2020
df_binding.drop(columns=['resolution', 'release_year', 'lig_name'], inplace=True)


pdb_codes = df_binding.index # pdbcodes

# merge with binding data to get unique protids that are validated:
df = df_pid.merge(df_binding, on='PDBCode') # pids + binding


df_unique = df['prot_id'].drop_duplicates() # index col is the PDB code
os.makedirs(os.path.dirname(cmap_p('')), exist_ok=True)
seqs = multi_save_cmaps([(code, pid) for code, pid in df_unique.items()],
                    pdb_p=pdb_p,
                    cmap_p=cmap_p,
                    overwrite=False)

assert len(seqs) == len(df_unique), 'Some codes failed to create contact maps'

df_seq = pd.DataFrame.from_dict(seqs, orient='index', columns=['prot_seq'])
df_seq.index.name = 'prot_id'

#%%############# Get ligand info #############
# Extracting SMILE strings:
dict_smi = PDBbindProcessor.get_SMILE(pdb_codes,
                                        dir=lambda x: f'{data_root}/{x}/{x}_ligand.sdf')
df_smi = pd.DataFrame.from_dict(dict_smi, orient='index', columns=['SMILE'])
df_smi.index.name = 'PDBCode'

df_smi = df_smi[df_smi.SMILE.notna()]
num_missing = len(pdb_codes) - len(df_smi)
if  num_missing > 0:
    print(f'\t{num_missing} ligands failed to get SMILEs')
    pdb_codes = list(df_smi.index)

#%%############# FINAL MERGES #############
df = df.merge(df_smi, on='PDBCode') # + smiles
df = df.merge(df_seq, on='prot_id') # + prot_seq


# %%
