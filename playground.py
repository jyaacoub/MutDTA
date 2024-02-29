# - [ ] See issue with af_confs mismatched (likely [when we renamed the af_confs](https://github.com/jyaacoub/MutDTA/issues/80#issuecomment-1944250594) we accidently overwrote previous pids)
# 	- To solve this we need to regenerate the proper af_confs..
# 	- [ ] Check which are mis matched from the XY.csv and re generate those as well as any missing af_confs
# %%
from src.utils.loader import Loader
from src import config as cfg

loaders = Loader.load_DataLoaders(cfg.DATA_OPT.PDBbind, 
                                  cfg.PRO_FEAT_OPT.nomsa,
                                  cfg.PRO_EDGE_OPT.ring3,
                                  training_fold=0,
                                  ligand_feature=cfg.LIG_FEAT_OPT.original,
                                  ligand_edge=cfg.LIG_EDGE_OPT.binary)

for b in loaders['train']: break
#%%
from src.models.ring_mod import Ring3DTA
m = Ring3DTA(num_features_pro=54)

out = m(b['protein'], b['ligand'])

# %%
from src.data_prep.init_dataset import create_datasets
from src import config as cfg
import logging
logging.getLogger().setLevel(logging.INFO)

create_datasets(
    [cfg.DATA_OPT.PDBbind],
    [cfg.PRO_FEAT_OPT.nomsa],
    [cfg.PRO_EDGE_OPT.ring3],
    k_folds=5,
    overwrite=True
)

# %% Checking XY.csv and regenerate mismatches
from src.data_prep.feature_extraction.protein import multi_get_sequences, multi_save_cmaps
from src.data_prep.processors import PDBbindProcessor
from src.data_prep.datasets import BaseDataset
from tqdm import tqdm
import pandas as pd
import os
import logging
logging.getLogger().setLevel(logging.DEBUG)

DATA_ROOT   = '/cluster/home/t122995uhn/projects/data/pdbbind/v2020-other-PL/' 
AF_CONF_DIR = '/cluster/home/t122995uhn/projects/data/pdbbind/PDBbind_afConf/'
csv_dir     = '/cluster/home/t122995uhn/projects/data/PDBbindDataset/nomsa_ring3_original_binary/full/'
csv_clean   = f'{csv_dir}/cleaned_XY.csv'
csv_fp      = f'{csv_dir}/XY.csv'
pdb_p = lambda code: os.path.join(DATA_ROOT, code, f'{code}_protein.pdb')
def cmap_p(pid):
    # cmap is saved in seperate directory under pdbbind/v2020-other-PL/cmaps/
    # file names are unique protein ids...
    # check to make sure arg is a pid
    if df is not None and pid in df.index:
        pid = df.loc[pid]['prot_id']
    return os.path.join(DATA_ROOT, 'cmaps', f'{pid}.npy')

#%%
# Get prot ids data:
df_pid = PDBbindProcessor.get_name_data(f'{DATA_ROOT}/index/INDEX_general_PL_name.2020')
df_pid.drop(columns=['release_year','prot_name'], inplace=True)
# contains col: prot_id
# some might not have prot_ids available so we need to use PDBcode as id instead
missing_pid = df_pid.prot_id == '------'
df_pid[missing_pid] = df_pid[missing_pid].assign(prot_id = df_pid[missing_pid].index)

# Get binding data:
df_binding = PDBbindProcessor.get_binding_data(f'{DATA_ROOT}/index/INDEX_general_PL_data.2020') # _data.2020
df_binding.drop(columns=['resolution', 'release_year', 'lig_name'], inplace=True)
pdb_codes = df_binding.index # pdbcodes

# merge with binding data to get unique protids that are validated:
df_pid_bind = df_pid.merge(df_binding, on='PDBCode')

############## validating codes #############
valid_codes = [c for c in pdb_codes if os.path.isfile(pdb_p(c))]
    
pdb_codes = valid_codes
assert len(pdb_codes) > 0, 'Too few PDBCodes, need at least 1...'
        
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
    
df = df_pid_bind.merge(df_smi, on='PDBCode')

############## Getting protein seq: #############
df_seqs = pd.DataFrame.from_dict(multi_get_sequences(pdb_codes, pdb_p), 
                                orient='index',
                                columns=['prot_seq'])
df_seqs.index.name = 'PDBCode'
df_seqs_pid = df_pid.merge(df_seqs, on='PDBCode')
# merge pids with sequence to get unique prots by seq length
df_unique = BaseDataset.get_unique_prots(df_seqs_pid)

# os.makedirs(os.path.dirname(cmap_p('')), exist_ok=True)
# %% TODO: use unique pids for cmaps...
seqs = multi_save_cmaps(
            [(code, pid) for code, pid in df_unique['prot_id'].items()],
            pdb_p=pdb_p,
            cmap_p=cmap_p,
            overwrite=True)


# %%
df = pd.read_csv(csv_fp, index_col=0)
df_clean = pd.read_csv(csv_clean, index_col=0)

df_unique = BaseDataset.get_unique_prots(df)
df_unique_clean = BaseDataset.get_unique_prots(df_clean)

mismatched = df_unique.index[:100] != df_unique_clean.index[:100]

print(df_unique.index[:100][mismatched])
print(df_unique_clean.index[:100][mismatched])

#%%
df = pd.read_csv(csv_clean, index_col=0)
dfr = df.reset_index(drop=False)
unique_pro = dfr[['prot_id']].drop_duplicates(keep='first')

dfr_i = dfr.set_index('code')
df_unique = dfr_i.iloc[unique_pro.index]
'3i3d' in df_unique.index

# mismatched 3i3d with B8LFD6 (slice at [500:625])
#                              529                                                        586
# PAVPKWSIKKWLSLPGETRPLILCEYAHA A GNSLGGFAKYWQAFRQYPRLQGGFVWDWVDQSLIKYDENGNPWSAYGGDFGDTPND R QFCMNGLVFADRTPHPALTEAKHQQQFFQFRLSGQTIE
# PAVPKWSIKKWLSLPGETRPLILCEYAHA M GNSLGGFAKYWQAFRQYPRLQGGFVWDWVDQSLIKYDENGNPWSAYGGDFGDTPND A QFCMNGLVFADRTPHPALTEAKHQQQFFQFRLSGQTIE
# %%
from src.utils.residue import Chain

ALL_LN_DIR = '/cluster/home/t122995uhn/projects/data/pdbbind/pdbbind_af2_out/all_ln/'
max_seq_len = 2400
        
# Filter proteins greater than max length
df_new = df[df['prot_seq'].str.len() <= max_seq_len]
pro_filtered = len(df) - len(df_new)
logging.info(f'Filtered out {pro_filtered} proteins greater than max length of {max_seq_len}')
df = df_new

missing_conf = set()
files = [f for f in os.listdir(ALL_LN_DIR) if f.endswith('.pdb')]
unique_df = BaseDataset.get_unique_prots(df)
# unique_df.sort_index()

#%%
for code, (pid, seq) in tqdm(unique_df[['prot_id', 'prot_seq']].iterrows(),
        desc='Filtering out proteins with missing PDB files for multiple confirmations',
        total=len(unique_df)):
    
    af_confs = [os.path.join(ALL_LN_DIR, f) for f in files \
                                if f.startswith(pid)]
    
    # need at least 2 confimations...
    if len(af_confs) <= 1:
        missing_conf.add(code)
        continue
    
    af_seq = Chain(af_confs[0]).sequence
    if seq != af_seq:
        logging.warning(f'Mismatched sequence for {pid}')
        missing_conf.add(code)
        continue


# %% create symbolic link of missing files into new folder
REMAIN_DIR = '/cluster/home/t122995uhn/projects/data/pdbbind/a3m_unique_prot/remainder/'
AF2_OUT_DIR = '/cluster/home/t122995uhn/projects/data/pdbbind/pdbbind_af2_out/'
ALL_LN_DIR = '/cluster/home/t122995uhn/projects/data/pdbbind/pdbbind_af2_out/all_ln/'

from src.utils.residue import Chain
import logging
logging.getLogger().setLevel(logging.INFO)
mismatched = {}
missing = {}
safe = {}
files = [f for f in os.listdir(ALL_LN_DIR) if f.endswith('.pdb')]
for code, (pid, seq) in tqdm(df_unique[['prot_id', 'prot_seq']].iterrows(),
                             total=len(df_unique)):
    filt = [f for f in files if f.startswith(pid)]
    afs = [os.path.join(ALL_LN_DIR, f) for f in filt]
    
    if len(afs) == 0:
        logging.debug(f'Missing confs for {pid}')
        missing[code] = pid
        continue
        
    af_seq = Chain(afs[0]).sequence
    if seq != af_seq:
        logging.debug(f'Mismatched sequence for {pid}')
        mismatched[code] = (pid, seq, af_seq)
        continue
    
    safe[code] = pid
    
# %%
