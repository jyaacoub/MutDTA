# %%
from src.data_prep.datasets import BaseDataset
import pandas as pd
DATA_ROOT = '/cluster/home/t122995uhn/projects/data/v2020-other-PL/'
csv_fp = '/cluster/home/t122995uhn/projects/data/PDBbindDataset/nomsa_ring3_original_binary/full/XY.csv'
df = pd.read_csv(csv_fp, index_col=0)
df_unique = BaseDataset.get_unique_prots(df)

# %%
import os
from glob import glob
def pdb_p(code):
    return os.path.join(DATA_ROOT, code, f'{code}_protein.pdb')

msa_p = lambda c: f'/cluster/home/t122995uhn/projects/data/pdbbind/PDBbind_a3m/{c}.msa.a3m'

from src.utils.residue import Chain
from tqdm import tqdm
import logging
logging.getLogger().setLevel(logging.INFO)
mismatched = {}
missing = {}
safe = {}

for code, (pid, seq) in tqdm(df_unique[['prot_id', 'prot_seq']].iterrows(),
                             total=len(df_unique)):
    fp = msa_p(code)
    # Check that a3m is present
    if not os.path.exists(fp): 
        missing[code] = fp
        logging.debug(f'{code} does not have a {fp}.')
        continue
        
    # Check that sequence matches
    with open(fp, 'r') as f:
        msa_seq = None
        for i in range(2): # target will be the second seq
            msa_seq = f.readline()
        msa_seq = msa_seq.strip()
    
    if seq != msa_seq:
        mismatched[code] = fp
        logging.debug(f'{code} does not match')
        
    safe[code] = fp

logging.info(f'{len(mismatched)} mismatches')
logging.info(f'{len(missing)} missing')
logging.info(f'{len(safe)} safe')

#%% cp from src to dst
from shutil import copyfile
msa_src = lambda c: f'/cluster/home/t122995uhn/projects/data/pdbbind/PDBbind_a3m/{c}.msa.a3m'
msa_dst = lambda c: f'/cluster/home/t122995uhn/projects/data/pdbbind/a3m_unique_prot/all/{c}.a3m'

for code, src in tqdm(safe.items()):
    copyfile(src, msa_dst(code))
