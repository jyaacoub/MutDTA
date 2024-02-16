# - [ ] See issue with af_confs mismatched (likely [when we renamed the af_confs](https://github.com/jyaacoub/MutDTA/issues/80#issuecomment-1944250594) we accidently overwrote previous pids)
# 	- To solve this we need to regenerate the proper af_confs..
# 	- [ ] Check which are mis matched from the XY.csv and re generate those as well as any missing af_confs

# %% Checking XY.csv and regenerate mismatches
from src.data_prep.datasets import PDBbindProcessor, BaseDataset
import pandas as pd
DATA_ROOT = '/cluster/home/t122995uhn/projects/data/v2020-other-PL/' 
AF_CONF_DIR = '/cluster/home/t122995uhn/projects/data/pdbbind/PDBbind_afConf/'
csv_fp = '/cluster/home/t122995uhn/projects/data/PDBbindDataset/nomsa_ring3_original_binary/full/XY.csv'
df = pd.read_csv(csv_fp, index_col=0)
df_unique = BaseDataset.get_unique_prots(df)

# %%
import os
from glob import glob
def pdb_p(code):
    return os.path.join(DATA_ROOT, code, f'{code}_protein.pdb')

def af_conf_files(pid) -> list[str]:
    return glob(f'{AF_CONF_DIR}/{pid}_model_*.pdb')

# %%
from src.utils.residue import Chain
from tqdm import tqdm
import logging
logging.getLogger().setLevel(logging.INFO)
mismatched = {}
missing = {}
safe = {}
for code, (pid, seq) in tqdm(df_unique[['prot_id', 'prot_seq']].iterrows(),
                             total=len(df_unique)):
    afs = af_conf_files(pid)
    
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
    
# %% Checking that safe is actually safe
# the df_unique should match up pid with correct pdb

for code, pid in safe.items():
    s0 = Chain(pdb_p(code)).sequence
    s1 = Chain(af_conf_files(pid)[0]).sequence
    
    assert s0 == s1, f"mismatch for af conf and pdb on ({code}, {pid})"


# %%
# saving to output
with open('mismatched_codes.csv', 'w') as f:
    f.write('code,pid,seq,af_seq\n')
    for code, (pid, seq, af_seq) in mismatched.items():
        f.write(f'{code},{pid},{seq},{af_seq}\n')
        
with open('missing_codes.csv', 'w') as f:
    f.write('code,pid\n')
    for code, pid in missing.items():
        f.write(f'{code},{pid}\n')
        
with open('safe_codes.csv', 'w') as f:
    f.write('code,pid\n')
    for code, pid in safe.items():
        f.write(f'{code},{pid}\n')
# %%
import pandas as pd
df_mm = pd.read_csv('mismatched_codes.csv', index_col=0)
df_m = pd.read_csv('missing_codes.csv', index_col=0)
df_s = pd.read_csv('safe_codes.csv', index_col=0)

# %%
import os
from tqdm import tqdm
msa = lambda c: f'/cluster/projects/kumargroup/msa/output/{c}.msa.a3m'

no_msa = []
for code in df_mm.index:
    if not os.path.exists(msa(code)):
        no_msa.append(code)

# %%
