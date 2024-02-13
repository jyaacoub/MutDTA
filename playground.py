#%%###################
# INIT PDBbind dataset with ring3 features:
from src.data_prep.init_dataset import create_datasets
from src import config as cfg
import logging
logging.getLogger().setLevel(logging.ERROR)

create_datasets([cfg.DATA_OPT.PDBbind], [cfg.PRO_FEAT_OPT.nomsa], 
                [cfg.PRO_EDGE_OPT.ring3], k_folds=5, overwrite=True)

# %%

import pandas as pd
from pathlib import Path

xy = f'{Path.home()}/projects/data/DavisKibaDataset/davis/nomsa_af2/full/XY.csv'
df = pd.read_csv(xy, index_col=0)
#%%
from src.utils.loader import Loader
from src import config as cfg

ds = Loader.load_dataset(cfg.DATA_OPT.PDBbind, cfg.PRO_FEAT_OPT.nomsa, cfg.PRO_EDGE_OPT.ring3,
                         subset='full')



#%%
from tqdm import tqdm
import re
from glob import glob

missing_conf = set()
unique_df = ds.get_unique_prots(ds.df)
for code in tqdm(unique_df.index,
        desc='Filtering out proteins with missing PDB files for multiple confirmations',
        total=len(unique_df)):
            # removing () from string since file names cannot include them and localcolabfold replaces them with _
    af_confs = ds.af_conf_files(code)
    
    # need at least 2 confimations...
    if len(af_confs) <= 1:
        missing_conf.add(code)

#%%
































#%%###############################################
################# HHBLITS RUN SCRIPT #############
##################################################

from src.utils.seq_alignment import MSARunner
from tqdm import tqdm
import pandas as pd
import os
data_name = 'davis'
data_dir = f'/cluster/home/t122995uhn/projects/data/DavisKibaDataset/{data_name}/'

csv = f'{data_dir}/nomsa_binary_original_binary/full/XY.csv'
df = pd.read_csv(csv, index_col=0)

#################### Get unique proteins:
# sorting by sequence length before dropping so that we keep the longest protein sequence instead of just the first.
df['seq_len'] = df['prot_seq'].str.len()
df = df.sort_values(by='seq_len', ascending=False)

# create new numerated index col for ensuring the first unique uniprotID is fetched properly 
df.reset_index(drop=False, inplace=True)
unique_pro = df[['prot_id']].drop_duplicates(keep='first')

# reverting index to code-based index
df.set_index('code', inplace=True)
unique_df = df.iloc[unique_pro.index]

#%%########################## Get job partition
num_arrays = 100
array_idx = 0 #${SLURM_ARRAY_TASK_ID}
partition_size = len(unique_df) / num_arrays
start, end = int(array_idx*partition_size), int((array_idx+1)*partition_size)

unique_df = unique_df[start:end]

raw_dir = f'{data_dir}/raw'

#%%#################################### create fastas
fa_dir = os.path.join(raw_dir, f'{data_name}_fa')
os.makedirs(fa_dir, exist_ok=True)
MSARunner.csv_to_fasta_dir(csv_or_df=unique_df, out_dir=fa_dir)

#%%##################################### Run hhblits
aln_dir = os.path.join(raw_dir, f'{data_name}_aln')
os.makedirs(aln_dir, exist_ok=True)

# finally running
for _, (prot_id, pro_seq) in tqdm(
                unique_df[['prot_id', 'prot_seq']].iterrows(), 
                desc='Running hhblits',
                total=len(unique_df)):
    in_fp = os.path.join(fa_dir, f"{prot_id}.fasta")
    out_fp = os.path.join(aln_dir, f"{prot_id}.a3m")
    
    if not os.path.isfile(out_fp):
        MSARunner.hhblits(in_fp, out_fp)