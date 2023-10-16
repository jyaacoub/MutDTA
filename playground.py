#%%
from tqdm import tqdm
import pandas as pd
import numpy as np
import os, re

from glob import glob
from src.data_processing.datasets import BaseDataset
from src.feature_extraction.protein_edges import get_target_edge_weights

data = 'davis'
data_dir = '/cluster/home/t122995uhn/projects/data/'
csv = f'{data_dir}/DavisKibaDataset/davis/nomsa_af2-anm/full/XY.csv'
raw_dir = f'{data_dir}/{data}/'
af_conf_dir = f'../colabfold/{data}_af2_out/'

def af_conf_files(code) -> list[str]:
    code = re.sub(r'[()]', '_', code)
    return glob(f'{af_conf_dir}/out?/{code}_unrelaxed_rank_*.pdb')

def pdb_p(code, safe=True):
    code = re.sub(r'[()]', '_', code)
    # davis and kiba dont have their own structures so this must be made using 
    # af or some other method beforehand.
    file = glob(os.path.join(af_conf_dir, f'highQ/{code}_unrelaxed_rank_001*.pdb'))
    # should only be one file
    assert not safe or len(file) == 1, f'Incorrect pdb pathing, {len(file)}# of structures for {code}.'
    return file[0] if len(file) >= 1 else None


df = pd.read_csv(csv, index_col=0)
#################### Get unique proteins:
unique_df = BaseDataset.get_unique_prots(df)

#%%######################### Get job partition
num_arrays = 140
array_idx = 0#${SLURM_ARRAY_TASK_ID}
partition_size = len(unique_df) / num_arrays
start, end = int(array_idx*partition_size), int((array_idx+1)*partition_size)

unique_df_part = unique_df[start:end]

print(unique_df_part.index)
##################################### Run hhblits
np_dir = os.path.join(raw_dir, 'edge_weights', 'af2-anm')
os.makedirs(np_dir, exist_ok=True)

# running
for code, (prot_id, pro_seq) in tqdm(
                unique_df_part[['prot_id', 'prot_seq']].iterrows(), 
                desc='Running edgw',
                total=len(unique_df_part)):
    out_fp = os.path.join(np_dir, f"{code}.npy")
    af_confs = af_conf_files(code)
    
    if not os.path.isfile(out_fp):
        pro_edge_weight = get_target_edge_weights(pdb_p(code), pro_seq, 
                                            edge_opt='af2-anm',
                                            n_modes=5, n_cpu=2,
                                            af_confs=af_confs)
        np.save(out_fp, pro_edge_weight)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%
from src.data_processing.init_dataset import create_datasets

create_datasets(data_opt=['davis'], # 'PDBbind' 'kiba' davis
                feat_opt=['nomsa'],    # nomsa 'msa' 'shannon']
                edge_opt=['af2-anm'], # for anm and af2 we need structures! (see colabfold-highQ)
                pro_overlap=False,
                #/home/jyaacoub/projects/data/
                #'/cluster/home/t122995uhn/projects/data/'
                data_root_dir='/cluster/home/t122995uhn/projects/data/')

# dataset = PlatinumDataset(
#                 save_root=f'../data/PlatinumDataset/',
#                 data_root=f'../data/PlatinumDataset/raw',
#                 aln_dir=None,
#                 cmap_threshold=8.0,
#                 feature_opt='nomsa',
#                 edge_opt='binary',
#                 )
# exit()

#%%
import pandas as pd
csv = '/cluster/home/t122995uhn/projects/data/PlatinumDataset/nomsa_binary/full/XY.csv'
pd.read_csv(csv, index_col=0)

#%%
import os
from glob import glob

data = 'davis'
fp = f'/cluster/home/t122995uhn/projects/data/DavisKibaDataset/{data}/nomsa_binary/full/XY.csv'
done_dir = f'/cluster/home/t122995uhn/projects/colabfold/{data}_af2_out/highQ/*.done.txt'

df = pd.read_csv(fp)

#%%
# replace '(' with '_' in prot_id col
df['pid_fix'] = df['prot_id'].str.replace(r'[()]', '_', regex=True)
unique_pids = df['pid_fix'].unique()
print(f'{len(unique_pids)} unique proteins')

def getID(fn):
    return os.path.splitext(os.path.basename(fn))[0].split('.done')[0]

done_IDs =  {getID(filename) for filename in glob(done_dir)}

remaining = [id for id in unique_pids if id not in done_IDs]

print(len(remaining), 'proteins remaining')

done_rows = df.iloc[df[~df['pid_fix'].isin(remaining)][['pid_fix']].drop_duplicates().index]
remaining_rows = df.iloc[df[df['pid_fix'].isin(remaining)][['pid_fix']].drop_duplicates().index]
remaining_rows['len'] = remaining_rows['prot_seq'].str.len()

max_prot_len_done = done_rows['prot_seq'].str.len().max()
max_prot_len = remaining_rows['len'].max()
print(f'{max_prot_len} is the max protein length for remaining structures.\n{max_prot_len_done} for done structures')
remaining_rows[['pid_fix', 'len']]

#%% get pids of those above 2149
files_to_remove = remaining_rows[remaining_rows['len'] > 2000]['prot_id']

for i in range(1,4):
    print('\nPART', i)
    p3p = f'/cluster/home/t122995uhn/projects/colabfold/{data}_a3m/part{i}'

    for f in files_to_remove:
        try:
            os.remove(os.path.join(p3p, f'{f}.a3m'))
            print(f'{f} REMOVED')
        except FileNotFoundError:
            pass
            # print(f'{f} doesnt exist')