# # %%
# from src.utils.loader import Loader
# from src.utils import config as cfg

# model = Loader.init_test_model()

# loaders = Loader.load_DataLoaders(data='davis', pro_feature='nomsa', edge_opt='binary', path=cfg.DATA_ROOT, 
#                                         ligand_feature=None, ligand_edge=None,
#                                         batch_train=1,
#                                         datasets=['test'])
# # %%
# for b in loaders['test']: break
# # %%
# model(b['protein'], b['ligand'])

# %%
from typing import Any
from src.utils.af_clust import AF_Clust
dir_p = f"/cluster/home/t122995uhn/projects/colabfold"

# # %% EGFR
# pid = "EGFR"
# msa = f"{dir_p}/in_a3m_misc/{pid}/{pid}.a3m"
# af = AF_Clust(keyword="test-"+pid, input_msa=msa, output_dir=dir_p+ f"/in_a3m/{pid}/")

#%% davis 
print("DAVIS sample")
pid = 'ABL1(E255K)'
msa = f"{dir_p}/davis_a3m/part1/{pid}.a3m"
af = AF_Clust(keyword="test-"+pid, input_msa=msa, output_dir=f"{dir_p}/davis_a3m/test/")

#%% kiba
print("KIBA sample")
pid = 'O14920'
msa = f"{dir_p}/kiba_a3m/part1/{pid}.a3m"
af = AF_Clust(keyword="test-"+pid, input_msa=msa, output_dir=f"{dir_p}/davis_a3m/test/")


# %% PDBBind
print("PDBBIND sample")
pid = '1a1e'
msa = f"{dir_p}/pdbbind_a3m/{pid}.msa.a3m"
af = AF_Clust(keyword="test-"+pid, input_msa=msa, output_dir=f"{dir_p}/test_af_clust/")

# %%



# %% RUN MSA:
from src.utils.seq_alignment import MSARunner
from tqdm import tqdm
import pandas as pd
import os
csv = '/cluster/home/t122995uhn/projects/data/PlatinumDataset/nomsa_binary/full/XY.csv'
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

########################## Get job partition
num_arrays = 100
array_idx = 0
partition_size = len(unique_df) / num_arrays
start, end = int(array_idx*partition_size), int((array_idx+1)*partition_size)

unique_df = unique_df[start:end]

raw_dir = '/cluster/home/t122995uhn/projects/data/PlatinumDataset/raw'

#################################### create fastas
fa_dir = os.path.join(raw_dir, 'platinum_fa')
os.makedirs(fa_dir, exist_ok=True)
MSARunner.csv_to_fasta_dir(csv_or_df=unique_df, out_dir=fa_dir)

##################################### Run hhblits
aln_dir = os.path.join(raw_dir, 'platinum_aln')
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
        
        
        
        
        
        
        
        
# %%
from src.utils.enum import CustomStrEnum

ModelOptions = CustomStrEnum('ModelOptions', ["DG"])

print(ModelOptions('DG'))
# class ModelOptions(BaseEnum):
#     DG = 'DG'
#     DGI = 'DGI'
#     ED = 'ED'
#     EDA = 'EDA'
#     EDI = 'EDI'
#     EDAI = 'EDAI'
#     EAT = 'EAT'
#     CD = 'CD'
#     CED = 'CED'
#     SPD = 'SPD'
    


input_opt = 'DG' # handles error checking 
ModelOptions(input_opt) == ModelOptions.DG
# %%
