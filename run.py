#%%
import os, random, itertools, math, pickle, json, time
import config


from tqdm import tqdm
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.models.prior_work import DGraphDTA, DGraphDTAImproved
from src.models.mut_dta import EsmDTA
from src.data_processing import PDBbindDataset, DavisKibaDataset, train_val_test_split
from src.models import train, test, CheckpointSaver
from src.models.utils import print_device_info
from src.data_analysis import get_metrics
from src.feature_extraction.protein import multi_save_cmaps

#%% 
pdb_codes = os.listdir('../data/v2020-other-PL/')
# filter out readme and index folders
pdb_codes = [p for p in pdb_codes if p != 'index' and p != 'readme']

assert len(pdb_codes) > 0, 'Too few PDBCodes, need at least 1...'

# creating contact maps:
seqs = multi_save_cmaps(pdb_codes,
                  pdb_p=lambda x: f'../data/v2020-other-PL/{x}/{x}_protein.pdb',
                  cmap_p=lambda x: f'../data/v2020-other-PL/{x}/{x}.npy',
                  processes=2)
        
print('DONE!')
exit()
#%%
dataset = PDBbindDataset(save_root='../data/PDBbindDataset/nomsa',
                 data_root='../data/v2020-other-PL/',
                 aln_dir=None, 
                 cmap_threshold=8.0,
                 shannon=False
                 )

import submitit

import torch
from transformers import AutoTokenizer, EsmModel


from src.models.utils import print_device_info

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print_device_info(device)

#%% Testing esm model 
# based on https://github.com/facebookresearch/esm#main-models-you-should-use-
# I should be using ESM-MSA-1b (esm1b_t33_650M_UR50S)
# from https://github.com/facebookresearch/esm/tree/main#pre-trained-models-
# Luke used esm2_t6_8M_UR50D for his experiments

# https://huggingface.co/facebook/esm2_t33_650M_UR50D is <10GB
# https://huggingface.co/facebook/esm2_t36_3B_UR50D is 11GB
df = pd.read_csv('../data/DavisKibaDataset/davis_msa/processed/XY.csv', index_col=0)
prot_seqs = list(df['prot_seq'].unique())

# %%
def run_esm(prot_seqs):
    start_t = time.time()
    # this will raise a warning since lm head is missing but that is okay since we are not using it:
    device_ids = [torch.device(f'cuda:{x}') for x in range(torch.cuda.device_count())]
    print(device_ids)
    
    esm_tok = AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
    esm_mdl = EsmModel.from_pretrained('facebook/esm2_t6_8M_UR50D').to(device_ids[0])
    print('Time to load:', time.time()-start_t)
    start_t = time.time()
    
    tok = esm_tok(prot_seqs, return_tensors='pt', padding=True)
    print('Time to tok:', time.time()-start_t)
    start_t = time.time()
    
    out = torch.nn.DataParallel(esm_mdl, device_ids=device_ids)(**tok)
    print('Time to inf:', time.time()-start_t)
    print(out.last_hidden_state.shape)
    
    return out.last_hidden_state.to(torch.device('cpu'))

#%%
# the AutoExecutor class is your interface for submitting function to a cluster or run them locally.
# The specified folder is used to dump job information, logs and result when finished
# %j is replaced by the job id at runtime
num_gpus = 3
jobs = []
for num_gpus in range(1,4):
    print(num_gpus)
    log_folder = f"log_test/Test_gpu_{num_gpus}"
    executor = submitit.AutoExecutor(folder=log_folder)

    executor.update_parameters(timeout_min=30, 
                            slurm_partition="gpu",
                            slurm_account='kumargroup_gpu',
                            slurm_mem='10G',
                            slurm_gres=f'gpu:v100:{num_gpus}',
                            slurm_job_name=f'Test_gpu_{num_gpus}')
    
    # The submission interface is identical to concurrent.futures.Executor
    job = executor.submit(run_esm, prot_seqs[:15])  # will compute add(5, 7)
    print('state:', job.state)
    print('id:', job.job_id)  # ID of your job
    jobs.append(job)
    
#%%
output = jobs[0].result()  # waits for the submitted function to complete and returns its output

# %%
for j in jobs:
    j.cancel()

# %%
