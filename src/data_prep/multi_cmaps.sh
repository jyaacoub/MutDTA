#!/bin/bash
#SBATCH -t 120
#SBATCH -o ./%j.out
#SBATCH --job-name=create_cmap_CA

#SBATCH -p all
#SBATCH --mem=500M
#SBATCH --cpus-per-task=2

#SBATCH --array=2-14
#SBATCH --ntasks=1

#this will create contact maps for all proteins in PDBbind

source /cluster/home/t122995uhn/projects/MutDTA/.venv/bin/activate

# each process does 1000 cmaps

python - << EOF
import os
from src.data_prep.feature_extraction.protein import create_save_cmaps
from src.data_prep import PDBbindProcessor, Downloader
import pandas as pd

pdb_path = '/cluster/home/t122995uhn/projects/data/v2020-other-PL/'

pdb_codes = os.listdir(pdb_path)
# filter out readme and index folders
pdb_codes = [p for p in pdb_codes if p != 'index' and p != 'readme'][::-1][1000*${SLURM_ARRAY_TASK_ID}:1000*(${SLURM_ARRAY_TASK_ID}+1)]

print('[1000*${SLURM_ARRAY_TASK_ID}:1000*(${SLURM_ARRAY_TASK_ID}+1)]')

exit
#%%
create_save_cmaps(pdb_codes,
                  pdb_p=lambda x: f'{pdb_path}/{x}/{x}_protein.pdb',
                  cmap_p=lambda x: f'{pdb_path}/{x}/{x}_cmap_CB.npy')

EOF