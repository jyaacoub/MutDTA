#!/bin/bash
#SBATCH -t 120
#SBATCH -o /cluster/projects/kumargroup/jean/slurm-outputs/models/%x-%j.out
#SBATCH --job-name=create_cmap_CA

#SBATCH -p all
#SBATCH --mem=500M
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

#this will create contact maps for all proteins in PDBbind

source /cluster/home/t122995uhn/projects/MutDTA/.venv/bin/activate

python - << EOF
from src.models.helpers.contact_map import get_contact
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os

PDBbind = '/cluster/projects/kumargroup/jean/data/refined-set'
path = lambda c: f'{PDBbind}/{c}/{c}_protein.pdb'
cmap_p = lambda c: f'{PDBbind}/{c}/{c}_contact_CA.npy'

for code in tqdm(os.listdir(PDBbind)):
    if os.path.isdir(os.path.join(PDBbind, code)) and code not in ["index", "readme"]:
        try:
            cmap = get_contact(path(code), CA_only=True)
        except Exception as e:
            print(code)
            raise e
        np.save(cmap_p(code), cmap)

EOF