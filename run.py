#%% testing download of sequences from FASTA files
import requests as r
from src.data_processing.pdbbind import get_prot_seq, save_prot_seq, excel_to_csv, prep_save_data
import pandas as pd
import os, re


# %%
# prep_save_data(csv_path='data/PDBbind/raw/P-L_refined_set_all.csv',
#                prot_seq_csv='data/prot_seq.csv',
#                save_path='data/PDBbind/kd_only', Kd_only=True)
# %%
from src.docking_helpers.utils import download_PDBs, download_SDFs

# %% ex:
download_SDFs(['VMW'], save_path='data/structures/ligands/')