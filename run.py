#%% testing download of sequences from FASTA files
import requests as r
from src.data_processing.pdbbind import get_prot_seq, save_prot_seq, excel_to_csv, prep_save_data
import pandas as pd
import os, re


# %%
import openbabel as ob

conv = ob.OBConversion()
conv.SetInAndOutFormats("smi", "pdbqt")
conv.AddOption("gen3D", conv.GENOPTIONS)

mol = ob.OBMol()
conv.ReadString(mol, "C1=CC=CS1")
# %%
from src.docking_helpers.utils import download_PDBs, download_SDFs

# %% ex:
download_SDFs(['VWW'], save_path='data/structures/ligands/')
# %%
