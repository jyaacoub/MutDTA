#%%
import os
from src.feature_extraction.protein import create_save_cmaps
from src.data_processing import PDBbindProcessor, Downloader
import pandas as pd

pdb_path = '/home/jyaacoub/projects/data/v2020-other-PL'

pdb_codes = os.listdir(pdb_path)
# filter out readme and index folders
pdb_codes = [p for p in pdb_codes if p != 'index' and p != 'readme']
#%%
# create_save_cmaps(pdb_codes,
#                   pdb_p=lambda x: f'{pdb_path}/{x}/{x}_protein.pdb',
#                   cmap_p=lambda x: f'{pdb_path}/{x}/{x}_cmap_CB.npy')



#%% v2020-other-PL/index/INDEX_general_PL_data.2020  contains all we need for binding data
# includes pkd values
# # ==============================================================================
# # List of protein-ligand complexes with known binding data in PDBbind v.2020
# # 19443 protein-ligand complexes in total, sorted by binding data
# # Latest update: July 2021
# # PDB code, resolution, release year, -logKd/Ki, Kd/Ki, reference, ligand name
# # ==============================================================================
# 3zzf  2.20  2012   0.40  Ki=400mM      // 3zzf.pdf (NLG)
# 3gww  2.46  2009   0.45  IC50=355mM    // 3gwu.pdf (SFX)

index_file = '../data/v2020-other-PL/index/INDEX_general_PL_data.2020'
df_binding = PDBbindProcessor.get_binding_data(index_file)
df_binding.to_csv('./data/PDBbind/general_PL_data.csv')

#%%
dict_smi = PDBbindProcessor.get_SMILE(df_binding.index,
                           dir=lambda x: f'{pdb_path}/{x}/{x}_ligand.sdf',)

df_smi = pd.DataFrame.from_dict(dict_smi, orient='index', columns=['SMILE'])
df_smi.index.name = 'PDBCode'
print(df_smi.head())

# 1979 missing unique lig_names out of 19443 
print(f'{len(df_smi[df_smi.SMILE.isna()])} missing SMILEs out of {len(df_smi)} = {len(df_smi[df_smi.SMILE.isna()])/len(df_smi)*100:.2f}%')

missing = df_smi[df_smi.SMILE.isna()].merge(df_binding, on='PDBCode')
#-> 3178 incomplete pdb codes
print(f'{len(missing.lig_name.unique())} unique lig_names missing out of {len(df_smi)} = {len(missing.lig_name.unique())/len(df_smi)*100:.2f}%')
#%%
# getting missing SMILEs from Cactus
# some dont appear to be in Cactus...
downloaded_smi = Downloader.get_SMILE(list(df_smi[df_smi.SMILE.isna()].index))

#%% get remaining SMILEs from PDB
import requests as r
from urllib.parse import quote
url = lambda l: f'https://www.rcsb.org/ligand/{quote(l)}'
for lig in df_binding.lig_name.unique():
    res = r.get(url(lig))
    if res.status_code > 400: print(lig)

#%%

# RMSD code for comparing two PDB files
# #%%
# import os

# from Bio.PDB import PDBParser, PDBIO, Select
# from src.data_processing import Downloader


# def is_het(residue):
#     res = residue.id[0]
#     return res != " " and res != "W"


# class ResidueSelect(Select):
#     def __init__(self, chain, residue):
#         self.chain = chain
#         self.residue = residue

#     def accept_chain(self, chain):
#         return chain.id == self.chain.id

#     def accept_residue(self, residue):
#         """ Recognition of heteroatoms - Remove water molecules """
#         return residue == self.residue and is_het(residue)


# def extract_ligands(pdb_code, file):
#     """ Extraction of the heteroatoms of .pdb files """ #NOTE: from PDBbind docs (prt3) -> peptides shorter than 20 residues are cosidered to be ligands
#     pdb = PDBParser().get_structure(pdb_code, file)
#     io = PDBIO()
#     io.set_structure(pdb)
#     i=0
#     for model in pdb:
#         for chain in model:
#             for residue in chain:
#                 if not is_het(residue):
#                     continue
#                 print(f"saving {chain} {residue}")
#                 io.save(f"lig_{pdb_code}_{i}.pdb", ResidueSelect(chain, residue))
#                 i+=1

# #%% Main
# code = '1a1e'
# file = Downloader.get_file_obj(code)

# extract_ligands(code, file)
# file.close()

# with open(f'{code}.pdb', 'w') as f:
#     f.write(Downloader.get_file_obj(code).read())
# %%
