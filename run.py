#%%
import os
from src.feature_extraction.protein import create_save_cmaps, get_sequence
from src.data_processing import PDBbindProcessor, Downloader
from tqdm import tqdm
import pandas as pd

pdb_path = '../data/v2020-other-PL/'

pdb_codes = os.listdir(pdb_path)
# filter out readme and index folders
pdb_codes = [p for p in pdb_codes if p != 'index' and p != 'readme']
#%%
seqs = create_save_cmaps(pdb_codes,
                  pdb_p=lambda x: f'{pdb_path}/{x}/{x}_protein.pdb',
                  cmap_p=lambda x: f'{pdb_path}/{x}/{x}_cmap_CB_lone.npy')

# exit()

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

index_file = f'{pdb_path}/index/INDEX_general_PL_data.2020'
df_binding = PDBbindProcessor.get_binding_data(index_file)
df_binding.to_csv('./data/PDBbind/general_PL_data.csv')

#%%
dict_smi = PDBbindProcessor.get_SMILE(df_binding.index,
                           dir=lambda x: f'{pdb_path}/{x}/{x}_ligand.sdf') #WARNING: some fail with mol2 and some fail with sdf...

df_smi = pd.DataFrame.from_dict(dict_smi, orient='index', columns=['SMILE'])
df_smi.index.name = 'PDBCode'
print(df_smi.head())

# better way to do this:
# code='4obv'
# m = Chem.MolFromMol2File(f'{pdb_path}/{code}/{code}_ligand.mol2')
# Chem.MolToSmiles(m, isomericSmiles=False) # setting this to False means each SMILE is unique (canonical)


#%%-> 3178 incomplete pdb codes
print(f'{len(df_smi[df_smi.SMILE.isna()])} total missing SMILEs out of \t' +
      f'{len(df_smi)} = {len(df_smi[df_smi.SMILE.isna()])/len(df_smi)*100:.2f}%')

missing = df_smi[df_smi.SMILE.isna()].merge(df_binding, on='PDBCode')
num_missing_ln = len(missing.lig_name.unique())
total_ln = len(df_binding.lig_name.unique())
# -> 1979 missing unique lig_names 
print(f'{num_missing_ln} unique lig_names missing out of \t'+
      f'{total_ln} = {num_missing_ln/total_ln*100:.2f}%')

#%% getting prot seqs

# %% Saving with SMILEs
df = df_smi[df_smi.SMILE.notna()].merge(df_binding, on='PDBCode')
# df.drop(columns=['lig_name'], inplace=True)


# %% getting prot seqs
seqs ={}
for code in tqdm(df.index):
      seq, _ = get_sequence(f'{pdb_path}/{code}/{code}_protein.pdb')
      seqs[code] = seq
   
#%%   
df_seq = pd.DataFrame.from_dict(seqs, orient='index', columns=['prot_seq'])
df_seq.index.name = 'PDBCode'

#%% merging seqs with df and saving
#change column name

df = df.merge(df_seq, on='PDBCode')
# mv SMILE to end
df = df[[c for c in df.columns if c not in ['prot_seq', 'SMILE']] + ['SMILE', 'prot_seq']]
df.to_csv('./data/PDBbind/general_XY.csv')

#%%
# getting missing SMILEs from Cactus
# # some dont appear to be in Cactus...
# downloaded_smi = Downloader.get_SMILE(list(missing.lig_name.unique()))

# #%% get remaining SMILEs from PDB
# import requests as r
# from urllib.parse import quote
# url = lambda l: f'https://www.rcsb.org/ligand/{quote(l)}'
# for lig in df_binding.lig_name.unique():
#     res = r.get(url(lig))
#     if res.status_code > 400: print(lig)

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
# cols are: run,cindex,pearson,spearman,mse,mae,rmse
import matplotlib.pyplot as plt
df_res = pd.read_csv('results/model_media/DGraphDTA_stats.csv')[6:]
df_res.sort_values(by='run', inplace=True)

df_res.loc[-1] = ['vina', 0.68,0.508,0.520,17.812,3.427,4.220] # hard coded vina results

for col in df_res.columns[1:]:
    plt.figure()
    bars = plt.bar(df_res['run'],df_res[col])
    bars[0].set_color('green')
    bars[2].set_color('black')
    bars[-1].set_color('red')
    plt.title(col)
    plt.xlabel('run')
    plt.xticks(rotation=30)
    # plt.ylim((0.2, 0.8))
    plt.show()
# %%
