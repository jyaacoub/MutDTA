#%%
import os
from src.feature_extraction.protein import create_save_cmaps

pdb_path = '/home/jyaacoub/projects/data/v2020-other-PL'

pdb_codes = os.listdir(pdb_path)
# filter out readme and index folders
pdb_codes = [p for p in pdb_codes if p != 'index' and p != 'readme']
#%%
create_save_cmaps(pdb_codes,
                  pdb_p=lambda x: f'{pdb_path}/{x}/{x}_protein.pdb',
                  cmap_p=lambda x: f'{pdb_path}/{x}/{x}_cmap_CB.npy')













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
#     """ Extraction of the heteroatoms of .pdb files """
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
