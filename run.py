#%% testing download of sequences from FASTA files
import requests as r
from data_processing.format import get_prot_seq, save_prot_seq
import pandas as pd

#%% protID -> seq
protID = "P55055"
url = lambda x: f'https://rest.uniprot.org/uniprotkb/{x}.fasta'
# d = get_prot_seq([protID])
# print(d)
# save_prot_seq(d, overwrite=True)

# %% xlsx file to csv
df = pd.read_excel('data/P-L_refined_set_all.xlsx', header=1, index_col=0)
df[['PDB code', 'Affinity Data', 'Release Year',
    'Protein Name', 'Ligand Name', 
    'UniProt AC', 'Canonical SMILES']]
# %%
