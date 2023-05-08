#%% testing download of sequences from FASTA files
import requests as r
from data_processing.format import get_prot_seq, save_prot_seq, excel_to_csv, prep_save_data
import pandas as pd
import os, re

# %% xlsx file to csv
# excel_to_csv()

# %% protID -> seq
# protID = "P55055"
# url = lambda x: f'https://rest.uniprot.org/uniprotkb/{x}.fasta'
# d = get_prot_seq([protID])
# print(d)
# save_prot_seq(d, overwrite=True)
csv_path='data/raw/P-L_refined_set_all.csv'
prot_seq_csv='data/prot_seq.csv'

X, Y = prep_save_data(csv_path, prot_seq_csv, save_path='data/')

# %%
df_raw = pd.read_csv(csv_path)

# # filter out complexes with 2+ proteins or none at all
# df = df_raw[lambda x: x['protID'].fillna('').str.split().apply(lambda x: len(x)==1)]

# #%% getting protein sequences
# if not os.path.exists(prot_seq_csv):
#     seq = get_prot_seq(df['protID'])
#     # default save path in 'data/prot_seq.csv'
#     save_prot_seq(seq, save_path=prot_seq_csv)
#     seq = pd.Series(seq, name='prot_seq')
#     seq.index.name = 'protID'
#     seq = pd.DataFrame(seq)
# else: 
#     seq = pd.read_csv(prot_seq_csv)

# # merge protein sequences with df on protID
# df = df.merge(seq, on='protID')

# pd.DataFrame(seq)
# # %% Unify affinity metrics to be same units (uM)
# conv = {
#     'mM': 1000,
#     'uM': 1,
#     'nM': 1e-3,
#     'pM': 1e-6,
#     'fM': 1e-9,
# }
# def convert_affinity(a):
#     if a[-2:] not in conv:
#         raise ValueError(f'Unknown affinity unit: {a[-2:]} in {a}')
#     else:
#         k, v = re.split(r'=|<=|>=', a)
#         v = float(v[:-2]) * conv[v[-2:]]
#         return v
    
# df.affinity = df.affinity.apply(convert_affinity)

# # %% Save PDBCode,prot_seq,SMILE to X and PDBCode,affinity to Y
# X = df[['PDBCode', 'prot_seq', 'SMILE']]
# Y = df[['PDBCode', 'affinity']]
# .save('data/X.csv')


# %%
