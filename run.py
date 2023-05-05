#%% testing download of sequences from FASTA files
import requests as r
from data_processing.format import get_prot_seq, save_prot_seq, excel_to_csv
import pandas as pd

# %% xlsx file to csv
# excel_to_csv()

# %% protID -> seq
# protID = "P55055"
# url = lambda x: f'https://rest.uniprot.org/uniprotkb/{x}.fasta'
# d = get_prot_seq([protID])
# print(d)
# save_prot_seq(d, overwrite=True)
csv_path='data/P-L_refined_set_all.csv'

# %%
df_raw = pd.read_csv(csv_path)

# filter out complexes with 2+ proteins or none at all
df = df_raw[lambda x: x['protID'].fillna('').str.split().apply(lambda x: len(x)==1)]

#%% getting protein sequences
seq = get_prot_seq(df['protID'][:1000])
# save_prot_seq(seq)

# pd.DataFrame(seq)
# %%
