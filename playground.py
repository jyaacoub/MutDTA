#%%
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

df = pd.read_csv("../data/TCGA_BRCA_Mutations.csv")
df = df.loc[df['SWISSPROT'].dropna().index]
df[['Gene', 'SWISSPROT']].head()

# %%
from src.utils.pdb import pdb2uniprot
df2 = pd.read_csv("../data/PlatinumDataset/nomsa_binary_original_binary/full/cleaned_XY.csv", index_col=0)
df2['pdb_id'] = df2.prot_id.str.split("_").str[0]

uniprots = pdb2uniprot(df2.pdb_id.unique())
df_pid = pd.DataFrame(list(uniprots.items()), columns=['pdbID', 'uniprot'])
# %%
# find overlap between uniprots and swissprot ids
uniprots = set(uniprots) # convert to set for faster lookup
df['uniprot'] = df['SWISSPROT'].str.split('.').str[0]
# Merge the original df with uni_to_pdb_df
df = df.merge(df_pid, on='uniprot', how='left')

# %%
df[~df['pdbID'].isna()]
# %%
