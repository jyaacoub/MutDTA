#%%
import os
import pandas as pd
from src.data_prep.processors import Processor
root_dir = '../data/DavisKibaDataset/davis'
df = pd.read_csv(f"{root_dir}/nomsa_binary_original_binary/full/XY.csv")

df_unique = df.loc[df[['code']].drop_duplicates().index]
df_unique.drop(['SMILE', 'pkd', 'prot_id'], axis=1, inplace=True)
df_unique['code'] = df_unique['code'].str.upper()
df_unique.columns = ['Gene name', 'prot_seq']
#%%
df_mart = pd.read_csv('../downloads/biomart_hsapiens.tsv', sep='\t')
df_mart = df_mart.loc[df_mart[['Gene name', 'UniProtKB/Swiss-Prot ID']].dropna().index]
df_mart['Gene name'] = df_mart['Gene name'].str.upper()
df_mart = df_mart.drop_duplicates(subset=['UniProtKB/Swiss-Prot ID'])

#%%
dfm = df_unique.merge(df_mart, on='Gene name', how='left')

dfm[['Gene name', 'UniProtKB/Swiss-Prot ID']].to_csv('../downloads/davis_biomart_matches.csv')
#%%


# %%
from src.utils.pdb import pdb2uniprot

for root_dir in ['../data/PDBbindDataset', '../data/DavisKibaDataset/davis', 
                 '../data/DavisKibaDataset/kiba', '../data/PlatinumDataset']:
    print(root_dir)
    DB = {root_dir.split("/")[-1]}
    fp = f'../downloads/{DB}_pids.csv'
    if not os.path.exists(fp):
        df2 = pd.read_csv(f"{root_dir}/nomsa_binary_original_binary/full/XY.csv", index_col=0)
        df2['pdb_id'] = df2.prot_id.str.split("_").str[0]
        df2 = df2[['pdb_id']]

        df2.to_csv(f'../downloads/{DB}_pids.csv')
    else:
        df2 = pd.read_csv(fp, index_col=0)
        
    if DB == 'PlatinumDataset': # needs internet connection to run
        uniprots = pdb2uniprot(df2.pdb_id.unique())
        df_pid = pd.DataFrame(list(uniprots.items()), columns=['pdbID', 'uniprot'])

        df_pid.to_csv(f'../downloads/{DB}_uniprot.csv')
    

# %%
exit()
# find overlap between uniprots and swissprot ids
uniprots = set(uniprots) # convert to set for faster lookup
df['uniprot'] = df['SWISSPROT'].str.split('.').str[0]
# Merge the original df with uni_to_pdb_df
df = df.merge(df_pid, on='uniprot', how='left')

# %%
df[~df['pdbID'].isna()]
# %%


