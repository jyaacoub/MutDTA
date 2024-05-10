#%% 1-2. download TCGA and gather proteins for dbs 
import os
import pandas as pd

df_prots = pd.read_csv('../downloads/all_prots.csv')
df_tcga = pd.read_csv('../downloads/TCGA_ALL.maf', sep='\t')

#%% 3. Pre filtering
import matplotlib.pyplot as plt
df_tcga = df_tcga[df_tcga['Variant_Classification'] == 'Missense_Mutation']
df_tcga['seq_len'] = pd.to_numeric(df_tcga['Protein_position'].str.split('/').str[1])
df_tcga = df_tcga[df_tcga['seq_len'] < 5000]
df_tcga['seq_len'].plot.hist(bins=100, title="sequence length histogram capped at 5K")
plt.show()
df_tcga = df_tcga[df_tcga['seq_len'] < 1200]
df_tcga['seq_len'].plot.hist(bins=100, title="sequence length after capped at 1.2K")

#%% 4. Merging df_prots with TCGA
df_tcga['uniprot'] = df_tcga['SWISSPROT'].str.split('.').str[0]

dfm = df_tcga.merge(df_prots[df_prots.db != 'davis'], 
                    left_on='uniprot', right_on='prot_id', how='inner')

# for davis we have to merge on HUGO_SYMBOLS
dfm_davis = df_tcga.merge(df_prots[df_prots.db == 'davis'], 
                          left_on='Hugo_Symbol', right_on='prot_id', how='inner')

dfm = pd.concat([dfm,dfm_davis], axis=0)

del dfm_davis # to save mem

# %% 5. Post filtering step
# 5.1. Filter for only those sequences with matching sequence length (to get rid of nonmatched isoforms)
# seq_len_x is from tcga, seq_len_y is from our dataset 
tmp = len(dfm)
dfm = dfm[dfm.seq_len_x == dfm.seq_len_y]
print(f"Filter #1 (seq_len)     : {tmp:5d} - {tmp-len(dfm):5d} = {len(dfm):5d}")

# 5.2. Filter out those that dont have the same reference seq according to the "Protein_position" and "Amino_acids" col
# - reference sequence is in colum "prot_seq" 
# - `56-Protein_position` tells us `<mutation location>/seqlen`
# 	- Match only proteins that have matching `seqlen`
# - `57-Amino_acids` tells us `<reference AA>/<mutated AA>`. 
# 	- match only proteins with the same `reference AA` at `mutation location` (see above)
 
# Extract mutation location and reference amino acid from 'Protein_position' and 'Amino_acids' columns
dfm['mt_loc'] = pd.to_numeric(dfm['Protein_position'].str.split('/').str[0])
dfm[['ref_AA', 'mt_AA']] = dfm['Amino_acids'].str.split('/', expand=True)

dfm['db_AA'] = dfm.apply(lambda row: row['prot_seq'][row['mt_loc']-1], axis=1)
                         
# Filter #2: Match proteins with the same reference amino acid at the mutation location
tmp = len(dfm)
dfm = dfm[dfm['db_AA'] == dfm['ref_AA']]
print(f"Filter #2 (ref_AA match): {tmp:5d} - {tmp-len(dfm):5d} = {len(dfm):5d}")

# %% final seq len distribution
df_tcga['seq_len'].plot.hist(bins=100, title='TCGA final filtering for db matches')

# %%
