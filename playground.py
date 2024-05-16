#%% 1.Gather data for davis,kiba and pdbbind datasets
import os
import pandas as pd
import matplotlib.pyplot as plt
from src.analysis.utils import combine_dataset_pids
from src import config as cfg
df_prots = combine_dataset_pids(dbs=[cfg.DATA_OPT.davis, cfg.DATA_OPT.PDBbind], # just davis and pdbbind for now
                                subset='test')


#%% 2. Load TCGA data
df_tcga = pd.read_csv('../downloads/TCGA_ALL.maf', sep='\t')

#%% 3. Pre filtering
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
# allow for some error due to missing amino acids from pdb file in PDBbind dataset
#   - assumption here is that isoforms will differ by more than 50 amino acids
dfm = dfm[(dfm.seq_len_y <= dfm.seq_len_x) & (dfm.seq_len_x<= dfm.seq_len_y+50)]
print(f"Filter #1 (seq_len)     : {tmp:5d} - {tmp-len(dfm):5d} = {len(dfm):5d}")

# 5.2. Filter out those that dont have the same reference seq according to the "Protein_position" and "Amino_acids" col
 
# Extract mutation location and reference amino acid from 'Protein_position' and 'Amino_acids' columns
dfm['mt_loc'] = pd.to_numeric(dfm['Protein_position'].str.split('/').str[0])
dfm = dfm[dfm['mt_loc'] < dfm['seq_len_y']]
dfm[['ref_AA', 'mt_AA']] = dfm['Amino_acids'].str.split('/', expand=True)

dfm['db_AA'] = dfm.apply(lambda row: row['prot_seq'][row['mt_loc']-1], axis=1)
                         
# Filter #2: Match proteins with the same reference amino acid at the mutation location
tmp = len(dfm)
dfm = dfm[dfm['db_AA'] == dfm['ref_AA']]
print(f"Filter #2 (ref_AA match): {tmp:5d} - {tmp-len(dfm):5d} = {len(dfm):5d}")
print('\n',dfm.db.value_counts())

# %% final seq len distribution

n_bins = 25
lengths = dfm.seq_len_x
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

# Plot histogram
n, bins, patches = ax.hist(lengths, bins=n_bins, color='blue', alpha=0.7)
ax.set_title('TCGA final filtering for db matches')

# Add counts to each bin
for count, x, patch in zip(n, bins, patches):
    ax.text(x + 0.5, count, str(int(count)), ha='center', va='bottom')

ax.set_xlabel('Sequence Length')
ax.set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# %%
