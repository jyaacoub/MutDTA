# %%
import pandas as pd
import logging
DATA_ROOT = '../data'
biom_df = pd.read_csv(f'{DATA_ROOT}/tcga/mart_export.tsv', sep='\t')
biom_df.rename({'Gene name': 'gene'}, axis=1, inplace=True)
biom_df['PDB ID'] = biom_df['PDB ID'].str.lower()

# %% merge on PDB ID
pdb_df = pd.read_csv(f'{DATA_ROOT}/PDBbindDataset/nomsa_binary_original_binary/full/XY.csv')
pdb_df = pdb_df.merge(biom_df.drop_duplicates('PDB ID'), left_on='code', right_on="PDB ID", how='left')
pdb_df.drop(['PDB ID', 'UniProtKB/Swiss-Prot ID'], axis=1, inplace=True)

# %% merge on prot_id: - gene_y
pdb_df = pdb_df.merge(biom_df.drop_duplicates('UniProtKB/Swiss-Prot ID'), 
              left_on='prot_id', right_on="UniProtKB/Swiss-Prot ID", how='left')
pdb_df.drop(['PDB ID', 'UniProtKB/Swiss-Prot ID'], axis=1, inplace=True)


#%%
biom_pdb_match_on_pdbID = pdb_df.gene_x.dropna().drop_duplicates()
print('                            match on PDB ID:', len(biom_pdb_match_on_pdbID))

biom_pdb_match_on_prot_id = pdb_df.gene_y.dropna().drop_duplicates()
print('                           match on prot_id:', len(biom_pdb_match_on_prot_id))


biom_concat = pd.concat([biom_pdb_match_on_pdbID,biom_pdb_match_on_prot_id]).drop_duplicates()
print('\nCombined match (not accounting for aliases):', len(biom_concat))

# cases where both pdb ID and prot_id match can cause issues if gene_x != gene_y resulting in double counting 
# in above concat
pdb_df['gene'] = pdb_df.gene_x.combine_first(pdb_df.gene_y)
print(' pdb_df.gene_x.combine_first(pdb_df.gene_y):', len(pdb_df['gene'].dropna().drop_duplicates()))
# case where we match on prot_id and PDB ID can cause issues with mismatched counts due to 
# different names for the gene (e.g.: due to aliases)
print("\n           num genes where gene_x != gene_y:",
      len(pdb_df[pdb_df['gene_x'] != pdb_df['gene_y']].dropna().drop_duplicates(['gene_x','gene_y'])))
print(f'\n   Total number of entries with a gene name: {len(pdb_df[~pdb_df.gene.isna()])}/{len(pdb_df)}')

# %% matching with kiba gene names as our starting test set
kiba_test_df = pd.read_csv('/cluster/home/t122995uhn/projects/MutDTA/splits/kiba_test.csv')
kiba_test_df = kiba_test_df[['gene']].drop_duplicates()

# only 171 rows from merging with kiba...
pdb_test_df = pdb_df.merge(kiba_test_df, on='gene', how='inner').drop_duplicates(['code', 'SMILE'])
print('Number of entries after merging gene names with kiba test set:', len(pdb_test_df))
print('                                              Number of genes:', len(pdb_test_df.gene.drop_duplicates()))

# %% adding any davis test set genes
davis_df = pd.read_csv('/cluster/home/t122995uhn/projects/MutDTA/splits/davis_test.csv')
davis_test_prots = set(davis_df.prot_id.str.split('(').str[0])
pdb_davis_gene_overlap = pdb_df[pdb_df.gene.isin(davis_test_prots)].gene.value_counts()
print("Total # of gene overlap with davis TEST set:", len(pdb_davis_gene_overlap))
print("                       # of entries in pdb:", pdb_davis_gene_overlap.sum())

pdb_test_df = pd.concat([pdb_test_df, pdb_df[pdb_df.gene.isin(davis_test_prots)]],
                        axis=0).drop_duplicates(['code', 'SMILE'])
print("# of entries in test set after adding davis genes: ", len(pdb_test_df))

#%% CONTINUE TO GET FROM OncoKB:
onco_df = pd.read_csv("/cluster/home/t122995uhn/projects/downloads/oncoKB_DrugGenePairList.csv")

pdb_join_onco = set(pdb_test_df.merge(onco_df.drop_duplicates("gene"), on="gene", how="left")['gene'])

#%%
remaining_onco = onco_df[~onco_df.gene.isin(pdb_join_onco)].drop_duplicates('gene')

# match with remaining pdb:
remaining_onco_pdb_df = pdb_df.merge(remaining_onco, on='gene', how="inner")
counts = remaining_onco_pdb_df.value_counts('gene')

print(counts)
print("total entries in pdb with remaining (not already in test set) onco genes", counts.sum())
# this only gives us 93 entries... so adding it to the rest would only give us 171+93=264 total entries

pdb_test_df = pd.concat([pdb_test_df, remaining_onco_pdb_df], axis=0).drop_duplicates(['code', 'SMILE'])
print("Combined pdb test set with remaining OncoKB genes entries:", len(pdb_test_df)) # 264 only




# %% Random sample to get the rest
# code is from balanced_kfold_split function
from collections import Counter
import numpy as np

# Get size for each dataset and indices
dataset_size = len(pdb_df)
test_size = int(0.1 * dataset_size) # 1626
indices = list(range(dataset_size))

# getting counts for each unique protein
prot_counts = pdb_df['code'].value_counts().to_dict()
prots = list(prot_counts.keys())
np.random.shuffle(prots)

# manually selected prots:
test_prots = set(pdb_test_df.code)
# increment count by number of samples in test_prots
count = sum([prot_counts[p] for p in test_prots])

#%%
## Sampling remaining proteins for test set (if we are under the test_size) 
for p in prots: # O(k); k = number of proteins
    if count + prot_counts[p] < test_size:
        test_prots.add(p)
        count += prot_counts[p]

additional_prots = test_prots - set(pdb_test_df.code)
print('additional codes to add:', len(additional_prots))
print('                  count:', count)

#%% ADDING FINAL PROTS
rand_sample_df = pdb_df[pdb_df.code.isin(additional_prots)]
pdb_test_df = pd.concat([pdb_test_df, rand_sample_df], axis=0).drop_duplicates(['code'])

pdb_test_df.drop(['cancerType', 'drug'], axis=1, inplace=True)
print('Final test dataset for pdbbind:')
pdb_test_df

#%% saving
pdb_test_df.rename({"gene_x":"gene_matched_on_pdb_id", "gene_y": "gene_matched_on_uniprot_id"}, axis=1, inplace=True)
pdb_test_df.to_csv('/cluster/home/t122995uhn/projects/MutDTA/splits/pdbbind_test.csv', index=False)
