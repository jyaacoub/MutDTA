#%% 1.Gather data for davis,kiba and pdbbind datasets
import os
import pandas as pd
import matplotlib.pyplot as plt
from src.analysis.utils import combine_dataset_pids
from src import config as cfg
ROOT_DIR = "../downloads"
kumar_db = False
merge_by_prot_id = False

# making sure NA doesnt get dropped due to pandas parsing it as NaN
tcga_tss = pd.read_csv(f'{ROOT_DIR}/tcga_code_tables/tissueSourceSite.tsv', sep='\t', keep_default_na=False, na_values=['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A', '#N/A', 'N/A', 'n/a','', '#NA', 'NULL', 'null', 'NaN', '-NaN', 'nan', '-nan', ''])
tcga_tss['Study Name'] = tcga_tss['Study Name'].str.strip()
tcga_bcr = pd.read_csv(f'{ROOT_DIR}/tcga_code_tables/bcrBatchCode.tsv', sep='\t', keep_default_na=False, na_values=['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A', '#N/A', 'N/A', 'n/a','', '#NA', 'NULL', 'null', 'NaN', '-NaN', 'nan', '-nan', ''])
tcga_codes = tcga_tss.merge(tcga_bcr.drop_duplicates(subset='Study Name'), on='Study Name', how='left')
tcga_codes = tcga_codes[['TSS Code', 'Study Abbreviation']]

#%% Load up db
if kumar_db:
    df_tcga = pd.DataFrame()
    for f in os.listdir('/home/jean/projects/data/tcga_kumars/TCGA_hg38/'):
        fp = os.path.join('/home/jean/projects/data/tcga_kumars/TCGA_hg38/', f)
        print('\n', '-'*20)
        print(f)
        df_tmp = pd.read_csv(fp, sep='\t', dtype=str)

        # dropping unnneccessary columns
        df_tmp = df_tmp[['Tumor_Sample_Barcode', 'Hugo_Symbol', 'SWISSPROT', 
                         'Variant_Type', 'Variant_Classification']]
        df_tmp['case'] = df_tmp['Tumor_Sample_Barcode'].str[:12]
        df_tmp['uniprot'] = df_tmp['SWISSPROT'].str.split('_').str[0]
        df_tcga = pd.concat([df_tcga, df_tmp], axis=0)
else:
    df_tcga = pd.read_csv(f'/cluster/home/t122995uhn/projects/data/tcga/mc3/mc3.v0.2.8.PUBLIC.maf', 
                      sep='\t', na_filter=False)
    df_tcga = df_tcga[['Tumor_Sample_Barcode', 'Hugo_Symbol', 'SWISSPROT', 
                        'Variant_Type', 'Variant_Classification']]
    df_tcga['case'] = df_tcga['Tumor_Sample_Barcode'].str[:12]
    df_tcga['uniprot'] = df_tcga['SWISSPROT'].str.split('_').str[0]


# merge with tcga codes
# Using second id to match with TSS code for cancer type
df_tcga['TSS Code'] = df_tcga['Tumor_Sample_Barcode'].str.split('-').str[1]
df_tcga = df_tcga.merge(tcga_codes, on='TSS Code', how='left')

# 3. Drop duplicates
df_tcga_uni=df_tcga.drop_duplicates(subset='Tumor_Sample_Barcode')
print(len(df_tcga_uni))
print(df_tcga_uni['Study Abbreviation'].value_counts())


#%% 4. Merging df_prots with TCGA
df_prots = pd.read_csv(f'{ROOT_DIR}/test_prots_gene_names.csv')
# df_prots = df_prots[df_prots.db != 'BindingDB']

if merge_by_prot_id:
    dfm = df_tcga.merge(df_prots[df_prots.db != 'davis'], 
                        left_on='uniprot', right_on='prot_id', how='inner')

    # for davis we have to merge on HUGO_SYMBOLS
    dfm_davis = df_tcga.merge(df_prots[df_prots.db == 'davis'], 
                            left_on='Hugo_Symbol', right_on='prot_id', how='inner')

    dfm = pd.concat([dfm,dfm_davis], axis=0)
    del dfm_davis # to save mem
else: # merge by gene name
    dfm = df_tcga.merge(df_prots, 
                            left_on='Hugo_Symbol', right_on='gene_name', how='inner')

dfm['Study Abbreviation'].value_counts()


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

# %% Getting updated sequences
def apply_mut(row):
    ref_seq = list(row['prot_seq'])
    ref_seq[row['mt_loc']-1] = row['mt_AA']
    return ''.join(ref_seq)

dfm['mt_seq'] = dfm.apply(apply_mut, axis=1)


# %%
dfm.to_csv("/cluster/home/t122995uhn/projects/data/tcga/tcga_maf_davis_pdbbind.csv")