#%% 1.Gather data for davis,kiba and pdbbind datasets
import os, logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.analysis.utils import combine_dataset_pids
from src import config as cfg

# get all prots
def add_gene_name(df, biomart="/cluster/home/t122995uhn/projects/data/tcga/mart_export.tsv"):
    bdf = pd.read_csv(biomart, sep='\t')
    bdf['PDB ID'] = bdf['PDB ID'].str.lower()
    
    df_davis = df[df.db == 'davis']
    df_davis['gene'] = df_davis['code']
    
    df_pdbbind = df[df.db == 'PDBbindDataset'].merge(bdf.drop_duplicates(subset='PDB ID'), 
                                                     left_on='code',   right_on="PDB ID", how="left")
    df_kiba = df[df.db == 'kiba'].merge(bdf.drop_duplicates(subset='UniProtKB/Swiss-Prot ID'), 
                                        left_on='prot_id',right_on="UniProtKB/Swiss-Prot ID", how="left")
    
    df_pdb_kiba = pd.concat([df_pdbbind, df_kiba], axis=0)
    df_pdb_kiba.drop(['PDB ID', 'UniProtKB/Swiss-Prot ID'], inplace=True, axis=1)
    df_pdb_kiba.rename({'Gene name':'gene'}, inplace=True, axis=1)
        
    return pd.concat([df_pdb_kiba, df_davis], axis=0)
    


def load_TCGA(tcga_maf= "../data/tcga/mc3/mc3.v0.2.8.PUBLIC.maf", tcga_code_tables_dir='../data/tcga_code_tables/'):
    # making sure NA doesnt get dropped due to pandas parsing it as NaN
    tcga_codes_kwargs = dict(sep='\t', keep_default_na=False, 
                       na_values=['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A', '#N/A', 'N/A', 
                                  'n/a', '', '#NA', 'NULL', 'null', 'NaN', '-NaN', 'nan', '-nan', ''])
    
    tcga_tss = pd.read_csv(f'{tcga_code_tables_dir}/tissueSourceSite.tsv',**tcga_codes_kwargs)
    tcga_tss['Study Name'] = tcga_tss['Study Name'].str.strip()
    tcga_bcr = pd.read_csv(f'{tcga_code_tables_dir}/bcrBatchCode.tsv', **tcga_codes_kwargs)
    tcga_codes = tcga_tss.merge(tcga_bcr.drop_duplicates(subset='Study Name'), on='Study Name', how='left')
    tcga_codes = tcga_codes[['TSS Code', 'Study Abbreviation']]

    # Load up db
    df_tcga = pd.read_csv(tcga_maf, sep='\t', na_filter=False, 
                    usecols=['Tumor_Sample_Barcode', 'Hugo_Symbol', 'SWISSPROT',
                        'Variant_Type', 'Variant_Classification', 'TREMBL'])
    # merge with tcga codes
    # Using second id to match with TSS code for cancer type
    df_tcga['TSS Code'] = df_tcga['Tumor_Sample_Barcode'].str[5:7]
    df_tcga = df_tcga.merge(tcga_codes, on='TSS Code', how='left')
    return df_tcga

def plot_tcga_heat_map(prots_df=None, tcga_df=None, merged_df=None, top=10, title_prot_subset="all proteins",
                       title_postfix='', axis=None, show=True):
    """Returns merged dataframe of prots and tcga maf file on "gene" column"""
    if isinstance(prots_df, str):
        prots_df = add_gene_name(pd.read_csv(prots_df))
        ''

    assert not (prots_df is None and tcga_df is None) or merged_df is not None, "Either provide a merged dataframe or both prots_df and tcga_df"
    
    if merged_df is None:
        logging.debug("Merging TCGA MAF with proteins")
        merged_df = tcga_df.merge(prots_df, left_on='Hugo_Symbol', right_on='gene', how='inner')

    # narrow heat map to just the top cancers/genes
    top_x_cancers = list(merged_df.value_counts('Study Abbreviation').index[:top])
    top_x_genes = list(merged_df.value_counts('Hugo_Symbol').index[:top])

    filtered_merged_df = merged_df[merged_df['Study Abbreviation'].isin(top_x_cancers)]
    filtered_merged_df = filtered_merged_df[filtered_merged_df['Hugo_Symbol'].isin(top_x_genes)]

    grps = filtered_merged_df.groupby(['Study Abbreviation', 'Hugo_Symbol'])

    logging.debug("Building matrix with cancers, genes as the rows and columns respectively")
    matrix = np.zeros((len(top_x_cancers), len(top_x_genes)))

    for (cancer, gene), v in grps.groups.items():
        i_cancer = top_x_cancers.index(cancer)
        i_gene = top_x_genes.index(gene)
        matrix[i_cancer, i_gene] = len(v)
        
    logging.debug("Plotting heatmap:")
    if axis is None:
        _, axis = plt.subplots(figsize=(12,8))
    heatmap = axis.pcolor(matrix, cmap=plt.cm.Blues)

    # Set ticks at the center of each cell
    axis.set_xticks(np.arange(matrix.shape[1]) + 0.5, minor=False)
    axis.set_yticks(np.arange(matrix.shape[0]) + 0.5, minor=False)

    # Set tick labels
    axis.set_xticklabels(top_x_genes, minor=False)
    axis.set_yticklabels(top_x_cancers, minor=False)
    axis.tick_params('x', labelrotation=45)
    
    axis.set_ylabel('Cancer Type')
    axis.set_xlabel('Gene Name')
    axis.set_title(f"Cancer type - gene counts for {title_prot_subset}{title_postfix}")
    plt.colorbar(heatmap)
    if show: plt.show()
    
    return merged_df

def plot_combined_heatmap(df_tcga=None):
    if not df_tcga:
        df_tcga = load_TCGA()
        df_tcga['case'] = df_tcga['Tumor_Sample_Barcode'].str[:12]
    # df_tcga_uni['uniprot'] = df_tcga_uni['SWISSPROT'].str.split('_').str[0]
    # df_tcga_uni['uniprot2'] = df_tcga_uni['TREMBL'].str.split(',').str[0].str.split('_').str[0]

    cases = [True,False]
    test_df = pd.read_csv('../downloads/test_prots_gene_names.csv').rename({'gene_name':'gene'}, axis=1)
    csvs = {
        'all proteins': "../downloads/all_prots.csv", 
        'test proteins with BindingDB': test_df,
        'test proteins': test_df[test_df.db != 'BindingDB'],
        }

    _, axes = plt.subplots(len(csvs),len(cases), figsize=(12*len(cases),8*len(csvs)))

    for i, drop_duplicates in enumerate(cases):
        df_tcga_uni = df_tcga.drop_duplicates(subset='Tumor_Sample_Barcode') if drop_duplicates else df_tcga
        
        for j, k in enumerate(csvs.keys()):
            merged_df = plot_tcga_heat_map(csvs[k], df_tcga_uni, merged_df=None, 
                                        top=20,
                                        title_prot_subset=k, 
                                        title_postfix=' (unique cases)' if drop_duplicates else '',
                                        axis=axes[j][i], show=False)
            
    plt.tight_layout()

# %%
df_tcga = load_TCGA()
#%%
plot_tcga_heat_map(merged_df=df_tcga, title_prot_subset="ENTIRE TCGA MAF", top=20)

# %%
