from typing import List
import requests as r
import pandas as pd
import os, re
from tqdm import tqdm
from urllib.parse import quote
from general import get_prot_seq, save_prot_seq

def excel_to_csv(xlsx_path='data/P-L_refined_set_all.xlsx'):
    """
    Converts the PDBbind Xls file into a CSV file with the following cols:
    ID,PDBCode,affinity,year,prot_name,lig_name,protID,SMILE
    """
    df = pd.read_excel(xlsx_path, header=1, index_col=0)
    df = df[['PDB code', 'Affinity Data', 'Release Year',
        'Protein Name', 'Ligand Name', 
        'UniProt AC', 'Canonical SMILES']]
    df.rename(columns={'PDB code': 'PDBCode', 
                       'Affinity Data': 'affinity', 
                       'Release Year': 'year',
                        'Protein Name': 'prot_name', 
                        'Ligand Name': 'lig_name',
                        'UniProt AC': 'protID', 
                        'Canonical SMILES': 'SMILE'}, inplace=True)
    
    df.to_csv(''.join(xlsx_path.split('.')[:-1])+'.csv')
    
def prep_data_for_mdl(data_path='data/'):
    """
    Preps data for model training

    Args:
        data_path (str, optional): The path to the X and Y csv files. 
                    Defaults to 'data/'.
    """
    pass

def prep_save_data(csv_path='data/raw/P-L_refined_set_all.csv', 
                   prot_seq_csv='data/prot_seq.csv', 
                   save_path='data/', Kd_only=False) -> tuple[pd.DataFrame]:
    """
    This file prepares and saves X and Y csv files for the model to learn from
    X data will contain cols:
    PDBCode,prot_seq,SMILE
    
    Y file will contain cols:
    PDBCode,affinity (in uM)

    Args:
        csv_path (str, optional): Path to unfiltered csv. Defaults to 
                                    'data/raw/P-L_refined_set_all.csv'.
        prot_seq_csv (str, optional): Path to csv containing pdbID and 
                                        corresponding sequences. Defaults 
                                        to 'data/prot_seq.csv'.
        save_path (str, optional): Path to save X and Y csv files. Defaults 
                                    to 'data/'.
        Kd_only (bool, optional): Whether to only use Kd values. Defaults to False.
    
    returns:
        tuple(pd.DataFrame, pd.DataFrame): X and Y dataframes
    """
    
    df_raw = pd.read_csv(csv_path)
    
    # filter out complexes with 2+ proteins or none at all
    df = df_raw[lambda x: x['protID'].fillna('').str.split().apply(lambda x: len(x)==1)]

    # getting protein sequences and saving them 
    if not os.path.exists(prot_seq_csv):
        seq = get_prot_seq(df['protID'])
        # default save path in 'data/prot_seq.csv'
        save_prot_seq(seq, save_path=prot_seq_csv)
        seq = pd.Series(seq, name='prot_seq')
        seq.index.name = 'protID'
        seq = pd.DataFrame(seq)
    else: 
        seq = pd.read_csv(prot_seq_csv)
    
    # merge protein sequences with df on protID
    df = df.merge(seq, on='protID') # inner join and left join are the same here
        
    # Unify affinity metrics to be same units (uM)
    conv = {
        'mM': 1000,
        'uM': 1,
        'nM': 1e-3,
        'pM': 1e-6,
        'fM': 1e-9,
    }
    def convert_affinity(a):
        if a[-2:] not in conv:
            raise ValueError(f'Unknown affinity unit: {a[-2:]} in {a}')
        else:
            k, v = re.split(r'=|<=|>=', a)
            v = float(v[:-2]) * conv[v[-2:]]
            return v
    
    if Kd_only:
        df = df[df.affinity.str.contains('Kd')]
    
    df.affinity = df.affinity.apply(convert_affinity)
    
    # Saving to csv without index
    x = df[['PDBCode', 'prot_seq', 'SMILE']]
    x.to_csv(save_path+'X.csv', index=False)
    y = df[['PDBCode', 'affinity']]
    y.to_csv(save_path+'Y.csv', index=False)
    return x, y

if __name__ == '__main__':
    # Exploring the data
    csv_path='data/PDBbind/raw/P-L_refined_set_all.csv'
    df_raw = pd.read_csv(csv_path)
    df = df_raw[lambda x: x['protID'].fillna('').str.split().apply(lambda x: len(x)==1)]
    df_a = df.affinity

    # unify affinity metrics to be same units (uM)
    conv = {
        'mM': 1000,
        'uM': 1,
        'nM': 1e-3,
        'pM': 1e-6,
        'fM': 1e-9,
    }
    const = {
        'Ki':[],
        'Kd':[]
    }

    for a in df_a:
        k, v = re.split(r'=|<=|>=', a)
        v = float(v[:-2]) * conv[v[-2:]]
        if 'Ki' in a:
            const['Ki'].append(v)
        elif 'Kd' in a:
            const['Kd'].append(v)
        else:
            raise ValueError(f'Unknown affinity metric: {a}')

    # removing outliers (Ki or Kd > 1e-3)
    for k in const:
        const[k] = [x for x in const[k] if x < 1e-3]

    # plotting distribution of Ki and Kd values
    import matplotlib.pyplot as plt
    plt.hist(const['Ki'], bins=10, alpha=0.8, color='r')
    plt.hist(const['Kd'], bins=10, alpha=0.8, color='g')

    plt.legend(['Ki', 'Kd'])
    print('Ki:',len(const['Ki']), 'Kd:', len(const['Kd']))

    # statistical test to see if Ki and Kd are from the same distribution
    from scipy.stats import ranksums
    listl = const['Ki']
    list2 = const['Kd']
    u_stat, p_val = ranksums(listl, list2)
    print('p-value:', p_val)
    print('u-stat:', u_stat)