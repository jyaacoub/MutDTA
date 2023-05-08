from typing import List
import requests as r
import pandas as pd
import os, re
from tqdm import tqdm

def sdf_to_SMILE():
    """
    This function will convert an SDF file into the SMILE format using RDKit.
    (for drug molecules)
    """
    pass

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
                   save_path='data/') -> tuple[pd.DataFrame]:
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
        
    df.affinity = df.affinity.apply(convert_affinity)
    
    # Saving to csv without index
    x = df[['PDBCode', 'prot_seq', 'SMILE']]
    x.to_csv(save_path+'X.csv', index=False)
    y = df[['PDBCode', 'affinity']]
    y.to_csv(save_path+'Y.csv', index=False)
    return x, y

def get_prot_seq(protIDs: List[str], 
                  url=lambda x: f'https://rest.uniprot.org/uniprotkb/{x}.fasta') -> dict:
    """
    Fetches FASTA files from given url and returns dict with {ID: seq}
    
    URL is passed in as a callable function which accepts a string (the protID) and returns a url 
    to download that file.
        e.g. for uniprot: lambda x: f'https://rest.uniprot.org/uniprotkb/{x}.fasta'    
    """
    prot_seq = {}
    for protID in tqdm(protIDs, 'Downloading protein sequences'):
        if protID in prot_seq: continue
        FASTA = r.get(url(protID)).text
        prot_seq[protID] = ''.join(FASTA.split('\n')[1:])
        
    return prot_seq
    
def save_prot_seq(prot_dict: dict, save_path="data/prot_seq.csv", overwrite=False) -> None:
    """
    Given the protein dict where keys are IDs and values are seq
    this saves it as a csv.
    """
    assert (overwrite) or (not os.path.exists(save_path)), "File already exists! Change name or delete existing file."
    
    with open(save_path, 'w') as f:
        f.write('protID,prot_seq\n')
        for k,v in prot_dict.items():
            f.write(f'{k},{v}\n')