from typing import List
import requests as r
import pandas as pd
import os

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
#TODO: fetch organism info (to filter for only human proteins)?

def prep_data(csv_path='data/P-L_refined_set_all.csv'):
    """
    This file prepares X and Y data files for the model to learn from
    X data will contain cols:
    PDBCode,prot_seq,SMILE
    
    Y file will contain cols:
    PDBCode,affinity
    """
    
    df_raw = pd.read_csv(csv_path)
    
    # filter out complexes with 2+ proteins or none at all
    df = df_raw[lambda x: x['protID'].fillna('').str.split().apply(lambda x: len(x)==1)]
    
    
    #TODO: unify affinity metrics to be same units
    #TODO: filter out protein complexes with multiple protein structures
    pass

def get_prot_seq(protIDs: List[str], 
                  url=lambda x: f'https://rest.uniprot.org/uniprotkb/{x}.fasta') -> dict:
    """
    Fetches FASTA files from given url and returns dict with {ID: seq}
    
    URL is passed in as a callable function which accepts a string (the protID) and returns a url 
    to download that file.
        e.g. for uniprot: lambda x: f'https://rest.uniprot.org/uniprotkb/{x}.fasta'    
    """
    prot_seq = {}
    for protID in protIDs:
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