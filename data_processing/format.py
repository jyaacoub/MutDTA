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

def excel_to_csv(xlsx_path):
    """
    Converts the PDBbind Xls file into a CSV file with the following cols:
    PDBCode,prot_name,ligand_name,protID,SMILE
    """
    file = pd.read_excel(xlsx_path)
    pass

#TODO: fetch organism info (to filter for only human proteins)
#TODO: filter out protein complexes with multiple protein structures

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
        f.write('protID,sequence\n')
        for k,v in prot_dict.items():
            f.write(f'{k},{v}\n')