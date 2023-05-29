from typing import List
import requests as r
import pandas as pd
import os, re
from tqdm import tqdm
from urllib.parse import quote

def get_mutated_seq(HGVSc: List[str], 
                    url=lambda x: f'https://mutalyzer.nl/api/normalize/{x}') -> dict:
    """
    Fetches mutated sequences from given url and returns dict with {HGVSc: seq}
    
    URL is passed in as a callable function which accepts a string (the HGVSc) and returns a url 
    to download that file.
        e.g. for mutalyzer: lambda x: f'https://mutalyzer.nl/api/normalize/{x}'
        
    Returned json is of the form:
        {protein: {predicted: "<Mutated Seq>",…},…}   
        
    to get native (non-mutated) sequence, use "reference" instead of "predicted"
    """
    mut_seq = {}
    for mut in tqdm(HGVSc, 'Downloading mutated sequences'):
        if mut in mut_seq: continue
        mut_seq[mut] = r.get(url(mut)).json()['protein']['predicted']
        
    return mut_seq

def get_SMILE(drug_names: List[str],
              url=lambda x: f'https://cactus.nci.nih.gov/chemical/structure/{quote(x)}/smiles') -> dict:
    """Gets SMILE strings from given drug names"""
    drug_SMILEs = {}
    for drug_name in tqdm(drug_names, 'Downloading SMILE strings'):
        if drug_name in drug_SMILEs: continue
        drug_SMILEs[drug_name] = r.get(url(drug_name)).text
    
    return drug_SMILEs

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