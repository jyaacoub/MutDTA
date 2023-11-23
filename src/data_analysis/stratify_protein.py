# To resolve https://github.com/jyaacoub/MutDTA/issues/57
# list of protein names for davis is in ../data/davis/proteins.txt 
# or downloaded from https://staff.cs.utu.fi/~aatapa/data/DrugTarget/target_gene_names.txt

from typing import Iterable
from src.utils import config as cfg
import pandas as pd
import re

def kinbase_to_df(fasta_fp:str=f'{cfg.DATA_ROOT}/misc/Human_kinase_domain.fasta'):
    """
    Converts the KinBase fasta file containing all Human Kinase Domains into a 
    dataframe for easy parsing.
    
    The Human_kinase_domain can be retrieved from:
    https://web.archive.org/web/20230517032418/http://kinase.com/kinbase/FastaFiles/Human_kinase_domain.fasta
        Sample of the FASTA headers:
            >TTBK2_Hsap (CK1/TTBK) *
            >TTBK1_Hsap (CK1/TTBK)
            >TSSK4_Hsap (CAMK/TSSK)
            >TSSK3_Hsap (CAMK/TSSK)
        - The first couple characters before the underscore are the protein names (*e.g.: TTBK2).
        - The characters after the underscore are the species (*e.g.: Hsap == homo sapiens).
        - Most importantly, the characters between the parenthesis are the protein family and 
          subgroups in that order (*e.g.: CK1/TTBK).

    Parameters
    ----------
    `fasta_fp` : str, optional
        The path to the downloaded fasta file path, by default f'{cfg.DATA_ROOT}/misc/Human_kinase_domain.fasta'
    """
    prots = {}
    with open(fasta_fp, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            if line[0] == '>': # header
                seq = lines[i+1].strip()
                name = re.search(r'^>(.+?)_Hsap', line).group(1)
                # all in the fasta has a protein family discriptor with at least 2 elements
                protein_family = re.search(r'\((.*)\)', line).group(1)
                main_family, subgroup = protein_family.split('/')[:2]
                
                prots[name] = (main_family, subgroup, seq)
                
                i+=2
            else:
                i+=1
    # convert to dataframe
    df = pd.DataFrame.from_dict(prots, orient='index', columns=['main_family', 'subgroup', 'seq'])
    df.index.name = 'protein_name'
    return df

def check_davis_names(davis_prots:dict, df:pd.DataFrame) -> list:
    """
    Checks davis protein names against fasta file containing human kinase domains. 
    Returns list of proteins in davis that are also found in the fasta.
    
    NOTE: that for some in davis they have mutation information in brackets after 
    the protein name, but we only need the protein name for this function.
    Example:
        ABL1(F317I)p -> ABL1
    
    There are also some with "-alpha?" (or "beta" etc..) where ? is a number, we 
    can't ignore these and must include them by appending to the protein name "a?".
        
    Parameters
    ----------
    `davis_prots` : dict
        Dictionary of davis protein names (keys) and their associated sequence (values).
    `df` : pd.DataFrame
        The dataframe containing the human kinase domains (see kinbase_to_df()).
    """
    
    df = kinbase_to_df() if df is None else df
    
    greek = {'alpha', 'beta', 'gamma', 'delta', 'epsilon'} # for checking if protein name has greek letter
    
    found_prots = {}
    for k in davis_prots.keys():
        name = k.split('(')[0]
        alpha = ''
        if '-' in name:
            name, alpha = name.split('-')
        
        # removing any 'p' at the end of the name which indicates phosphorylation
        if name[-1] == 'p':
            name = name[:-1]
        
        # getting alpha info if it exists
        if len(alpha) >= 1:
            alpha_name, alpha_num = re.search(r'([a-z]*)(\d*)', alpha).groups()
            if alpha_name in greek:
                name += f'{alpha_name[0]}{alpha_num}'
            
        # checking if name is in the dataframe
        if name in df.index:
            found_prots[k] = (name, df.loc[name, 'main_family'], df.loc[name, 'subgroup'])
        else:
            print(f'MISSING: {k}-{name}')
            
    return found_prots
            
if __name__ == '__main__':
    import json
    from src.data_analysis.stratify_protein import check_davis_names, kinbase_to_df

    prot_dict = json.load(open('/home/jyaacoub/projects/data/davis/proteins.txt', 'r'))
    
    df = kinbase_to_df()
    prots = check_davis_names(prot_dict, df)
