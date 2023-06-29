"""
Processors for processing data from various sources.

Primary use is for handling data at the high level if you wish to extract features,
see `src/feature_extraction/` instead.

e.g.: here we might want to use prep_save_data to get the X and Y csv files for training.

"""

from typing import Iterable, List, Tuple
from urllib.parse import quote
import requests as r
import os, re

import pandas as pd
from tqdm import tqdm

from rdkit import RDLogger
from rdkit.Chem.PandasTools import LoadSDF

class Processor:

    @staticmethod
    def get_SMILE(info_df: pd.DataFrame,  # can download from https://files.rcsb.org/ligands/view/SQ9_ideal.sdf
                dir=lambda x: f'/home/jyaacoub/projects/data/refined-set/{x}/{x}_ligand.sdf') -> dict:
        """
        Uses rdkit to converts sdf files to SMILE strings and returns dict with {lig_name: SMILE}, 
        given `info_df` which contains the PDBcodes as the index and the corresponding drug names 
        in column 'lig_name'.

        Parameters
        ----------
        `info_df` : pd.DataFrame
            Dataframe containing PDBcodes as the index and the corresponding drug names in column 'lig_name'.
        `dir` : _type_, optional
            Callable function that returns the path to read sdf files from, 
            by default lambda x :f'/home/jyaacoub/projects/data/refined-set/{x}/{x}_ligand.sdf'

        Returns
        -------
        dict
            Dict with {lig_name: SMILE}
        """
        drug_smi = {}
        RDLogger.DisableLog('rdApp.*') # supress rdkit warnings
        for code, row in tqdm(info_df.iterrows(), total=len(info_df), 
                              desc='Extracting SMILE strings from sdf'):
            lig_name = row['lig_name']
            if lig_name in drug_smi and drug_smi[lig_name] is not None: continue
            try:
                drug_smi[lig_name] = LoadSDF(dir(code), smilesName='smile')['smile'][0]
            except KeyError as e:
                drug_smi[lig_name] = None
        RDLogger.EnableLog('rdApp.*')
            
        return drug_smi
    
    @staticmethod
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
                

class PDBbindProcessor(Processor):
    @staticmethod
    def excel_to_csv(xlsx_path='data/PDBbind/raw/P-L_refined_set_all.xlsx') -> None:
        """
        Converts the PDBbind Xls file into a CSV file with the following cols:
        ID,PDBCode,affinity,year,prot_name,lig_name,protID,SMILE

        Parameters
        ----------
        `xlsx_path` : str, optional
            Path to PDBbind xls file, by default 'data/PDBbind/raw/P-L_refined_set_all.xlsx'
        """
        df = pd.read_excel(xlsx_path, header=1, index_col=0, dtype=str)
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
        
        df.to_csv(''.join(xlsx_path.split('.')[:-1]) + '.csv')
    
    @staticmethod
    def prep_data_for_mdl(data_path='data/'):
        """
        Preps data for model training

        Parameters
        ----------
        `data_path` : str, optional
            The path to the X and Y csv files, by default 'data/'

        Raises
        ------
        NotImplementedError
            This method is not yet implemented for this class.
        """
        raise NotImplementedError
    
    @staticmethod
    def get_binding_data(index_file : str) -> pd.DataFrame:
        """
        Extracts binding data from given index file and returns it as a dataframe.
        Sample file header:
        
            PDB code, resolution, release year, -logKd/Ki, Kd/Ki, reference, ligand name 
            3zzf  2.20  2012   0.40  Ki=400mM      // 3zzf.pdf (NLG)        

        Parameters
        ----------
        `index_file` : str
            Path to the v2020-other-PL/index/INDEX_general_PL_data.2020 from PDBbind

        Returns
        -------
        pd.DataFrame
            With cols:
            
            PDBCode,resolution,release_year,pkd,lig_name
        """
        data = {}
        with open(index_file, 'r') as f:
            for line in f.readlines():
                if line.startswith('#'): continue
                code = line[:4]
                try:
                    res = line[5:10].strip()
                    res = float(res) if res != 'NMR' else None
                    year = int(line[11:16])
                    pkd = float(line[17:23])
                    lig_name = re.search(r'\((.*)\)', line).group(1)
                    data[code] = [res, year, pkd, lig_name]
                except ValueError as e:
                    print(f'Error with line: {line}')
                    raise e
        
        df = pd.DataFrame.from_dict(data, orient='index', 
                                    columns=['resolution', 'release_year', 'pkd', 'lig_name'])
        df.index.name = 'PDBCode'
        
        return df
    
    @staticmethod
    def prep_save_data(csv_path='data/PDBbind/raw/P-L_refined_set_all.csv', 
                        prot_seq_csv='data/prot_seq.csv', 
                        save_path='data/PDBbind/kd_only', Kd_only=True) -> Tuple[pd.DataFrame]:
        """
        This file prepares and saves X, Y, and unique_lig csv files for the model to learn from
        X data will contain cols:
        PDBCode,prot_seq,SMILE
        
        Y file will contain cols:
        PDBCode,affinity (in uM)
        
        unique_lig file will contain cols (this will be used for docking prep).
        lig_name,SMILE.

        Parameters
        ----------
        `csv_path` : str, optional
            Path to unfiltered csv, by default 'data/PDBbind/raw/P-L_refined_set_all.csv'
        `prot_seq_csv` : str, optional
            Path to csv containing PDB codes and corresponding sequences, by default 'data/prot_seq.csv'
        `save_path` : str, optional
            Path to save X and Y csv files, by default 'data/PDBbind/kd_only'
        `Kd_only` : bool, optional
            Whether to only use Kd values, by default True

        Returns
        -------
        tuple[pd.DataFrame]
            Dataframe containing PDB codes, protein sequences, SMILES, and affinity values.

        Raises
        ------
        ValueError
            Unknown affinity unit.
        """
        
        df_raw = pd.read_csv(csv_path, dtype=str)
        
        # filter out complexes with 2+ proteins or none at all
        df = df_raw[lambda x: x['protID'].fillna('').str.split().apply(lambda x: len(x)==1)]

        # getting protein sequences and saving them 
        if not os.path.exists(prot_seq_csv):
            seq = PDBbindProcessor.get_prot_seq(df['protID'])
            # default save path in 'data/prot_seq.csv'
            PDBbindProcessor.save_prot_seq(seq, save_path=prot_seq_csv)
            seq = pd.Series(seq, name='prot_seq')
            seq.index.name = 'protID'
            seq = pd.DataFrame(seq)
        else: 
            seq = pd.read_csv(prot_seq_csv, dtype=str)
        
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
        x = df[['PDBCode', 'protID', 'lig_name','prot_seq', 'SMILE']]
        x.to_csv(save_path+'/X.csv', index=False)
        y = df[['PDBCode', 'affinity']]
        y.to_csv(save_path+'/Y.csv', index=False)
        
        # unique lig_names + SMILE
        lig_names = df[['lig_name', 'SMILE']].drop_duplicates('lig_name')
        lig_names.to_csv(save_path+'/unique_lig.csv', index=False)
        
        return df

if __name__ == '__main__':
    # prep data from raw
    PDBbindProcessor.prep_save_data()
    
    exit()
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