"""
Processors for processing data from various sources.

Primary use is for handling data at the high level if you wish to extract features,
see `src/feature_extraction/` instead.

e.g.: here we might want to use prep_save_data to get the X and Y csv files for training.

"""

from argparse import ArgumentError as ArgError
from collections import OrderedDict
from ctypes import ArgumentError
from typing import Callable, Iterable, List, Tuple
from urllib.parse import quote
import requests as r
import os, re

import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.PandasTools import LoadSDF

from src.feature_extraction.utils import ResInfo

class Processor:    
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
    
    @staticmethod
    def get_sequence(pdb_file: str, check_missing=False, 
                 select_largest=True) -> Tuple[str, OrderedDict]:
        """
        Given a pdb file path this will return the residue sequence for that structure
        (could be missing residues) and the residue dict in order of seq# that contains coords.

        Args:
            pdb_file (str): path to .pdb file to process.
            check_missing (bool, optional): Adds check to ensure all residues are available. 
                                    Defaults to False.
            select_largest (bool, optional): If True, only the largest chain is used. Otherwise
                        returns the first chain. Defaults to True.
            
        Returns:
            Tuple[str, OrderedDict]: the sequence of residues and the residue dict in order of seq#.
        """

        # read and filter
        with open(pdb_file, 'r') as f:
            lines = f.readlines()
            chains = OrderedDict() # chain dict of dicts
            ter = 0 # chain terminator
            chains[0] = OrderedDict() # first chain
            curr_res, prev_res = None, None
            for line in lines:
                if (line[:6].strip() == 'TER'): # TER indicates new chain "terminator"
                    ter += 1
                    chains[ter] = OrderedDict()
                    curr_res, prev_res = None, None
                
                if (line[:6].strip() != 'ATOM'): continue # skip non-atom lines
                
                # make sure res# is in order and not missing
                prev_res = curr_res
                curr_res = int(line[22:26])
                if check_missing:
                    assert prev_res is None or \
                        curr_res == prev_res or \
                        curr_res == prev_res+1, \
                            f"Invalid order or missing residues: {prev_res} -> {curr_res} in {pdb_file}"
                                
                # only want CA and CB atoms
                atm_type = line[12:16].strip()
                if atm_type not in ['CA', 'CB']: continue
                icode = line[26].strip() # dumb icode because residues will sometimes share the same res num 
                                # (https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html)
                
                # Glycine has no CB atom, so we save both 
                key = f"{curr_res}_{icode}"
                assert atm_type not in chains[ter].get(key, {}), f"Duplicate {atm_type} for residue {key} in {pdb_file}"
                # adding atom to residue
                chains[ter].setdefault(key, {})[atm_type] = np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])])
                
                # Saving residue name
                assert ("name" not in chains[ter].get(key, {})) or \
                    (chains[ter][key]["name"] == line[17:20].strip()), \
                                            f"Inconsistent residue name for residue {key} in {pdb_file}"
                chains[ter][key]["name"] = line[17:20].strip()
                
        # getting sequence of largest chain
        chain_opt = 0
        if select_largest:
            for i in range(len(chains)):
                if len(chains[i]) > len(chains[chain_opt]): chain_opt = i
            
        return_chain = chains[chain_opt]
        seq = '' # sequence of residues based on pdb file
        for res in return_chain:
            seq += ResInfo.pep_to_code[return_chain[res]["name"]]
                
        return seq, return_chain


class PDBbindProcessor(Processor):
    @staticmethod
    def get_SMILE(IDs: Iterable[str],
                dir: Callable[[str], str] = 
                lambda x: f'/home/jyaacoub/projects/data/refined-set/{x}/{x}_ligand.mol2') -> dict:
        """
        Uses rdkit to converts sdf (or mol2) files to SMILE strings and returns dict with {ID: SMILE}.
        lig_names are not unique and so IDs must be something other than ligand names pdbcodes 
        (e.g.: https://en.wikipedia.org/wiki/K-mer)

        Parameters
        ----------
        `IDs` : Iterable[str]
            Iterable of codes to pass on to `dir` to get the path to the sdf file.
        `dir` : Callable[[str], str], optional
            Callable function that returns the path to read sdf (or mol2) files from, 
            by default lambda x: f'/home/jyaacoub/projects/data/refined-set/{x}/{x}_ligand.mol2'

        Returns
        -------
        dict
            Dict with {ID: SMILE}
        """
        drug_smi = {}
        RDLogger.DisableLog('rdApp.*') # supress rdkit warnings
        for id in tqdm(IDs, desc='Extracting SMILE strings'):
            assert id not in drug_smi, f'duplicate ID: {id}'
            fp = dir(id)
            try:
                if fp.endswith('.sdf'):
                    drug_smi[id] = LoadSDF(fp, smilesName='smile',
                        # setting this to False means each SMILE is unique (canonical)
                                           isomericSmiles=False)['smile'][0]
                elif fp.endswith('.mol2'):
                    m=Chem.MolFromMol2File(fp)
                    drug_smi[id] = Chem.MolToSmiles(m, 
                                                    isomericSmiles=False) 
                else:
                    raise ValueError(f'Invalid file type: {fp}')
            except KeyError as e:
                drug_smi[id] = None
            except ArgumentError as e:
                print(f'Error with {id}')
                raise e
        RDLogger.EnableLog('rdApp.*')
            
        return drug_smi
    
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
    def get_name_data(index_name_file: str) -> pd.DataFrame:
        """
        Extracts PDB code, release year, Uniprot ID, protein name
        from INDEX_general_PL_name.2020 file from PDBbind and returns 
        it as a dataframe.
        
        Input file has cols:
            PDB code, release year, Uniprot ID, protein name
            
        e.g. line:
            `6mu1  2018  P29994  INOSITOL 1,4,5-TRISPHOSPHATE RECEPTOR TYPE 1`

        Parameters
        ----------
        `index_name_file` : str
            Path to the v2020-other-PL/index/INDEX_general_PL_name.2020 from PDBbind

        Returns
        -------
        pd.DataFrame
            With cols:
            
            PDBCode,release_year,prot_id,prot_name
        """
        data = {}
        with open(index_name_file, 'r') as f:
            for line in f.readlines():
                if line.startswith('#'): continue
                code = line[:4]
                try:
                    year = int(line[5:10])
                    prot_id = line[11:18].strip()
                    prot_name = line[18:].strip()
                    # print('year:', year, 
                    #       '\nprot_id:', prot_id, 
                    #       '\nprot_name:', prot_name)
                    data[code] = [year, prot_id, prot_name]
                except ValueError as e:
                    print(f'Error with line: {line}')
                    raise e
        
        df = pd.DataFrame.from_dict(data, orient='index', 
                                    columns=['release_year', 'prot_id', 'prot_name'])
        df.index.name = 'PDBCode'
        
        return df
         
    @staticmethod
    def get_binding_data(index_data_file : str) -> pd.DataFrame:
        """
        Extracts binding data from given index file and returns it as a dataframe.
        Sample file header:
        
            PDB code, resolution, release year, -logKd/Ki, Kd/Ki, reference, ligand name 
            3zzf  2.20  2012   0.40  Ki=400mM      // 3zzf.pdf (NLG)        

        Parameters
        ----------
        `index_data_file` : str
            Path to the v2020-other-PL/index/INDEX_general_PL_data.2020 from PDBbind

        Returns
        -------
        pd.DataFrame
            With cols:
            
            PDBCode,resolution,release_year,pkd,lig_name
        """
        data = {}
        with open(index_data_file, 'r') as f:
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