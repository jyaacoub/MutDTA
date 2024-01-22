"""
Processors for processing data from various sources.

Primary use is for handling data at the high level if you wish to extract features,
see `src/feature_extraction/` instead.

e.g.: here we might want to use prep_save_data to get the X and Y csv files for training.

"""
from typing import Callable, Iterable, Tuple

from argparse import ArgumentError
from urllib.parse import quote

import os, re, subprocess
from multiprocessing import Pool

import numpy as np
import requests as r
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.PandasTools import LoadSDF


class Processor:
    @staticmethod
    def check_aln_lines(fp:str, limit=None):
        if not os.path.isfile(fp): return False
        
        with open(fp, 'r') as f:
            lines = f.readlines()
            seq_len = len(lines[0])
            limit = len(lines) if limit is None else limit
            
            for i,l in enumerate(lines[:limit]):
                if l[0] == '>' or len(l) != seq_len:
                    return False
        return True
    
    @staticmethod
    def csv_to_fasta_dir(csv_or_df:str or pd.DataFrame, out_dir:str):
        """
        Given a list of sequences from a csv file under 'prot_seq' column,
        and 'prot_id' column for protein names, this will create fastas for each 
        unique protein with the code as the file name and fasta header.
        Args:
            csv_p (strorpd.DataFrame): csv path or pandas dataframe with 
            'prot_id' and 'prot_seq' columns.
            out_dir (str): output directory for fasta files
        """
        if isinstance(csv_or_df, str):
            df = pd.read_csv(csv_or_df)
        elif isinstance(csv_or_df, pd.DataFrame):
            df = csv_or_df
        else:
            raise ValueError("csv_p should be a file path or a pandas DataFrame.")

        os.makedirs(out_dir, exist_ok=True)
        
        # sorting by sequence length before dropping so that we keep the longest protein sequence instead of just the first.
        df['seq_len'] = df['prot_seq'].str.len()
        df = df.sort_values(by='seq_len', ascending=False)
        
        # create new numerated index col for ensuring the first unique uniprotID is fetched properly 
        df.reset_index(drop=False, inplace=True)
        unique_pro = df[['prot_id']].drop_duplicates(keep='first')
        
        # reverting index to code-based index
        df.set_index('code', inplace=True)
        
        # getting the unique proteins
        unique_df = df.iloc[unique_pro.index]
        
        # finally creating Fastas
        for _, (prot_id, pro_seq) in tqdm(
                        unique_df[['prot_id', 'prot_seq']].iterrows(), 
                        desc='Creating fastas',
                        total=len(unique_df)):
            
            fasta_fp = os.path.join(out_dir, f"{prot_id}.fasta")
            with open(fasta_fp, "w") as f:
                f.write(f">{prot_id}\n{pro_seq}")
    
    @staticmethod
    def fasta_to_aln_file(in_fp, out_fp):
        """
        Removes lines from the input Fasta file that start with '>' and saves the result in the output file.
        """
        # Read the input file
        with open(in_fp, 'r') as file:
            lines = file.readlines()

        # Initialize an empty result list
        result = []

        # Iterate through the lines and exclude lines starting with '>'
        for line in lines:
            if not line.startswith('>'):
                result.append(line)

        # Join the result list into a single string
        output_text = ''.join(result)

        # Write the result to the output file
        with open(out_fp, 'w') as file:
            file.write(output_text)
            
    @staticmethod
    def fasta_to_aln_dir(in_dir:str, out_dir:str=None, silent=True):
        """
        Removes lines starting with '>' from all files in the input directory and saves
        the modified content in corresponding output files in the output directory.

        Args:
            in_dir (str): Input dir for a3m fasta files
            out_dir (str, optional): Output dir, if none then will save in the same dir as input. 
                Defaults to None.
            silent (bool, optional): to silent tqdm output. Defaults to True.
        """
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
        else:
            out_dir = in_dir    
            
        # Get a list of files in the input directory
        input_files = os.listdir(in_dir)

        # Iterate through each input file
        for in_fn in tqdm(input_files, "Converting fastas to aln files",
                          disable=silent):
            # Construct the full path of the input file
            in_fp = os.path.join(in_dir, in_fn)

            # Skip directories if any
            if os.path.isdir(in_fp): continue

            # Construct the full path of the output file
            start, _ = os.path.splitext(in_fn)
            out_fp = os.path.join(out_dir, f"{start}.aln")
                
            if os.path.isfile(out_fp): continue
            
            Processor.fasta_to_aln_file(in_fp, out_fp)
    
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

class MSARunner(Processor):
    hhsuite_bin_dir = '/cluster/tools/software/centos7/hhsuite/3.3.0/bin'
    bin_hhblits = f'{hhsuite_bin_dir}/hhblits'
    bin_hhfilter = f'{hhsuite_bin_dir}/hhfilter'
    UniRef30_dir = '/cluster/projects/kumargroup/mslobody/Protein_Communities/01_MSA/databases/UniRef30_2020_06'

    @staticmethod    
    def hhblits(f_in:str, f_out:str, n_cpus=6, n_iter:int=2,
                bin_path:str=None, dataset:str=None) -> subprocess.CompletedProcess:
        # hhblits works on a single sequence at once
        # can pass FASTA file
        
        cmd = f"{bin_path or MSARunner.bin_hhblits}" + \
                f" -i {f_in}" + \
                f" -oa3m {f_out}" + \
                f" -d {dataset or MSARunner.UniRef30_dir}" + \
                f" -cpu {n_cpus}" + \
                f" -n {n_iter}"
        return subprocess.run(cmd, capture_output=True, check=True, shell=True)

    @staticmethod    
    def hhfilter(f_in: str, f_out:str=None, 
                 max_seq_id:int=90, bin_path:str=None) -> subprocess.CompletedProcess:
        assert '.a3m' in f_in, 'Input file must be in a3m format'
        
        f_out = f_in.split('.a3m')[0] + '_filtered.a3m' if f_out is None else f_out
        
        cmd = f'{bin_path or MSARunner.bin_hhfilter} -id {max_seq_id} -i {f_in} -o {f_out}'
        return subprocess.run(cmd, capture_output=True, check=True, shell=True)

    @staticmethod
    def clean_msa(f_in: str, f_out:str):
        # getting sequences from file
        with open(f_in, 'r') as f:
            lines = f.readlines()
            seq = [l for l in lines if l[0] != '>']
        
        # writing sequences to file without lowercase letters 
        # (hhblits adds lowercase letters to indicate insertions)
        with open(f_out, 'w') as f:
            for l in seq:
                # removing lowercase letters
                new_l = ''
                for c in l:
                    if c.isupper() or c == '-':
                        new_l += c
                f.write(new_l+'\n')

    @staticmethod
    def check_aln_lines(fp:str, limit=None):
        if not os.path.isfile(fp): return False
        
        with open(fp, 'r') as f:
            lines = f.readlines()
            seq_len = len(lines[0])
            limit = len(lines) if limit is None else limit
            
            for i,l in enumerate(lines[:limit]):
                if l[0] == '>' or len(l) != seq_len:
                    return False
        return True

    @staticmethod
    def process_msa(f_in:str, f_out:str, hhfilter_bin:str=None):
        MSARunner.hhfilter(f_in, f_out, bin_path=hhfilter_bin)
        # overwrites filtered msa with cleaned msa
        MSARunner.clean_msa(f_in=f_out, f_out=f_out)
        MSARunner.check_aln_lines(f_out)

    @staticmethod    
    def process_msa_dir(in_dir:str, out_dir:str=None,
                        postfix:str='.msa.a3m', hhfilter_bin:str=None):
        """
        Processes msas after hhblits has been run:
        1. Filters the MSA using hhfilter.
        2. Removes lowercase letters and '>' labels from the MSA and saves it as 
        '<pdb>_clean.a3m'.

        Parameters
        ----------
        `hhfilter_bin` : str
            binary path to hhfilter see: https://github.com/soedinglab/hh-suite
        `dir_p` : str
            directory path to the MSA files.
        `postfix` : str, optional
            postfix pattern for msa files, by default '.msa.a3m'
        """
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
        else:
            out_dir = in_dir
        msas = [f for f in os.listdir(in_dir) if f.endswith(postfix)]
        
        for msa in tqdm(msas, 'Filtering and cleaning MSAs'):
            f_in = f'{in_dir}/{msa}'
            f_out = f'{out_dir}/{msa[:-len(postfix)]}.aln'
            if not os.path.isfile(f_out):
                MSARunner.process_msa(f_in, f_out, hhfilter_bin)

    @staticmethod    
    def process_msa_file(args):
        hhfilter_bin, f_in, f_out = args
        if not os.path.isfile(f_out):
            MSARunner.process_msa(f_in, f_out, hhfilter_bin)

    @staticmethod
    def multiprocess_msa_dir(in_dir:str, out_dir:str=None,postfix:str='.msa.a3m',
                            hhfilter_bin:str=None):
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
        else:
            out_dir = in_dir
        msas = [f for f in os.listdir(in_dir) if f.endswith(postfix)]

        # Create a Pool for multiprocessing
        with Pool() as pool:
            args_list = [(hhfilter_bin, os.path.join(in_dir, msa), 
                        os.path.join(out_dir, 
                                    msa[:-len(postfix)]) + '.aln') for msa in msas]
            with tqdm(total=len(args_list), desc='Processing MSAs') as pbar:
                for _ in pool.imap_unordered(MSARunner.process_msa_file, args_list):
                    pbar.update(1)
    

if __name__ == '__main__':
    from src.data_processing.processors import MSARunner
    from tqdm import tqdm
    import pandas as pd
    import os
    from src.data_processing.datasets import BaseDataset
    csv = '/cluster/home/t122995uhn/projects/data/PlatinumDataset/nomsa_binary/full/XY.csv'
    df = pd.read_csv(csv, index_col=0)
    #################### Get unique proteins:
    unique_df = BaseDataset.get_unique_prots(df)
    
    ########################## Get job partition
    num_arrays = 100
    array_idx = 0#${SLURM_ARRAY_TASK_ID}
    partition_size = len(unique_df) / num_arrays
    start, end = int(array_idx*partition_size), int((array_idx+1)*partition_size)
    
    unique_df = unique_df[start:end]
    
    raw_dir = '/cluster/home/t122995uhn/projects/data/PlatinumDataset/raw'

    #################################### create fastas
    fa_dir = os.path.join(raw_dir, 'platinum_fa')
    os.makedirs(fa_dir, exist_ok=True)
    MSARunner.csv_to_fasta_dir(csv_or_df=unique_df, out_dir=fa_dir)

    ##################################### Run hhblits
    aln_dir = os.path.join(raw_dir, 'platinum_aln')
    os.makedirs(aln_dir, exist_ok=True)

    # finally running
    for _, (prot_id, pro_seq) in tqdm(
                    unique_df[['prot_id', 'prot_seq']].iterrows(), 
                    desc='Running hhblits',
                    total=len(unique_df)):
        in_fp = os.path.join(fa_dir, f"{prot_id}.fasta")
        out_fp = os.path.join(aln_dir, f"{prot_id}.a3m")
        
        if not os.path.isfile(out_fp):
            MSARunner.hhblits(in_fp, out_fp)