"""
Assuming hhblits has been run, this script will filter the MSA and convert it to FASTA format.
then run convert to convert the MSA to FASTA format.

hhfilter -id 90 -i outputs/10gs.msa.a3m -o outputs/10gs_filtered.a3m
reformat.pl ./outputs/10gs_filtered.a3m ./outputs/10gs_filtered.fas -r # reformat is not needed!
# just use process_msa to remove lowercase amino acids.
"""
#%%
import os, subprocess
from typing import Iterable, Tuple
from tqdm import tqdm
from multiprocessing import Pool
from src.feature_extraction.protein import get_pfm
from src.data_processing.processors import Processor

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
    

# %%
if __name__ == '__main__':
    import pandas as pd
    csv = '/cluster/home/t122995uhn/projects/data/PlatinumDataset/nomsa_binary/full/XY.csv'
    df = pd.read_csv(csv, index_col=0)