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


def hhblits(bin_path, f_in, f_out):
    pass

def hhfilter(bin_path: str, f_in: str,
             f_out: str=None) -> subprocess.CompletedProcess:
    assert 'a3m' in f_in, 'Input file must be in a3m format'
    f_out = f_in.split('.a3m')[0] + '_filtered.a3m' if f_out is None else f_out
    
    cmd = f'{bin_path} -id 90 -i {f_in} -o {f_out}'
    return subprocess.run(cmd, capture_output=True,
                          shell=True, check=True)

def clean_msa(f_in: str, f_out:str):
    assert '.a3m' in f_in, 'Input file must be in a3m format'
    
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

def process_msa(hhfilter_bin:str, f_in:str, f_out:str):
    hhfilter(hhfilter_bin, f_in, f_out)
    # overwrites filtered msa with cleaned msa
    clean_msa(f_in=f_out, f_out=f_out)
    check_aln_lines(f_out)

def process_msa_dir(hhfilter_bin:str, dir_p:str, postfix:str='.msa.a3m'):
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
    msas = [f for f in os.listdir(dir_p) if f.endswith(postfix)]
    
    for msa in tqdm(msas, 'Filtering and cleaning MSAs'):
        f_in = f'{dir_p}/{msa}'
        f_out = f'{dir_p}/{msa[:-len(postfix)]}_cleaned.a3m'
        process_msa(hhfilter_bin, f_in, f_out)
    
def multi_process_msa_dir(hhfilter_bin:str, dir_p:str, 
                          postfix:str='.msa.a3m', processes:int=4):
    # create list of arguments for multiprocessing
    #   -> tuple of (hhfilter_bin, f_in, f_out)
    msas = [(hhfilter_bin, f'{dir_p}/{f}', 
             f'{dir_p}/{f[:-len(postfix)]}_cleaned.a3m') \
                 for f in os.listdir(dir_p) if f.endswith(postfix)]
    
    
    with Pool(processes=processes) as pool:
        starargs = lambda args: process_msa(*args)
        # using tqdm to show progress bar
        list(tqdm(pool.imap(starargs, msas), 
                  total=len(msas), 
                  desc='Filtering and cleaning MSAs'))
    
def create_pfm_np_files(aln_dir, processes=4):
    """
    Creates a .npy file for each MSA in the given directory.
    """
    files = [f for f in os.listdir(aln_dir) if f.endswith('.a3m') or f.endswith('.aln')]
    # adding directory path to file names
    files = [f'{aln_dir}/{f}' for f in files]
    
    with Pool(processes=processes) as pool:
        # using tqdm to show progress bar
        list(tqdm(pool.imap(get_pfm, files), 
                  total=len(files), 
                  desc='Creating PFM files'))
    
    
#%%
if __name__ == '__main__':
    # dir_p = '/home/jyaacoub/projects/data/msa/outputs'
    # hhfilter_bin = '/home/jyaacoub/miniconda3/bin/hhfilter'
    # postfix = '.msa.a3m'
    # process_msa_dir(hhfilter_bin, dir_p, postfix)
    dir_p = '/cluster/home/t122995uhn/projects/data'
    
    create_pfm_np_files(dir_p)
# %%
