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


def hhblits(f_in:str, f_out:str, n_cpus=6, 
            bin_path ='/cluster/tools/software/centos7/hhsuite/3.3.0/bin/hhblits',
            dataset='/cluster/projects/kumargroup/mslobody/Protein_Communities/01_MSA/databases/UniRef30_2020_06'
            ) -> subprocess.CompletedProcess:
    # hhblits works on a single sequence at once
    # can pass FASTA file
    
    cmd = f"hhblits" + \
            f" -i {f_in}" + \
            f" -oa3m {f_out}" + \
            f" -d {dataset}" + \
            f" -cpu {n_cpus}" + \
            f" -n 2"

    return subprocess.run(cmd, capture_output=True, check=True, shell=True)

def hhfilter(bin_path: str, f_in: str,
             f_out: str=None) -> subprocess.CompletedProcess:
    assert 'a3m' in f_in, 'Input file must be in a3m format'
    f_out = f_in.split('.a3m')[0] + '_filtered.a3m' if f_out is None else f_out
    
    cmd = f'{bin_path} -id 90 -i {f_in} -o {f_out}'
    return subprocess.run(cmd, capture_output=True,
                          shell=True, check=True)

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
    
def process_msa_dir(hhfilter_bin:str, in_dir:str, out_dir:str=None,
                    postfix:str='.msa.a3m'):
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
            process_msa(hhfilter_bin, f_in, f_out)
    
def process_msa_file(args):
    hhfilter_bin, f_in, f_out = args
    if not os.path.isfile(f_out):
        process_msa(hhfilter_bin, f_in, f_out)

def multiprocess_msa_dir(hhfilter_bin: str, in_dir: str, out_dir:str=None, 
                         postfix: str = '.msa.a3m'):
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
            for _ in pool.imap_unordered(process_msa_file, args_list):
                pbar.update(1)
    

# %%
