from typing import Tuple
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool

from src.utils.residue import ResInfo, one_hot

def get_pfm(aln_file: str, target_seq: str=None, overwrite=False) -> Tuple[np.array, int]:
    """ Returns position frequency matrix of amino acids based on MSA for each node in sequence"""
    with open(aln_file, 'r') as f:
        lines = f.readlines()
        
    # first line is target seq
    target_seq = lines[0].strip() if target_seq is None else target_seq 
        
    save_p = aln_file+'.pfm.npy'
    if not overwrite and os.path.isfile(save_p):
        return np.load(save_p), len(lines)
    
    # initializing matrix and counting up amino acids
    # matrix is Lx21 where L is the length of the protein
    # 21 is the number of amino acids + X (unknown) 
    pfm = np.zeros((len(target_seq), len(ResInfo.amino_acids)), dtype=np.int64)
    for line in lines:
        line = line.strip()
        
        # counting up the amino acids at each position
        res_indices = np.array([ResInfo.res_to_i.get(res, -1) for res in line])
        valid_indices = res_indices != -1
        pfm[np.where(valid_indices), res_indices[valid_indices]] += 1
            
    # saving as numpy array
    np.save(save_p, pfm)
    
    return pfm, len(lines)
    
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

# target aln file save in data/dataset/aln
def target_to_feature(target_seq):    
    pro_hot = np.zeros((len(target_seq), len(ResInfo.amino_acids)))
    pro_property = np.zeros((len(target_seq), 12))
    for i in range(len(target_seq)):
        # if 'X' in pro_seq:
        #     print(pro_seq)
        pro_hot[i,] = one_hot(target_seq[i], ResInfo.amino_acids)
        pro_property[i,] = residue_features(target_seq[i])
    
    return pro_hot, pro_property

def residue_features(residue):
    feats = [residue in ResInfo.aliphatic, residue in ResInfo.aromatic,
            residue in ResInfo.polar_neutral, residue in ResInfo.acidic_charged,
            residue in ResInfo.basic_charged,
            
            ResInfo.weight[residue], ResInfo.pka[residue], 
            ResInfo.pkb[residue], ResInfo.pkx[residue],
            ResInfo.pl[residue], ResInfo.hydrophobic_ph2[residue], 
            ResInfo.hydrophobic_ph7[residue]]
    return np.array(feats)


if __name__ == '__main__':
    # dir_p = '/home/jyaacoub/projects/data/msa/outputs'
    # hhfilter_bin = '/home/jyaacoub/miniconda3/bin/hhfilter'
    # postfix = '.msa.a3m'
    # process_msa_dir(hhfilter_bin, dir_p, postfix)
    dir_p = '/cluster/home/t122995uhn/projects/data'
    
    create_pfm_np_files(dir_p)