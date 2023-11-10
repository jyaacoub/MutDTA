import json
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

def run_foldseek(foldseek,
                  path,
                  chains: list = None,
                  process_id: int = 0,
                  plddt_path: str = None,
                  plddt_threshold: float = 70.) -> dict:
    """
    
    Args:
        foldseek: Binary executable file of foldseek
        path: Path to pdb file
        chains: Chains to be extracted from pdb file. If None, all chains will be extracted.
        process_id: Process ID for temporary files. This is used for parallel processing.
        plddt_path: Path to plddt file. If None, plddt will not be used.
        plddt_threshold: Threshold for plddt. If plddt is lower than this value, the structure will be masked.

    Returns:
        seq_dict: A dict of structural seqs. The keys are chain IDs. The values are tuples of
        (seq, struc_seq, combined_seq).
    """
    assert os.path.exists(foldseek), f"Foldseek not found: {foldseek}"
    assert os.path.exists(path), f"Pdb file not found: {path}"
    assert plddt_path is None or os.path.exists(plddt_path), f"Plddt file not found: {plddt_path}"
    
    tmp_save_path = f"get_struc_seq_{process_id}.tsv"
    cmd = f"{foldseek} structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
    os.system(cmd)

    seq_dict = {}
    name = os.path.basename(path)
    with open(tmp_save_path, "r") as r:
        for i, line in enumerate(r):
            desc, seq, struc_seq = line.split("\t")[:3]
            
            # Mask low plddt
            if plddt_path is not None:
                with open(plddt_path, "r") as r:
                    plddts = np.array(json.load(r)["confidenceScore"])
                    
                    # Mask regions with plddt < threshold
                    indices = np.where(plddts < plddt_threshold)[0]
                    np_seq = np.array(list(struc_seq))
                    np_seq[indices] = "#"
                    struc_seq = "".join(np_seq)
            
            name_chain = desc.split(" ")[0]
            chain = name_chain.replace(name, "").split("_")[-1]

            if chains is None or chain in chains:
                if chain not in seq_dict:
                    combined_seq = "".join([a + b.lower() for a, b in zip(seq, struc_seq)])
                    seq_dict[chain] = (seq, struc_seq, combined_seq)
        
    os.remove(tmp_save_path)
    os.remove(tmp_save_path + ".dbtype")
    return seq_dict

if __name__ == '__main__':
    # dir_p = '/home/jyaacoub/projects/data/msa/outputs'
    # hhfilter_bin = '/home/jyaacoub/miniconda3/bin/hhfilter'
    # postfix = '.msa.a3m'
    # process_msa_dir(hhfilter_bin, dir_p, postfix)
    dir_p = '/cluster/home/t122995uhn/projects/data'
    
    create_pfm_np_files(dir_p)