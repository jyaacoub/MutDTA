import json
from typing import Tuple
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool

from src.utils.residue import ResInfo, one_hot
from src.utils import config as cfg

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

def get_foldseek_onehot(combined_seq):
    # the target sequence includes 3Di foldseek tokens alternating with the actual sequence
    # so we divide by 2 to get the length of the actual sequence
    fld_hot = np.zeros((len(combined_seq)//2, len(ResInfo.foldseek_tokens)))
    for i in range(1, len(combined_seq), 2):
        fld_hot[i // 2,] = one_hot(combined_seq[i], ResInfo.foldseek_tokens)
    return fld_hot

def run_foldseek(pdb_fp:str, foldseek_bin:str=cfg.FOLDSEEK_BIN,
                 chains: list = None,
                 plddt_fp: str = None,
                 plddt_threshold: float = 70.) -> dict:
    """
    Adapted from https://github.com/westlake-repl/SaProt/blob/main/utils/foldseek_util.py
    
    Args:
        path: Path to pdb file
        foldseek: Binary executable file of foldseek
        chains: Chains to be extracted from pdb file. If None, all chains will be extracted.
        plddt_path: Path to plddt file. If None, plddt will not be used.
            Example: colabfold/kiba_af2_out/highQ/O00141_scores_rank_001_alphafold2_ptm_model_1_seed_000.json
        plddt_threshold: Threshold for plddt. If plddt is lower than this value, the structure will be masked.

    Returns:
        seq_dict: A dict of structural seqs. The keys are chain IDs. The values are tuples of
        (seq, struc_seq, combined_seq).
    """
    assert os.path.exists(foldseek_bin), f"Foldseek not found: {foldseek_bin}"
    assert os.path.exists(pdb_fp), f"Pdb file not found: {pdb_fp}"
    assert plddt_fp is None or os.path.exists(plddt_fp), f"Plddt file not found: {plddt_fp}"
    
    # save in same location as pdb file
    tmp_save_path = f"{pdb_fp}.foldseek.txt"
    
    # run foldseek only if the output file doesn't already exist
    if not os.path.exists(tmp_save_path):   
        cmd = f"{foldseek_bin} structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {pdb_fp} {tmp_save_path}"
        os.system(cmd) # this is a blocking call
        # remove dbtype file which is created by foldseek for some reason #TODO: why?
        os.remove(tmp_save_path + ".dbtype")

    # extract seqs from foldseek output
    seq_dict = {}
    name = os.path.basename(pdb_fp)
    with open(tmp_save_path, "r") as r:
        for i, line in enumerate(r):
            desc, seq, struc_seq = line.split("\t")[:3]
            
            # Mask low plddt
            if plddt_fp is not None:
                with open(plddt_fp, "r") as r:
                    plddts = np.array(json.load(r)["plddt"]) # NOTE: updated from "confidenceScore"
                    
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
    
    return seq_dict

if __name__ == '__main__':
    # dir_p = '/home/jyaacoub/projects/data/msa/outputs'
    # hhfilter_bin = '/home/jyaacoub/miniconda3/bin/hhfilter'
    # postfix = '.msa.a3m'
    # process_msa_dir(hhfilter_bin, dir_p, postfix)
    dir_p = '/cluster/home/t122995uhn/projects/data'
    
    create_pfm_np_files(dir_p)