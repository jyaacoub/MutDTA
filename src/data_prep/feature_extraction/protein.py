from multiprocessing import Pool
from typing import Callable, Iterable
import numpy as np
from tqdm import tqdm

import os, math
import pandas as pd
from src.utils import config as cfg
from src.utils.residue import ResInfo, Chain
from src.data_prep.feature_extraction.protein_nodes import get_pfm, target_to_feature, get_foldseek_onehot, run_foldseek
from src.data_prep.feature_extraction.protein_edges import get_target_edge

########################################################################
###################### Protein Feature Extraction ######################
########################################################################
def target_to_graph(target_sequence:str, contact_map:str or np.array, 
                    threshold=10.5, pro_feat='nomsa', aln_file:str=None,
                    pdb_fp:str=None, pddlt_fp:str=None) -> tuple[np.array,np.array]:
    """
    Feature extraction for protein sequence using contact map to generate
    edge index and node features.

    Parameters
    ----------
    `target_sequence` : str
        Sequence of the target protein.
    `contact_map` : str | np.array
        File path for contact map or the actual map itself.
    `threshold` : float, optional
        Threshold for what defines an edge, anything under this value 
        is considered an edge. Passing in a negative value will flip 
        this to be anything **above** the value (useful for when the cmap 
        is probability based, anything above 0.5 is considered in contact),
        by default 10.5
    `pro_feat` : str, optional
        Type of protein node feature to use, by default 'nomsa'
    `aln_file` : str, optional
        Path to alignment file for PSSM matrix, by default None
    `pdb_fp` : str, optional
        Path to pdb file for foldseek feature, by default None
    `pddlt_fp` : str, optional
        Path to pddlt (confidence) file for foldseek if pdb is a predicted structure, by default None

    Returns
    -------
    Tuple[np.array]
        tuple of (target_sequence, target_feature, target_edge_index)
    """
    assert pro_feat in cfg.PRO_FEAT_OPT, \
        f'Invalid protein feature option: {pro_feat}, must be one of {cfg.PRO_FEAT_OPT}'
    
    edge_index, _ = get_target_edge(target_sequence, contact_map, threshold)
    
    # Geting simple residue one hot and property features.
    pro_hot, pro_property = target_to_feature(target_sequence) # shapes=Lx21 and Lx12
    
    # getting node features
    if pro_feat == 'nomsa':
        #NOTE: DGraphDTA never uses pssm due to logic error 
        # (see: https://github.com/595693085/DGraphDTA/issues/16)
        # returns Lx21 matrix of amino acid distribution for each node
        pssm = np.zeros((len(target_sequence), len(ResInfo.amino_acids)))
        line_count = 1
        target_feature = np.concatenate((pssm, pro_hot, pro_property), axis=1)
    elif pro_feat == 'msa' or pro_feat == 'shannon':
        # get pssm matrix from alignment file
        pssm, line_count = get_pfm(aln_file, target_sequence)
        if pro_feat == 'shannon':
            def entropy(col):
                ent = 0.0
                for base in np.where(col > 0)[0]: # all bases being used
                    n_i = col[base]
                    P_i = n_i/line_count # number of res of type i/ total res in col
                    ent -= P_i*(math.log(P_i,2)) # entropy calc
                # "1 -" so that the larger the value the more conserved that amino acid is. 
                return 1 - (ent / math.log2(21)) # divided by log2(21) which is the max entropy score for any 21 dimension vector
                
            pssm = np.apply_along_axis(entropy, axis=1, arr=pssm)
            pssm = pssm.reshape((len(target_sequence),1))
        else: # normal pssm
            pseudocount = 0.8 # pseudocount to avoid divide by 0
            pssm = (pssm + pseudocount / 4) / (float(line_count) + pseudocount)
        target_feature = np.concatenate((pssm, pro_hot, pro_property), axis=1)
    elif pro_feat == 'foldseek':
        # returns {chain: [seq, struct_seq, combined_seq]} dict
        seq_dict = run_foldseek(pdb_fp, plddt_fp=pddlt_fp)
        
        # use matching sequence from foldseek
        combined_seq = None
        for c in seq_dict:
            if seq_dict[c][0] == target_sequence:
                combined_seq = seq_dict[c][2]
                break
        assert combined_seq is not None, f'Could not find matching foldseek 3Di sequence for {pdb_fp}'
        
        # input sequences should now include 3di tokens
        pro_hot_3di = get_foldseek_onehot(combined_seq)
        target_feature = np.concatenate((pro_hot, pro_hot_3di), axis=1)
        
        # updating target sequence to include 3di tokens
        target_sequence = combined_seq
    else:
        raise NotImplementedError(f'Invalid protein feature option: {pro_feat}')
    
    return target_sequence, target_feature, edge_index


######################################################################
#################### CONTACT MAP EXTRACTION/PREP: ####################
######################################################################
def create_save_cmaps(pdbcodes: Iterable[str]|Iterable[tuple[str]], 
                      pdb_p: Callable[[str], str],
                      cmap_p: Callable[[str], str],
                      overwrite:bool=False) -> dict:
    """
    Given a list of PDBcodes, this will create and save the contact maps for each.
    Example path callable functions:
        pdb_p = lambda c: f'/path/to/refined-set/{c}/{c}_protein.pdb'
        cmap_p = lambda c: f'path/to/refined-set/{c}/{c}_contact_CB.npy'

    Parameters
    ----------
    `pdbcodes` : Iterable[str]
        list of PDBcodes to create contact maps for. or tuple containing pdbcodes and protids
    `pdb_p` : Callable[[str], str]
        function to get pdb file path from pdbcode.
    `cmap_p` : Callable[[str], str]
        function to get cmap save file path from pdbcode.
    Returns

    -------
    dict
        dictionary of sequences for each pdbcode
    """
    seqs = {}
    for elmt in tqdm(pdbcodes, 'Getting protein seq & contact maps'):
        if isinstance(elmt, str):
            code = elmt
            pid = elmt
        else:
            code, pid = elmt
        
        chain = Chain(pdb_p(code))
        seqs[pid] = chain.getSequence()
        # only get cmap if it doesnt exist
        if not os.path.isfile(cmap_p(pid)) or overwrite:
            cmap = chain.get_contact_map()
            np.save(cmap_p(pid), cmap)
    return seqs

def _save_cmap(args):
    pdb_f, cmap_f, overwrite = args
    try:
        chain = Chain(pdb_f)
        seq = chain.getSequence()
    except KeyError as e:
        raise KeyError(f'Error with {pdb_f}') from e
    
    if os.path.isfile(cmap_f) or overwrite:
        return seq
        
    # only get cmap if it doesnt exist
    cmap = chain.get_contact_map()
    np.save(cmap_f, cmap)
    return seq
    
def multi_save_cmaps(pdbcodes: Iterable[str]|Iterable[tuple[str]], 
                      pdb_p: Callable[[str], str],
                      cmap_p: Callable[[str], str],
                      overwrite:bool=False,
                      processes=None) -> dict: 
    # by default uses same number of processes as in system
    
    # pdb_f, cmap_f, overwrite
    if isinstance(pdbcodes[0], str):
        args = [[pdb_p(code), cmap_p(code), overwrite] for code in pdbcodes]
    else:
        args = [[pdb_p(code), cmap_p(pid), overwrite] for code, pid in pdbcodes]
    
    with Pool(processes=processes) as pool:
        seqs = list(tqdm(pool.imap(_save_cmap, args),
                  total=len(args),
                  desc='Creating and saving cmaps'))
    
    # order is maintained (see https://stackoverflow.com/questions/41273960/python-3-does-pool-keep-the-original-order-of-data-passed-to-map)
    if isinstance(pdbcodes[0], str):
        return {code: seq for code, seq in zip(pdbcodes, seqs)}
    return {pid: seq for (_, pid), seq in zip(pdbcodes, seqs)}
    


def create_aln_files(df_seq: pd.DataFrame, aln_p: Callable[[str], str]):
    """
    Creates alignment files for all PDBbind structures.
    
    df_seq: dataframe with index as PDBcodes and column 'prot_seq' as sequence
    """
    raise NotImplementedError("This function is not complete (see yumika)")

if __name__ == "__main__":
    print("hi")
