from multiprocessing import Pool
from typing import Callable, Iterable
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import os, math
import pandas as pd
from src.utils.residue import ResInfo, Chain
from src.feature_extraction.protein_nodes import get_pfm, target_to_feature
from src.feature_extraction.protein_edges import get_target_edge

########################################################################
###################### Protein Feature Extraction ######################
########################################################################
def target_to_graph(target_sequence:str, contact_map:str or np.array, 
                    threshold=10.5, aln_file:str=None, shannon=False) -> tuple[np.array,np.array]:
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
    `aln_file` : str, optional
        Path to alignment file for PSSM matrix, by default None
    `shannon` : bool, optional
        If True, shannon entropy instead of PSSM matrix is used for 
        protein features, by default False.

    Returns
    -------
    Tuple[np.array]
        tuple of (target_feature, target_edge_index)
    """
    edge_index, _ = get_target_edge(target_sequence, contact_map, threshold)
    # getting node features
    
    # aln_dir = 'data/' + dataset + '/aln'
    if aln_file is not None:
        pssm, line_count = get_pfm(aln_file, target_sequence)
    else:
        #NOTE: DGraphDTA never uses pssm due to logic error 
        # (see: https://github.com/595693085/DGraphDTA/issues/16)
        # returns Lx21 matrix of amino acid distribution for each node
        pssm = np.zeros((len(target_sequence), len(ResInfo.amino_acids)))
        line_count = 1
    
    if shannon:
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
        #TODO: *1 if max prob matches target seq node *-1 otherwise
    else:
        pseudocount = 0.8
        pssm = (pssm + pseudocount / 4) / (float(line_count) + pseudocount)
    
    pro_hot, pro_property = target_to_feature(target_sequence) # shapes=Lx21 and Lx12
    target_feature = np.concatenate((pssm, pro_hot, pro_property), axis=1)
    
    return target_feature, edge_index


######################################################################
#################### CONTACT MAP EXTRACTION/PREP: ####################
######################################################################
def create_save_cmaps(pdbcodes: Iterable[str], 
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
        list of PDBcodes to create contact maps for.
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
    for code in tqdm(pdbcodes, 'Getting protein seq & contact maps'):
        chain = Chain(pdb_p(code))
        seqs[code] = chain.getSequence()
        # only get cmap if it doesnt exist
        if not os.path.isfile(cmap_p(code)) or overwrite:
            cmap = chain.get_contact_map()
            np.save(cmap_p(code), cmap)
    return seqs

def _save_cmap(args):
    pdb_f, cmap_f, overwrite = args
    # skip if already created
    if os.path.isfile(cmap_f) and not overwrite: return
    try:
        cmap = Chain(pdb_f).get_contact_map()
    except KeyError as e:
        raise KeyError(f'Error with {pdb_f}') from e
    np.save(cmap_f, cmap)
    
def multi_save_cmaps(pdbcodes: Iterable[str], 
                      pdb_p: Callable[[str], str],
                      cmap_p: Callable[[str], str],
                      processes=8) -> dict:
    
    #pdb_f, cmap_f, overwrite
    args = [[pdb_p(code), cmap_p(code), True] for code in pdbcodes]
    with Pool(processes=processes) as pool:
        print('Starting process')
        list(tqdm(pool.imap(_save_cmap, args),
                  total=len(args),
                  desc='Creating and saving cmaps'))
    


def create_aln_files(df_seq: pd.DataFrame, aln_p: Callable[[str], str]):
    """
    Creates alignment files for all PDBbind structures.
    
    df_seq: dataframe with index as PDBcodes and column 'prot_seq' as sequence
    """
    raise NotImplementedError("This function is not complete (see yumika)")

if __name__ == "__main__":
    print("hi")
