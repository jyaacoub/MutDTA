from typing import Callable, Iterable, Tuple
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from tqdm import tqdm

import os, math
import pandas as pd
from src.feature_extraction import ResInfo, one_hot

########################################################################
###################### Protein Feature Extraction ######################
########################################################################
# pconsc4 predicted contact map save in data/dataset/pconsc4
def get_target_edge_index(target_sequence:str, contact_map:str or np.array,
                          threshold=10.5):
    """
    Returns edge index for target sequence given a contact map.

    Parameters
    ----------
    `target_sequence` : str
        Sequence of the target protein.
    `contact_map` : strornp.array
        File path for contact map or the actual map itself.
    `threshold` : float, optional
        Threshold for what defines an edge, by default 10.5
        
    Returns
    -------
    np.array
        edge index for target sequence.
    """
    # loading up contact map if it is a file path
    if type(contact_map) == str: contact_map = np.load(contact_map)
    
    target_size = len(target_sequence)
    assert contact_map.shape[0] == contact_map.shape[1], 'contact map is not square'
    # its ok if it is smaller, but not larger (due to missing residues in pdb)
    assert contact_map.shape[0] == target_size, \
            f'contact map size does not match target sequence size,'+\
            f'{contact_map.shape[0]} != {target_size}'
    
    
    # adding self loop then thresholding
    # contact_map += np.matrix(np.eye(contact_map.shape[0])) # Self loop
    # NOTE: the self loop is implied since the diagonal is already 0 (for real cmaps)
    if threshold >= 0.0:
        index_row, index_col = np.where(contact_map <= threshold)
    else: # negative threshold flips the sign
        index_row, index_col = np.where(contact_map >= abs(threshold))
    assert index_row.max() < target_size and index_col.max() < target_size, \
        'contact map size does not match target sequence size'
    
    # converting edge matrix to edge index for pytorch geometric
    target_edge_index = np.array([[i,j] for i, j in zip(index_row, index_col)])
    return target_edge_index

def target_to_graph(target_sequence:str, contact_map:str or np.array, 
                    threshold=10.5, aln_file:str=None, shannon=False):
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
    target_edge_index = get_target_edge_index(target_sequence, contact_map, threshold)
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
                ent -= P_i*(math.log(P_i,2))
            return ent
            
        pssm = np.apply_along_axis(entropy, axis=1, arr=pssm)
        pssm = pssm.reshape((len(target_sequence),1))
        #TODO: *1 if max prob matches target seq node *-1 otherwise
    else:
        pseudocount = 0.8
        pssm = (pssm + pseudocount / 4) / (float(line_count) + pseudocount)
    
    pro_hot, pro_property = target_to_feature(target_sequence) # shapes=Lx21 and Lx12
    target_feature = np.concatenate((pssm, pro_hot, pro_property), axis=1)
    
    return target_feature, target_edge_index

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


def get_contact(residues: OrderedDict, CA_only=True, check_missing=False,
                display=False, title="Residue Contact Map") -> np.array:
    """
    Given the residue chain dict this will return the residue contact map for that structure.
        See: `get_sequence` for details on getting the residue chain dict.

    Parameters
    ----------
    `residues` : OrderedDict
        Residue chain dict extracted from pdb file.
    `CA_only` : bool, optional
        If true then only use alpha carbon for distance calculation. Otherwise follow DGraphDTA 
        definition using CB for all except glycine, by default True
    `check_missing` : bool, optional
        Checks to ensure no residues are missing, by default False
    `display` : bool, optional
        If true will display the contact map, by default False
    `title` : str, optional
        Title for cmap plot, by default "Residue Contact Map"

    Returns
    -------
    Tuple[np.array, str]
        residue contact map as a matrix

    Raises
    ------
    KeyError
        KeyError if a non-glycine residue is missing CB atom.
    """
    
    # getting coords from residues
    coords = []
    if CA_only:
        for res in residues.values():
            coords.append(res["CA"])
    else:
        for code in residues:
            res = residues[code]
            try:
                coords.append(res["CB"] if res["name"] != "GLY" else res["CA"])
            except KeyError as e:
                # CB missing in residue that is not GLY
                if check_missing:
                    print(e)
                    print(code)
                    print(res)
                    raise e
                else:
                    coords.append(res['CA'])
                
    
    # Main loop to calc matrix
    m = np.zeros((len(coords), len(coords)), np.float32)
    for row, r1 in enumerate(coords):
        for col, r2 in enumerate(coords):
            # lower triangle is all we need
            if col >= row: break
            m[row, col] = np.sqrt(np.sum((r1-r2)**2)) # L2 distance
            # duplicating for visual purposes
            m[col, row] = m[row, col]
    
    if display:
        plt.imshow(m)
        plt.title(title)
        plt.show()
        
    return m

def create_save_cmaps(pdbcodes: Iterable[str], 
                      pdb_p: Callable[[str], str],
                      cmap_p: Callable[[str], str],
                      CA_only=False,
                      check_missing=False) -> dict:
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
    `CA_only` : bool, optional
        If true then we use CA as point for distance measure, by default False
    `check_missing` : bool, optional
        Dont allow any missing residues, by default False

    Returns
    -------
    dict
        dictionary of sequences for each pdbcode
    """
    seqs = {}
    for pdbcode in tqdm(pdbcodes, 'Getting Sequences + Contact Maps'):
        seqs[pdbcode], res = get_sequence(pdb_p(pdbcode), 
                                check_missing=check_missing, 
                                select_largest=True)
        # only get cmap if it doesnt exist
        if not os.path.isfile(cmap_p(pdbcode)):
            cmap = get_contact(res,
                            CA_only=CA_only, # CB is needed by DGraphDTA
                            check_missing=check_missing)
            np.save(cmap_p(pdbcode), cmap)
        
    return seqs


def create_aln_files(df_seq: pd.DataFrame, aln_p: Callable[[str], str]):
    """
    Creates alignment files for all PDBbind structures.
    
    df_seq: dataframe with index as PDBcodes and column 'prot_seq' as sequence
    """
    raise NotImplementedError("This function is not complete (see yumika)")

if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    PDBbind = '/cluster/projects/kumargroup/jean/data/refined-set'
    path = lambda c: f'{PDBbind}/{c}/{c}_protein.pdb'
    cmap_p = lambda c: f'{PDBbind}/{c}/{c}_contact_CA.npy'

    # %% main loop to create and save contact maps
    cmaps = {}
    for code in tqdm(os.listdir(PDBbind)[32:100]):
        if os.path.isdir(os.path.join(PDBbind, code)) and code not in ["index", "readme"]:
            try:
                cmap, _ = get_contact(path(code), CA_only=True)
                cmaps[code] = cmap
            except Exception as e:
                print(code)
                raise e
            # np.save(cmap_p(code), cmap)
            
    #%% Displaying:
    r,c = 10,10
    f, ax = plt.subplots(r,c, figsize=(15, 15))
    i=0
    threshold = None
    for i, code in enumerate(cmaps.keys()):
        cmap = cmaps[code] if threshold is None else cmaps[code] < threshold
        ax[i//c][i%c].imshow(cmap)
        ax[i//c][i%c].set_title(code)
        ax[i//c][i%c].axis('off')
        i+=1
        
        
    #%% Create and save just sequences:
    df_x = pd.read_csv('data/PDBbind/kd_ki/X.csv', index_col=0) 
    df_seq = pd.DataFrame(index=df_x.index, columns=['seq'])
    for code in tqdm(df_x.index, 'Getting experimental sequences'):
        seq, _ = get_sequence(path(code), check_missing=False, raw=False, select_largest=True)
        df_seq.loc[code]['seq'] = seq
        
    # save sequences
    df_seq.to_csv('data/PDBbind/kd_ki/pdb_seq_lrgst.csv')
