from typing import Callable, Iterable, Tuple
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from tqdm import tqdm

import pandas as pd

RES_CODE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
    'ASX': 'B', 'CYS': 'C', 'GLU': 'E', 'GLN': 'Q',
    
    'GLX': 'Z', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
    
    'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V'
}

def get_sequence(pdb_file: str, check_missing=False, raw=False, 
                 select_largest=True) -> Tuple[str, OrderedDict]:
    """
    Given a pdb file path this will return the residue sequence for that structure
    (could be missing residues) and the residue dict in order of seq# that contains coords.

    Args:
        pdb_file (str): path to .pdb file to process.
        check_missing (bool, optional): Adds check to ensure all residues are available. 
                                Defaults to False.
        raw (bool, optional): If True, no splitting is done to isolate a single structure 
                    and the contact map is produced for the entire pdb file. 
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
        seq += RES_CODE[return_chain[res]["name"]]
            
    return seq, return_chain


def get_contact(pdb_file: str, CA_only=True, check_missing=False,
                display=False, title="Residue Contact Map", 
                raw=False) -> Tuple[np.array, str]:
    """
    Given a pdb file path this will return the residue contact map for that structure.
    
    Examples: 1a1e has multiple main structures and so it creates a quadrant, 
    1o0n is a single structure and doesnt create quadrants

    Args:
        pdb_file (str): path to .pdb file to process.
        CA_only (bool, optional): if true only use alpha carbon for calc distance. Otherwise 
                                follow DGraphDTA definition, using CB for all except glycine.
        check_missing (bool, optional): Checking to ensure all residues are available. 
                                Defaults to False.
        display (bool, optional): if true will display contact map. Defaults to False.
        title (str, optional): title for plot. Defaults to "Residue Contact Map".
        raw (bool, optional): If True, no splitting is done to isolate a single structure 
                    and the contact map is produced for the entire pdb file. 
                    Defaults to False.
        
    Returns:
        Tuple[np.array, str]: residue contact map as a matrix and the sequence of residues.
    """
    # getting sequence and residue dict
    seq, residues = get_sequence(pdb_file, check_missing=check_missing, raw=raw)
            
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
        
    return m, seq

def create_save_cmaps(pdbcodes: Iterable[str], 
                      pdb_p: Callable[[str], str],
                      cmap_p: Callable[[str], str]):
    """
    Given a list of PDBcodes, this will create and save the contact maps for each.
    Example path callable functions:        
        pdb_p = lambda c: f'/path/to/refined-set/{c}/{c}_protein.pdb'
        cmap_p = lambda c: f'path/to/refined-set/{c}/{c}_contact_CB.npy'

    Args:
        pdbcodes (Iterable(str)): list of PDBcodes to create contact maps for.
        pdb_p (Callable[[str], str]): function to get pdb file path from pdbcode.
        cmap_p (Callable[[str], str]): function to get cmap save file path from pdbcode.
    """
    for pdbcode in tqdm(pdbcodes, 'Generating contact maps+saving'):
        cmap, _ = get_contact(pdb_p(pdbcode), # pdbcode is index
                        CA_only=False, # CB is needed by DGraphDTA
                        check_missing=False)
        np.save(cmap_p(pdbcode), cmap)
        


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
