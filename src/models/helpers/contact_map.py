import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

def get_contact(pdb_file: str, CA_only=True, check_missing=False,
                display=False, title="Residue Contact Map") -> np.array:
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
        
    Returns:
        np.array: residue contact map as a matrix.
    """

    # read and filter
    with open(pdb_file, 'r') as f:
        lines = f.readlines()
        residues = OrderedDict() # residue dict
        
        ## read residues into res dict with the following format
        ## res = {ter#_res# : {CA: [x, y, z], CB: [x, y, z], name: resname},...}
        ter = 0 # prefix to indicate TER grouping
        curr_res, prev_res = None, None
        for line in lines:
            if (line[:6].strip() == 'TER'): # TER indicates new chain "terminator"
                ter += 1
            
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
            key = f"{ter}_{curr_res}_{icode}"
            assert atm_type not in residues.get(key, {}), f"Duplicate {atm_type} for residue {key} in {pdb_file}"
            # adding atom to residue
            residues.setdefault(key, {})[atm_type] = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])])
            
            # Saving residue name
            assert ("name" not in residues.get(key, {})) or \
                (residues[key]["name"] == line[17:20].strip()), \
                                        f"Inconsistent residue name for residue {key} in {pdb_file}"
            residues[key]["name"] = line[17:20].strip()
            
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