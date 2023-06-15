import numpy as np
import matplotlib.pyplot as plt

def get_contact(pdb_file: str, CA_only=True, display=False, title="Residue Contact Map") -> np.array:
    """
    Given a pdb file path this will return the residue contact map for that structure.
    
    Examples: 1a1e has multiple main structures and so it creates a quadrant, 
    1o0n is a single structure and doesnt create quadrants

    Args:
        pdb_file (str): path to .pdb file to process.
        CA_only (bool, optional): if true only use alpha carbon for calc distance. Otherwise 
                                follow DGraphDTA definition, using CB for all except glycine.
        display (bool, optional): if true will display contact map. Defaults to False.

    Returns:
        np.array: residue contact map as a matrix.
    """

    # read and filter
    with open(pdb_file, 'r') as f:
        lines = f.readlines()
        coords = []
        if CA_only:
            for line in lines:
                if (line[:6].strip() == 'ATOM' and 
                    line[12:16].strip()=='CA'): # getting alpha carbons only
                    #                           x                   y                   z
                    coords.append(np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]))
        else:
            next_res = None # res# number
            gly_CA = [] # buffer to save CA values in case of glycine 
            for line in lines:
                # reset res counter at TER
                if (line[:6].strip() == 'TER'): next_res = None
                if (line[:6].strip() != 'ATOM'): continue
                
                curr_res = int(line[22:26])
                assert curr_res <= next_res, f"Missing residue #{next_res}"
                if ((line[12:16].strip() == 'CB') and 
                    (next_res is None or curr_res == next_res)):
                    next_res = curr_res + 1
                    #                           x                   y                   z
                    coords.append(np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]))
                #TODO: add exception for glycine using buffer or if statment to check if it is glycine 
                # if line[17:20] =="GLY"
                    
                    
            
            
    
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