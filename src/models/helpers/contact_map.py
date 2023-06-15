import numpy as np
import matplotlib.pyplot as plt

def get_contact(pdb_file: str, display=False, title="Residue Contact Map") -> np.array:
    """
    Given a pdb file path this will return the residue contact map for that structure.
    
    Examples: 1a1e has multiple main structures and so it creates a quadrant, 
    1o0n is a single structure and doesnt create quadrants

    Args:
        pdb_file (str): path to .pdb file to process.
        display (bool, optional): if true will display contact map. Defaults to False.

    Returns:
        np.array: residue contact map as a matrix.
    """

    # read and filter
    with open(pdb_file, 'r') as f:
        lines = f.readlines()
        coords = []
        for line in lines:
            if (line[:6].strip() == 'ATOM' and 
                line[12:16].strip()=='CA'): # getting alpha carbons only
                #                           x                   y                   z
                coords.append(np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]))
            # elif l[:3] == 'TER': #NOTE: add this to split structures with multiple complexes (e.g.: 1a1e)
            #     break
            
    
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