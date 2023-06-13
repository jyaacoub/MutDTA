import numpy as np
from io import StringIO
from Bio.PDB import PDBParser, Structure as BioStructure
import matplotlib.pyplot as plt

# biopy code from: https://warwick.ac.uk/fac/sci/moac/people/students/peter_cock/python/protein_contact_map/
def calc_dist_matrix(chain_one: BioStructure, chain_two: BioStructure):
    # TODO: instead of relying on biopython we can just get the coors of all the CA ATOMs (alpha carbons) 
    # and pass that into the calc_dist_matrix function directly as a list
    """Returns a matrix of C-alpha distances between two chains"""
    answer = np.zeros((len(chain_one), len(chain_two)), np.float32)
    for row, residue_one in enumerate(chain_one):
        for col, residue_two in enumerate(chain_two):
            # using alpha carbon as center point for distance measurment 
            c1, c2  = residue_one["CA"].coord,  residue_two["CA"].coord
            answer[row, col] = np.sqrt(np.sum((c1-c2)**2)) # L2 distance 
    return answer

def get_contact(pdb_file: str, display=False) -> np.array:
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

    with open(pdb_file, 'r') as f:
        lines = f.readlines()
        clean = []
        for l in lines:
            if l[:4] == 'ATOM':
                clean.append(l)
            # elif l[:3] == 'TER': #NOTE: add this to split structures with multiple complexes (e.g.: 1a1e)
            #     break
        
        # Getting structure
        structure = PDBParser().get_structure('', StringIO(''.join(clean)))
        res = [r for r in structure.get_residues()]

    # Calc matrix
    m = calc_dist_matrix(res, res) # distance based on alpha carbon
    if display:
        plt.imshow(m)
        plt.title("Residue Contact Map")
        plt.show()