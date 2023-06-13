#%%
import pandas as pd
import numpy as np
from io import StringIO
from Bio.PDB import PDBParser, Structure as BioStructure
from src.models.helpers.contact_map import get_contact
import matplotlib.pyplot as plt

# biopy code from: https://warwick.ac.uk/fac/sci/moac/people/students/peter_cock/python/protein_contact_map/
def calc_residue_dist(residue_one, residue_two):
    """Returns the C-alpha distance between two residues"""
    diff_vector  = residue_one["CA"].coord - residue_two["CA"].coord
    return np.sqrt(np.sum(diff_vector * diff_vector)) # this is the L2 (euclidian) distance 

def calc_dist_matrix(chain_one: BioStructure, chain_two: BioStructure):
    """Returns a matrix of C-alpha distances between two chains"""
    answer = np.zeros((len(chain_one), len(chain_two)), np.float32)
    for row, residue_one in enumerate(chain_one):
        for col, residue_two in enumerate(chain_two):
            answer[row, col] = calc_residue_dist(residue_one, residue_two)
    return answer

#%% 1a1e has multiple main structures and so it creates a quadrant 
# 1o0n is a single structure and doesnt create quadrants 
# code = "1o0n" #"1a1e"
for code in ["1o0n", "1a1e"]:
    path = f'/cluster/projects/kumargroup/jean/data/refined-set/{code}/{code}_protein.pdb'

    with open(path, 'r') as f:
        lines = f.readlines()
        clean = [l for l in lines if l[:4] == 'ATOM']
        
        # Getting structure
        structure = PDBParser().get_structure(code, StringIO(''.join(clean)))
        res = [r for r in structure.get_residues()]

    # Calc matrix
    m = calc_dist_matrix(res, res) # distance based on alpha carbon
    plt.imshow(m)
    plt.title(code)
    plt.show()


# #%% Filtering out database to get only PDBs that didnt have errors
# err = pd.read_csv('/cluster/home/t122995uhn/projects/MutDTA/src/docking/bash_scripts/pdb_error.txt',names=['PDBCode'], index_col=0)
# full = pd.read_csv('/cluster/home/t122995uhn/projects/MutDTA/data/PDBbind/kd_ki/info.csv', index_col=0)

# full_m = full.merge(err, how='left', on='PDBCode', indicator=True) # indicator gives _merge col
# full_m[full_m._merge == 'left_only']
# %%
