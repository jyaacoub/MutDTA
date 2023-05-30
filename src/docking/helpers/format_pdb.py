from typing import List
from io import StringIO

import pandas as pd
import numpy as np

def split_structure(file_path='sample_data/1a1e.pdbqt', save='all') -> List[str]:
    """
    Splits a pdbqt file if duplicate protein structures are present and saves the 
    largest structure to a new file under the same name postfixed by "-split.pdbqt".
    
        "TER" - determines the end of a protein structure as per the 
        pdb format.
        
    For more information on the pdb format see:
    http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#TER
    
    args:
        file_path (str): path to pdbqt file (processed PDB file using 
        AutoDockTools - prepare_receptor4.py).
        save (str): save option depending on needs:
            'all', saves all structures to new files with additional postfix "-split-<length>.pdbqt".
            'mains', saves main protein and ligand (if present) to new files.
            'largest', saves largest structure to new file.
    
    returns:
        List[str]: list of structures present in the pdbqt file.
    """
    assert save in ['all', 'mains', 'largest'], f'Invalid save option: {save}'
    extens = file_path.split('.')[-1]
    print('extracting structures from:', file_path)
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        structures = []
        i=0
        while i < len(lines):
            structure = []
            while lines[i][:3] != 'TER':
                structure.append(lines[i])
                i += 1
            structure.append(lines[i])
            structures.append(structure)
            i += 1
        
    # saving all structures to new files
    if save.lower() == 'all':
        saved_files = {}
        for structure in structures:
            # Making sure no files are overwritten by adding postfix number count
            fp = f'{file_path.split(".pdb")[0]}-split-{len(structure)}'
            postfix = f'-{len(structure)}_{saved_files.get(len(structure), 0)}.{extens}'
            
            with open(fp + postfix, 'w') as f:
                f.writelines(structure)
            saved_files[len(structure)] = saved_files.get(len(structure), 0) + 1
            
    elif save.lower() == 'mains':
        # saving main protein structure to new file
        lrgst=max(structures, key=len)
        fp = f'{file_path.split(".pdb")[0]}-split-{len(lrgst)}_receptor.{extens}'
        with open(fp, 'w') as f:
            f.writelines(lrgst)
        print('wrote main receptor file: ', fp)
            
        prot_df = get_df(lrgst)
        protein_center = prot_df[['x', 'y', 'z']].mean().values
        
        
        # saving closest structure to new file as the ligand
        if len(structures) > 1:
            # finding closest ligand structure to protein center
            lig_structure = None
            lig_df = None
            lig_dist = np.inf
            for structure in structures:
                if structure == lrgst: continue
                curr_lig_df = get_df(structure)
                curr_lig_center = curr_lig_df[['x', 'y', 'z']].mean().values
                
                curr_dist = np.linalg.norm(protein_center - curr_lig_center)
                if curr_dist < lig_dist:
                    lig_dist = curr_dist
                    lig_structure = structure
                    lig_df = curr_lig_df
                
            fp = f'{file_path.split(".pdb")[0]}-split-{len(structure)}_ligand.{extens}'
            with open(fp, 'w') as f:
                f.writelines(lig_structure)

            print('wrote ligand file: ', fp)
            
            # Saving binding pocket info in conf.txt file
            with open('/'.join(file_path.split('/')[:-1]) + '/conf.txt', 'w') as f:
                f.write(f"center_x = {protein_center[0]}\n"\
                        f"center_y = {protein_center[1]}\n"\
                        f"center_z = {protein_center[2]}\n"\
                        f"size_x = {(lig_df['x'].max() - lig_df['x'].min())/2 + 20}\n"\
                        f"size_y = {(lig_df['y'].max() - lig_df['y'].min())/2 + 20}\n"\
                        f"size_z = {(lig_df['z'].max() - lig_df['z'].min())/2 + 20}\n")
                
    
    elif save.lower() == 'largest':
        lrgst=max(structures, key=len)
        # saving largest structure to new file
        fp = f'{file_path.split(".pdb")[0]}-split-{len(structure)}.{extens}'
        with open(fp, 'w') as f:
            f.writelines(lrgst)
    
    return structures


def get_df(lines:List[str]) -> pd.DataFrame:
    """
    Returns a pandas DataFrame of pdb file.
    
    Columns (in order that they appear in the PDB format)
        'atom_name', 'count', 'atom_type', 'res_name', 'chain_name', 'res_num', 'x', 'y', 'z'
    
    Example of a line from PDB:
        ATOM      1  N   ILE A 146      57.904  24.527  16.458  *1.00  39.85     0.626 N
        everything after * is ignored.
    
    args:
        lines (List[str]): list of lines from pdb file.
    """
    cols = ['atom_name', 'count', 'atom_type', 
            'res_name', 'chain_name', 'res_num', 
            'x', 'y', 'z']
    strIO = StringIO(''.join(lines))
    df = pd.read_csv(strIO, sep=r'[ ]+', header=None, names=cols, 
                     usecols=cols, engine='python')
    df.dropna(inplace=True)
    # removing lines that are not atoms
    clean_df = df[(df['atom_name'] == 'ATOM') | (df['atom_name'] == 'HETATM')]
    # converting to float
    return clean_df.astype({'x':float, 'y':float, 'z':float})
