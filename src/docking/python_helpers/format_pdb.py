from typing import List
from io import StringIO

import pandas as pd
import numpy as np

import os, re, argparse

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
            
        prot_df = get_coords(lrgst)
        protein_center = prot_df[['x', 'y', 'z']].mean().values
        
        
        # saving closest structure to new file as the ligand
        if len(structures) > 1:
            # finding closest ligand structure to protein center
            lig_structure = None
            lig_df = None
            lig_dist = np.inf
            for structure in structures:
                if structure == lrgst: continue
                curr_lig_df = get_coords(structure)
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

def get_structure_dict(lines:List[str]) -> dict:
    residues = {} # residue dict
    
    ## read residues into res dict with the following format
    ## res = {ter#_res# : {CA: [x, y, z], CB: [x, y, z], name: resname},...}
    ter = 0 # prefix to indicate TER grouping
    curr_res = None # res# number
    
    for line in lines:
        if (line[:6].strip() == 'TER'): # TER indicates new chain "terminator"
            ter += 1
        
        if (line[:6].strip() != 'ATOM'): continue
        
        # make sure res# is in order and not missing
        prev_res = curr_res
        curr_res = int(line[22:26])
        assert curr_res >= prev_res, f"Missing residue #{prev_res+1} OR out of order"
        
        # only want CA and CB atoms
        atm_type = line[12:16].strip()
        if atm_type not in ['CA', 'CB']: continue
        
        # Glycine has no CB atom, so we save both 
        key = f"{ter}_{curr_res}"
        assert atm_type not in residues.get(key, {}), f"Duplicate {atm_type} for residue {key}"
        # adding atom to residue
        residues.setdefault(key, {})[atm_type] = np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])])
        
        # Saving residue name
        assert ("name" not in residues.get(key, {})) or \
               (residues[key]["name"] == line[17:20].strip()), \
                                                f"Inconsistent residue name for residue {key}"
        residues[key]["name"] = line[17:20].strip()
    return residues


def get_atom_df(lines:List[str]) -> pd.DataFrame:
    """
    Same as `get_coords()` but with additional data (* below), however this is limited to 
    only "ATOM" records.
    
    See: http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html
        ATOM Format:
        COLUMNS        DATA  TYPE    FIELD        DEFINITION
        -------------------------------------------------------------------------------------
        1 -  6         Record name   "ATOM  "
        7 - 11         Integer       serial       Atom  serial number.
       *13 - 16        Atom          name         Atom name.
        17             Character     altLoc       Alternate location indicator.
        18 - 20        Residue name  resName      Residue name.
        22             Character     chainID      Chain identifier.
       *23 - 26        Integer       resSeq       Residue sequence number.
       *31 - 38        Real(8.3)     x            Orthogonal coordinates for X in Angstroms.
       *39 - 46        Real(8.3)     y            Orthogonal coordinates for Y in Angstroms.
       *47 - 54        Real(8.3)     z            Orthogonal coordinates for Z in Angstroms.
                        [...]
    """
    coords = []
    for line in lines:
        if (line[:6].strip() == 'ATOM'):
            #                    name           resSeq              
            coords.append([line[12:16].strip(), int(line[22:26]),
                           #            x                   y                   z
                           float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return pd.DataFrame(coords, columns=['name', 'res_num', 'x', 'y', 'z'])

def get_coords(lines:List[str]) -> pd.DataFrame:
    """
    Returns a numpy array of coordinates from a pdb file.
    See: http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html
    
        ATOM Format:
        COLUMNS        DATA  TYPE    FIELD        DEFINITION
        -------------------------------------------------------------------------------------
       *1 -  6         Record name   "ATOM  "
        7 - 11         Integer       serial       Atom  serial number.
       *13 - 16        Atom          name         Atom name.
        17             Character     altLoc       Alternate location indicator.
       *23 - 26        Integer       resSeq       Residue sequence number.
       *31 - 38        Real(8.3)     x            Orthogonal coordinates for X in Angstroms.
       *39 - 46        Real(8.3)     y            Orthogonal coordinates for Y in Angstroms.
       *47 - 54        Real(8.3)     z            Orthogonal coordinates for Z in Angstroms.
                        [...]
        
        HETATM Format:
        COLUMNS       DATA  TYPE     FIELD         DEFINITION
        -----------------------------------------------------------------------
        1 - 6        Record name    "HETATM"
                        [...]
        31 - 38       Real(8.3)      x             Orthogonal coordinates for X.
        39 - 46       Real(8.3)      y             Orthogonal coordinates for Y.
        47 - 54       Real(8.3)      z             Orthogonal coordinates for Z.
                        [...]
    * Arguments we care about
    
    args:
        lines (List[str]): list of lines from pdb file.
        
    returns:
        pd.DataFrame: pandas DataFrame of pdb file.
    """
    coords = []
    for line in lines:
        if (line[:6].strip() == 'ATOM') or (line[:6].strip() == 'HETATM'):
            # coords           x                     y                   z
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return pd.DataFrame(coords, columns=['x', 'y', 'z'])

def clean_pdb(lines: List[str]): #NOTE: this is no longer needed and instead is just done in the PDBbind_prepare.sh script via 'grep ATOM *'
    """
    This file will clean PDB of "waters, ligands, cofactors, ions deemed unnecessary for the docking".
    According to https://autodock-vina.readthedocs.io/en/latest/docking_basic.html#preparing-the-receptor
    
    Mainly removing HETATM lines.
    """
    return [l for l in lines if l[:4] == 'ATOM']

if __name__ == "__main__": # calling from cli will split pdbqt
    parser = argparse.ArgumentParser(description='Split pdbqt file into protein and ligand')
    parser.add_argument('-r', metavar='--receptor', type=str, help='path to pdbqt file', 
                        required=True)
    parser.add_argument('-v', help='To validate and view protein and ligand',
                        required=False, default=False, action='store_true')

    parser.add_argument('-s', metavar='--save', type=str, 
                        help='What structures to save ("all", "mains", or "largest")',
                        required=False, default='mains')
    args = parser.parse_args()

    if args.s.lower() not in ['all', 'mains', 'largest']:
        parser.error(f'Invalid save option: {parser.parse_args().s}')

    r_path = '/'.join(args.r.split('/')[:-1])

    # making sure save path exists
    os.makedirs(r_path, exist_ok=True)
    # splitting pdbqt file
    split_structure(file_path=args.r, save=args.s.lower())
    

    # viewing protein and ligand if requested
    if args.v == 'y':
        dfP, dfL = None, None
        for f_name in os.listdir(r_path):
            if re.search(r'split[\w\-]*.pdbqt',f_name):
                if re.search('ligand',f_name):
                    dfL = get_atom_df(open(f'{r_path}/{f_name}','r').readlines()) 
                else:
                    dfP = get_atom_df(open(f'{r_path}/{f_name}','r').readlines()) 
                    
        if dfP is not None and dfL is not None:
            from data_analysis.plot_structures import plot_together
            plot_together(dfL,dfP)
        else:
            print('Split failed, missing protein or ligand file')
            exit(1)
