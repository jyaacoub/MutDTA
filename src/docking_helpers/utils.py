from typing import List
from tqdm import tqdm
import os
import requests as r

def download_PDBs(PDBCodes: List[str], save_path='./',
                  url=lambda x: f'https://files.rcsb.org/download/{x}.pdb') -> dict:
    """
    Fetches pdb files from given url and returns set of successfully downloaded PDBs.
    
    URL is passed in as a callable function which accepts a string (the protID) and returns a url 
    to download that file.
    For pdbbind:
        lambda x: f'http://www.pdbbind.org.cn/v2007/{x}/{x}_complex.pdb'
    For pdb:
        lambda x: f'https://files.rcsb.org/download/{x}.pdb'
        
    """
    pdb_codes = {}
    for PDBcode in tqdm(PDBCodes, 'Downloading PDB complex'):
        if PDBcode in pdb_codes: continue
        fp = f'{save_path}/{PDBcode}/{PDBcode}.pdb'
        # checking to make sure that we didnt already download file
        if os.path.isfile(fp): 
            pdb_codes[PDBcode] = 'already downloaded'
            continue
            
        os.makedirs(f'{save_path}/{PDBcode}/')
        with open(fp, 'w') as f:
            f.write(r.get(url(PDBcode)).text)
        pdb_codes[PDBcode] = 'downloaded'
        
    return pdb_codes


def download_SDFs(ligand_names: List[str], save_path='./data/structures/ligands',
                  url=lambda x: f'https://files.rcsb.org/ligands/download/{x}_ideal.sdf') -> dict:
    """
    Fetches SDF files from PDB site and returns a dict of the successfully downloaded SDFs.
    
    URL is passed in as a callable function which accepts a string (the ligand name) and returns a url 
    to download that file.
    e.g.:
    For pdb:
        lambda x: f'https://files.rcsb.org/download/{x}.pdb'
        
    """
    lig_names = {}
    for name in tqdm(ligand_names, 'Downloading PDB complex'):
        if name in lig_names: continue
        fp = f'{save_path}/{name}/{name}.sdf'
        # checking to make sure that we didnt already download file
        if os.path.isfile(fp): 
            lig_names[name] = 'already downloaded'
            continue
            
        os.makedirs(f'{save_path}/{name}/')
        with open(fp, 'w') as f:
            f.write(r.get(url(name)).text)
        lig_names[name] = 'downloaded'
        
    return lig_names

if __name__ == '__main__':
    # downloading pdbs from X.csv list
    import pandas as pd
    import json

    X_path = './data/PDBbind/kd_ki/X.csv'
    X = pd.read_csv(X_path)

    codes = X['PDBCode'].values

    downloaded_codes = download_PDBs(codes[:10], './data/structures')

    with open('./data/downloaded_codes.json', 'w') as f:
        json.dump(downloaded_codes, f)