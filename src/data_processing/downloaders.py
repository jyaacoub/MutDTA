from typing import Iterable, List
from tqdm import tqdm
import os
import requests as r

class Downloader:
    @staticmethod
    def download(IDs: Iterable[str], 
                 save_path=lambda x:'./data/structures/ligands/{x}.sdf', 
                 url=lambda x: f'https://files.rcsb.org/ligands/download/{x}_ideal.sdf') -> dict:
        """
        Generalized download function for downloading any file type from any site.
        
        URL and save_path are passed in as callable functions which accept a string (the ID)
        and return a url or save path for that file.

        Parameters
        ----------
        `IDs` : Iterable[str]
            List of IDs to download
        `save_path` : Callable[str,], optional
            Callable fn that returns the save path for file, by default 
            lambda x :'./data/structures/ligands/{x}.sdf'
        `url` : Callable[str,], optional
            Callable fn that returns the url to download file, by default 
            lambda x :f'https://files.rcsb.org/ligands/download/{x}_ideal.sdf'
            
        Returns
        -------
        dict
            status of each ID (whether it was downloaded or not)        
        """
        
        ID_status = {}
        for name in tqdm(IDs, 'Downloading files'):
            if name in ID_status: continue
            # checking to make sure that we didnt already download file
            if os.path.isfile(save_path(name)): 
                ID_status[name] = 'already downloaded'
                continue
                
            os.makedirs(save_path(name)[:-len(name)])
            with open(save_path(name), 'w') as f:
                f.write(r.get(url(name)).text)
            ID_status[name] = 'downloaded'
            
        return ID_status
    
    @staticmethod
    def download_PDBs(PDBCodes: List[str], save_dir='./') -> dict:
        """
        Wrapper of `Downloader.download` for downloading PDB files.
        Fetches PDB files from https://files.rcsb.org/download/{PDBCode}.pdb.           
        """
        save_path = lambda x: f'{save_dir}/{x}.pdb',
        url = lambda x: f'https://files.rcsb.org/download/{x}.pdb'
        
        return Downloader.download(PDBCodes,save_path=save_path, url=url)
    
    @staticmethod
    def download_SDFs(ligand_names: List[str], 
                      save_dir='./data/structures/ligands/') -> dict:
        """
        Wrapper of `Downloader.download` for downloading SDF files. 
        Fetches SDF files from
        https://files.rcsb.org/ligands/download/{ligand_name}_ideal.sdf.
        """
        save_path = lambda x: f'{save_dir}/{x}.sdf'
        url = lambda x: f'https://files.rcsb.org/ligands/download/{x}_ideal.sdf'
        
        return Downloader.download(ligand_names, save_path=save_path, url=url)
    

if __name__ == '__main__':
    # downloading pdbs from X.csv list
    import pandas as pd
    import json

    X_path = './data/PDBbind/kd_ki/X.csv'
    X = pd.read_csv(X_path)

    codes = X['PDBCode'].values

    downloaded_codes = Downloader.download_PDBs(codes[:10], './data/structures')

    with open('./data/downloaded_codes.json', 'w') as f:
        json.dump(downloaded_codes, f)