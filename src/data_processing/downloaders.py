from typing import Iterable, List
import os
import requests as r
from io import StringIO
from urllib.parse import quote

from tqdm import tqdm

class Downloader:
    @staticmethod
    def get_prot_seq(protIDs: List[str], 
                    url=lambda x: f'https://rest.uniprot.org/uniprotkb/{x}.fasta') -> dict:
        """
        Fetches FASTA files from given url and returns dict with {ID: seq}
        
        URL is passed in as a callable function which accepts a string (the protID) and returns a url 
        to download that file.
            e.g. for uniprot: lambda x: f'https://rest.uniprot.org/uniprotkb/{x}.fasta'    
        """
        prot_seq = {}
        for protID in tqdm(protIDs, 'Downloading protein sequences'):
            if protID in prot_seq: continue
            FASTA = r.get(url(protID)).text
            prot_seq[protID] = ''.join(FASTA.split('\n')[1:])
            
        return prot_seq
    
    @staticmethod
    def get_mutated_seq(HGVSc: List[str], 
                        url=lambda x: f'https://mutalyzer.nl/api/normalize/{x}') -> dict:
        """
        Fetches mutated sequences from given url and returns dict with {HGVSc: seq}
        
        URL is passed in as a callable function which accepts a string (the HGVSc) and returns a url 
        to download that file.
            e.g. for mutalyzer: lambda x: f'https://mutalyzer.nl/api/normalize/{x}'
            
        Returned json is of the form:
            {protein: {predicted: "<Mutated Seq>",…},…}   
            
        to get native (non-mutated) sequence, use "reference" instead of "predicted"
        """
        mut_seq = {}
        for mut in tqdm(HGVSc, 'Downloading mutated sequences'):
            if mut in mut_seq: continue
            mut_seq[mut] = r.get(url(mut)).json()['protein']['predicted']
            
        return mut_seq
    
    @staticmethod
    def get_SMILE(drug_names: Iterable[str],
                url=lambda x: f'https://cactus.nci.nih.gov/chemical/structure/{quote(x)}/smiles') -> dict:
        """Gets SMILE strings from given drug names and returns dict with {drug_name: SMILE}"""
        # wont work with some drug names and so its better to use download_sdf then parse sdf to get smile
        drug_SMILEs = {}
        for drug_name in tqdm(drug_names, 'Downloading SMILE strings'):
            if drug_name in drug_SMILEs: continue
            drug_SMILEs[drug_name] = r.get(url(drug_name)).text
        
        return drug_SMILEs

    @staticmethod
    def get_file_obj(ID: str, url=lambda x: f'https://files.rcsb.org/download/{x}.pdb') -> StringIO:
        """
        Returns a file object for the given ID after downloading it from the given url.

        Parameters
        ----------
        `ID` : str
            The ID of the file to download
        `url` : _type_, optional
            The url to download the file, by default lambdax:f'https://files.rcsb.org/download/{x}.pdb'

        Returns
        -------
        StringIO
            The file object.
        """
        return StringIO(r.get(url(ID)).text)
        
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
        `save_path` : Callable[[str], str], optional
            Callable fn that returns the save path for file, by default 
            lambda x :'./data/structures/ligands/{x}.sdf'
        `url` : Callable[[str], str], optional
            Callable fn that returns the url to download file, by default 
            lambda x :f'https://files.rcsb.org/ligands/download/{x}_ideal.sdf'
            
        Returns
        -------
        dict
            status of each ID (whether it was downloaded or not)        
        """
        
        ID_status = {}
        for id in tqdm(IDs, 'Downloading files'):
            if id in ID_status: continue
            fp = save_path(id)
            # checking to make sure that we didnt already download file
            if os.path.isfile(fp): 
                ID_status[id] = 'already downloaded'
                continue
                
            os.makedirs(os.path.dirname(fp), 
                        exist_ok=True)
            with open(fp, 'w') as f:
                resp = r.get(url(id))
                if resp.status_code >= 400: 
                    ID_status[id] = resp.status_code
                else:
                    ID_status[id] = 'downloaded'
                    f.write(resp.text)
        return ID_status
    
    @staticmethod
    def download_PDBs(PDBCodes: Iterable[str], save_dir='./') -> dict:
        """
        Wrapper of `Downloader.download` for downloading PDB files.
        Fetches PDB files from https://files.rcsb.org/download/{PDBCode}.pdb.           
        """
        save_path = lambda x: os.path.join(save_dir, f'{x}.pdb')
        url = lambda x: f'https://files.rcsb.org/download/{x}.pdb'
        return Downloader.download(PDBCodes, save_path=save_path, url=url)
    
    @staticmethod
    def download_predicted_PDBs(UniProtID: Iterable[str], save_dir='./') -> dict:
        """Downloads pdbs given uniprotIDs from alphafold predictions"""
        save_path = lambda x: os.path.join(save_dir, f'{x}.pdb')
        url = lambda x: f'https://alphafold.ebi.ac.uk/files/AF-{x}-F1-model_v4.pdb'
        return Downloader.download(UniProtID, save_path=save_path, url=url)
    
    @staticmethod
    def download_SDFs(ligand_names: List[str], 
                      save_dir='./data/structures/ligands/') -> dict:
        """
        Wrapper of `Downloader.download` for downloading SDF files. 
        Fetches SDF files from
        https://files.rcsb.org/ligands/download/{ligand_name}_ideal.sdf.
        """
        save_path = lambda x: os.path.join(save_dir, f'{x}.sdf')
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