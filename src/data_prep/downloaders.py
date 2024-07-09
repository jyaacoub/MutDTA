from typing import Iterable, List, Callable
import os, time
import requests as r
from io import StringIO
from urllib.parse import quote

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

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
    def download_single_file(id: str, save_path: Callable[[str], str], url: Callable[[str], str], 
                             url_backup: Callable[[str], str], max_retries=3) -> tuple:
        """
        Helper function to download a single file.
        """
        fp = save_path(id)
        # Check if the file already exists
        if os.path.isfile(fp):
            return id, 'already downloaded'

        os.makedirs(os.path.dirname(fp), exist_ok=True)
        
        def fetch_url(url):
            retries = 0
            while retries <= max_retries:
                resp = r.get(url)
                if resp.status_code == 503:
                    wait_time = 2 ** retries  # Exponential backoff
                    time.sleep(wait_time)
                elif resp.status_code < 500:
                    return resp
                retries += 1
            return resp  # Return the last response after exhausting retries

        resp = fetch_url(url(id))
        if resp.status_code >= 400 and url_backup:
            logging.debug(f'{id}-{resp.status_code} {resp}')
            resp = fetch_url(url_backup(id))

        if resp.status_code >= 400:
            logging.debug(f'\tbkup{id}-{resp.status_code} {resp}')
            return id, resp.status_code
        else:
            with open(fp, 'w') as f:
                f.write(resp.text)
            return id, 'downloaded'    

    @staticmethod
    def download(IDs: Iterable[str], 
                save_path=lambda x: f'./data/structures/ligands/{x}.sdf', 
                url=lambda x: f'https://files.rcsb.org/ligands/download/{x}_ideal.sdf',
                tqdm_desc='Downloading files',
                url_backup=None, # for if the first url fails
                tqdm_disable=False,
                max_workers=None,
                **kwargs) -> dict:
        """
        Generalized multithreaded download function for downloading any file type from any site.
        
        Parameters
        ----------
        IDs : Iterable[str]
            List of IDs to download
        save_path : Callable[[str], str], optional
            Callable fn that returns the save path for file, by default 
            lambda x :'./data/structures/ligands/{x}.sdf'
        url : Callable[[str], str], optional
            Callable fn that returns the url to download file, by default 
            lambda x :f'https://files.rcsb.org/ligands/download/{x}_ideal.sdf'
        max_workers : int, optional
            Number of threads to use for downloading files.
        
        Returns
        -------
        dict
            status of each ID (whether it was downloaded or not)
        """
        ID_status = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(Downloader.download_single_file, id, save_path, 
                                       url, url_backup, **kwargs): id for id in IDs}
            for future in tqdm(as_completed(futures), desc=tqdm_desc, total=len(IDs), disable=tqdm_disable):
                id, status = future.result()
                ID_status[id] = status
        return ID_status
    
    @staticmethod
    def download_PDBs(PDBCodes: Iterable[str], save_dir='./', **kwargs) -> dict:
        """
        Wrapper of `Downloader.download` for downloading PDB files.
        Fetches PDB files from https://files.rcsb.org/download/{PDBCode}.pdb.           
        """
        save_path = lambda x: os.path.join(save_dir, f'{x}.pdb')
        url = lambda x: f'https://files.rcsb.org/download/{x}.pdb'
        return Downloader.download(PDBCodes, save_path=save_path, url=url, **kwargs)
    
    @staticmethod
    def download_predicted_PDBs(UniProtID: Iterable[str], save_dir='./') -> dict:
        """Downloads pdbs given uniprotIDs from alphafold predictions"""
        save_path = lambda x: os.path.join(save_dir, f'{x}.pdb')
        url = lambda x: f'https://alphafold.ebi.ac.uk/files/AF-{x}-F1-model_v4.pdb'
        return Downloader.download(UniProtID, save_path=save_path, url=url)
    
    @staticmethod
    def download_SDFs(ligand_ids: List[str],
                      save_dir='./data/structures/ligands/',
                      max_workers=None,
                      **kwargs) -> dict:
        """
        Wrapper of `Downloader.download` for downloading SDF files. 
        Fetches SDF files from 
        https://files.rcsb.org/ligands/download/{ligand_name}_ideal.sdf.
        
        Where ligand_id is either the CID, CHEMBL id, or simply the ligand name. Will look at the first 
        ligand_id in the list and determine which type it is.
        
        ## Different urls for different databases
        For CID we can use the following url (identifiable by the fact that it is a number)
            - https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/CID/11314340/record/SDF?record_type=3d
        
        For CHEMBL ids we can use the following url (identifiable by the fact that it starts with "CHEMBL")
            - https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/xref/registryID/CHEMBL390156/record/sdf?record_type=3d
        
        For ligand names we can use the following url 
            - https://files.rcsb.org/ligands/download/{ligand_name}_ideal.sdf. (e.g.: NLG_ideal.sdf)
            OR
            - https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{ligand_name}/record/SDF?record_type=3d
        """
        urls = {'CID': lambda x: f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/CID/{x}/record/SDF?record_type=3d',
                'CHEMBL': lambda x: f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/xref/registryID/{x}/record/sdf?record_type=3d',
                'name': lambda x: f'https://files.rcsb.org/ligands/download/{x}_ideal.sdf'}
        
        
        lid = ligand_ids[0]
        if lid.isdigit():
            url = urls['CID']
        elif lid.startswith('CHEMBL'):
            url = urls['CHEMBL']
        else:
            url = urls['name']
        
        save_path = lambda x: os.path.join(save_dir, f'{x}.sdf')        
        url_backup=lambda x: url(x).split('?')[0]  # fallback to 2d conformer structure
        return Downloader.download(ligand_ids, save_path=save_path, url=url, url_backup=url_backup,
                                   tqdm_desc='Downloading ligand sdfs', max_workers=max_workers, **kwargs)
        
    @staticmethod
    def download_pocket_seq(prot_ids: list[str], save_dir='./data/prot_pockets', max_worker=None, **kwargs) -> dict:
        """
        Fetches pocket sequences from the given protein IDs (Gene names or UniProt).
        """
        assert issubclass(type(prot_ids), list) and type(prot_ids[0]) == str, 'prot_ids should be a list of strings'
        url = lambda x: f'https://klifs.net/api/kinase_ID?kinase_name={x}&species=HUMAN'
        save_path = lambda x: os.path.join(save_dir, f'{x}.json')
        # os.makedirs(save_dir, exist_ok=True)
        return Downloader.download(prot_ids, save_path=save_path, url=url, max_workers=max_worker, **kwargs)

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