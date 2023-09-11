from collections import Counter, OrderedDict
import json, pickle, re, os, abc
import tarfile
from typing import Iterable
import requests
import urllib.request

import torch
from torch.utils import data
import torch_geometric as torchg
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.residue import Chain, ResInfo
from src.feature_extraction.ligand import smile_to_graph
from src.feature_extraction.protein import create_save_cmaps, get_contact_map, target_to_graph, get_target_edge_weights
from src.feature_extraction.process_msa import check_aln_lines
from src.data_processing.processors import PDBbindProcessor
from src.utils import config as cfg

# See: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
# for details on how to create a dataset
class BaseDataset(torchg.data.InMemoryDataset, abc.ABC):
    FEATURE_OPTIONS = cfg.PRO_FEAT_OPT
    EDGE_OPTIONS = cfg.EDGE_OPT
    
    def __init__(self, save_root:str, data_root:str, aln_dir:str,
                 cmap_threshold:float, feature_opt='nomsa',
                 edge_opt='binary',
                 subset=None, 
                 overwrite=False, *args, **kwargs):
        """
        Base class for datasets. This class is used to create datasets for 
        graph models. Subclasses only need to define the `pre_process` method
        and raw_file_names property. The `pre_process` method is used to create
        an XY.csv file that contains the binding data and the sequences for 
        proteins and ligands.

        Parameters
        ----------
        `save_root` : str, optional
            Path to processed dir, by default '../data/DavisKibaDataset/'
        `data_root` : str, optional
            Path to raw data files, by default '../data/davis_kiba/davis/'
        `aln_dir` : str, optional
            Path to sequence alignment directory with files of the name 
            '{code}_cleaned.a3m'. If set to None then no PSSM calculation is 
            done and is set to zeros, by default '../data/davis_kiba/davis/aln/'.
        `cmap_threshold` : float, optional
            Threshold for contact map creation, DGraphDTA use probability based 
            cmaps so we use negative to indicate this. (see `feature_extraction.protein.
            target_to_graph` for details), by default -0.5.
        `feature_opt` : str, optional
            Choose from ['nomsa', 'msa', 'shannon']
        `edge_opt` : str, optional
            Choose from ['simple', 'binary', 'anm', 'af2']
        `subset` : str, optional
            If you want to name this dataset or load an existing version of this dataset 
            that is under a different name. For distributed training this is useful since 
            you can save subsets and load only those samples for DDP leaving the DDP 
            implementation untouched, by default 'full'.
            
            
        *args and **kwargs sent to superclass `torch_geometric.data.InMemoryDataset`.
        """
        subset = subset or 'full'
        if subset != 'full':
            data_p = os.path.join(save_root, subset)
            assert os.path.isdir(data_p), f"{data_p} Subset does not exist,"+\
                "please create subset before initialization."
        self.subset = subset
        
        self.data_root = data_root
        self.cmap_threshold = cmap_threshold
        
        self.shannon = False
        
        assert feature_opt in self.FEATURE_OPTIONS, \
            f"Invalid feature_opt '{feature_opt}', choose from {self.FEATURE_OPTIONS}"
        if feature_opt == 'nomsa':
            self.aln_dir = None # none treats it as np.zeros
        elif feature_opt == 'msa':
            self.aln_dir =  aln_dir # path to sequence alignments
        elif feature_opt == 'shannon':
            self.aln_dir = aln_dir
            self.shannon = True
            
        assert edge_opt in self.EDGE_OPTIONS, \
            f"Invalid edge_opt '{edge_opt}', choose from {self.EDGE_OPTIONS}"
        self.edge_opt = edge_opt
        self.overwrite = overwrite
        
        super(BaseDataset, self).__init__(save_root, *args, **kwargs)
        self.load()
    
    @abc.abstractmethod
    def pdb_p(self, code):
        """path to pdbfile for a particular protein"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def cmap_p(self, code):
        raise NotImplementedError
    
    @abc.abstractmethod
    def aln_p(self, code):
        # path to cleaned input alignment file
        raise NotImplementedError
    
    @property
    def raw_dir(self) -> str:
        return self.data_root
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.subset)
    
    @property
    def processed_file_names(self):
        # XY.csv cols: PDBCode,pkd,SMILE,prot_seq
        # XY is created in pre_process
        return ['XY.csv','data_pro.pt','data_mol.pt']
    
    @property
    def index(self):
        return self._indices
    
    @property
    def ligands(self):
        # returns unique ligands in dataset
        return self.df['SMILE'].unique()
    
    @property
    def proteins(self):
        # returns unique proteins in dataset
        return self.df['prot_id'].unique()
    
    def get_protein_counts(self) -> Counter:
        # returns dict of protein counts
        return Counter(self.df['prot_id'])
        
    def load(self):
        self.df = pd.read_csv(self.processed_paths[0], index_col=0)
        self._indices = self.df.index
        self._data_pro = torch.load(self.processed_paths[1])
        self._data_mol = torch.load(self.processed_paths[2])
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx) -> dict:
        row = self.df.iloc[idx]
        code = row.name
        prot_id = row['prot_id']
        lig_seq = row['SMILE']
        
        return {'code': code, 'prot_id': prot_id, 
                'y': row['pkd'],
                'protein': self._data_pro[prot_id],
                'ligand': self._data_mol[lig_seq]}
        
    def save_subset(self, idxs:Iterable[int]|data.Sampler|data.DataLoader, 
                    subset_name:str)->str:
        """Saves a subset of the dataset that can be loaded up as its own seperate dataset"""
        if issubclass(idxs.__class__, data.DataLoader):
            idxs = idxs.sampler
        if issubclass(idxs.__class__, data.Sampler):
            idxs = idxs.indices
        
        # getting subset df, prots, and ligs
        sub_df = self.df.iloc[idxs]
        sub_prots = {k:self._data_pro[k] for k in sub_df['prot_id']}
        sub_lig = {k:self._data_mol[k] for k in sub_df['SMILE']}
        
        # saving to new dir
        path = os.path.join(self.root, subset_name)
        os.makedirs(path, exist_ok=True)
        sub_df.to_csv(os.path.join(path, self.processed_file_names[0]))
        torch.save(sub_prots, os.path.join(path, self.processed_file_names[1]))
        torch.save(sub_lig, os.path.join(path, self.processed_file_names[2]))
        return path
    
    def load_subset(self, subset_name:str):
        path = os.path.join(self.root, subset_name)
        
        # checking if processed files exist
        assert os.path.isdir(path), f"Subset {subset_name} does not exist!"
        
        for f in self.processed_file_names:
            new_fp = os.path.join(path, f)
            assert os.path.isfile(new_fp), f"Missing processed file: {f}!"
        
        # all checks successfully passed, now we can load up the subset:
        self.subset = subset_name
        self.load()
               
    def process(self):
        """
        This method is used to create the processed data files after feature extraction.
        
        Note about protein and ligand duplicates:
        - We create graphs using the pdb/sdf file from the first instance of that prot_id/smile in the csv
          all future instances will just reference back to that in `self.__getitem__`
        """
        if not os.path.isfile(self.processed_paths[0]):
            self.df = self.pre_process()
        else:
            self.df = pd.read_csv(self.processed_paths[0], index_col=0)
            print(f'{self.processed_paths[0]} file found, using it to create the dataset')
        print(f'Number of codes: {len(self.df)}')
        
        ###### Get Protein Graphs ######
        processed_prots = {}
        
        unique_pro = self.df[['prot_id']].drop_duplicates()
        unique_df = self.df.loc[unique_pro.index]
        for code, (prot_id, pro_seq) in tqdm(
                        unique_df[['prot_id', 'prot_seq']].iterrows(), 
                        desc='Creating protein graphs',
                        total=len(unique_df)):
            pro_feat = torch.Tensor() # for adding additional features
            # extra_feat is Lx54 or Lx34 (if shannon=True)
            try:
                extra_feat, pro_edge = target_to_graph(pro_seq, np.load(self.cmap_p(code)),
                                                            threshold=self.cmap_threshold,
                                                            aln_file=self.aln_p(code),
                                                            shannon=self.shannon)
            except AssertionError as e:
                raise Exception(f"error on protein creation for code {code}") from e
            pro_edge_weight = get_target_edge_weights(pro_edge, self.pdb_p(code), 
                                                      pro_seq, n_modes=10, n_cpu=2,
                                                      edge_opt=self.edge_opt)
            
            pro_feat = torch.cat((pro_feat, torch.Tensor(extra_feat)), axis=1)
            
            if pro_edge_weight is None:
                pro = torchg.data.Data(x=torch.Tensor(pro_feat),
                                    edge_index=torch.LongTensor(pro_edge),
                                    pro_seq=pro_seq, # protein sequence for downstream esm model
                                    prot_id=prot_id)
            else:
                pro = torchg.data.Data(x=torch.Tensor(pro_feat),
                                    edge_index=torch.LongTensor(pro_edge),
                                    edge_weight=torch.Tensor(pro_edge_weight),
                                    pro_seq=pro_seq, # protein sequence for downstream esm model
                                    prot_id=prot_id)
            processed_prots[prot_id] = pro
        
        ###### Get Ligand Graphs ######
        processed_ligs = {}
        errors = []
        for lig_seq in tqdm(self.df['SMILE'].unique(), 
                            desc='Creating ligand graphs'):
            if lig_seq not in processed_ligs:
                try:
                    mol_feat, mol_edge = smile_to_graph(lig_seq)
                except ValueError:
                    errors.append(f'L-{lig_seq}')
                    continue
                
                lig = torchg.data.Data(x=torch.Tensor(mol_feat),
                                    edge_index=torch.LongTensor(mol_edge),
                                    lig_seq=lig_seq)
                processed_ligs[lig_seq] = lig
            
        print(f'{len(errors)} codes failed to create graphs')
        
        ###### Save ######
        print('Saving...')
        torch.save(processed_prots, self.processed_paths[1])
        torch.save(processed_ligs, self.processed_paths[2])


class PDBbindDataset(BaseDataset): # InMemoryDataset is used if the dataset is small and can fit in CPU memory
    def __init__(self, save_root='../data/PDBbindDataset/nomsa', 
                 data_root='../data/v2020-other-PL', 
                 aln_dir=None,
                 cmap_threshold=8.0, feature_opt='nomsa', *args, **kwargs):
        """
        Subclass of `torch_geometric.data.InMemoryDataset`.
        Dataset for PDBbind data. This dataset is used to train graph models.

        Parameters
        ----------
        `save_root` : str, optional
            Path to processed dir, by default '../data/pytorch_PDBbind/'
        `bind_root` : str, optional
            Path to raw pdbbind files, by default '../data/v2020-other-PL'
        `aln_dir` : str, optional
            Path to sequence alignment directory with files of the name 
            '{code}_cleaned.a3m'. If set to None then no PSSM calculation is 
            done and is set to zeros, by default None.
        `cmap_threshold` : float, optional
            Threshold for contact map creation, by default 8.0
        `feature_opt` : bool, optional
            choose from ['nomsa', 'msa', 'shannon']
            
        *args and **kwargs sent to superclass `src.data_processing.datasets.BaseDataset`.
        """   
        super(PDBbindDataset, self).__init__(save_root, data_root=data_root,
                                             aln_dir=aln_dir, cmap_threshold=cmap_threshold,
                                             feature_opt=feature_opt, *args, **kwargs)
    
    def pdb_p(self, code):
        return os.path.join(self.data_root, code, f'{code}_protein.pdb')
    
    def cmap_p(self, code):
        return os.path.join(self.data_root, code, f'{code}.npy')
    
    def aln_p(self, code):
        # see feature_extraction/process_msa.py for details on how the alignments are cleaned
        if self.aln_dir is None:
            # dont use aln if none provided (will set to zeros)
            return None
        return os.path.join(self.aln_dir, f'{code}_cleaned.a3m')
        
    # for data augmentation override the transform method
    @property
    def raw_file_names(self):
        """
        Index files from pdbbind are needed for this dataset:
        
            "INDEX_general_PL_data.2020": List of the "general set" of protein-small 
            ligand complexes with formatted binding data.  

            "INDEX_general_PL_name.2020": List of the "general set" of protein-small 
            ligand complexes with protein names and UniProt IDs.
            
        Paths are accessed via self.raw_paths attribute, which adds the raw_dir to the
        file names. For example:
            self.raw_paths[0] = self.raw_dir + '/index/INDEX_general_PL_data.2020'

        Returns
        -------
        List[str]
            List of file names.
        """
        return ['index/INDEX_general_PL_data.2020', 
                'index/INDEX_general_PL_name.2020']
        
    def pre_process(self):
        """
        This method is used to create the processed data files for feature extraction.
        
        It creates a XY.csv file that contains the binding data and the sequences for 
        both ligand and proteins. with the following columns: 
        code,SMILE,pkd,prot_seq,prot_id
        
        It also generates and saves contact maps for each protein in the dataset.

        Returns
        -------
        pd.DataFrame
            The XY.csv dataframe.
        """
        ############# Read index files to get binding and protID data #############
        # Get binding data:
        df_binding = PDBbindProcessor.get_binding_data(self.raw_paths[0]) # _data.2020
        df_binding.drop(columns=['resolution', 'release_year', 'lig_name'], inplace=True)
        
        # Get prot ids data:
        df_pid = PDBbindProcessor.get_name_data(self.raw_paths[1]) # _name.2020
        df_pid.drop(columns=['release_year','prot_name'], inplace=True)
        # contains col: prot_id
        # some might not have prot_ids available so we need to use PDBcode as id instead
        missing_pid = df_pid.prot_id == '------'
        df_pid[missing_pid] = df_pid[missing_pid].assign(prot_id = df_pid[missing_pid].index)
        
        ############# get pdb codes based on data root dir #############
        pdb_codes = os.listdir(self.data_root)
        # filter out readme and index folders
        pdb_codes = [p for p in pdb_codes if p != 'index' and p != 'readme']
        
        ############# creating MSA: #############
        #NOTE: assuming MSAs are already created, since this would take a long time to do.
        # create_aln_files(df_seq, self.aln_p)
        if self.aln_dir is not None:
            valid_codes =  [c for c in pdb_codes if os.path.isfile(self.aln_p(c))]
            # filters out those that do not have aln file
            print(f'Number of codes with aln files: {len(valid_codes)} out of {len(pdb_codes)}')
            pdb_codes = valid_codes
        
        assert len(pdb_codes) > 0, 'Too few PDBCodes, need at least 1...'
            
        
        ############# Getting protein seq & contact maps: #############
        #TODO: replace this to save cmaps by protID instead
        seqs = create_save_cmaps(pdb_codes,
                          pdb_p=self.pdb_p,
                          cmap_p=self.cmap_p,
                          overwrite=self.overwrite)
        
        assert len(seqs) == len(pdb_codes), 'Some codes failed to create contact maps'
        df_seq = pd.DataFrame.from_dict(seqs, orient='index', columns=['prot_seq'])
        df_seq.index.name = 'PDBCode'
        
        ############## Get ligand info #############
        # Extracting SMILE strings:
        dict_smi = PDBbindProcessor.get_SMILE(pdb_codes,
                                              dir=lambda x: f'{self.data_root}/{x}/{x}_ligand.sdf')
        df_smi = pd.DataFrame.from_dict(dict_smi, orient='index', columns=['SMILE'])
        df_smi.index.name = 'PDBCode'
        
        df_smi = df_smi[df_smi.SMILE.notna()]
        num_missing = len(pdb_codes) - len(df_smi)
        if  num_missing > 0:
            print(f'\t{num_missing} ligands failed to get SMILEs')
            pdb_codes = list(df_smi.index)
        
        ############# MERGE #############
        df = df_pid.merge(df_binding, on='PDBCode') # pids + binding
        df = df.merge(df_smi, on='PDBCode') # + smiles
        df = df.merge(df_seq, on='PDBCode') # + prot_seq
        
        # changing index name to code (to match with super class):
        df.index.name = 'code'
        df.to_csv(self.processed_paths[0])
        return df

  
class DavisKibaDataset(BaseDataset):
    def __init__(self, save_root='../data/DavisKibaDataset/', 
                 data_root='../data/davis_kiba/davis/', 
                 aln_dir='../data/davis_kiba/davis/aln/',
                 cmap_threshold=-0.5, feature_opt='nomsa', *args, **kwargs):
        """
        InMemoryDataset for davis or kiba. This dataset is used to train graph models.

        Parameters
        ----------
        `save_root` : str, optional
            Path to processed dir, by default '../data/DavisKibaDataset/'
        `data_root` : str, optional
            Path to raw data files, by default '../data/davis_kiba/davis/'
        `aln_dir` : str, optional
            Path to sequence alignment directory with files of the name 
            '{code}_cleaned.a3m'. If set to None then no PSSM calculation is 
            done and is set to zeros, by default '../data/davis_kiba/davis/aln/'.
        `cmap_threshold` : float, optional
            Threshold for contact map creation, DGraphDTA use probability based 
            cmaps so we use negative to indicate this. (see `feature_extraction.protein.
            target_to_graph` for details), by default -0.5.
        `feature_opt` : bool, optional
            choose from ['nomsa', 'msa', 'shannon']
        *args and **kwargs sent to superclass `src.data_processing.datasets.BaseDataset`.
        """
        super(DavisKibaDataset, self).__init__(save_root, data_root=data_root,
                                               aln_dir=aln_dir, cmap_threshold=cmap_threshold,
                                               feature_opt=feature_opt, *args, **kwargs)
    # def pdb_p(self, code):
    #     return super().pdb_p(code) #TODO: download pdb structures using protID
    
    def cmap_p(self, code):
        return os.path.join(self.data_root, 'pconsc4', f'{code}.npy')
    
    def aln_p(self, code):
        # using existing cleaned alignment files
        # see feature_extraction/process_msa.py
        if self.aln_dir is None:
            # dont use aln if none provided (will set to zeros)
            return None
        return os.path.join(self.aln_dir, f'{code}.a3m')
    
    @property
    def raw_file_names(self):
        """
        Files needed for this dataset.
        
        `proteins.txt`: 
            dict of protein sequences with keys as codes.
        `ligands_can.txt`: 
            dict of ligand sequences with keys as ligand ID.
        `Y`:
            Binding affinity data with shape (lg, pr) where lg is the
            number of ligands and pr is the number of proteins.
        

        Returns
        -------
        List[str]
            List of file names.
        """
        return ['proteins.txt',
                'ligands_can.txt',
                'Y']
    
    def pre_process(self):
        """
        This method is used to create the processed data files for feature extraction.
        
        It creates a XY.csv file that contains the binding data and the sequences for 
        both ligand and proteins. with the following columns: 
        code,SMILE,pkd,prot_seq
        
        It also generates and saves contact maps for each protein in the dataset.

        Returns
        -------
        pd.DataFrame
            The XY.csv dataframe.
        """
        prot_seq = json.load(open(self.raw_paths[0], 'r'), object_hook=OrderedDict)
        codes = list(prot_seq.keys())
        prot_seq = list(prot_seq.values())
        
        # get ligand sequences (order is important since they are indexed by row in affinity matrix):
        ligand_seq = json.load(open(self.raw_paths[1], 'r'), 
                               object_hook=OrderedDict)
        ligand_seq = list(ligand_seq.values())
        
        # Get binding data:
        affinity_mat = pickle.load(open(self.raw_paths[2], 'rb'), encoding='latin1')
        lig_r, prot_c = np.where(~np.isnan(affinity_mat)) # index values corresponding to non-nan values
        
        # checking alignment files present for each code:
        no_aln = []
        if self.aln_dir is not None:
            no_aln = [c for c in codes if (not check_aln_lines(self.aln_p(c)))]
                    
            # filters out those that do not have aln file
            print(f'Number of codes with invalid aln files: {len(no_aln)} out of {len(codes)}')
            
        # Checking that contact maps are present for each code:
        #       (Created by psconsc4)
        no_cmap = [c for c in codes if not os.path.isfile(self.cmap_p(c))]
        print(f'Number of codes without cmap files: {len(no_cmap)} out of {len(codes)}')
        
        invalid_codes = set(no_aln + no_cmap)
        # filtering out invalid codes:
        lig_r = [r for i,r in enumerate(lig_r) if codes[prot_c[i]] not in invalid_codes]
        prot_c = [c for c in prot_c if codes[c] not in invalid_codes]
        
        # creating binding dataframe:
        #   code,SMILE,pkd,prot_seq
        df = pd.DataFrame({
           'code': [codes[c] for c in prot_c],
           'SMILE': [ligand_seq[r] for r in lig_r],
           'prot_seq': [prot_seq[c] for c in prot_c]
        })
        # adding binding data:
        if 'davis' in re.split(r'/+',self.data_root)[-2:]:
            print('davis dataset, taking -log10 of pkd')
            # davis affinity values are in nM, so we take -log10(*1e-9) to get pKd
            df['pkd'] = [-np.log10(y*1e-9) for y in affinity_mat[lig_r, prot_c]]
        else:
            df['pkd'] = affinity_mat[lig_r, prot_c]
            
        # adding prot_id column (in the case of davis and kiba datasets, the code is the prot_id)
        # Note: this means the code is not unique (unlike pdbbind)
        df['prot_id'] = df['code']
        df.set_index('code', inplace=True,
                     verify_integrity=False)
        
        df.to_csv(self.processed_paths[0])
        return df
    
    
class PlatinumDataset(BaseDataset):
    CSV_LINK = 'https://biosig.lab.uq.edu.au/platinum/static/platinum_flat_file.csv'
    PBD_LINK = 'https://biosig.lab.uq.edu.au/platinum/static/platinum_processed_pdb_files.tar.gz'
    def __init__(self, save_root: str, data_root: str, 
                 aln_dir: str=None, cmap_threshold: float=8.0, 
                 feature_opt='nomsa', mutated=True, *args, **kwargs):
        """
        Dataset class for the Platinum dataset.

        Parameters
        ----------
        `save_root` : str
            Where to save the processed data.
        `data_root` : str
            Where the raw data is stored from the Platinum dataset 
            (from: https://biosig.lab.uq.edu.au/platinum/).
        `aln_dir` : str, optional
            MSA alignments for each protein in the dataset, by default None
        `cmap_threshold` : float, optional
            Threshold for contact map creation, by default 8.0
        `feature_opt` : str, optional
            Choose from ['nomsa', 'msa', 'shannon'], by default 'nomsa'
        """
        options = ['nomsa']
        if feature_opt not in options:
            raise ValueError(f'Invalid feature_opt: {feature_opt}. '+\
                        f'Only {options} is currently supported for Platinum dataset.')
        
        # Platinum dataset is essentially two datasets in one
        # (one for mutations and one for wildtype) and is why we need to
        # specify mutations_only.
        self.mutated = mutated # only use mutated sequences and data
        if mutated:
            save_root = save_root + '_mut'
        
        if aln_dir is not None:
            print('WARNING: aln_dir is not used for Platinum dataset, no support for MSA alignments.')
        
        super().__init__(save_root, data_root, None, cmap_threshold, 
                         feature_opt, *args, **kwargs)
        
    def cmap_p(self, code):
        return os.path.join(self.raw_dir, 'contact_maps', f'{code}.npy')
    
    def aln_p(self, code):
        return None # no support for MSA alignments
        # raise NotImplementedError('Platinum dataset does not have MSA alignments.')
    
    @property
    def raw_file_names(self):
        """call by self.raw_paths to get the proper file paths"""
        # in order of download/creation:
        return ['platinum_flat_file.csv',# downloaded from website
                'platinum_pdb']
    
    def download(self):
        """Download the raw data for the dataset."""
        if not os.path.isfile(self.raw_paths[0]):
            print('CSV file not found, downloading from website...')
            urllib.request.urlretrieve(self.CSV_LINK, self.raw_paths[0])
        
        df = pd.read_csv(self.raw_paths[0])
        
        # Downloading pdb files:
        os.makedirs(self.raw_paths[1], exist_ok=True)
        print('Downloading pdb files from PLATINUM website...')
        try:
            temp_fp, msg  = urllib.request.urlretrieve(self.PBD_LINK)
            # check if successfully downloaded:
            if msg['Content-Type'] != 'application/x-tar':
                raise ValueError('Error downloading pdb files from PLATINUM website, '+\
                    'content type is not tar.gz:\n' + str(msg))
                # NOTE: alternate approach is to download directly from PDB using the pdb codes:
                # pdb_status = Downloader.download_PDBs(df['affin.pdb_id'], self.raw_paths[1])
                # print(f'Downloaded {len(pdb_status)} unique pdb files out of {len(df)}...')
            else:
                # extracting files:
                with tarfile.open(temp_fp, 'r:gz') as tar:
                    tar.extractall(self.raw_dir)
            os.remove(temp_fp)
        except requests.HTTPError as e:
            raise ValueError('Error downloading pdb files from PLATINUM website:\n' + str(e))
        
    def pre_process(self):
        """
        This method is used to create the processed data files for feature extraction.
        
        It creates a XY.csv file that contains the binding data and the sequences for 
        both ligand and proteins. with the following columns: 
        code,SMILE,pkd,prot_seq
        
        It generates and saves contact maps for each protein in the dataset.
        
        And also generates mutated sequences for each protein in the dataset.   

        Returns
        -------
        pd.DataFrame
            The XY.csv dataframe.
        """
        df_raw = pd.read_csv(self.raw_paths[0])
        os.makedirs(os.path.join(self.raw_dir, 'contact_maps'), exist_ok=True)
        
        # Getting sequences and cmaps:
        prot_seq = {}
        for i, row in tqdm(df_raw.iterrows(), 
                           desc='Getting sequences and cmaps',
                           total=len(df_raw)):
            mut = row['mutation']
            pdb = row['affin.pdb_id']
            t_chain = row['affin.chain']
            
            # Getting sequence from pdb file:
            pdb_fp = f'{self.raw_paths[1]}/{pdb}.pdb'
            chain = Chain(pdb_fp, t_chain=t_chain)
            
            # Getting and saving contact map:
            if not os.path.isfile(self.cmap_p(pdb)):
                cmap = get_contact_map(chain)
                np.save(self.cmap_p(i), cmap)
            
            mut_seq = chain.get_mutated_seq(chain, mut.split('/'), reversed=False)
            ref_seq = chain.getSequence()
            
            # getting mutated sequence:
            if self.mutated:
                prot_seq[i] = (pdb, mut_seq)
            else:
                prot_seq[i] = (pdb, ref_seq)
                
        df_seq = pd.DataFrame.from_dict(prot_seq, orient='index', 
                                        columns=['prot_id', 'prot_seq'])
        
        # NOTE: ligand sequences and binding data are already in the csv file
        # 'affin.lig_id', 'lig.canonical_smiles', 'affin.k_wt', 'affin.k_mt'
        if self.mutated:
            # parsing out '>' and '<' from binding data for pure numbers:
            df_binding = df_raw['affin.k_mt'].str.extract(r'(\d+\.*\d+)', 
                                                      expand=False).astype(float)
        else:
            df_binding = df_raw['affin.k_wt']
        
        # adjusting units for binding data from nM to pKd:
        df_binding = -np.log10(df_binding*1e-9)
        
        # merging dataframes:
        df = pd.DataFrame({
            'lig_id': df_raw['affin.lig_id'],
            'prot_id': df_seq['prot_id'],
            
            'pkd': df_binding,
            
            'prot_seq': df_seq['prot_seq'],
            'SMILE': df_raw['lig.canonical_smiles']
        }, index=df_raw.index)
        df.index.name = 'code'
        
        df.to_csv(self.processed_paths[0])
        return df