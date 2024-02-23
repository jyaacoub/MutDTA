from collections import Counter, OrderedDict
from glob import glob
import json, pickle, re, os, abc
import logging
import shutil
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

from src.utils import config as cfg
from src.utils.residue import Chain, Ring3Runner
from src.utils.exceptions import DatasetNotFound
from src.data_prep.feature_extraction.ligand import smile_to_graph
from src.data_prep.feature_extraction.protein import (multi_save_cmaps, 
                                                      multi_get_sequences, 
                                                      target_to_graph,)
from src.data_prep.feature_extraction.protein_edges import get_target_edge_weights
from src.data_prep.processors import PDBbindProcessor, Processor
from src.data_prep.downloaders import Downloader


# See: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
# for details on how to create a dataset
class BaseDataset(torchg.data.InMemoryDataset, abc.ABC):
    EDGE_OPTIONS = cfg.PRO_EDGE_OPT
    FEATURE_OPTIONS = cfg.PRO_FEAT_OPT
    LIGAND_EDGE_OPTIONS = cfg.LIG_EDGE_OPT
    LIGAND_FEATURE_OPTIONS = cfg.LIG_FEAT_OPT
    
    def __init__(self, save_root:str, data_root:str, aln_dir:str,
                 cmap_threshold:float, feature_opt='nomsa',
                 edge_opt='binary', 
                 af_conf_dir:str=None,
                 subset:str=None, 
                 overwrite=False, 
                 max_seq_len:int=None,
                 only_download=False,
                 ligand_feature:str='original', 
                 ligand_edge:str='binary',
                 verbose=False,
                 *args, **kwargs):
        """
        Base class for datasets. This class is used to create datasets for 
        graph models. Subclasses only need to define the `pre_process` method
        and raw_file_names property. The `pre_process` method is used to create
        an XY.csv file that contains the binding data and the sequences for 
        proteins and ligands.

        Parameters
        ----------
        `save_root` : str
            Path to processed dir, by default '../data/DavisKibaDataset/'
        `data_root` : str
            Path to raw data files, by default '../data/davis_kiba/davis/'
        `aln_dir` : str
            Path to sequence alignment directory with files of the name 
            '{code}_cleaned.a3m'. If set to None then no PSSM calculation is 
            done and is set to zeros, by default '../data/davis_kiba/davis/aln/'.
        `cmap_threshold` : float
            Threshold for contact map creation, DGraphDTA use probability based 
            cmaps so we use negative to indicate this. (see `feature_extraction.protein.
            target_to_graph` for details), by default -0.5.
        `feature_opt` : str, optional
            Choose from ['nomsa', 'msa', 'shannon']
        `edge_opt` : str, optional
            Choose from ['simple', 'binary', 'anm', 'af2']
        `af_conf_dir`: str, optional
            Path to parent dir of output dirs for af configs, by default None. Output dirs 
            must be names 'out?' where ? is the seed number given to localcolabfold. This 
            argument is the parent directory for these 'out?' dirs.
        `subset` : str, optional
            If you want to name this dataset or load an existing version of this dataset 
            that is under a different name. For distributed training this is useful since 
            you can save subsets and load only those samples for DDP leaving the DDP 
            implementation untouched, by default 'full'.
        `max_seq_len` : int, optional
            The max protein sequence length that your system is able to handle, 2149 is 
            the max sequence length from PDBbind, davis and kiba have sequence lengths 
            of 2500+ which wont run on a 32GB gpu with a batch size of 32. This is also 
            applied retroactively to existing datasets (see load fn), default is 2150. 
        `only_dowwnload` : bool, optional
            If you only want to download the raw files and not prepare the dataset set 
            this to true, by default False. 
            
        *args and **kwargs sent to superclass `torch_geometric.data.InMemoryDataset`.
        """
        self.verbose = verbose
        self.data_root = data_root
        self.cmap_threshold = cmap_threshold
        self.overwrite = overwrite
        max_seq_len = max_seq_len or 100000
        assert max_seq_len >= 100, 'max_seq_len cant be smaller than 100.'
        self.max_seq_len = max_seq_len
        
        # checking feature and edge options
        assert feature_opt in self.FEATURE_OPTIONS, \
            f"Invalid feature_opt '{feature_opt}', choose from {self.FEATURE_OPTIONS}"
            
        self.pro_feat_opt = feature_opt
        self.aln_dir = None # none treats it as np.zeros
        if feature_opt in ['msa', 'shannon']:
            self.aln_dir =  aln_dir # path to sequence alignments
            
        assert edge_opt in self.EDGE_OPTIONS, \
            f"Invalid edge_opt '{edge_opt}', choose from {self.EDGE_OPTIONS}"
        self.pro_edge_opt = edge_opt
        
        # check ligand options:
        ligand_feature = ligand_feature or 'original'
        ligand_edge = ligand_edge or 'binary'
        assert ligand_feature in self.LIGAND_FEATURE_OPTIONS, \
            f"Invalid ligand_feature '{ligand_feature}', choose from {self.LIGAND_FEATURE_OPTIONS}"
        self.ligand_feature = ligand_feature
        assert ligand_edge in self.LIGAND_EDGE_OPTIONS, \
            f"Invalid ligand_edge '{ligand_edge}', choose from {self.LIGAND_EDGE_OPTIONS}"
        self.ligand_edge = ligand_edge
        
        # Validating subset
        subset = subset or 'full'
        save_root = os.path.join(save_root, f'{self.pro_feat_opt}_{self.pro_edge_opt}_{self.ligand_feature}_{self.ligand_edge}') # e.g.: path/to/root/nomsa_anm
        if self.verbose: print('save_root:', save_root)
        
        if subset != 'full':
            data_p = os.path.join(save_root, subset)
            if not os.path.isdir(data_p):
                raise DatasetNotFound(f"{data_p} Subset does not exist, please create (using split) before initialization.")
        self.subset = subset
        
        # checking af2 conf dir if we are creating the dataset from scratch
        if not os.path.isdir(save_root) and (self.pro_edge_opt in cfg.OPT_REQUIRES_CONF):
            assert af_conf_dir is not None, f"{self.pro_edge_opt} edge selected but no af_conf_dir provided!"
            assert os.path.isdir(af_conf_dir), f"AF configuration dir doesnt exist, {af_conf_dir}"
        self.af_conf_dir = af_conf_dir
        
        self.only_download = only_download
        self.df = None # dataframe for csv of raw strings for SMILE protein sequence and affinity
        super(BaseDataset, self).__init__(save_root, *args, **kwargs)
        self.load()
    
    @abc.abstractmethod
    def pdb_p(self, code) -> str:
        """path to pdbfile for a particular protein"""
        raise NotImplementedError
    
    def pddlt_p(self, code) -> str:
        """path to plddt file for a particular protein"""
        return None
    
    @abc.abstractmethod
    def cmap_p(self, code) -> str:
        raise NotImplementedError
        
    @abc.abstractmethod
    def aln_p(self, code) -> str:
        # path to cleaned input alignment file
        raise NotImplementedError
    
    def edgew_p(self, code) -> str:
        """Also includes edge_attr"""
        dirname = os.path.join(self.raw_dir, 'edge_weights', self.pro_edge_opt)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        return os.path.join(dirname, f'{code}.npy')
    
    def af_conf_files(self, code) -> list[str]:
        # removing () from string since file names cannot include them and localcolabfold replaces them with _
        code = re.sub(r'[()]', '_', code)
        # localcolabfold has 'unrelaxed' as the first part after the code/ID.
        # output must be in out directory
        return glob(f'{self.af_conf_dir}/out?/{code}*_alphafold2_ptm_model_*.pdb')
    
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
        return ['XY.csv','data_pro.pt','data_mol.pt','cleaned_XY.csv']
    
    @property
    def index(self):
        return self._indices
    
    @property
    def ligands(self):
        """returns unique ligands in dataset"""
        return self.df['SMILE'].unique()
    
    @property
    def proteins(self):
        """returns unique proteins in dataset"""
        return self.df['prot_id'].unique()
    
    def get_protein_counts(self) -> Counter:
        """returns dict of protein counts from `collections.Counter` object"""
        return Counter(self.df['prot_id'])
    
    @staticmethod
    def get_unique_prots(df, verbose=True, keep_len=False) -> pd.DataFrame:
        """Gets the unique proteins from a dataframe by their protein id"""
        # get index name for later
        idx_name = df.index.name
        
        # sorting by sequence length before dropping so that we keep the longest protein sequence instead of just the first.
        df['seq_len'] = df['prot_seq'].str.len()
        df = df.sort_values(by='seq_len', ascending=False)
        
        # create new numerated index col for ensuring the first unique uniprotID is fetched properly 
        df.reset_index(drop=False, inplace=True)
        unique_pro = df[['prot_id']].drop_duplicates(keep='first')
        # reverting index to code-based index
        df.set_index(idx_name, inplace=True)
        unique_df = df.iloc[unique_pro.index]
        
        if verbose: logging.info(f'{len(unique_df)} unique proteins')
        if not keep_len: df.drop('seq_len', axis='columns', inplace=True)
        return unique_df
    
    def load(self): 
        # loading cleaned XY.csv file
        self.df = pd.read_csv(self.processed_paths[3], index_col=0)
        
        self._indices = self.df.index
        self._data_pro = torch.load(self.processed_paths[1])
        self._data_mol = torch.load(self.processed_paths[2])
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx) -> dict:
        row = self.df.iloc[idx] #WARNING: idx must be a list in future versions on pandas since it is deprecated
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
    
    def save_subset_folds(self, idxs:Iterable[Iterable[int]]|Iterable[data.Sampler]|Iterable[data.DataLoader],
                          subset_name:str) -> list[str]:
        """
        Saves multiple folds of the same dataset for some subset (e.g.: training or val).
        Name of each fold will be `subset_name` + fold number (e.g.: train0, train1, ...).
        """
        paths = []
        for i, idx in enumerate(idxs):
            p = self.save_subset(idx, f'{subset_name}{i}')
            paths.append(p)
        return paths        
    
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
        
    def _create_protein_graphs(self, df, node_feat, edge):
        processed_prots = {}
        unique_df = self.get_unique_prots(df)
        
        # Multiprocessed ring3 creation if chosen
        if edge == cfg.PRO_EDGE_OPT.ring3:
            logging.info('Getting confs list for ring3.')
            confs = [self.af_conf_files(code) for code in unique_df.index]
            Ring3Runner.run_multiprocess(pdb_fps=confs)
        
        for code, (prot_id, pro_seq) in tqdm(
                        unique_df[['prot_id', 'prot_seq']].iterrows(), 
                        desc='Creating protein graphs',
                        total=len(unique_df)):
            pro_feat = torch.Tensor() # for adding additional features
            # extra_feat is Lx54 or Lx34 (if shannon=True)
            try:
                pro_cmap = np.load(self.cmap_p(code))
                # updated_seq is for updated foldseek 3di combined seq
                updated_seq, extra_feat, edge_idx = target_to_graph(target_sequence=pro_seq, 
                                                                    contact_map=pro_cmap,
                                                                    threshold=self.cmap_threshold, 
                                                                    pro_feat=node_feat, 
                                                                    aln_file=self.aln_p(code),
                                                                    # For foldseek feats
                                                                    pdb_fp=self.pdb_p(code),
                                                                    pddlt_fp=self.pddlt_p(code))
            except Exception as e:
                raise Exception(f"error on protein graph creation for code {code}") from e
            
            pro_feat = torch.cat((pro_feat, torch.Tensor(extra_feat)), axis=1)
            
            # get multiple configurations if available/needed
            if edge in cfg.OPT_REQUIRES_CONF:
                af_confs = self.af_conf_files(code)
            else: 
                af_confs = None
            
            # Check to see if edge weights already generated:
            pro_edge_weight = None
            if edge != 'binary':
                if os.path.isfile(self.edgew_p(code)) and not self.overwrite:
                    pro_edge_weight = np.load(self.edgew_p(code))
                else:
                    # includes edge_attr like ring3
                    pro_edge_weight = get_target_edge_weights(self.pdb_p(code), pro_seq, 
                                                        edge_opt=edge,
                                                        cmap=pro_cmap,
                                                        n_modes=5, n_cpu=4,
                                                        af_confs=af_confs)
                    np.save(self.edgew_p(code), pro_edge_weight)
                
                if len(pro_edge_weight.shape) == 2:
                    pro_edge_weight = torch.Tensor(pro_edge_weight[edge_idx[0], edge_idx[1]])
                elif len(pro_edge_weight.shape) == 3: # edge attr!
                    pro_edge_weight = torch.Tensor(pro_edge_weight[edge_idx[0], edge_idx[1], :])
        
            pro = torchg.data.Data(x=torch.Tensor(pro_feat),
                                edge_index=torch.LongTensor(edge_idx),
                                pro_seq=updated_seq, # Protein sequence for downstream esm model
                                prot_id=prot_id,
                                edge_weight=pro_edge_weight)
            
            processed_prots[prot_id] = pro
            
        return processed_prots
    
    def _create_ligand_graphs(self, df, node_feat, edge):
        processed_ligs = {}
        errors = []
        for lig_seq in tqdm(df['SMILE'].unique(), 
                            desc='Creating ligand graphs'):
            if lig_seq not in processed_ligs:
                try:
                    mol_feat, mol_edge = smile_to_graph(lig_seq, 
                                                        lig_feature=node_feat, 
                                                        lig_edge=edge)
                except ValueError:
                    errors.append(f'L-{lig_seq}')
                    continue
                except AttributeError as e:
                    raise Exception(f'Error on graph creation for ligand {lig_seq}.') from e
                
                lig = torchg.data.Data(x=torch.Tensor(mol_feat),
                                    edge_index=torch.LongTensor(mol_edge),
                                    lig_seq=lig_seq)
                processed_ligs[lig_seq] = lig
        
        if len(errors) > 0:
            logging.warning(f'{len(errors)} ligands failed to create graphs')
        return processed_ligs
        
    def clean_XY(self, df, max_seq_len=None):
        max_seq_len = self.max_seq_len
        
        # Filter proteins greater than max length
        df_new = df[df['prot_seq'].str.len() <= max_seq_len]
        pro_filtered = len(df) - len(df_new)
        if pro_filtered > 0 and self.verbose:
            logging.info(f'Filtered out {pro_filtered} proteins greater than max length of {max_seq_len}')
        df = df_new
        
        # Filter out proteins that are missing pdbs for confirmations
        missing_conf = set()
        if self.pro_edge_opt in cfg.OPT_REQUIRES_CONF:
            unique_df = self.get_unique_prots(df)
            for code in tqdm(unique_df.index,
                    desc='Filtering out proteins with missing PDB files for multiple confirmations',
                    total=len(unique_df)):
                af_confs = self.af_conf_files(code)
                # need at least 2 confimations...
                if len(af_confs) <= 1:
                    missing_conf.add(code)
        
        filtered_df = df[~df.index.isin(missing_conf)]    
        logging.debug(f'Number of codes: {len(filtered_df)}/{len(df)}')     
        
        return filtered_df   
               
    def process(self):
        """
        This method is used to create the processed data files after feature extraction.
        
        Note about protein and ligand duplicates:
        - We create graphs using the pdb/sdf file from the first instance of that prot_id/smile in the csv
          all future instances will just reference back to that in `self.__getitem__`
        """
        if self.only_download:
            return
        
        ####### checking for XY.csv #######
        def file_real(fp):
            # file exists and is not empty
            return os.path.isfile(fp) and not (os.path.getsize(fp) <= 50)
        
        if file_real(self.processed_paths[3]): # cleaned_XY found
            self.df = pd.read_csv(self.processed_paths[3], index_col=0)
            logging.info(f'{self.processed_paths[3]} file found, using it to create the dataset')
        elif file_real(self.processed_paths[0]): # raw XY found
            self.df = pd.read_csv(self.processed_paths[0], index_col=0)
            logging.info(f'{self.processed_paths[0]} file found, using it to create the dataset')
        else:
            logging.info('Creating dataset from scratch!')
            self.df = self.pre_process()
            logging.info('Created XY.csv file')
        
        # creating clean_XY.csv
        if not file_real(self.processed_paths[3]): 
            self.df = self.clean_XY(self.df)
            self.df.to_csv(self.processed_paths[3])
            logging.info('Created cleaned_XY.csv file')
            
        
        ###### Get Protein Graphs ######
        processed_prots = self._create_protein_graphs(self.df, self.pro_feat_opt, self.pro_edge_opt)
        
        ###### Get Ligand Graphs ######
        processed_ligs = self._create_ligand_graphs(self.df, self.ligand_feature, self.ligand_edge)
        
        ###### Save ######
        logging.info('Saving...')
        torch.save(processed_prots, self.processed_paths[1])
        torch.save(processed_ligs, self.processed_paths[2])


class PDBbindDataset(BaseDataset): # InMemoryDataset is used if the dataset is small and can fit in CPU memory
    def __init__(self, save_root=f'{cfg.DATA_ROOT}/PDBbindDataset', 
                 data_root=f'{cfg.DATA_ROOT}/pdbbind/v2020-other-PL', 
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
            Path to raw pdbbind files, by default '../data/pdbbind/v2020-other-PL'
        `aln_dir` : str, optional
            Path to sequence alignment directory with files of the name 
            '{code}_cleaned.a3m'. If set to None then no PSSM calculation is 
            done and is set to zeros, by default None.
        `cmap_threshold` : float, optional
            Threshold for contact map creation, by default 8.0
        `feature_opt` : bool, optional
            choose from ['nomsa', 'msa', 'shannon']
            
        *args and **kwargs sent to superclass `src.data_prep.datasets.BaseDataset`.
        """   
        super(PDBbindDataset, self).__init__(save_root, data_root=data_root,
                                             aln_dir=aln_dir, cmap_threshold=cmap_threshold,
                                             feature_opt=feature_opt, *args, **kwargs)
    
    def af_conf_files(self, pid) -> list[str]:
        if self.df is not None and pid in self.df.index:
            pid = self.df.loc[pid]['prot_id']
        return glob(f'{self.af_conf_dir}/{pid}_model_*.pdb')
    
    def pdb_p(self, code):
        return os.path.join(self.data_root, code, f'{code}_protein.pdb')
    
    def cmap_p(self, pid):
        # cmap is saved in seperate directory under pdbbind/v2020-other-PL/cmaps/
        # file names are unique protein ids...
        # check to make sure arg is a pid
        if self.df is not None and pid in self.df.index:
            pid = self.df.loc[pid]['prot_id']
        return os.path.join(self.data_root, 'cmaps', f'{pid}.npy')
    
    def aln_p(self, code):
        # see feature_extraction/process_msa.py for details on how the alignments are cleaned
        if self.aln_dir is None:
            # dont use aln if none provided (will set to zeros)
            return None
        # aln_dir has a3m files.
        return os.path.join(os.path.dirname(self.aln_dir), f'{code}.aln')
        
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
        
        ############## Read index files to get binding and protID data #############
        # Get prot ids data:
        df_pid = PDBbindProcessor.get_name_data(self.raw_paths[1]) # _name.2020
        df_pid.drop(columns=['release_year','prot_name'], inplace=True)
        # contains col: prot_id
        # some might not have prot_ids available so we need to use PDBcode as id instead
        missing_pid = df_pid.prot_id == '------'
        df_pid[missing_pid] = df_pid[missing_pid].assign(prot_id = df_pid[missing_pid].index)
        
        # Get binding data:
        df_binding = PDBbindProcessor.get_binding_data(self.raw_paths[0]) # _data.2020
        df_binding.drop(columns=['resolution', 'release_year', 'lig_name'], inplace=True)
        pdb_codes = df_binding.index # pdbcodes
        
        ############## validating codes #############
        if self.aln_dir is not None: # create msa if 'msaF' is selected
            #NOTE: assuming MSAs are already created, since this would take a long time to do.
            
            # filters out those that do not have aln file or have empty files
            valid_codes = [c for c in pdb_codes if (os.path.isfile(self.aln_p(c)) and \
                                                    os.stat(self.aln_p(c)).st_size > 50)]
            print(f'Number of codes with aln files: {len(valid_codes)} out of {len(pdb_codes)}')
        else: # check if exists
            valid_codes = [c for c in pdb_codes if os.path.isfile(self.pdb_p(c))]
            
        pdb_codes = valid_codes
        assert len(pdb_codes) > 0, 'Too few PDBCodes, need at least 1...'
        
        # merge with binding data to get unique protids that are validated:
        df = df_pid.merge(df_binding, on='PDBCode') # pids + binding
        
        ############## Getting protein seq: #############
        df_seqs = pd.DataFrame.from_dict(multi_get_sequences(pdb_codes, self.pdb_p), 
                                        orient='index',
                                        columns=['prot_seq'])
        df_seqs.index.name = 'PDBCode'
        df_seqs_pid = df_pid.merge(df_seqs, on='PDBCode')
        # merge pids with sequence to get unique prots by seq length
        df_unique = self.get_unique_prots(df_seqs_pid)
        
        ############## Getting contact maps: #############
        os.makedirs(os.path.dirname(self.cmap_p('')), exist_ok=True)
        seqs = multi_save_cmaps(
                    [(code, pid) for code, pid in df_unique['prot_id'].items()],
                    pdb_p=self.pdb_p,
                    cmap_p=self.cmap_p,
                    overwrite=self.overwrite)
        
        assert len(seqs) == len(df_unique), 'Some codes failed to create contact maps'
                
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
        
        ############## FINAL MERGES #############
        # pkd + seq + pi
        df = df_binding.merge(df_seqs_pid, on="PDBCode")        
        # + SMILES
        df = df.merge(df_smi, on='PDBCode')

        # changing index name to code (to match with super class):
        df.index.name = 'code'
                
        df.to_csv(self.processed_paths[0])
        return df

  
class DavisKibaDataset(BaseDataset):
    def __init__(self, save_root=f'{cfg.DATA_ROOT}/DavisKibaDataset/', 
                 data_root=f'{cfg.DATA_ROOT}/davis_kiba/davis/', 
                 aln_dir=f'{cfg.DATA_ROOT}/davis_kiba/davis/aln/',
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
        *args and **kwargs sent to superclass `src.data_prep.datasets.BaseDataset`.
        """
        super(DavisKibaDataset, self).__init__(save_root, data_root=data_root,
                                               aln_dir=aln_dir, cmap_threshold=cmap_threshold,
                                               feature_opt=feature_opt, *args, **kwargs)
    
    def af_conf_files(self, code) -> list[str]:
        """Davis has issues since prot_ids are not really that unique"""
        # removing () from string since file names cannot include them and localcolabfold replaces them with _
        code = re.sub(r'[()]', '_', code)
        # localcolabfold has 'unrelaxed' as the first part after the code/ID.
        # output must be in out directory
        return glob(f'{self.af_conf_dir}/out?/{code}_unrelaxed*_alphafold2_ptm_model_*.pdb')    
    
    def pdb_p(self, code, safe=True):
        code = re.sub(r'[()]', '_', code)
        # davis and kiba dont have their own structures so this must be made using 
        # af or some other method beforehand.
        if (self.pro_edge_opt not in cfg.STRUCT_EDGE_OPT) and \
            (self.pro_feat_opt not in cfg.STRUCT_PRO_FEAT_OPT): return None
        
        file = glob(os.path.join(self.af_conf_dir, f'highQ/{code}_unrelaxed_rank_001*.pdb'))
        # should only be one file
        assert not safe or len(file) == 1, f'Incorrect pdb pathing, {len(file)}# of structures for {code}.'
        return file[0] if len(file) >= 1 else None
    
    def pddlt_p(self, code, safe=True):
        # this contains confidence scores for each predicted residue position in the protein
        pdb_p = self.pdb_p(code, safe=safe)
        if pdb_p is None: return None
        # from O00141_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb
        # to   O00141_scores_rank_001_alphafold2_ptm_model_1_seed_000.json
        return pdb_p.replace('unrelaxed', 'scores').replace('.pdb', '.json')
    
    def cmap_p(self, code):
        return os.path.join(self.data_root, 'pconsc4', f'{code}.npy')
    
    def aln_p(self, code):
        # using existing cleaned alignment files
        # see feature_extraction/process_msa.py
        if self.aln_dir is None:
            # dont use aln if none provided (will set to zeros)
            return None
        return os.path.join(self.aln_dir, f'{code}.aln')
        
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
    
    def download_structures(self):
        # proteins.txt file:
        unique_prots = json.load(open(self.raw_file_names[0], 'r')).keys()
        # [...]
        # TODO: send unique prots to uniprot for structure search
        # TODO: make flexible depending on which dataset is being used (kiba vs davis have different needs)
        
        ##### Map to PDB structural files
        # downloaded from https://www.uniprot.org/id-mapping/bcf1665e2612ea050140888440f39f7df822d780/overview
        df = pd.read_csv(f'{self.data_root}/kiba_mapping_pdb.tsv', sep='\t')
        # getting only first hit for each unique PDB-ID
        df = df.loc[df[['From']].drop_duplicates().index]

        # getting missing/unmapped prot ids
        missing = [prot_id for prot_id in unique_prots if prot_id not in df['From'].values]

        ##### download pdb files
        save_dir = f'{self.data_root}/structures'
        Downloader.download_PDBs(df['To'].values, save_dir=save_dir)

        # retrieve missing structures from AlphaFold:
        Downloader.download_predicted_PDBs(missing, save_dir=save_dir)

        # NOTE: some uniprotIDs map to the same structure, so we copy them to ensure each has its own file.

        # copying to new uniprot id file names
        for _, row in df.iterrows():
            uniprot = row['From']
            pdb = row['To']
            # finding pdb file
            f_in = f'{save_dir}/{pdb}.pdb'
            f_out = f'{save_dir}/{uniprot}.pdb'
            if not os.path.isfile(f_in):
                print('Missing', f_in)
            elif not os.path.isfile(f_out):
                shutil.copy(f_in, f_out)


        # removing old pdb files.
        for i, row in df.iterrows():
            pdb = row['To']
            f_in = f'{save_dir}/{pdb}.pdb'
            if os.path.isfile(f_in):
                os.remove(f_in)
    
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
            no_aln = [c for c in codes if (not Processor.check_aln_lines(self.aln_p(c)))]
                    
            # filters out those that do not have aln file
            print(f'Number of codes with invalid aln files: {len(no_aln)} / {len(codes)}')
                    
        # Checking that contact maps are present for each code:
        #       (Created by psconsc4)
        no_cmap = [c for c in codes if not os.path.isfile(self.cmap_p(c))]
        print(f'Number of codes without cmap files: {len(no_cmap)} out of {len(codes)}')
        
        # Checking that structure and af_confs files are present if required:
        no_confs = []
        if self.pro_edge_opt in cfg.STRUCT_EDGE_OPT or self.pro_feat_opt in cfg.STRUCT_PRO_FEAT_OPT:
            if self.pro_feat_opt == 'foldseek':
                # we only need HighQ structures for foldseek
                no_confs = [c for c in codes if (self.pdb_p(c, safe=False) is None)]
            else:
                no_confs = [c for c in codes if (
                (self.pdb_p(c, safe=False) is None) or # no highQ structure
                    (len(self.af_conf_files(c)) < 2))]    # only if not for foldseek
           
            # WARNING: TEMPORARY FIX FOR DAVIS (TESK1 highQ structure is mismatched...)
            no_confs.append('TESK1')
           
            print(f'Number of codes missing af2 configurations: {len(no_confs)} / {len(codes)}')
           
        invalid_codes = set(no_aln + no_cmap + no_confs)
        # filtering out invalid codes:
        lig_r = [r for i,r in enumerate(lig_r) if codes[prot_c[i]] not in invalid_codes]
        prot_c = [c for c in prot_c if codes[c] not in invalid_codes]
        
        assert len(prot_c) > 10, f"Not enough proteins in dataset, {len(prot_c)} total."
        
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
            
        # adding prot_id column (in the case of davis and kiba datasets, the code is the prot_id/name)
        # Note: this means the code is not unique (unlike pdbbind)
        df['prot_id'] = df['code']
        df.set_index('code', inplace=True,
                     verify_integrity=False)
        
        # NOTE: codes are not unique but its okay since we use idx positions when getting 
        # binding info (see BaseDataset.__getitem__)
        df.to_csv(self.processed_paths[0])
        return df
    
    
class PlatinumDataset(BaseDataset):
    CSV_LINK = 'https://biosig.lab.uq.edu.au/platinum/static/platinum_flat_file.csv'
    PBD_LINK = 'https://biosig.lab.uq.edu.au/platinum/static/platinum_processed_pdb_files.tar.gz'
    def __init__(self, save_root: str, data_root: str=None, 
                 aln_dir: str=None, cmap_threshold: float=8.0, 
                 feature_opt='nomsa', *args, **kwargs):
        """
        Dataset class for the Platinum dataset.
        NOTE: Platinum dataset is essentially two datasets in one (one for mutations and one for wildtype)

        Parameters
        ----------
        `save_root` : str
            Where to save the processed data.
        `data_root` : str, optional
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
        
        
        if aln_dir is not None:
            print('WARNING: aln_dir is not used for Platinum dataset, no support for MSA alignments.')
        
        super().__init__(save_root, data_root, None, cmap_threshold, 
                         feature_opt, *args, **kwargs)
    
    def pdb_p(self, code):
        code = code.split('_')[0] # removing additional information for mutations.
        return os.path.join(self.raw_paths[1], f'{code}.pdb')
    
    def cmap_p(self, code, map=True): 
        # map is important so that we map the provided code to the correct pdb ID
        if map:
            code = self.df.loc[code]['prot_id']
        return os.path.join(self.raw_dir, 'contact_maps', f'{code}.npy')
    
    def aln_p(self, code):
        return None # no support for MSA alignments
        # raise NotImplementedError('Platinum dataset does not have MSA alignments.')
    
    @property
    def raw_file_names(self):
        """call by self.raw_paths to get the proper file paths"""
        # in order of download/creation:
        return ['platinum_flat_file.csv',# downloaded from website
                'platinum_pdb',
                'platinum_sdf']
    
    def download(self):
        """Download the raw data for the dataset."""
        # Download flat CSV file containing binding info, PDB IDs, ligand IDs, etc...
        if not os.path.isfile(self.raw_paths[0]):
            print('CSV file not found, downloading from website...')
            urllib.request.urlretrieve(self.CSV_LINK, self.raw_paths[0])
            df_raw = pd.read_csv(self.raw_paths[0])
            # removing broken structures
            df_raw = df_raw[~df_raw['lig.canonical_smiles'].str.contains('broken')]
            df_raw.index.name = 'raw_idx'
            df_raw.to_csv(self.raw_paths[0])
        else:
            df_raw = pd.read_csv(self.raw_paths[0])
            
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
        
        # validate that all files are there and download any missing pdbs:
        for i, row in tqdm(df_raw.iterrows(), 
                           desc='Checking pdb files and downloading missing',
                           total=len(df_raw)):
            pdb_wt = row['mut.wt_pdb']
            pdb_mt = row['mut.mt_pdb']
            
            # Check if PDBs available
            missing_wt = pdb_wt == 'NO'
            missing_mt = pdb_mt == 'NO'
            assert not (missing_mt and missing_wt), f'missing pdbs for both mt and wt on idx {i}'
            
            # Download if file doesnt exist:
            if not missing_wt and (not os.path.isfile(self.pdb_p(pdb_wt))):
                Downloader.download_PDBs([pdb_wt], save_dir=self.raw_paths[1], tqdm_disable=True)
            if not missing_mt and (not os.path.isfile(self.pdb_p(pdb_mt))):
                Downloader.download_PDBs([pdb_mt], save_dir=self.raw_paths[1], tqdm_disable=True)
            
        # Download corrected SMILEs since the ones provided in the csv file have issues 
        # (see https://github.com/jyaacoub/MutDTA/issues/27)
        os.makedirs(self.raw_paths[2], exist_ok=True)
        print('Downloading SDF files for ligands.')
        Downloader.download_SDFs(ligand_names=df_raw['affin.lig_id'].unique(),
                                save_dir=self.raw_paths[2])
        
        # Fixing smiles in csv file using downloaded sdf files        
        smiles_dict = Processor.get_SMILE(df_raw['affin.lig_id'].unique(),
                            lambda x: os.path.join(self.raw_paths[2], f'{x}.sdf'))
        df_raw['smiles'] = df_raw['affin.lig_id'].map(smiles_dict)
        df_raw.to_csv(self.raw_paths[0])
        
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
        # fixing pkd values for binding affinity
        df_raw['affin.k_mt'] = df_raw['affin.k_mt'].str.extract(r'(\d+\.*\d+)', 
                                                      expand=False).astype(float)
        # adjusting units for binding data from nM to pKd:
        df_raw['affin.k_mt'] = -np.log10(df_raw['affin.k_mt']*1e-9)
        df_raw['affin.k_wt'] = -np.log10(df_raw['affin.k_wt']*1e-9)
        # Getting sequences and cmaps:
        prot_seq = {}
        for i, row in tqdm(df_raw.iterrows(), 
                           desc='Getting sequences and cmaps',
                           total=len(df_raw)):
            muts = row['mutation'].split('/') # split to account for multiple point mutations
            pdb_wt = row['mut.wt_pdb']
            pdb_mt = row['mut.mt_pdb'] 
            t_chain = row['affin.chain']
            # Check if PDBs available
            missing_wt = pdb_wt == 'NO'
            missing_mt = pdb_mt == 'NO'
            assert not (missing_mt and missing_wt), f'missing pdbs for both mt and wt on idx {i}'
            pdb_wt = pdb_mt if missing_wt else pdb_wt
            pdb_mt = pdb_wt if missing_mt else pdb_mt
            
            try:
                chain_wt = Chain(self.pdb_p(pdb_wt), t_chain=t_chain)
                chain_mt = Chain(self.pdb_p(pdb_mt), t_chain=t_chain)
                
                # Getting sequences:
                if missing_wt:
                    mut_seq = chain_wt.sequence
                    ref_seq = chain_wt.get_mutated_seq(muts, reversed=True)
                else:
                    mut_seq = chain_wt.get_mutated_seq(muts, reversed=False)
                    ref_seq = chain_wt.sequence
                
                # creating protein unique IDs for computational speed up by avoiding redundant compute
                wt_id = f'{pdb_wt}_wt'
                mt_id = f'{pdb_mt}_{"-".join(muts)}'
                if pdb_mt != pdb_wt and mut_seq != chain_mt.sequence:
                    # print(f'Mutated doesnt match with chain for {i}:{self.pdb_p(pdb_wt)} and {self.pdb_p(pdb_mt)}')
                    # using just the wildtype protein structure to avoid mismatches with graph network
                    mt_id = f'{pdb_wt}_{"-".join(muts)}'
                    chain_mt = chain_wt
                
                # Getting and saving cmaps under the unique prot_ID
                if not os.path.isfile(self.cmap_p(wt_id, map=False)):
                    np.save(self.cmap_p(wt_id, map=False), chain_wt.get_contact_map())
                
                if not os.path.isfile(self.cmap_p(mt_id, map=False)):
                    np.save(self.cmap_p(mt_id, map=False), chain_mt.get_contact_map())
                
            except Exception as e:
                raise Exception(f'Error with idx {i} on {pdb_wt} wt and {pdb_mt} mt.') from e
                
            # Saving sequence and additional relevant info
            mt_pkd = row['affin.k_mt']
            wt_pkd = row['affin.k_wt']
            lig_id = row['affin.lig_id']
            smiles = row['smiles']
            
            # Using index number for ID since pdb is not unique in this dataset.
            prot_seq[f'{i}_mt'] = (mt_id, lig_id,  mt_pkd, smiles, mut_seq)
            prot_seq[f'{i}_wt'] = (wt_id, lig_id, wt_pkd, smiles, ref_seq)
                
        df = pd.DataFrame.from_dict(prot_seq, orient='index', 
                                        columns=['prot_id', 'lig_id', 
                                                 'pkd', 'SMILE', 'prot_seq'])
        df.index.name = 'code'
        df.to_csv(self.processed_paths[0])
        return df