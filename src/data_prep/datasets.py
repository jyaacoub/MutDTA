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
from multiprocessing import Pool, cpu_count

from src.data_prep.feature_extraction.gvp_feats import GVPFeaturesProtein, GVPFeaturesLigand
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
        max_seq_len = max_seq_len or 2400
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
        self.save_path = save_root
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
        self.alphaflow = self.pro_edge_opt in cfg.OPT_REQUIRES_AFLOW_CONF
        
        super(BaseDataset, self).__init__(save_root, *args, **kwargs)
        self.load()
    
    @abc.abstractmethod
    def pdb_p(self, code) -> str:
        """path to pdbfile for a particular protein"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def sdf_p(self, code) -> str:
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
    
    @abc.abstractmethod
    def af_conf_files(self, code) -> list[str]:
        raise NotImplementedError
    
    def edgew_p(self, code) -> str:
        """Also includes edge_attr"""
        dirname = os.path.join(self.raw_dir, 'edge_weights', self.pro_edge_opt)
        os.makedirs(dirname, exist_ok=True)
        return os.path.join(dirname, f'{code}.npy')
    
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
    def get_unique_prots(df, keep_len=False) -> pd.DataFrame:
        """Gets the unique proteins from a dataframe by their protein id"""
        # sorting by sequence length to keep the longest protein sequence instead of just the first.
        if 'sort_order' in df:
            # ensures that the right codes are always present in unique_pro
            df.sort_values(by='sort_order', inplace=True)
        else:
            df = df.assign(seq_len=df['prot_seq'].str.len())
            df.sort_values(by='seq_len', ascending=False, inplace=True)
            df = df.assign(sort_order=[i for i in range(len(df))])
        
        # Get unique protid codes
        idx_name = df.index.name
        df.reset_index(drop=False, inplace=True)
        unique_pro = df[['prot_id']].drop_duplicates(keep='first')
        
        # reverting index to code-based index
        df.set_index(idx_name, inplace=True)
        unique_df = df.iloc[unique_pro.index]
        
        logging.info(f'{len(unique_df)} unique proteins')
        if not keep_len and 'seq_len' in df: df.drop('seq_len', axis='columns', inplace=True)
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
        sub_df.to_csv(os.path.join(path, self.processed_file_names[0])) # redundant save since it is not used and mainly just for tracking prots.
        sub_df.to_csv(os.path.join(path, self.processed_file_names[3])) # clean_XY.csv
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
        
    @staticmethod    
    def process_protein_multiprocessing(args):
        """
        Checks if protein has conf file and correct sequence, returns:
            - None, None - if it has a conf file and is correct
            - pid,  None - is missing a conf file
            - pid,  matching_code - has the correct number of conf files but is not the correct sequence.
            - None, matching_code - correct seq, # of confs, but under a different file name
        """
        pid, seq, af_conf_dir, is_pdbbind, files = args
        MIN_MODEL_COUNT = 5
        
        af_confs = []
        if is_pdbbind:
            fp = os.path.join(af_conf_dir, f'{pid}.pdb')
            if os.path.exists(fp):
                af_confs = [os.path.join(af_conf_dir, f'{pid}.pdb')]
        else:
            af_confs = [os.path.join(af_conf_dir, f) for f in files if f.startswith(pid)]
        
        if len(af_confs) == 0:
            return pid, None
        
        # either all models in one pdb file (alphaflow) or spread out across multiple files (AF2 msa subsampling)
        model_count = len(af_confs) if len(af_confs) > 1 else Chain.get_model_count(af_confs[0])
        
        if model_count < MIN_MODEL_COUNT:
            return pid, None
        
        af_seq = Chain(af_confs[0]).sequence
        if seq != af_seq:
            logging.debug(f'Mismatched sequence for {pid}')
            return pid, af_seq
        
        return None, None
            
    @staticmethod
    def check_missing_confs(df_unique:pd.DataFrame, af_conf_dir:str, is_pdbbind=False):
        logging.debug(f'Getting af_confs from {af_conf_dir}')

        missing = set()
        mismatched = {}
        # total of 3728 unique proteins with alphaflow confs (named by pdb ID)
        files = None
        if not is_pdbbind:
            files = [f for f in os.listdir(af_conf_dir) if f.endswith('.pdb')]

        with Pool(processes=cpu_count()) as pool:
            tasks = [(pid, seq, af_conf_dir, is_pdbbind, files) \
                            for _, (pid, seq) in df_unique[['prot_id', 'prot_seq']].iterrows()]

            for pid, correct_seq in tqdm(pool.imap_unordered(BaseDataset.process_protein_multiprocessing, tasks), 
                            desc='Filtering out proteins with missing PDB files for multiple confirmations', 
                            total=len(tasks)):
                if correct_seq is not None:
                    mismatched[pid] = correct_seq
                elif pid is not None: # just pid -> missing af files
                    missing.add(pid)
        
        return missing, mismatched
    
    def clean_XY(self, df:pd.DataFrame, max_seq_len=None):        
        max_seq_len = self.max_seq_len
        
        # Filter proteins greater than max length
        df_new = df[df['prot_seq'].str.len() <= max_seq_len]
        pro_filtered = len(df) - len(df_new)
        if pro_filtered > 0 and self.verbose:
            logging.info(f'Filtered out {pro_filtered} proteins greater than max length of {max_seq_len}')
        df = df_new
        
        df_unique = self.get_unique_prots(df)
        # merge sequences so that they are all the same for all matching prot_id
        df = df.drop('prot_seq', axis=1)
        idx_name = df.index.name
        df.reset_index(drop=False, inplace=True)
        df = df.merge(df_unique[['prot_id', 'prot_seq']], 
                      on='prot_id')
        df.set_index(idx_name, inplace=True)
        
        # Filter out proteins that are missing pdbs for confirmations
        missing = set()
        mismatched = {}
        if self.pro_edge_opt in cfg.OPT_REQUIRES_CONF:
            missing, mismatched = self.check_missing_confs(df_unique, self.af_conf_dir, self.__class__ is PDBbindDataset)
        
        if len(missing) > 0:
            filtered_df = df[~df.prot_id.isin(missing)]
            logging.warning(f'{len(missing)} missing pids')
        else:
            filtered_df = df
        
        if len(mismatched) > 0:
            filtered_df = filtered_df[~filtered_df.prot_id.isin(mismatched)]
            # adjust all in dataframe to the ones with the correct pid
            logging.warning(f'{len(mismatched)} mismatched pids')
            
            
        logging.debug(f'Number of codes: {len(filtered_df)}/{len(df)}')
        
        # we are done filtering if ligand doesnt need filtering
        if not (self.ligand_edge in cfg.OPT_REQUIRES_SDF or 
                self.ligand_feature in cfg.OPT_REQUIRES_SDF):
            return filtered_df
        
        ###########
        # filter ligands
        # removing rows with ligands that have missing sdf files:
        unique_lig = filtered_df[['lig_id']].drop_duplicates()
        missing = set()
        for code, (lig_id,) in tqdm(unique_lig.iterrows(), desc='dropping missing sdfs from df', 
                                    total=len(unique_lig)):
            fp = self.sdf_p(code, lig_id=lig_id)
            if (not os.path.isfile(fp) or 
                os.path.getsize(fp) <= 20):
                missing.add(lig_id)
        
        logging.debug(f'{len(missing)}/{len(unique_lig)} missing ligands')
        filtered_df = filtered_df[~filtered_df.lig_id.isin(missing)]
        logging.debug(f'Number of codes after ligand filter: {len(filtered_df)}/{len(df)}')
        return filtered_df
    
    def _create_protein_graphs(self, df, node_feat, edge):
        processed_prots = {}
        unique_df = self.get_unique_prots(df)
        
        # Multiprocessed ring3 creation if chosen
        if edge in cfg.OPT_REQUIRES_RING3:
            logging.info('Getting confs list for ring3.')
            files = [f for f in os.listdir(self.af_conf_dir) if f.endswith('.pdb')]
            confs = [[os.path.join(self.af_conf_dir, f) for f in files if f.startswith(pid)] for pid in unique_df.prot_id]
            Ring3Runner.run_multiprocess(pdb_fps=confs)
        
        for code, (prot_id, pro_seq) in tqdm(
                        unique_df[['prot_id', 'prot_seq']].iterrows(), 
                        desc='Creating protein graphs',
                        total=len(unique_df)):
            
            if node_feat == cfg.PRO_FEAT_OPT.gvp:
                # gvp has its own unique graph to support the architecture's implementation.
                coords = Chain(self.pdb_p(code), grep_atoms={'CA', 'N', 'C'}).getCoords(get_all=True)
                processed_prots[prot_id] = GVPFeaturesProtein().featurize_as_graph(code, coords, pro_seq)
                continue
            
            pro_feat = torch.Tensor() # for adding additional features
            # extra_feat is Lx54 or Lx34 (if shannon=True)
            try:
                pro_cmap = np.load(self.cmap_p(prot_id))
                # updated_seq is for updated foldseek 3di combined seq
                aln_file = self.aln_p(code) if node_feat in cfg.OPT_REQUIRES_MSA_ALN else None
                updated_seq, extra_feat, edge_idx = target_to_graph(target_sequence=pro_seq, 
                                                                    contact_map=pro_cmap,
                                                                    threshold=self.cmap_threshold, 
                                                                    pro_feat=node_feat, 
                                                                    aln_file=aln_file,
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
                                                        af_confs=af_confs) # NOTE: this will handle if af_confs is a single file w/ multiple models
                    np.save(self.edgew_p(code), pro_edge_weight)
                
                if len(pro_edge_weight.shape) == 2:
                    pro_edge_weight = torch.Tensor(pro_edge_weight[edge_idx[0], edge_idx[1]])
                elif len(pro_edge_weight.shape) == 3: # has edge attr!
                    pro_edge_weight = torch.Tensor(pro_edge_weight[edge_idx[0], edge_idx[1], :])
        
            pro = torchg.data.Data(x=torch.Tensor(pro_feat),
                                edge_index=torch.LongTensor(edge_idx),
                                pro_seq=updated_seq, # Protein sequence for downstream esm model
                                prot_id=prot_id,
                                edge_weight=pro_edge_weight)
            
            processed_prots[prot_id] = pro
            
        return processed_prots
    
    def _create_ligand_graphs(self, df:pd.DataFrame, node_feat, edge):
        processed_ligs = {}
        errors = []
        if node_feat == cfg.LIG_FEAT_OPT.gvp:
            for code, (lig_seq, lig_id) in tqdm(df[['SMILE', 'lig_id']].iterrows(), desc='Creating ligand graphs', 
                                      total=len(df)):
                processed_ligs[lig_seq] = GVPFeaturesLigand().featurize_as_graph(self.sdf_p(code,lig_id=lig_id))
            return processed_ligs
        
        for lig_seq in tqdm(df['SMILE'].unique(), desc='Creating ligand graphs'):
            if lig_seq not in processed_ligs:
                try:
                    mol_feat, mol_edge = smile_to_graph(lig_seq, lig_feature=node_feat, lig_edge=edge)
                except ValueError:
                    errors.append(f'L-{lig_seq}')
                    continue
                except AttributeError as e:
                    raise Exception(f'Error on graph creation for ligand {lig_seq}.') from e
                
                lig = torchg.data.Data(x=torch.Tensor(mol_feat), edge_index=torch.LongTensor(mol_edge),
                                    lig_seq=lig_seq)
                processed_ligs[lig_seq] = lig
        
        if len(errors) > 0:
            logging.warning(f'{len(errors)} ligands failed to create graphs')
        return processed_ligs
        
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
            # WARNING: HOT fix so that it is still compatible with prev datasets
            # return
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
    
    def af_conf_files(self, pid, map_to_pid=True) -> list[str]|str:
        if self.df is not None and pid in self.df.index and map_to_pid:
            pid = self.df.loc[pid]['prot_id']
            
        if self.alphaflow:
            fp = os.path.join(self.af_conf_dir, f'{pid}.pdb')
            fp = fp if os.path.exists(fp) else None
            return fp
            
        return glob(f'{self.af_conf_dir}/{pid}_model_*.pdb')
    
    def pdb_p(self, code):
        return os.path.join(self.data_root, code, f'{code}_protein.pdb')
    
    def sdf_p(self, code, **kwargs):        
        return os.path.join(self.data_root, code, f'{code}_ligand.sdf')
    
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
        df_binding.drop(columns=['resolution', 'release_year'], inplace=True)
        df_binding.rename({'lig_name':'lig_id'}, inplace=True, axis=1)
        pdb_codes = df_binding.index # pdbcodes
        
        ############## validating codes #############
        logging.debug('Validating Codes')
        if self.aln_dir is not None: # create msa if 'msaF' is selected
            #NOTE: assuming MSAs are already created, since this would take a long time to do.
            
            # filters out those that do not have aln file or have empty files
            valid_codes = [c for c in pdb_codes if (os.path.isfile(self.aln_p(c)) and \
                                                    os.stat(self.aln_p(c)).st_size > 50)]
            print(f'Number of codes with aln files: {len(valid_codes)} out of {len(pdb_codes)}')
        else: # check if exists
            valid_codes = [c for c in pdb_codes if os.path.isfile(self.pdb_p(c))]
            
        pdb_codes = valid_codes
        logging.debug(f"{len(pdb_codes)} valid PDBs with existing files.")
        assert len(pdb_codes) > 0, 'Too few PDBCodes, need at least 1...'
                
        ############## Get ligand info #############
        # WARNING: THIS SHOULD ALWAYS COME BEFORE GETTING PROTEIN SEQUEINCES. ORDER MATTERS 
        # BECAUSE LIGAND INFO REDUCES NUMBER OF PDBS DUE TO MISSING SMILES.
        # Extracting SMILE strings:
        dict_smi = PDBbindProcessor.get_SMILE(pdb_codes, dir=self.sdf_p)
        df_smi = pd.DataFrame.from_dict(dict_smi, orient='index', columns=['SMILE'])
        df_smi.index.name = 'PDBCode'
        
        df_smi = df_smi[df_smi.SMILE.notna()]
        num_missing = len(pdb_codes) - len(df_smi)
        if  num_missing > 0:
            print(f'\t{num_missing} ligands failed to get SMILEs')
            pdb_codes = list(df_smi.index)
        
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
        """Davis has issues since prot_ids are not really unique"""
        if self.alphaflow:
            fp = f'{self.af_conf_dir}/{code}.pdb'
            fp = fp if os.path.exists(fp) else None
            return fp
            
        # removing () from string since localcolabfold replaces them with _
        code = re.sub(r'[()]', '_', code)
        
        # localcolabfold has 'unrelaxed' as the first part after the code/ID.
        # output must be in out directory
        return glob(f'{self.af_conf_dir}/out?/{code}_unrelaxed*_alphafold2_ptm_model_*.pdb')    
    
    def sdf_p(self, code, lig_id):
        # code is just a placeholder since other datasets (pdbbind) need it.
        return os.path.join(self.data_root, 'lig_sdf', f'{lig_id}.sdf')
    
    def pdb_p(self, code, safe=True):
        if self.alphaflow:
            return self.af_conf_files(code)
        
        code = re.sub(r'[()]', '_', code)
        # davis and kiba dont have their own structures so this must be made using 
        # af or some other method beforehand.
        if (self.pro_edge_opt not in cfg.OPT_REQUIRES_PDB) and \
            (self.pro_feat_opt not in cfg.OPT_REQUIRES_PDB): return None
        
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
        lig_dict = json.load(open(self.raw_paths[1], 'r'), 
                               object_hook=OrderedDict)
        lig_id = list(lig_dict.keys())
        ligand_seq = list(lig_dict.values())
        
        # Get binding data:
        # for davis this matrix should contain no nan values
        affinity_mat = pickle.load(open(self.raw_paths[2], 'rb'), encoding='latin1')
        lig_r, prot_c = np.where(~np.isnan(affinity_mat)) # index values corresponding to non-nan values
        
        # checking alignment files present for each code:
        no_aln = []
        if self.aln_dir is not None:
            no_aln = [c for c in codes if (not Processor.check_aln_lines(self.aln_p(c)))]
                    
            # filters out those that do not have aln file
            print(f'Number of codes with valid aln files: {len(codes)-len(no_aln)} / {len(codes)}')
                    
        # Checking that contact maps are present for each code:
        #       (Created by psconsc4)
        no_cmap = [c for c in codes if not os.path.isfile(self.cmap_p(c))]
        print(f'Number of codes with cmap files: {len(codes)-len(no_cmap)} / {len(codes)}')
        
        # Checking that structure and af_confs files are present if required:
        no_confs = []
        if self.pro_edge_opt in cfg.OPT_REQUIRES_PDB or \
            self.pro_feat_opt in cfg.OPT_REQUIRES_PDB:
            if self.pro_feat_opt == cfg.PRO_FEAT_OPT.foldseek:
                # we only need HighQ structures for foldseek
                no_confs = [c for c in codes if (self.pdb_p(c, safe=False) is None)]
            else:
                if self.alphaflow:
                    # af_conf_files will be different for alphaflow (single file)
                    no_confs = [c for c in codes if (
                        (self.pdb_p(c, safe=False) is None) or # no highQ structure
                            (Chain.get_model_count(self.af_conf_files(c)) < 5))] # single file needs Chain.get_model_count
                else:
                    no_confs = [c for c in codes if (
                        (self.pdb_p(c, safe=False) is None) or # no highQ structure
                            (len(self.af_conf_files(c)) < 5))] # only if not for foldseek
           
            # WARNING: TEMPORARY FIX FOR DAVIS (TESK1 highQ structure is mismatched...)
            if not self.alphaflow: no_confs.append('TESK1')
           
            logging.warning(f'Number of codes missing {"aflow" if self.alphaflow else "af2"} ' + \
                            f'conformations: {len(no_confs)} / {len(codes)}')
           
        invalid_codes = set(no_aln + no_cmap + no_confs)
        # filtering out invalid codes and storing their index vals.
        lig_r = [r for i,r in enumerate(lig_r) if codes[prot_c[i]] not in invalid_codes]
        prot_c = [c for c in prot_c if codes[c] not in invalid_codes]
        
        assert len(prot_c) > 10, f"Not enough proteins in dataset, {len(prot_c)} total from {self.af_conf_dir}"
        
        # creating binding dataframe:
        #   code,SMILE,pkd,prot_seq
        df = pd.DataFrame({
           'code': [codes[c] for c in prot_c],
           'lig_id': [lig_id[r] for r in lig_r],
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
    # website: https://biosig.lab.uq.edu.au/platinum/
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
    
    def af_conf_files(self, pid, map=True) -> list[str]:
        """
        Multiple confirmations are needed to generate edge attributes/weights 
        (see cfg.OPT_REQUIRES_CONF)
        """
        if self.df is not None and pid in self.df.index:
            pid = self.df.loc[pid]['prot_id']
            
        if self.alphaflow:
            fp = f'{self.af_conf_dir}/{pid}.pdb'
            fp = fp if os.path.exists(fp) else None
            return fp
            
        return glob(f'{self.af_conf_dir}/{pid}_model_*.pdb')
    
    def sdf_p(self, code, lig_id) -> str:
        """Needed for gvp ligand branch (uses coordinate info)"""
        return os.path.join(self.raw_paths[2], f'{lig_id}.sdf')
    
    def pdb_p(self, pid, id_is_pdb=False):
        if id_is_pdb:
            fp = os.path.join(self.raw_paths[1], f'{pid}.pdb')
        else:
            fp = f'{self.af_conf_dir}/{pid}.pdb'
            
        fp = fp if os.path.exists(fp) else None
        return fp
        
    def cmap_p(self, prot_id):
        return os.path.join(self.raw_dir, 'contact_maps', f'{prot_id}.npy')
    
    def aln_p(self, code):
        raise NotImplementedError('Platinum dataset does not have MSA alignments.')
    
    @property
    def raw_file_names(self):
        """call by self.raw_paths to get the proper file paths"""
        # in order of download/creation:
        return ['platinum_flat_file.csv',# downloaded from website
                'platinum_pdb',
                'platinum_sdf',
                'done_downloading.json']
    
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
            if not missing_wt and (self.pdb_p(pdb_wt, id_is_pdb=True is None)): # returns None if file doesnt exist
                Downloader.download_PDBs([pdb_wt], save_dir=self.raw_paths[1], tqdm_disable=True)
            if not missing_mt and (self.pdb_p(pdb_mt, id_is_pdb=True is None)):
                Downloader.download_PDBs([pdb_mt], save_dir=self.raw_paths[1], tqdm_disable=True)
            
        # Download corrected SMILEs since the ones provided in the csv file have issues 
        # (see https://github.com/jyaacoub/MutDTA/issues/27)
        os.makedirs(self.raw_paths[2], exist_ok=True)
        print(f'Downloading SDF files for ligands to {self.raw_paths[2]}')
        sdf_resp = Downloader.download_SDFs(ligand_ids=df_raw['affin.lig_id'].unique(),
                                save_dir=self.raw_paths[2])
        
        # Fixing smiles in csv file using downloaded sdf files        
        smiles_dict = Processor.get_SMILE(df_raw['affin.lig_id'].unique(),
                            lambda x: os.path.join(self.raw_paths[2], f'{x}.sdf'))
        df_raw['smiles'] = df_raw['affin.lig_id'].map(smiles_dict)
        df_raw.to_csv(self.raw_paths[0])
        
        # write to done_downloading.txt
        with open(self.raw_paths[3], 'a') as f: 
            f.write(json.dumps(sdf_resp))
    
    def _get_prot_structure(self, muts, pdb_wt, pdb_mt, t_chain):
        # Check if PDBs available (need at least 1 for sequence info)
        missing_wt = pdb_wt == 'NO'
        missing_mt = pdb_mt == 'NO'
        assert not (missing_mt and missing_wt), f'missing pdbs for both mt and wt'
        pdb_wt = pdb_mt if missing_wt else pdb_wt
        pdb_mt = pdb_wt if missing_mt else pdb_mt
        
        # creating protein unique IDs for computational speed up by avoiding redundant compute
        wt_id = f'{pdb_wt}_wt'
        mt_id = f'{pdb_mt}_{"-".join(muts)}'
        
        chain_wt = Chain(self.pdb_p(pdb_wt, id_is_pdb=True), t_chain=t_chain)
        chain_mt = Chain(self.pdb_p(pdb_mt, id_is_pdb=True), t_chain=t_chain)
        
        # Getting sequences:
        if missing_wt:
            mut_seq = chain_mt.sequence
            ref_seq = chain_mt.get_mutated_seq(muts, reversed=True)
        else:
            # get mut_seq from wt to confirm that mapping the mutations works
            mut_seq = chain_wt.get_mutated_seq(muts, reversed=False)
            ref_seq = chain_wt.sequence
        
        if pdb_mt != pdb_wt and mut_seq != chain_mt.sequence:
            # sequences dont match due to missing residues in either the wt or the mt pdb files (seperate files since pdb_mt != pdb_wt)
            # we can just use the wildtype protein structure to avoid mismatches with graph network (still the same mutated sequence tho)
            mt_id = f'{pdb_wt}_{"-".join(muts)}'
            
            # if we have aflow confs then we use those instead
            fp = self.pdb_p(mt_id, id_is_pdb=False)
            if fp is not None:
                chain_mt = Chain(fp, model=0) # no t_chain for alphaflow confs since theres only one input sequence.
            
            # final check to make sure this aflow conf is correct for the mt sequence.
            if mut_seq != chain_mt.sequence:
                logging.warning(f'Mismatched AA: Using wt STRUCTURE ({pdb_wt}) for mutated {mt_id}')
                chain_mt = chain_wt
        
        # Getting and saving cmaps under the unique protein ID
        if not os.path.isfile(self.cmap_p(wt_id)) or self.overwrite:
            np.save(self.cmap_p(wt_id), chain_wt.get_contact_map())
        
        if not os.path.isfile(self.cmap_p(mt_id)) or self.overwrite:
            np.save(self.cmap_p(mt_id), chain_mt.get_contact_map())
        
        return wt_id, ref_seq, mt_id, mut_seq
            
    def pre_process(self):
        """
        This method is used to create the processed data files for feature extraction.
        
        It creates a XY.csv file that contains the binding data and the sequences for 
        both ligand and proteins. with the following columns: 
        code,prot_id,lig_id,SMILE,pkd,prot_seq
            - code: unique identifier for each protein in the dataset (wild type and 
                    mutated have the same id with "_mt" or "_wt" suffix)
            - prot_id: unique identifier for each protein in the dataset
                       wt is just "{pdbid}_wt" and mt is "{pdbid}_mt-<mut1>-<mut2>..."
            - lig_id: unique identifier for each ligand in the dataset (from raw data)
            - SMILE: ligand smiles
            - pkd: binding affinity
            - prot_seq: protein sequence
        
        It generates and saves contact maps for each protein in the dataset.
        
        And also generates mutated sequences for each protein in the dataset.   

        Returns
        -------
        pd.DataFrame
            The XY.csv dataframe.
        """
        ### LOAD UP RAW CSV FILE + adjust values ###
        df_raw = pd.read_csv(self.raw_paths[0])
        # fixing pkd values for binding affinity
        df_raw['affin.k_mt'] = df_raw['affin.k_mt'].str.extract(r'(\d+\.*\d+)', 
                                                      expand=False).astype(float)
        # adjusting units for binding data from nM to pKd:
        df_raw['affin.k_mt'] = -np.log10(df_raw['affin.k_mt']*1e-9)
        df_raw['affin.k_wt'] = -np.log10(df_raw['affin.k_wt']*1e-9)
        
        ### GETTING SEQUENCES AND CMAPS ###
        os.makedirs(os.path.join(self.raw_dir, 'contact_maps'), exist_ok=True)
        prot_seq = {}
        for i, row in tqdm(df_raw.iterrows(), 
                           desc='Getting sequences and cmaps',
                           total=len(df_raw)):
            # NOTE: wild type and mutated sequences are processed in same iteration
            # but they are saved as SEPARATE ENTRIES in the final dataframe.
            
            muts = row['mutation'].split('/') # split to account for multiple point mutations
            pdb_wt = row['mut.wt_pdb']
            pdb_mt = row['mut.mt_pdb'] 
            t_chain = row['affin.chain']
            
            try:
                wt_id, ref_seq, mt_id, mut_seq = self._get_prot_structure(muts, pdb_wt, pdb_mt, t_chain)
            except Exception as e:
                raise Exception(f'Error with idx {i} on {pdb_wt} wt and {pdb_mt} mt.') from e
    
            # Using index number for ID since pdb is not unique in this dataset.
            prot_seq[f'{i}_mt'] = (mt_id, row['affin.lig_id'], row['affin.k_mt'], row['smiles'], mut_seq)
            prot_seq[f'{i}_wt'] = (wt_id, row['affin.lig_id'], row['affin.k_wt'], row['smiles'], ref_seq)
                
        df = pd.DataFrame.from_dict(prot_seq, orient='index', 
                                        columns=['prot_id', 'lig_id', 
                                                 'pkd', 'SMILE', 'prot_seq'])
        df.index.name = 'code'        
        df.to_csv(self.processed_paths[0])
        return df