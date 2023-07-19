from collections import OrderedDict
import json, pickle, re, os

import torch
import torch_geometric as torchg
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.feature_extraction import smile_to_graph, target_to_graph
from src.feature_extraction.process_msa import check_aln_lines
from src.feature_extraction.protein import create_save_cmaps
from src.data_processing import PDBbindProcessor

#  See: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
# for details on how to create a dataset
class BaseDataset(torchg.data.InMemoryDataset):
    def __init__(self, save_root:str, data_root:str, aln_dir:str,
                 cmap_threshold:float, shannon=True,  *args, **kwargs):
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
        `shannon` : bool, optional
            If True, shannon entropy instead of PSSM matrix is used for 
            protein features, by default False.
            
        *args and **kwargs sent to superclass `torch_geometric.data.InMemoryDataset`.
        """
        self.data_root = data_root
        self.aln_dir =  aln_dir # path to sequence alignments
        
        self.cmap_threshold = cmap_threshold
        self.shannon = shannon
        
        super(BaseDataset, self).__init__(save_root, *args, **kwargs)
        self.load()
        
    def cmap_p(self, code):
        return f'{self.data_root}/pconsc4/{code}.npy'
    
    def aln_p(self, code):
        # see feature_extraction/process_msa.py for details on how the alignments are cleaned
        if self.aln_dir is None:
            # dont use aln if none provided (will set to zeros)
            return None
        return f'{self.aln_dir}/{code}.aln'
    
    @property
    def processed_file_names(self):
        # XY.csv cols: PDBCode,pkd,SMILE,prot_seq
        # XY is created in pre_process
        return ['XY.csv','data_pro.pt','data_mol.pt']
    
    def load(self):
        self._data_pro = torch.load(self.processed_paths[1])
        self._data_mol = torch.load(self.processed_paths[2])
        
    def __len__(self):
        return len(self._data_mol)
    
    def __getitem__(self, idx):
        return self._data_pro[idx], self._data_mol[idx]
    
    def process(self):
        """
        This method is used to create the processed data files after feature extraction.
        """
        if not os.path.isfile(self.processed_paths[0]):
            df = self.pre_process()
        else:
            df = pd.read_csv(self.processed_paths[0])
            print(f'{self.processed_paths[0]} file found, using it to create the dataset')
        print(f'Number of codes: {len(df)}')
        
        # creating the dataset:
        data_list = []
        errors = []
        for idx in tqdm(df.index, 'Extracting node features and creating graphs'):
            code = df.loc[idx]['code']
            cmap = np.load(self.cmap_p(code))
            pro_seq = df.loc[idx]['prot_seq']
            lig_seq = df.loc[idx]['SMILE']
            label = df.loc[idx]['pkd']
            label = torch.Tensor([[label]])
            
            _, pro_feat, pro_edge = target_to_graph(pro_seq, cmap, 
                                                    threshold=self.cmap_threshold,
                                                    aln_file=self.aln_p(code),
                                                    shannon=self.shannon)
            try:
                _, mol_feat, mol_edge = smile_to_graph(lig_seq)
            except ValueError:
                errors.append(code)
                continue
            
            pro = torchg.data.Data(x=torch.Tensor(pro_feat),
                                edge_index=torch.LongTensor(pro_edge).transpose(1, 0),
                                y=label,
                                code=code,
                                prot_id=df.loc[idx]['prot_id'])
            lig = torchg.data.Data(x=torch.Tensor(mol_feat),
                                edge_index=torch.LongTensor(mol_edge).transpose(1, 0),
                                y=label,
                                code=code,
                                prot_id=df.loc[idx]['prot_id'])
            data_list.append([pro, lig])
            
        print(f'{len(errors)} codes failed to create graphs')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
            
        def collate(data_list):
            batchA = torchg.data.Batch.from_data_list([data[0] for data in data_list])
            batchB = torchg.data.Batch.from_data_list([data[1] for data in data_list])
            return batchA, batchB
        
        self._data_pro, self._data_mol = collate(data_list)
        
        print('Saving...')
        torch.save(self._data_pro, self.processed_paths[1])
        torch.save(self._data_mol, self.processed_paths[2])


class PDBbindDataset(BaseDataset): # InMemoryDataset is used if the dataset is small and can fit in CPU memory
    def __init__(self, save_root='../data/PDBbindDataset/msa', 
                 bind_root='../data/v2020-other-PL', 
                 aln_dir=None,
                 cmap_threshold=8.0, shannon=False, *args, **kwargs):
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
        `shannon` : bool, optional
            If True, shannon entropy instead of PSSM matrix is used for 
            protein features, by default False.
            
        *args and **kwargs sent to superclass `torch_geometric.data.InMemoryDataset`.
        """        
        super(PDBbindDataset, self).__init__(save_root, data_root=bind_root,
                                             aln_dir=aln_dir, cmap_threshold=cmap_threshold,
                                             shannon=shannon, *args, **kwargs)
        
    # for data augmentation override the transform method
    @property
    def raw_file_names(self):
        """
        Index files from pdbbind are needed for this dataset:
        
            "INDEX_general_PL_data.2020": List of the "general set" of protein-small 
            ligand complexes with formatted binding data.  

            "INDEX_general_PL_name.2020": List of the "general set" of protein-small 
            ligand complexes with protein names and UniProt IDs.

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
        PDBCode,resolution,release_year,pkd,lig_name
        
        It also generates and saves contact maps for each protein in the dataset.

        Returns
        -------
        pd.DataFrame
            The XY.csv dataframe.
        """
        pdb_codes = os.listdir(self.bind_dir)
        # filter out readme and index folders
        pdb_codes = [p for p in pdb_codes if p != 'index' and p != 'readme']
        
        # creating MSA:
        #NOTE: assuming MSAs are already created, since this would take a long time to do.
        # create_aln_files(df_seq, self.aln_p)
        if self.aln_dir is not None:
            valid_codes =  [c for c in pdb_codes if os.path.isfile(self.aln_p(c))]
            # filters out those that do not have aln file #NOTE: TEMPORARY
            print(f'Number of codes with aln files: {len(valid_codes)} out of {len(pdb_codes)}')
            pdb_codes = valid_codes
        
        assert len(pdb_codes) > 0, 'Too few PDBCodes, need at least 1...'
        
        # creating contact maps:
        seqs = create_save_cmaps(pdb_codes,
                          pdb_p=lambda x: f'{self.bind_dir}/{x}/{x}_protein.pdb',
                          cmap_p=self.cmap_p)
        
        assert len(seqs) == len(pdb_codes), 'Some codes failed to create contact maps'
        df_seq = pd.DataFrame.from_dict(seqs, orient='index', columns=['prot_seq'])
        df_seq.index.name = 'PDBCode'
        
        # getting ligand sequences:
        dict_smi = PDBbindProcessor.get_SMILE(pdb_codes,
                                              dir=lambda x: f'{self.bind_dir}/{x}/{x}_ligand.sdf')
        df_smi = pd.DataFrame.from_dict(dict_smi, orient='index', columns=['SMILE'])
        df_smi.index.name = 'PDBCode'
        
        # Get binding data:
        df_binding = PDBbindProcessor.get_binding_data(self.raw_file_names[0])
        df_binding.drop(columns=['resolution', 'release_year', 'lig_name'], inplace=True)
        
        # Get prot ids data:
        df_pid = PDBbindProcessor.get_name_data(self.raw_file_names[1])
        df_pid.drop(columns=['release_year','prot_name'], inplace=True)
        # contains col: prot_id
        
        
        # merging dataframes:
        df_binding.merge(df_pid, on='PDBCode')
        
        df = df_smi[df_smi.SMILE.notna()].merge(df_binding, on='PDBCode')
        df = df.merge(df_seq, on='PDBCode')
        
        # changing index name to code (to match with super class):
        df.index.name = 'code'
        df.to_csv(self.processed_paths[0])
        return df
    
    
class DavisKibaDataset(BaseDataset):
    def __init__(self, save_root='../data/DavisKibaDataset/', 
                 data_root='../data/davis_kiba/davis/', 
                 aln_dir='../data/davis_kiba/davis/aln/',
                 cmap_threshold=-0.5, shannon=True, *args, **kwargs):
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
        `shannon` : bool, optional
            If True, shannon entropy instead of PSSM matrix is used for 
            protein features, by default False.
            
        *args and **kwargs sent to superclass `torch_geometric.data.InMemoryDataset`.
        """
        super(DavisKibaDataset, self).__init__(save_root, data_root=data_root,
                                               aln_dir=aln_dir, cmap_threshold=cmap_threshold,
                                               shannon=shannon, *args, **kwargs)
    
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
        return [f'{self.data_root}/proteins.txt',
                f'{self.data_root}/ligands_can.txt',
                f'{self.data_root}/Y']
    
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
        prot_seq = json.load(open(f'{self.data_root}/proteins.txt', 'r'), object_hook=OrderedDict)
        codes = list(prot_seq.keys())
        prot_seq = list(prot_seq.values())
        
        # get ligand sequences (order is important since they are indexed by row in affinity matrix):
        ligand_seq = json.load(open(f'{self.data_root}/ligands_can.txt', 'r'), 
                               object_hook=OrderedDict)
        ligand_seq = list(ligand_seq.values())
        
        # Get binding data:
        affinity_mat = pickle.load(open(f'{self.data_root}/Y', 'rb'), encoding='latin1')
        lig_r, prot_c = np.where(~np.isnan(affinity_mat)) # index values corresponding to non-nan values
        
        # checking alignment files present for each code:
        no_aln = []
        if self.aln_dir is not None:
            no_aln = [c for c in codes if (not check_aln_lines(self.aln_p(c)))]
                    
            # filters out those that do not have aln file
            print(f'Number of codes with invalid aln files: {len(no_aln)} out of {len(codes)}')
            
        # Checking that contact maps are present for each code:
        no_cmap = [c for c in codes if not os.path.isfile(self.cmap_p(c))]
        print(f'Number of codes without cmap files: {len(no_cmap)} out of {len(codes)}')
        
        invalid_codes = set(no_aln + no_cmap)
        # filtering out invalid codes:
        lig_r = [r for i,r in enumerate(lig_r) if codes[prot_c[i]] not in invalid_codes]
        prot_c = [c for c in prot_c if codes[c] not in invalid_codes]
        
        # creating binding dataframe:
        #code,SMILE,pkd,prot_seq
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
        
        df.to_csv(self.processed_paths[0])
        return df
    
    