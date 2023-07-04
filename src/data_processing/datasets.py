import torch, os
import numpy as np
import pandas as pd
from torch_geometric import data as geo_data
from tqdm import tqdm

from src.feature_extraction import smile_to_graph, target_to_graph
from src.feature_extraction.protein import create_save_cmaps
from src.data_processing import PDBbindProcessor

#  See: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
# for details on how to create a dataset
class PDBbindDataset(geo_data.InMemoryDataset): # InMemoryDataset is used if the dataset is small and can fit in CPU memory
    def __init__(self, save_root='../data/pytorch_PDBbind/', 
                 bind_root='../data/v2020-other-PL',  *args, **kwargs):
        """
        Dataset for PDBbind data. This dataset is used to train graph models.

        Parameters
        ----------
        `save_root` : str, optional
            Path to processed dir, by default '../data/pytorch_PDBbind/'
        `bind_root` : str, optional
            Path to raw pdbbind files, by default '../data/v2020-other-PL'
        """
        self.bind_dir = bind_root
        self.cmap_threshold = 8.0
        super(PDBbindDataset, self).__init__(save_root, *args, **kwargs)
        self.load()
        
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

    @property
    def processed_file_names(self):
        # XY.csv cols: PDBCode,pkd,SMILE,prot_seq
        return ['data_pro.pt','data_mol.pt', 'XY.csv']

    def load(self, path: str=None):
        path = self.processed_dir if path is None else path
        self._data_pro = torch.load(path + '/data_pro.pt')
        self._data_mol = torch.load(path + '/data_mol.pt')
        
    # def download(self):
    #     # geo_data.download_url('https://pdbbind.oss-cn-hangzhou.aliyuncs.com/download/PDBbind_v2020_other_PL.tar.gz', 
    #     #                         self.bind_dir)
    #     raise NotImplementedError(
    #         'Download not supported please download the data manually from PDBbind website')
            
    def process(self):
        pdb_codes = os.listdir(self.bind_dir)
        # filter out readme and index folders
        pdb_codes = [p for p in pdb_codes if p != 'index' and p != 'readme']
        # creating contact maps:
        cmap_p = lambda x: f'{self.bind_dir}/{x}/{x}_cmap_CB_lone.npy'
        seqs = create_save_cmaps(pdb_codes,
                          pdb_p=lambda x: f'{self.bind_dir}/{x}/{x}_protein.pdb',
                          cmap_p=cmap_p)
        
        assert len(seqs) == len(pdb_codes), 'Some codes failed to create contact maps'
        df_seq = pd.DataFrame.from_dict(seqs, orient='index', columns=['prot_seq'])
        df_seq.index.name = 'PDBCode'
        
        # getting ligand sequences:
        dict_smi = PDBbindProcessor.get_SMILE(pdb_codes,
                                              dir=lambda x: f'{self.bind_dir}/{x}/{x}_ligand.sdf')
        df_smi = pd.DataFrame.from_dict(dict_smi, orient='index', columns=['SMILE'])
        df_smi.index.name = 'PDBCode'
        
        # Get binding data:
        df_binding = PDBbindProcessor.get_binding_data(f'{self.bind_dir}/index/INDEX_general_PL_data.2020')
        df_binding.drop(columns=['resolution', 'release_year', 'lig_name'], inplace=True)
        
        # merging dataframes:
        df = df_smi[df_smi.SMILE.notna()].merge(df_binding, on='PDBCode')
        df = df.merge(df_seq, on='PDBCode')
        df.to_csv(self.processed_dir + '/XY.csv')
        
        
        # creating the dataset:
        data_list = []
        errors = []
        for code in tqdm(df.index, 'Extracting node features and creating graphs'):
            cmap = np.load(cmap_p(code))
            pro_seq = df.loc[code]['prot_seq']
            lig_seq = df.loc[code]['SMILE']
            label = df.loc[code]['pkd']
            label = torch.Tensor([[label]])
            
            pro_size, pro_feat, pro_edge = target_to_graph(pro_seq, cmap, 
                                                           threshold=self.cmap_threshold)
            try:
                mol_size, mol_feat, mol_edge = smile_to_graph(lig_seq)
            except ValueError:
                errors.append(code)
                continue
            
            pro = geo_data.Data(x=torch.Tensor(pro_feat),
                                edge_index=torch.LongTensor(pro_edge).transpose(1, 0),
                                y=label,
                                code=code)
            lig = geo_data.Data(x=torch.Tensor(mol_feat),
                                edge_index=torch.LongTensor(mol_edge).transpose(1, 0),
                                y=label,
                                code=code)
            data_list.append([pro, lig])
            
        print(f'{len(errors)} codes failed to create graphs')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
            
        def collate(data_list):
            batchA = geo_data.Batch.from_data_list([data[0] for data in data_list])
            batchB = geo_data.Batch.from_data_list([data[1] for data in data_list])
            return batchA, batchB
        
        self._data_pro, self._data_mol = collate(data_list)
        
        torch.save(self._data_pro, self.processed_dir + '/data_pro.pt')
        torch.save(self._data_mol, self.processed_dir + '/data_mol.pt')

    def __len__(self):
        return len(self._data_mol)

    def __getitem__(self, idx):
        return self._data_pro[idx], self._data_mol[idx]
    