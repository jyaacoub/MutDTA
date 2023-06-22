import torch, os
import numpy as np
import pandas as pd
from torch_geometric import data as geo_data

from src.models.helpers.feature_extraction import smile_to_graph, target_to_graph


def create_dataset_for_test(df_x: pd.DataFrame,
                            cmap_p=lambda c: f'data/PDBbind/{c}/{c}_contact_CB.npy'):
    """
    Creates a dataset for testing the model. This dataset is used to test the model

    Args:
        df_x (pd.DataFrame): dataframe containing the SMILE and protein sequence.
            Index is pdbcode.
        cmap_p (Callable, optional): contact map path. 
                Defaults to lambda c: f'data/PDBbind/{c}/{c}_contact_CB.npy'.

    Returns:
       DTADataset: dataset for testing the model
    """
    print('Getting ligand graphs...')
    df_x['smile_graph'] = df_x['SMILE'].apply(smile_to_graph) #TODO: fix this, "Try using .loc[row_indexer,col_indexer] = value instead"
    # 36 codes fail
    print('\n', sum(df_x.isna()['smile_graph']), 
            'codes failed out of', df_x.shape[0])
    df_x = df_x.dropna()

    print('Getting Target graphs...')
    def temp(row):
        cmap = np.load(cmap_p(row.name))
        return target_to_graph(row['prot_seq'], cmap, threshold=10.5)
    df_x['target_graph'] = df_x.apply(temp, axis=1) # TODO:
    # full warning message:
    #WARNING:/home/jyaacoub/projects/MutDTA/src/models/helpers/dataset_creation.py:24: SettingWithCopyWarning: 
    #WARNING: A value is trying to be set on a copy of a slice from a DataFrame.
    #WARNING: Try using .loc[row_indexer,col_indexer] = value instead
    #WARNING: See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    #WARNING:   df_x['smile_graph'] = df_x['SMILE'].apply(smile_to_graph)
    
    # since structures might be slightly different due to x-ray crystallography resolution            
    print('test entries', df_x.shape[0])
    
    test_dataset = DTADataset(df_x, root='data')

    return test_dataset


class DTADataset(geo_data.InMemoryDataset):
    def __init__(self, df, dataset='kd_ki', root='./tmp', *args, **kwargs):
        super(DTADataset, self).__init__(root, *args, **kwargs)
        self.dataset = dataset
        self.process(df)

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '_data_mol.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
            
    def process(self, df:pd.DataFrame):
        data_list_mol = []
        data_list_pro = []
        for pdbcode, row in df.iterrows():
            c_size, features, edge_index = row['smile_graph']
            target_size, target_features, target_edge_index = row['target_graph']
            labels = row['affinity']
            
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData_mol = geo_data.Data(x=torch.Tensor(features),
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                    y=torch.FloatTensor([labels]))
            GCNData_mol.__setitem__('c_size', torch.LongTensor([c_size]))

            GCNData_pro = geo_data.Data(x=torch.Tensor(target_features),
                                    edge_index=torch.LongTensor(target_edge_index).transpose(1, 0),
                                    y=torch.FloatTensor([labels]))
            GCNData_pro.__setitem__('target_size', torch.LongTensor([target_size]))
            # print(GCNData.target.size(), GCNData.target_edge_index.size(), GCNData.target_x.size())
            data_list_mol.append(GCNData_mol)
            data_list_pro.append(GCNData_pro)

        if self.pre_filter is not None:
            data_list_mol = [data for data in data_list_mol if self.pre_filter(data)]
            data_list_pro = [data for data in data_list_pro if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list_mol = [self.pre_transform(data) for data in data_list_mol]
            data_list_pro = [self.pre_transform(data) for data in data_list_pro]
        self.data_mol = data_list_mol
        self.data_pro = data_list_pro

    def __len__(self):
        return len(self.data_mol)

    def __getitem__(self, idx):
        return self.data_mol[idx], self.data_pro[idx]
    

#prepare the protein and drug pairs
def collate(data_list):
    batchA = geo_data.Batch.from_data_list([data[0] for data in data_list])
    batchB = geo_data.Batch.from_data_list([data[1] for data in data_list])
    return batchA, batchB