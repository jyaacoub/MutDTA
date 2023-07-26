import numpy as np
import pandas as pd
import torch
from collections import Counter
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset


# Creating data indices for training and validation splits:
def train_val_test_split(dataset: InMemoryDataset, 
                         train_split=.8, val_split=.1, 
                         shuffle_dataset=True, random_seed=None,
                         batch_train=128, use_refined=False, 
                         split_by_prot=True) -> tuple[DataLoader]:
    """
    Splits up InMemoryDataset into train, val, and test loaders.

    Parameters
    ----------
    `dataset` : InMemoryDataset
        The dataset to split
    `train_split` : float, optional
        How much goes to just training, by default .8
    `val_split` : float, optional
        How much for validation (remainder goes to test), by default .1
    `shuffle_dataset` : bool, optional
        self explainatory, by default True
    `random_seed` : _type_, optional
        seed for shuffle, by default None
    `batch_train` : int, optional
        size of batch, by default 128
    `use_refined` : bool, optional
        If true, the test set will only consist of refined samples for PDBbind, by default True
    `split_by_prot` : bool, optional
        If true, will take into consideration the proteins when splitting, by default True

    Returns
    -------
    tuple[DataLoader]
        Train, val, and test loaders
    """
    
    if random_seed is not None: 
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    tr_size = int(np.floor(train_split * dataset_size))
    v_size = int(np.floor(val_split * dataset_size))
    te_size = dataset_size - tr_size - v_size
    if shuffle_dataset:
        np.random.shuffle(indices) # in-place shuffle

    if use_refined:
        # test set will contain only refined complexes so that we can compare with vina
        vina_df = pd.read_csv('./results/PDBbind/vina_out/run10.csv', index_col=0)
        # cols are: PDBCode,vina_deltaG(kcal/mol),vina_kd(uM)
        test_indices = []
        train_val_indices = []
        for i in range(dataset_size):
            code = dataset[i]['code']
            if code in vina_df.index:
                test_indices.append(i)
            else:
                train_val_indices.append(i)
            # break once we have enough test indices to meet the size requirement
            if len(test_indices) >= te_size:
                train_val_indices += indices[i+1:]
                break
            
        # now split train_val_indices into train and val
        train_indices, val_indices = indices[:tr_size], indices[tr_size:]
        
    elif split_by_prot:
        prot_counts = dataset.get_protein_counts()
        prots = list(prot_counts.keys())
        np.random.shuffle(prots)
        
        # NOTE: We want to ensure a diverse set of proteins across train, val, and test
        # simple case is where proteins appear exactly once in the dataset
        #     Then we can simply split the proteins into train, val, and test
        
        # the following code doesnt consider diversity, it just splits the proteins into 
        # train, val, and test
        
        count = 0
        selected = {}
        for p in prots: # O(n)
            if count + prot_counts[p] <= tr_size:
                selected[p] = True
                count += prot_counts[p]
          
        train_indices = []
        val_test_indices = []
        for i in range(dataset_size): # O(n)
            if dataset[i]['prot_id'] in selected:
                train_indices.append(i)
            else:
                val_test_indices.append(i)
                
        val_indices, test_indices = val_test_indices[:v_size], val_test_indices[v_size:]
        
    else:
        # normal split into train_val and test
        train_indices, val_test_indices = indices[:tr_size], indices[tr_size:]
        val_indices, test_indices = val_test_indices[:v_size], val_test_indices[v_size:]

    # Creating PT data samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    t_count = len(train_sampler)
    v_count = len(val_sampler)
    te_count = len(test_sampler)
    
    print(f'  Train size: {t_count}')
    print(f'    Val size: {v_count}')
    print(f'   Test size: {te_count}')
    print(f'            = {t_count+v_count+te_count}')
    print(f'Dataset size: {dataset_size}')
    assert te_count > 0, 'Test set is empty, check that use_refined is set correctly'
    
    # create dataloaders
    train_loader = DataLoader(dataset, batch_size=batch_train, 
                            sampler=train_sampler, pin_memory=True)
    val_loader = DataLoader(dataset, batch_size=batch_train,
                            sampler=val_sampler, pin_memory=True)
    test_loader = DataLoader(dataset, batch_size=1, # batch size 1 for testing
                            sampler=test_sampler, pin_memory=True)
    
    return train_loader, val_loader, test_loader