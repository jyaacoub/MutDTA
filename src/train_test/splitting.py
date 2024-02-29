from typing import Tuple

import numpy as np
import pandas as pd

import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.loader import DataLoader

from src.models.utils import BaseModel
from src.data_prep.datasets import BaseDataset

# Creating data indices for training and validation splits:
def train_val_test_split(dataset: BaseDataset, 
                         train_split=.8, val_split=.1, 
                         shuffle_dataset=True, random_seed=None,
                         batch_train=128, use_refined=False, 
                         split_by_prot=True) -> tuple[DataLoader]:
    """
    Splits up InMemoryDataset into train, val, and test loaders.

    Parameters
    ----------
    `dataset` : BaseDataset
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
        # getting counts for each unique protein
        prot_counts = dataset.get_protein_counts()
        prots = list(prot_counts.keys())
        np.random.shuffle(prots)
        
        # selecting from unique proteins until we have enough for train
        count = 0
        selected = {}
        for p in prots: # O(k); k = number of proteins
            if count + prot_counts[p] <= tr_size:
                selected[p] = True
                count += prot_counts[p]
        
        # looping through dataset to get indices for train and val
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

def balanced_kfold_split(dataset: BaseDataset,
                         k_folds:int=5, test_split=.1,
                         shuffle_dataset=True, random_seed=None,
                         batch_train=128,
                         verbose=False) -> tuple[DataLoader]:
    """
    Same as train_val_test_split_kfold but we make considerations for the 
    fact that each protein might not show up in equal proportions (e.g.: 
    in PDBbind there are some proteins that appear only once and others 
    that appear hundreds of times).

    Parameters
    ----------
    `dataset` : BaseDataset
        The dataset to split
    `test_split` : float, optional
        What percentage of the dataset to use for testing, by default .1
    `k_folds` : int, optional
        Number of folds to split the training set into, by default 5
    `shuffle_dataset` : bool, optional
        self explainatory, by default True
    `random_seed` : _type_, optional
        seed for shuffle, by default None
    `batch_train` : int, optional
        size of batch, by default 128

    Returns
    -------
    tuple[DataLoader]
        Train, val, and test loaders
    """
    if random_seed is not None: 
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    # Get size for each dataset and indices
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    test_size = int(test_split * dataset_size)
    if shuffle_dataset:
        np.random.shuffle(indices) # in-place shuffle

    ########## Sampling for test set ##########
    # getting counts for each unique protein
    prot_counts = dataset.get_protein_counts()
    prots = list(prot_counts.keys())
    np.random.shuffle(prots)
    
    #### Getting test set 
    count = 0
    test_prots = {}
    for p in prots: # O(k); k = number of proteins
        if count + prot_counts[p] <= test_size:
            test_prots[p] = True
            count += prot_counts[p]
            
    # looping through dataset to get indices for test
    test_indices = [i for i in range(dataset_size) if dataset[i]['prot_id'] in test_prots]
    test_sampler = SubsetRandomSampler(test_indices)
    test_loader = DataLoader(dataset, batch_size=1, # batch size 1 for testing
                            sampler=test_sampler, pin_memory=True)
            
    # removing selected proteins from prots
    prots = [p for p in prots if p not in test_prots]
    print(f'Number of unique proteins in test set: {len(test_prots)} == {count} samples')
    
    ########## split remaining proteins into k_folds ##########
    # Steps for this basically follow Greedy Number Partitioning
    # tuple of (list of proteins, total weight, current-score):
    prot_folds = [[[], 0, -1] for i in range(k_folds)] 
    # score = fold.weight - abs(fold.weight/len(fold) - item.weight)
    prot_counts = sorted(list(prot_counts.items()), key=lambda x: x[1], reverse=True)
    for p, c in prot_counts:
        # Update scores for each fold
        for fold in prot_folds:
            f_len = len(fold[0])
            if f_len == 0:
                continue
            
            # calculate score for adding protein to fold
            fold[2] = fold[1] #- abs(fold[1]/f_len - c)
            
        # Finding optimal fold to add protein to (minimize score)
        best_fold = min(prot_folds, key=lambda x: x[2])
        
        # Add protein to fold
        best_fold[0].append(p)
        # update weight
        best_fold[1] += c
    
    if verbose:
        print(f'{"#":>10} | {"num_prots":^10} | {"total_count":^12} | {"final_score":^10}')
        print('-'*53)
        for i,f in enumerate(prot_folds): 
            print(f'{"Fold "+str(i):>10} | {len(f[0]):^10} | {f[1]:^12} | {f[2]:^10}')
    
    # convert folds to set after done selecting for faster lookup
    prot_folds = [set(f[0]) for f in prot_folds]

    ########## create train and val loaders ##########
    train_loaders, val_loaders = [], []
    for i, fold in enumerate(prot_folds):
        train_indices, val_indices = [], []
        for idx in range(dataset_size): # O(n)
            if dataset[idx]['prot_id'] in fold:
                val_indices.append(idx)
            elif dataset[idx]['prot_id'] not in test_prots:
                train_indices.append(idx)
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        train_loaders.append(DataLoader(dataset, batch_size=batch_train,
                                        sampler=train_sampler, pin_memory=True))
        val_loaders.append(DataLoader(dataset, batch_size=batch_train,
                                        sampler=val_sampler, pin_memory=True))
        
    t_count = len(train_sampler)
    v_count = len(val_sampler)
    te_count = len(test_sampler)
    
    print(f'       Folds: {len(train_loaders)}')
    print(f' Train0 size: {t_count}')
    print(f'   Val0 size: {v_count}')
    print(f'   Test size: {te_count}')
    print(f'            = {t_count+v_count+te_count}')
    print(f'Dataset size: {dataset_size}')
    assert te_count > 0, 'Test set is empty'
    
    return train_loaders, val_loaders, test_loader