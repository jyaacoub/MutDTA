import signal, os, subprocess
from typing import Tuple
from copy import deepcopy

import submitit

import numpy as np
import pandas as pd


import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset

from src.models.utils import BaseModel
from src.data_processing.datasets import DavisKibaDataset, BaseDataset


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


class CheckpointSaver:
    # Adapted from https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    def __init__(self, model:BaseModel, save_path=None, train_all=True, patience=30, 
                 min_delta=0.2, debug=False, dist_rank:int=None):
        """
        Early stopping and checkpoint saving class.

        Parameters
        ----------
        `model` : BaseModel
            The model to track.
        `save_path` : str, optional
            Path to save model checkpoints if None provided will save locally with name 
            of model class in the name of the file, by default None
        `train_all` : bool, optional
            Overrides early stopping behavior by never stopping if true, by default False
        `patience` : int, optional
            Number of epochs to wait for before stopping, by default 5
        `min_delta` : float, optional
            The minimum change in val loss to be considered a significant degradation, 
            by default 0.03
        `dist_rank` : int, optional
            Whether or not this is for a distributed run, if it is this number will indicate 
            the rank of this particular process, by default None.
        """
        self.train_all = train_all 
        self.patience = patience
        self.min_delta = min_delta
        self.debug = debug
        self.dist_rank = dist_rank
        
        self.new_model(model, save_path)
    
    @property
    def save_path(self):
        return self._save_path or \
            f'./{self.model.__class__.__name__}_{self.best_epoch}E.model'
            
    @save_path.setter
    def save_path(self, save_path:str):
        self._save_path = save_path
    
    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, model:BaseModel):
        self._model = model
        if model is not None:
            if isinstance(model, nn.DataParallel):
                self.best_model_dict = model.module.state_dict()
            else:
                self.best_model_dict = model.state_dict()

    def new_model(self, model:BaseModel, save_path:str):
        """Updates internal model and resets internal state"""
        self.model = model
        self.save_path = save_path
        self.best_epoch = 0
        self.stop_epoch = -1
        self._counter = 0
        self.min_val_loss = np.inf
        
    def early_stop(self, validation_loss, curr_epoch):
        """Check if early stopping condition is met. Call after each epoch."""
        if self.debug: return False
        assert self.model is not None, 'model is None, please set model first'
        # save model if validation loss is lower than previous best
        if validation_loss < self.min_val_loss:
            self.min_val_loss = validation_loss
            self._counter = 0
            if isinstance(self._model, nn.DataParallel):
                self.best_model_dict = deepcopy(self._model.module.state_dict())
            else:
                self.best_model_dict = deepcopy(self._model.state_dict())
            self.best_epoch = curr_epoch
            # saves new best state dict
            self.save(f'{self.save_path}_tmp', rm_tmp=False) 
        
        # early stopping if validation loss doesnt improve for `patience` epochs
        elif (not self.train_all) and \
            (validation_loss > (self.min_val_loss + self.min_delta)):
            self._counter += 1
            if self._counter >= self.patience:
                self.stop_epoch = curr_epoch
                return True
        return False

    def save(self, path:str=None, silent=False, rm_tmp=True):
        # only allow the main process to save models
        if self.dist_rank is None or self.dist_rank == 0:
            path = path or self.save_path
            # save model default path is model class name + best epoch
            torch.save(self.best_model_dict, path)
            if not silent: print(f'Model saved to: {path}')
            if rm_tmp and os.path.isfile(f'{path}_tmp'): 
                os.remove(f'{path}_tmp')
        elif not silent:
            print(f'WARNING: No saving on non-main process')
        
    def __repr__(self) -> str:
        return f'save path: {self.save_path}'+ \
               f'min val loss: {self.min_val_loss}'+ \
               f'stop epoch: {self.stop_epoch}'+ \
               f'best epoch: {self.best_epoch}'
    
def print_device_info(device:torch.device|int) -> torch.cuda.device:
    if (type(device) is int  and device == -1) or \
        (type(device) is torch.device and device.type != 'cuda'):
        print('Device is not cuda, no device info available')
        return None
    prop = torch.cuda.get_device_properties(device)
    r = torch.cuda.memory_reserved(device)
    a = torch.cuda.memory_allocated(device)
    f = r-a  # free inside reserved
    print(f'Device: {device} ({prop.name})')
    print(f'v{prop.major}.{prop.minor} ({prop.multi_processor_count} SMs)')
    print(f'\ttotal:     {prop.total_memory/1e9:<10.3f}GB')
    print(f'\treserved:  {r/1e6:<10.3f}MB')
    print(f'\tallocated: {a/1e6:<10.3f}MB')
    
    return prop


def debug(model: BaseModel, data_loader:DataLoader,
          device: torch.device) -> Tuple[torch.Tensor]:
    """
    Runs a single batch through model for testing that it has been 
    set up properly
    
    returns model.train() and model.eval() outputs
    """
    print('\tDEBUG MODE - Running single batch through model')
    for data in data_loader:
        batch_pro = data['protein'].to(device)
        batch_mol = data['ligand'].to(device)
        labels = data['y'].reshape(-1,1).to(device)
        
        # Forward pass
        model.train()
        train = model(batch_pro, batch_mol)
        model.eval()
        eval = model(batch_pro, batch_mol)
        print('\tDEBUG MODE - PASSED!')
        
        
        return train, eval

## ==================== Distributed Training ==================== ##
def init_node(args):    
    args.ngpus_per_node = torch.cuda.device_count()

    # find the common host name on all nodes
    cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
    stdout = subprocess.check_output(cmd.split())
    host_name = stdout.decode().splitlines()[0] # first node is the host
    args.dist_url = f'tcp://{host_name}:{args.port}'

    # distributed parameters
    args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
    args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
    
def init_dist_gpu(args):
    job_env = submitit.JobEnvironment()
    args.gpu = job_env.local_rank
    args.rank = job_env.global_rank

    # PyTorch calls to setup gpus for distributed training
    dist.init_process_group(backend='gloo', init_method=args.dist_url, 
                            world_size=args.world_size, rank=args.rank)
    
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    
    torch.cuda.set_device(args.gpu)
    # cudnn.benchmark = True # not needed since we include dropout layers
    dist.barrier()

    # # disabling printing if not master process:
    # import builtins as __builtin__
    # builtin_print = __builtin__.print

    # def print(*args, **kwargs):
    #     force = kwargs.pop('force', False)
    #     if (args.rank == 0) or force:
    #         builtin_print(*args, **kwargs)

    # __builtin__.print = print
