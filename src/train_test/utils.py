import signal, os, subprocess
from typing import Tuple

import submitit
import numpy as np

import torch
from torch import nn
import torch.distributed as dist
import numpy as np
import pandas as pd
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset

from src.models.utils import BaseModel


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

class CheckpointSaver:
    # adapted from https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    def __init__(self, model:BaseModel, save_path=None, train_all=False, patience=5, 
                 min_delta=0.03, save_freq:int=50, debug=False, dist_rank:int=None):
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
        `save_freq` : int, optional
            Number of epochs between saving checkpoints. Set to None to stop this behaviour, 
            by default 50.
        `dist_rank` : int, optional
            Whether or not this is for a distributed run, if it is this number will indicate 
            the rank of this particular process, by default None.
        """
        self.train_all = train_all 
        self.patience = patience
        self.min_delta = min_delta 
        self.save_freq = save_freq
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
                self.best_model_dict = self._model.module.state_dict()
            else:
                self.best_model_dict = self._model.state_dict()
            self.best_epoch = curr_epoch
        
        # early stopping if validation loss doesnt improve for `patience` epochs
        elif (not self.train_all) and \
            (validation_loss > (self.min_val_loss + self.min_delta)):
            self._counter += 1
            if self._counter >= self.patience:
                self.stop_epoch = curr_epoch
                return True
            
        if curr_epoch % self.save_freq == 0:
            self.save(f'{self.save_path}_tmp')
        return False

    def save(self, path:str=None, silent=False):
        # only allow the main process to save models
        if self.dist_rank is None or self.dist_rank == 0:
            path = path or self.save_path
            # save model default path is model class name + best epoch
            torch.save(self.best_model_dict, path)
            if not silent: print(f'Model saved to: {path}')
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
def handle_sigusr1(signum, frame):
    # requeues the job
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()    

def init_node(args):    
    args.ngpus_per_node = torch.cuda.device_count()

    # requeue job on SLURM preemption
    signal.signal(signal.SIGUSR1, handle_sigusr1)

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
