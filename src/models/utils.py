from typing import Tuple
import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader


class BaseModel(nn.Module):
    """
    Base model for printing summary
    """
    def __str__(self) -> str:
        main_str = super().__str__()
        # model size
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2


        main_str += f'\nmodel size: {size_all_mb:.3f}MB'
        return main_str
    
class CheckpointSaver:
    # adapted from https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    def __init__(self, model:BaseModel, save_path=None, train_all=False, 
                 patience=5, min_delta=0.03, save_freq:int=50):
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
        """
        self.train_all = train_all 
        self.patience = patience
        self.min_delta = min_delta 
        self.save_freq = save_freq
        
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
    
    @save_path.setter
    def model(self, model:BaseModel):
        self._model = model
        if model is not None:
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
        assert self.model is not None, 'model is None, please set model first'
        # save model if validation loss is lower than previous best
        if validation_loss < self.min_val_loss:
            self.min_val_loss = validation_loss
            self._counter = 0
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
        path = path or self.save_path   
        # save model default path is model class name + best epoch
        torch.save(self.best_model_dict, path)
        if not silent: print(f'Model saved to: {path}')
        
    def __repr__(self) -> str:
        return f'save path: {self.save_path}'+ \
               f'min val loss: {self.min_val_loss}'+ \
               f'stop epoch: {self.stop_epoch}'+ \
               f'best epoch: {self.best_epoch}'
    
def print_device_info(device:torch.device) -> torch.cuda.device:
    if device.type != 'cuda':
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
    for data in data_loader:
        batch_pro = data['protein'].to(device)
        batch_mol = data['ligand'].to(device)
        labels = data['y'].reshape(-1,1).to(device)
        
        # Forward pass
        model.train()
        train = model(batch_pro, batch_mol)
        model.eval()
        eval = model(batch_pro, batch_mol)
        
        return train, eval