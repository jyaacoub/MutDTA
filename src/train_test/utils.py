from typing import Tuple

import torch
from torch_geometric.loader import DataLoader

from src.models.utils import BaseModel

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
