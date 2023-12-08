from typing import Tuple

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.models.utils import BaseModel


def simple_train(model: BaseModel, optimizer:torch.optim.Optimizer, 
                 train_loader:DataLoader, device: torch.device,
                 epochs=10) -> dict:
    """
    simple training loop without any checkpointing, lr scheduling, early stopping, 
    or support for distributed training.

    Parameters
    ----------
    `model` : BaseModel
        Model to be trained.
    `optimizer` : torch.optim
        Optimizer to use.
    `train_loader` : DataLoader
        Data loader for training set.
    `device` : torch.device
        Device to train on.
    `epochs` : int, optional
        number of epochs, by default 10
        
    Returns
    -------
    dict
        Dictionary containing training and validation loss for each epoch.
    """
    CRITERION = torch.nn.MSELoss()
    
    model.train()
    for _ in range(epochs):
        # Training loop
        for data in train_loader:
            batch_pro = data['protein'].to(device)
            batch_mol = data['ligand'].to(device)
            labels = data['y'].reshape(-1,1).to(device)
            
            # Forward pass
            predictions = model(batch_pro, batch_mol)

            # Compute loss
            loss = CRITERION(predictions, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def simple_eval(model:BaseModel, data_loader:DataLoader, device:torch.device, 
                CRITERION:torch.nn.Module=None) -> float:
    """
    Run inference on the test set.

    Parameters
    ----------
    `model` : BaseModel
        The model to run inference on.
    `test_loader` : DataLoader
        The test set data loader.
    `device` : torch.device
        Device to run inference on.
    `CRITERION` : torch.nn, optional
        Loss function, by default None

    Returns
    -------
    Tuple[float, np.ndarray, np.ndarray]
        Tuple of test loss, predictions, and actual labels.
    """
    # After training, you can evaluate the model on the test set if needed
    model.eval()
    test_loss = 0.0
    CRITERION = CRITERION or torch.nn.MSELoss()
    
    with torch.no_grad():
        for data in data_loader:
            batch_pro = data['protein'].to(device)
            batch_mol = data['ligand'].to(device)
            labels = data['y'].reshape(-1,1).to(device)
            
            # Forward pass
            predictions = model(batch_pro, batch_mol)

            # Compute loss
            loss = CRITERION(predictions, labels) # y is labels
            test_loss += loss.item()

        # Compute average test loss
        test_loss /= len(data_loader)
    
    return test_loss