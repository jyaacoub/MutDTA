import numpy as np
import torch
from torch import nn

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
    def __init__(self, model, save_path=None, train_all=False, patience=5, min_delta=0.03):
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
        """
        self.model = model
        self.save_path = save_path
        self.train_all = train_all 
        self.patience = patience
        self.min_delta = min_delta 
        
        self.best_model_dict = model.state_dict()
        self.best_epoch = 0
        self.stop_epoch = -1
        
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss, curr_epoch):
        # save model if validation loss is lower than previous best
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_model_dict = self.model.state_dict()
            self.best_epoch = curr_epoch
        # early stopping if validation loss doesnt improve for `patience` epochs
        elif (not self.train_all) and \
            (validation_loss > (self.min_validation_loss + self.min_delta)):
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_epoch = curr_epoch
                return True
        return False

    def save(self):
        self.save_path = self.save_path or \
            f'./{self.model.__class__.__name__}_{self.best_epoch}E.model'
        torch.save(self.best_model_dict, self.save_path)
        print(f'Model saved to: {self.save_path}')
        
    def __repr__(self) -> str:
        return f'save path: {self.save_path}'+ \
               f'min val loss: {self.min_validation_loss}'+ \
               f'stop epoch: {self.stop_epoch}'+ \
               f'best epoch: {self.best_epoch}'
    
