import numpy as np
import torch
from torch import optim, nn
from torch_geometric.data.batch import DataBatch

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

from src.utils import config as cfg
from src.utils.loader import Loader
from src.models.utils import BaseModel
from src.analysis.metrics import concordance_index


# define the LightningModule
# note device calls not needed
# min and max epochs is set at trainer init
#NOTE: BETTER OPTION IS TO SET A "MAX_TIME"
class LitModel(L.LightningModule):
    def __init__(self, model:str|BaseModel, dataset:str,
                    pro_feature:str=None, pro_edge:str=None, dropout:float=None, 
                    ligand_feature:str=None, ligand_edge:str=None, learning_rate=1e-4, **kwargs):
        """
        Lightning wrapper for training models. Includes EarlyStopping, ModelCheckpoint callbacks 
        and ReduceLROnPlateau scheduler.

        Args:
            model (str | BaseModel): str for model (see cfg.model_opt) or the model itself
            
            dataset (str): dataset name to add to checkpoint path and avoid overwrites.
            # Additional features for model loading if one isnt passed in (see src.utils.loader.Loader).
            
            pro_feature (str, optional): _description_. Defaults to None.
            pro_edge (str, optional): _description_. Defaults to None.
            dropout (float, optional): _description_. Defaults to None.
            ligand_feature (str, optional): _description_. Defaults to None.
            ligand_edge (str, optional): _description_. Defaults to None.
            
            learning_rate (_type_, optional): _description_. Defaults to 1e-4.
        """
        super().__init__()
        
        assert dataset is not None and dataset in cfg.DATA_OPT, \
            f"Select dataset for checkpoint path from {cfg.DATA_OPT}"
        self.dataset = dataset
        
        if type(model) is str:
            self.model = Loader.init_model(model, pro_feature, pro_edge, dropout, 
                                        ligand_feature, ligand_edge)
            self.save_hyperparameters() # saves all parameters for easy loading later on.
        else:
            self.model = model
            # dont want to save the entire module since it would be too large and redundant
            self.save_hyperparameters(ignore=['model'])
            
        self.learning_rate = learning_rate
        self.loss = nn.MSELoss()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optim.ReduceLROnPlateau(optimizer, mode='min', patience=50, 
                                  threshold=1e-4, min_lr=5e-5, factor=0.8,
                                  verbose=True),
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
                "name":"lr"
            },
        }
        
    def configure_callbacks(self):
        """
        A list of callbacks which will extend the list of callbacks in the Trainer.
        Replaces any existing callbacks in Trainer.
        """
        early_stop = EarlyStopping(monitor="val_loss", mode='min', patience=50, min_delta=0.2)
        dirpath = f"{cfg.LIT_CHECKPOINTS}/{type(self.model).__name__}/{self.dataset}/",
        checkpoint = ModelCheckpoint(dirpath=dirpath,
                                     # uses logged metrics for file name
                                     filename='{epoch}-{val_loss:.2f}-{val_cindex:.2f}', 
                                     save_top_k=1, # only save best model
                                     save_on_train_epoch_end=True,
                                     monitor="val_loss", mode='min')
        return [early_stop, checkpoint]
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """To handle custom batch created by torch_geometric.data.DataBatch"""
        if isinstance(batch, dict):
            batch['protein'] = batch['protein'].to(device)
            batch['ligand'] = batch['ligand'].to(device)
            batch['y'] = batch['y'].to(device)
        else:
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        return batch
   
    def forward(self, *args, **kwargs): # for inference on single val
        """Run data_pro, data_mol through model"""
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        batch_pro = batch['protein']
        batch_mol = batch['ligand']
        labels = batch['y'].reshape(-1,1)
        
        # Forward pass
        predictions = self.model(batch_pro, batch_mol)

        # Compute loss
        train_loss = self.loss(predictions, labels)
        self.log("train_loss", train_loss, on_epoch=True)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        batch_pro = batch['protein']
        batch_mol = batch['ligand']
        labels = batch['y'].reshape(-1,1)
        
        # Forward pass
        predictions = self.model(batch_pro, batch_mol)
        
        # Compute loss
        val_loss = self.loss(predictions, labels)
        self.log("val_loss", val_loss, on_epoch=True)
        
        # compute cindex
        pred = np.append(pred, 
                        predictions.detach().cpu().numpy().flatten())
        actual = np.append(actual, 
                            labels.detach().cpu().numpy().flatten())
        val_cindex = concordance_index(actual, pred)
        self.log('val_cindex', val_cindex)
        return val_loss, val_cindex
    

    
    