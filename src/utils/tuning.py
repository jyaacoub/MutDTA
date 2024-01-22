from ray.air import session

import torch
from torch_geometric.loader import DataLoader

from src.data_processing.datasets import BaseDataset
from src.train_test.training import CheckpointSaver
from src.utils.loader import Loader
from src.train_test.training import train


def train_tune(config, model:str, pro_feature:str, train_dataset:BaseDataset, val_dataset:BaseDataset):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Loader.init_model(model, pro_feature, config['edge'], config['dropout'])
    model.to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                            shuffle=True,
                            num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                            shuffle=True,
                            num_workers=2)
    
    saver = CheckpointSaver(model, debug=True)
    for i in range(10): # 10 epochs
        logs = train(model, train_loader, val_loader, device, epochs=1, 
              lr_0=config['lr'], silent=True, saver=saver)
        val_loss = logs['val_loss'][0]

        # Send the current training result back to Tune
        session.report({"val_loss": val_loss})

        if i % 5 == 0:
            # This saves the model to the trial directory
            torch.save(model.state_dict(), "./model.pth")
    


