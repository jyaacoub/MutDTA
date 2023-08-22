import itertools
import gc
from typing import Tuple

from tqdm import tqdm
import torch
import numpy as np

from src.train_test.utils import train_val_test_split, CheckpointSaver
from src.models.utils import BaseModel
from src.models.prior_work import DGraphDTA
from src.data_processing.datasets import BaseDataset
from torch_geometric.loader import DataLoader
from src.utils.loader import Loader


def train(model: BaseModel, train_loader:DataLoader, val_loader:DataLoader, 
          device: torch.device, epochs=10, lr=0.001, 
          saver: CheckpointSaver=None, silent=False, **kwargs) -> dict:
    """
    Training loop for graph models.
    Note that **kwargs is used to pass any additional arguments to the optimizer.

    Parameters
    ----------
    `model` : BaseModel
        Model to be trained.
    `train_loader` : DataLoader
        Data loader for training set.
    `val_loader` : DataLoader
        Data loader for validation set.
    `device` : torch.device
        Device to train on.
    `epochs` : int, optional
        number of epochs, by default 10
    `lr` : float, optional
        learning rate, by default 0.001
    `saver` : CheckpointSaver, optional
        CheckpointSaver object that decides when to stop training and saves checkpoint 
        of best performant version of the model if none provided then will train for 
        all epochs, by default None
        
    Returns
    -------
    dict
        Dictionary containing training and validation loss for each epoch.
    """
    saver = saver or CheckpointSaver(model, train_all=True)
    CRITERION = torch.nn.MSELoss()
    OPTIMIZER = torch.optim.Adam(model.parameters(), lr=lr, **kwargs)

    logs = {'train_loss': [], 'val_loss': []}
    
    # pre training validation test:
    val_loss = test(model, val_loader, device, CRITERION)[0]
    # ensures that we save the best model arch even when loading from existing.
    saver.early_stop(val_loss, 0) 
    
    # validation loss before training
    if not silent:
        print(f"Epoch {0}/{epochs}: Val Loss: {val_loss:.4f} ")
    # we dont save it to logs since this will not be useful information
    #   - either very high or the same as prev model (redundant)
    
    for epoch in range(1, epochs+1):
        # Training loop
        model.train()
        train_loss = 0.0
        with tqdm(total=len(train_loader), 
                  desc=f"Epoch {epoch}/{epochs}", 
                  unit="batch", disable=silent) as progress_bar:
            for data in train_loader:
                batch_pro = data['protein'].to(device)
                batch_mol = data['ligand'].to(device)
                labels = data['y'].reshape(-1,1).to(device)
                
                # Forward pass
                predictions = model(batch_pro, batch_mol)

                # Compute loss
                loss = CRITERION(predictions, labels)
                train_loss += loss.item()

                # Backward pass and optimization
                OPTIMIZER.zero_grad()
                loss.backward()
                OPTIMIZER.step()

                # Update tqdm progress bar
                progress_bar.set_postfix({"Train Loss": train_loss / (progress_bar.n + 1)})
                progress_bar.update(1)
                del batch_pro, batch_mol, labels, predictions, loss
                gc.collect()
                torch.cuda.empty_cache()

            # Compute average training loss for the epoch
            train_loss /= len(train_loader)
        
        # Validation loop
        val_loss = test(model, val_loader, device, CRITERION)[0]

        logs['train_loss'].append(train_loss)
        logs['val_loss'].append(val_loss)
        
        if saver.early_stop(val_loss, epoch) and not silent:
            print(f'Early stopping at epoch {epoch}, best epoch was {saver.best_epoch}')
            break
        
        # Print training and validation loss for the epoch
        if not silent:
            print(f"Epoch {epoch}/{epochs}: Train Loss: {train_loss:.4f}, "+\
                f"Val Loss: {val_loss:.4f}, "+\
                f"Best Val Loss: {saver.min_val_loss:.4f} @ Epoch {saver.best_epoch}")

    logs['best_epoch'] = saver.best_epoch
    return logs


def test(model, test_loader, device, CRITERION=None) -> Tuple[float, np.ndarray, np.ndarray]:
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

    Returns
    -------
    Tuple[float, np.ndarray, np.ndarray]
        Tuple of test loss, predictions, and actual labels.
    """
    # After training, you can evaluate the model on the test set if needed
    model.eval()
    test_loss = 0.0
    CRITERION = CRITERION or torch.nn.MSELoss()
    
    pred = np.array([])
    actual = np.array([])
    
    with torch.no_grad():
        for data in test_loader:
            batch_pro = data['protein'].to(device)
            batch_mol = data['ligand'].to(device)
            labels = data['y'].reshape(-1,1).to(device)
            
            # Forward pass
            predictions = model(batch_pro, batch_mol)

            # Compute loss
            loss = CRITERION(predictions, labels) # y is labels
            test_loss += loss.item()
            
            # save predictions and actual labels
            pred = np.append(pred, 
                            predictions.detach().cpu().numpy().flatten())
            actual = np.append(actual, 
                               labels.detach().cpu().numpy().flatten())

        # Compute average test loss
        test_loss /= len(test_loader)
    
    return test_loss, pred, actual


def train_tune(config, train_dataset:BaseDataset, val_dataset:BaseDataset):
    from ray.air import session
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = config['model_cls'](dropout=config['dropout'])
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
              lr=config['lr'], silent=True, saver=saver)
        val_loss = logs['val_loss'][0]

        # Send the current training result back to Tune
        session.report({"val_loss": val_loss})

        if i % 5 == 0:
            # This saves the model to the trial directory
            torch.save(model.state_dict(), "./model.pth")
    


def grid_search(pdb_dataset, TRAIN_SPLIT=0.8, VAL_SPLIT=0.1, RAND_SEED=42,
                epoch_opt = [5],
                weight_opt = ['kiba', 'davis', 'random'],
                batch_size_opt = [32, 64, 128],
                lr_opt = [0.0001, 0.001, 0.01],
                dropout_opt = [0.1, 0.2, 0.3]):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    model_results = {}
    for WEIGHTS, BATCH_SIZE, \
        LEARNING_RATE, DROPOUT, NUM_EPOCHS in \
        itertools.product(weight_opt, batch_size_opt, \
                        lr_opt, dropout_opt, epoch_opt):
        MODEL_KEY = f'{WEIGHTS}W_{BATCH_SIZE}B_{LEARNING_RATE}LR_{DROPOUT}DO_{NUM_EPOCHS}E'
        print(f'\n\n{MODEL_KEY}')
        

        model = DGraphDTA(dropout=DROPOUT)
        model.to(device)
        assert WEIGHTS in weight_opt, 'WEIGHTS must be one of: kiba, davis, random'
        if WEIGHTS != 'random':
            model_file_name = f'results/model_checkpoints/prior_work/DGraphDTA_{WEIGHTS}_t2.model'
            model.load_state_dict(torch.load(model_file_name, map_location=device))

        train_loader, val_loader, test_loader = train_val_test_split(pdb_dataset, 
                            train_split=TRAIN_SPLIT, val_split=VAL_SPLIT,
                            shuffle_dataset=True, random_seed=RAND_SEED, 
                            batch_size=BATCH_SIZE)
        logs = train(model, train_loader, val_loader, device, 
                epochs=NUM_EPOCHS, lr=LEARNING_RATE)

        loss, pred, actual = test(model, test_loader, device)
        model_results[MODEL_KEY] = {'test_loss': loss, 
                                    'train_loss': logs['train_loss'][-1],
                                    'val_loss': logs['val_loss'][-1]}
    return model_results

