from tqdm import tqdm
import torch
import numpy as np

def train(model, train_loader, val_loader, device,
          epochs=10, lr=0.001, **kwargs) -> dict:
    """
    Training loop for graph models.
    Note that **kwargs is used to pass any additional arguments to the optimizer.

    Parameters
    ----------
    `model` : _type_
        Model to be trained.
    `train_loader` : _type_
        Data loader for training set.
    `val_loader` : _type_
        Data loader for validation set.
    `device` : _type_
        Device to train on.
    `epochs` : int, optional
        number of epochs, by default 10
    `lr` : float, optional
        learning rate, by default 0.001
        
    Returns
    -------
    dict
        Dictionary containing training and validation loss for each epoch.
    """
    CRITERION = torch.nn.MSELoss()
    OPTIMIZER = torch.optim.Adam(model.parameters(), lr=lr, **kwargs)

    logs = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Training loop
        with tqdm(total=len(train_loader), 
                  desc=f"Epoch {epoch+1}/{epochs}", 
                  unit="batch") as progress_bar:
            model.train()
            train_loss = 0.0
            for batch_pro, batch_mol in train_loader:
                batch_pro = batch_pro.to(device)
                batch_mol = batch_mol.to(device)
                
                # Forward pass
                predictions = model(batch_pro, batch_mol)

                # Compute loss
                loss = CRITERION(predictions, batch_pro.y) # y is labels
                train_loss += loss.item()

                # Backward pass and optimization
                OPTIMIZER.zero_grad()
                loss.backward()
                OPTIMIZER.step()

                # Update tqdm progress bar
                progress_bar.set_postfix({"Train Loss": train_loss / (progress_bar.n + 1)})
                progress_bar.update(1)

            # Compute average training loss for the epoch
            train_loss /= len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_pro, batch_mol in val_loader:
                batch_pro = batch_pro.to(device)
                batch_mol = batch_mol.to(device)
                
                # Forward pass
                predictions = model(batch_pro, batch_mol)

                # Compute loss
                loss = CRITERION(predictions, batch_pro.y) # y is labels
                val_loss += loss.item()

            # Compute average validation loss for the epoch
            val_loss /= len(val_loader)

        # Print training and validation loss for the epoch
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logs['train_loss'].append(train_loss)
        logs['val_loss'].append(val_loss)
    return logs


def test(model, test_loader, device):
    # After training, you can evaluate the model on the test set if needed
    model.eval()
    test_loss = 0.0
    CRITERION = torch.nn.MSELoss()
    
    pred = np.array([])
    actual = np.array([])
    
    with torch.no_grad():
        for batch_pro, batch_mol in test_loader:
            batch_pro = batch_pro.to(device)
            batch_mol = batch_mol.to(device)
            
            # Forward pass
            predictions = model(batch_pro, batch_mol)
            pred = np.append(pred, 
                            predictions.detach().cpu().numpy().flatten())
            actual = np.append(actual, 
                            batch_pro.y.detach().cpu().numpy().flatten())

            # Compute loss
            loss = CRITERION(predictions, batch_pro.y) # y is labels
            test_loss += loss.item()

        # Compute average test loss
        test_loss /= len(test_loader)

    print(f"Test Loss: {test_loss:.4f}")
    
    return pred, actual
