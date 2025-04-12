import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn.functional as F

class EloModel(nn.Module):
    def __init__(self, input_size):
        super(EloModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)  # Add batch normalization
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)   # Add batch normalization
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)   # Add batch normalization
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(32, 32)     
        self.bn4 = nn.BatchNorm1d(32)   # Add batch normalization
        self.fc5 = nn.Linear(32, 1)  # Output layer for Elo rating prediction


    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x


def train_model(model, train_dataset, val_dataset, hyperparams):
    """
    Train the model using the provided training and validation datasets with L2 regularization.
    
    Parameters:
        model (nn.Module): The neural network model.
        train_dataset (TensorDataset): The training dataset.
        val_dataset (TensorDataset): The validation dataset.
        hyperparams (dict): A dictionary containing hyperparameters including:
            - "l2_lambda": Strength of L2 regularization (weight decay).
    
    Returns:
        model (nn.Module): The trained model.
        training_losses (list): List of training losses over epochs.
        validation_losses (list): List of validation losses over epochs.
    """
    batch_size = hyperparams.get("batch_size", 64)
    epochs = hyperparams.get("epochs", 1000)
    learning_rate = hyperparams.get("learning_rate", 0.001)
    print_every = hyperparams.get("print_every", 10)
    best_model_path = hyperparams.get("best_model_path", "best_model.pth")
    l2_lambda = hyperparams.get("l2_lambda", 0.001)  # L2 regularization strength
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Apply L2 regularization via weight decay parameter
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    criterion = nn.MSELoss()

    # Implement early stopping
    patience = hyperparams.get("patience", 40)
    early_stopping_counter = 0
    
    best_val_loss = float('inf')
    training_losses = []
    validation_losses = []
    learning_rates = []

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=10, verbose=True
    )
    
    for epoch in tqdm(range(epochs), desc="Training epochs"):
        model.train()
        train_loss_epoch = 0.0
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item() * X_batch.size(0)
        
        train_loss_epoch /= len(train_loader.dataset)
        training_losses.append(train_loss_epoch)
        
        model.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                val_outputs = model(X_batch)
                val_loss = criterion(val_outputs, y_batch)
                val_loss_epoch += val_loss.item() * X_batch.size(0)
        
        val_loss_epoch /= len(val_loader.dataset)
        validation_losses.append(val_loss_epoch)

        # Step the scheduler based on validation loss
        scheduler.step(val_loss_epoch)
        
        # if (epoch + 1) % print_every == 0:
        #     print(f"Epoch [{epoch+1}/{epochs}], LR: {current_lr:.6f}, Training Loss: {train_loss_epoch:.4f}, Validation Loss: {val_loss_epoch:.4f}")
        
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            torch.save(model.state_dict(), best_model_path)
            #print(f"Epoch [{epoch+1}]: New best model saved with Validation Loss: {val_loss_epoch:.4f}")
            early_stop_counter = 0  # Reset counter when we find a better model
        else:
            early_stop_counter += 1
            
        # Early stopping check
        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    return model, training_losses, validation_losses, learning_rates

def evaluate_model(model, test_dataset, hyperparams):
    """
    Evaluate the model on the test dataset.
    
    Parameters:
        model (nn.Module): The neural network model.
        test_dataset (TensorDataset): The test dataset.
        hyperparams (dict): Dictionary of hyperparameters (for batch_size).
    
    Returns:
        test_loss (float): The average test loss.
    """
    batch_size = hyperparams.get("batch_size", 32)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.MSELoss()
    
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            batch_loss = criterion(outputs, y_batch)
            test_loss += batch_loss.item() * X_batch.size(0)
    test_loss /= len(test_loader.dataset)
    return test_loss
