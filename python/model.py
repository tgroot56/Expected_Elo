import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)  # Output layer for regression
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

def train_model(model, train_dataset, val_dataset, hyperparams):
    """
    Train the model using the provided training and validation datasets.
    
    Parameters:
        model (nn.Module): The neural network model.
        train_dataset (TensorDataset): The training dataset.
        val_dataset (TensorDataset): The validation dataset.
        hyperparams (dict): A dictionary containing hyperparameters such as:
            - "batch_size": Batch size for training.
            - "epochs": Number of epochs.
            - "learning_rate": Learning rate.
            - "print_every": Frequency (in epochs) to print progress.
            - "best_model_path": File path to save the best model.
    
    Returns:
        model (nn.Module): The trained model.
        validation_losses (list): List of validation losses over epochs.
    """
    batch_size = hyperparams.get("batch_size", 32)
    epochs = hyperparams.get("epochs", 1000)
    learning_rate = hyperparams.get("learning_rate", 0.001)
    print_every = hyperparams.get("print_every", 10)
    best_model_path = hyperparams.get("best_model_path", "best_model.pth")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    validation_losses = []
    
    for epoch in tqdm(range(epochs), desc="Training epochs"):
        model.train()
        train_loss_epoch = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item() * X_batch.size(0)
        train_loss_epoch /= len(train_loader.dataset)
        
        model.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                val_outputs = model(X_batch)
                batch_loss = criterion(val_outputs, y_batch)
                val_loss_epoch += batch_loss.item() * X_batch.size(0)
        val_loss_epoch /= len(val_loader.dataset)
        validation_losses.append(val_loss_epoch)
        
        if (epoch + 1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss_epoch:.4f}, Validation Loss: {val_loss_epoch:.4f}")
        
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch [{epoch+1}]: New best model saved with Validation Loss: {val_loss_epoch:.4f}")
    
    return model, validation_losses

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
