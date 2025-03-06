import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Dataset Loader
class DigitDataset(Dataset):
    def __init__(self, file_path):
        data = np.loadtxt(file_path)
        self.labels = torch.tensor(data[:, 0], dtype=torch.long)  # First column is the label
        self.images = torch.tensor(data[:, 1:], dtype=torch.float32).reshape(-1, 1, 16, 16)  # Reshape to 16x16
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Load data
train_dataset = DigitDataset('zip_train.txt')
test_dataset = DigitDataset('zip_test.txt')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Experiment with different parameter initializations
def initialize_weights(model, init_type="default"):
    for layer in model.modules():
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            if init_type == "default":
                nn.init.normal_(layer.weight, mean=0, std=0.01)
            elif init_type == "he":
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            elif init_type == "zero":
                nn.init.constant_(layer.weight, 0)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

# Experiment with different learning rates
def train_model(model, train_loader, learning_rate, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Learning Rate: {learning_rate} | Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}')

# Experiment with different batch sizes
def test_batch_sizes(models, train_dataset, batch_sizes=[16, 32, 64]):
    for batch_size in batch_sizes:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for model_name, model in models.items():
            print(f'Training {model_name} with Batch Size: {batch_size}')
            train_model(model, train_loader, learning_rate=0.001, epochs=3)

# Experiment with momentum values in optimizers
def test_momentum(models, train_loader, momentum_values=[0.5, 0.9, 0.99]):
    criterion = nn.CrossEntropyLoss()
    
    for momentum in momentum_values:
        for model_name, model in models.items():
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=momentum)
            model.train()
            print(f'Training {model_name} with Momentum: {momentum}')
            
            for epoch in range(3):
                total_loss = 0
                for images, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print(f'{model_name} | Momentum {momentum} | Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}')

# Run basic training to check output
if __name__ == "__main__":
    print("Running Task 2 Experiments...")
    from Task1 import FullyConnectedNN, LocallyConnectedNN, CNNModel  # Ensure this module exists
    
    models = {
        "FullyConnectedNN": FullyConnectedNN(),
        "LocallyConnectedNN": LocallyConnectedNN(),
        "CNNModel": CNNModel()
    }
    
    print("Testing Parameter Initialization...")
    for model_name, model in models.items():
        initialize_weights(model, "default")
        print(f"Initialized {model_name} with default weights.")
    
    print("Testing Learning Rates...")
    for lr in [0.0001, 0.001, 0.01]:
        print(f"Training with Learning Rate: {lr}")
        train_model(models["CNNModel"], train_loader, learning_rate=lr, epochs=3)
    
    print("Testing Batch Sizes...")
    test_batch_sizes(models, train_dataset, batch_sizes=[16, 32, 64])
    
    print("Testing Momentum...")
    test_momentum(models, train_loader, momentum_values=[0.5, 0.9, 0.99])
