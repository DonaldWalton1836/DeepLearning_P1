import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Dataset loader
class DigitDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        data = np.loadtxt(file_path)
        self.labels = torch.tensor(data[:, 0], dtype=torch.long)
        self.images = torch.tensor(data[:, 1:], dtype=torch.float32).reshape(-1, 1, 16, 16)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Training set and  Validation sets 
train_dataset = DigitDataset('zip_train.txt')
val_dataset   = DigitDataset('zip_test.txt')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Create MLP for best architecture search 
class FlexMLP(nn.Module):
    def __init__(self, input_dim, layer_sizes, output_dim=10):
        super(FlexMLP, self).__init__()
        layers = []
        current_dim = input_dim
        for size in layer_sizes:
            layers.append(nn.Linear(current_dim, size))
            layers.append(nn.ReLU())
            current_dim = size
        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1) 
        # Change size from (N, 16) to (N, 256)
        return self.net(x)

def train_once(model, train_loader, val_loader, epochs=2, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Train the model 
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images, labels
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Validate model accuracy 
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images, labels
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total

# Optimal Architecture search loop 
layer_candidates = [
    [128],           # Single hidden layer
    [128, 64],       # Two hidden layers
    [256, 128, 64],  # Three hidden layers
]

best_acc = 0
best_arch = None

for layers in layer_candidates:
    model = FlexMLP(input_dim=256, layer_sizes=layers)
    val_acc = train_once(model, train_loader, val_loader, epochs=3, lr=1e-3)
    # Print out the validation accuracy of a single architecture model 
    print(f"Architecture {layers}, Validation Accuracy: {val_acc:.2f}")
    
    if val_acc > best_acc:
        best_acc = val_acc
        best_arch = layers
 # Find the validation accuracy of the most efficient architecture model 
print(f"\nBest Architecture: {best_arch} => Validation Accuracy: {best_acc:.2f}")
