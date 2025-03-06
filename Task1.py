import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 1. Fully Connected Network (MLP)
class FullyConnectedNN(nn.Module):
    def __init__(self):
        super(FullyConnectedNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256, 512)  # First hidden layer
        self.fc2 = nn.Linear(512, 256)  # Second hidden layer
        self.fc3 = nn.Linear(256, 128)  # Additional hidden layer
        self.fc4 = nn.Linear(128, 10)   # Output layer
    
    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)  # No activation for logits
        return x

# 2. Locally Connected Network (No Weight Sharing)
class LocallyConnectedNN(nn.Module):
    def __init__(self):
        super(LocallyConnectedNN, self).__init__()
        self.local1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, groups=1)
        self.local2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, groups=1)
        self.local3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, groups=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 1 * 1, 64)
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.relu(self.local1(x))
        x = torch.relu(self.local2(x))
        x = torch.relu(self.local3(x))
        x = self.flatten(x)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

# 3. Convolutional Neural Network (Weight Sharing)
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 1 * 1, 64)
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.flatten(x)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

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

# Instantiate models
mlp_model = FullyConnectedNN()
local_model = LocallyConnectedNN()
cnn_model = CNNModel()

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_mlp = optim.Adam(mlp_model.parameters(), lr=0.001)
optimizer_local = optim.Adam(local_model.parameters(), lr=0.001)
optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=0.001)

# Training Function
def train(model, optimizer, train_loader, epochs=5):
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
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}')

# Evaluation Function
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total:.2f}%')

# Train and evaluate each model
print("Training Fully Connected Network (MLP)...")
train(mlp_model, optimizer_mlp, train_loader)
evaluate(mlp_model, test_loader)

print("Training Locally Connected Network...")
train(local_model, optimizer_local, train_loader)
evaluate(local_model, test_loader)

print("Training CNN...")
train(cnn_model, optimizer_cnn, train_loader)
evaluate(cnn_model, test_loader)