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

# Using train data and optimial value data "value"
train_dataset = DigitDataset('zip_train.txt')
value_dataset   = DigitDataset('zip_test.txt')  

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
value_loader   = DataLoader(value_dataset, batch_size=64, shuffle=False)

# Model defenintion (using MLP)
class MLP(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=10, dropout_rate=0.0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
    
     # Reshape size from (N,16) to (N, 256)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def train_eval_model(lr, dropout_rate):
    model = MLP(dropout_rate=dropout_rate)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop for model 
    for epoch in range(2):
        model.train()
        for images, labels in train_loader:
            images, labels = images, labels
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Validation to measure accuracy of hyperparameters 
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in value_loader:
            images, labels = images, labels
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    value_accuracy = 100.0 * correct / total
    return value_accuracy

# Coarse-to-fine search to automate values for hyperparameters 
# Using arbitrary values for dropout and learning rate 
coarse_lr_values = [1e-3, 1e-2, 1e-1]
coarse_dropout_values = [0.0, 0.3, 0.5]

best_accuracy = 0
best_lr = None
best_dropout = None

# Coarse search
for lr in coarse_lr_values:
    for do_rate in coarse_dropout_values:
        acc = train_eval_model(lr, do_rate)
        print(f"Coarse search: LR = {lr}, Dropout = {do_rate}, Value Accuracy = {acc:.2f}")
        if acc > best_accuracy:
            best_accuracy = acc
            best_lr = lr
            best_dropout = do_rate

# Fine search to locate the best LR and Dropout values 
fine_lr_range = [best_lr*0.5, best_lr, best_lr*2.0]
fine_dropout_range = [
    max(0.0, best_dropout - 0.1),
    best_dropout,
    min(0.9, best_dropout + 0.1)
]

for lr in fine_lr_range:
    for do_rate in fine_dropout_range:
        acc = train_eval_model(lr, do_rate)
        print(f"Fine: LR = {lr}, Dropout = {do_rate}, Value Accuracy = {acc:.2f}")
        if acc > best_accuracy:
            best_accuracy = acc
            best_lr = lr
            best_dropout = do_rate

# Print most optimal hyperparameters found 
print("\nThe Best Hyperparameters Found Are :")
print(f"Learning Rate: {best_lr}, Dropout Rate: {best_dropout}, Accuracy: {best_accuracy:.2f}")
