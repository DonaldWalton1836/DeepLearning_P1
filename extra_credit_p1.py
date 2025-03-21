
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

# Load training data
train_dataset = DigitDataset('zip_train.txt')
test_dataset  = DigitDataset('zip_test.txt')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model using convolutional neural network 
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32*4*4, 128) 
        self.fc2 = nn.Linear(128, 10)
 # Adjust based on final total size
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Initialize model 
model = SimpleCNN()

#  Adversarial training example generation using fast graident sign method (FGSM)
def fgsm_attack(model, images, labels, epsilon=0.1):
    images.requires_grad = True
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    
    model.zero_grad()
    loss.backward()
    
    data_grad = images.grad.data.sign()
    adv_images = images + epsilon * data_grad
    adv_images = torch.clamp(adv_images, 0, 1)
    
    return adv_images.detach()

# Trainng loop with adversarial example 
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images, labels

        # Generate adversarial examples
        adv_images = fgsm_attack(model, images, labels, epsilon=0.1)
        
        # Combine the original and adversarial data
        combined_images = torch.cat([images, adv_images], dim=0)
        combined_labels = torch.cat([labels, labels], dim=0)

        optimizer.zero_grad()
        outputs = model(combined_images)
        loss = criterion(outputs, combined_labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    # Compute current Epoch number and average training loss 
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}")

# Final Evaluation and model Accuracy test 
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images, labels
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100.0 * correct / total
print(f"Test Accuracy is : {accuracy:.2f}%")
