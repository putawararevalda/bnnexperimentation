import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import time
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import os

# Install bayesian-torch if not installed:
# pip install bayesian-torch

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn

import pickle

# Set seed for reproducibility
torch.manual_seed(0)

# 1. Define original CNN Architecture (deterministic)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # -> (32, 32, 32)
        x = self.pool(F.relu(self.conv2(x)))   # -> (64, 16, 16)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def compute_kl_loss(model):
    kl = 0.0
    for module in model.modules():
        if hasattr(module, 'kl_loss'):
            kl += module.kl_loss()
    return kl

# 2. Data Preparation (unchanged)
def load_data():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3444, 0.3809, 0.4082], std=[0.1809, 0.1331, 0.1137])
    ])

    dataset = datasets.EuroSAT(root='./data', transform=transform, download=True)
    #train_size = int(0.8 * len(dataset))
    #test_size = len(dataset) - train_size
    #train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    with open('datasplit/split_indices.pkl', 'rb') as f:
        split = pickle.load(f)
        train_dataset = Subset(dataset, split['train'])
        test_dataset = Subset(dataset, split['test'])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    return train_loader, test_loader

# 3. Training Function (add KL term from BNN layers)
def train_model(model, loader, optimizer, criterion, epochs=10):
    model.train()
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)

            nll_loss = criterion(output, target)
            kl = compute_kl_loss(model) / len(loader.dataset)  # normalize
            loss = nll_loss + kl

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} (NLL: {nll_loss.item():.4f}, KL: {kl.item():.4f})")

    return loss_history


# 4. Evaluation (same as before)
def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# 5. Main Routine
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_data()

    model = CNN()
    # Convert deterministic CNN to Bayesian CNN

    const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",  # Flipout or Reparameterization
        "moped_enable": False,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
        }
    
    dnn_to_bnn(
        model,
        const_bnn_prior_parameters
    )
    
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    loss_history = train_model(model, train_loader, optimizer, criterion, epochs=10)
    end = time.time()

    print(f"Training completed in {end - start:.2f} seconds")

    test_acc = evaluate_model(model, test_loader)

    os.makedirs("results_eurosat", exist_ok=True)
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Bayesian CNN Accuracy: {test_acc:.2f}%")
    plt.savefig("results_eurosat/bayesian_cnn_loss_plot.png")
    plt.close()

    model_path = "results_eurosat/bayesian_cnn_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
