import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import time
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import torch

import pickle

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import Adam

from tqdm import tqdm

import numpy as np
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

##################################

class BayesianCNNSingleFC(PyroModule):
    def __init__(self, num_classes):
        super().__init__()

        prior_mu = 0.
        #prior_sigma = 0.1 #accuracy 13.203704% 2 epochs
        #prior_sigma = 1. #accuracy 31% 2 epochs
        prior_sigma = torch.tensor(10., device=device) #accuracy 45% 10 epochs
        #prior_sigma = 100 #accuracy 21% 10 epochs

        self.conv1 = PyroModule[nn.Conv2d](3, 32, kernel_size=5, stride=1, padding=2)
        self.conv1.weight = PyroSample(dist.Normal(prior_mu, prior_sigma).expand([32, 3, 5, 5]).to_event(4))
        self.conv1.bias = PyroSample(dist.Normal(prior_mu, prior_sigma).expand([32]).to_event(1))

        self.conv2 = PyroModule[nn.Conv2d](32, 64, kernel_size=5, stride=1, padding=2) #initially padding=1 kernel_size=3, without stride
        self.conv2.weight = PyroSample(dist.Normal(prior_mu, prior_sigma).expand([64, 32, 5, 5]).to_event(4))
        self.conv2.bias = PyroSample(dist.Normal(prior_mu, prior_sigma).expand([64]).to_event(1))

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = PyroModule[nn.Linear](64 * 16 * 16, num_classes)
        self.fc1.weight = PyroSample(dist.Normal(prior_mu, prior_sigma).expand([num_classes, 64 * 16 * 16]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(prior_mu, prior_sigma).expand([num_classes]).to_event(1))

    def forward(self, x, y=None):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        logits = self.fc1(x)
        
        # THIS IS THE MISSING PIECE: Define the likelihood
        if y is not None:
            with pyro.plate("data", x.shape[0]):
                pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        
        return logits
    

def load_data(batch_size=54):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3444, 0.3809, 0.4082], std=[0.1809, 0.1331, 0.1137])
    ])

    dataset = datasets.EuroSAT(root='./data', transform=transform, download=False)

    torch.manual_seed(42)
    
    with open('datasplit/split_indices.pkl', 'rb') as f:
        split = pickle.load(f)
        train_dataset = Subset(dataset, split['train'])
        test_dataset = Subset(dataset, split['test'])

    # Add num_workers and pin_memory for faster data loading
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    return train_loader, test_loader


def train_svi(model, guide, svi, train_loader, num_epochs=10):
    # Clear parameter store only ONCE at the beginning
    pyro.clear_param_store()
    model.train()
    
    # Ensure model is on the correct device
    model.to(device)
    #guide.to(device)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            loss = svi.step(images, labels)
            epoch_loss += loss
            num_batches += 1
            
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} - ELBO Loss: {avg_loss:.4f}")

def predict_data(model, test_loader, num_samples=10):
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            logits_mc = torch.zeros(num_samples, images.size(0), model.fc1.out_features).to(device)

            for i in range(num_samples):
                guide_trace = pyro.poutine.trace(guide).get_trace()
                replayed_model = pyro.poutine.replay(model, trace=guide_trace)
                logits = replayed_model(images)
                logits_mc[i] = logits

            avg_logits = logits_mc.mean(dim=0)
            predictions = torch.argmax(avg_logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    return all_labels, all_predictions

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
            
    # make a mark to the diagonal
    plt.plot([0, cm.shape[1]-1], [0, cm.shape[0]-1], color='red', linestyle='--', linewidth=2)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def main():
    # 1. Initialize the model and guide
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the number of classes for EuroSAT dataset
    # EuroSAT has 10 classes
    num_classes = 10
    bayesian_model = BayesianCNNSingleFC(num_classes=num_classes).to(device)

    guide = AutoDiagonalNormal(bayesian_model)

    # 2. Optimizer and SVI - increase learning rate for better convergence
    optimizer = Adam({"lr": 1e-3})  # Increased from 1e-4 to 1e-3
    svi = pyro.infer.SVI(model=bayesian_model,
                        guide=guide,
                        optim=optimizer,
                        loss=pyro.infer.Trace_ELBO())
    
    pyro.clear_param_store()

    # Ensure model and guide are on the correct device
    bayesian_model.to(device)
    guide.to(device)

    train_loader, test_loader = load_data(batch_size=54)
    train_svi(bayesian_model, guide, svi, train_loader, num_epochs=100)

    all_labels, all_predictions = predict_data(bayesian_model, test_loader, num_samples=10)
    cm = confusion_matrix(all_labels, all_predictions)

    accuracy = np.trace(cm) / np.sum(cm)
    print(f"Accuracy from confusion matrix: {accuracy * 100:.6f}%")
    
    class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
               'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
    plot_confusion_matrix(cm, class_names)
    

if __name__ == "__main__":
    main()
    
