import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from custoptimizer import SGLD  # Assuming you have a custom SGLD optimizer

import matplotlib.pyplot as plt
from datetime import datetime
import json

#ensure reproducibility
torch.manual_seed(0)

# ----- 1. Bayesian Layer with Gaussian Weight Distribution -----
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_log_sigma = nn.Parameter(torch.Tensor(out_features, in_features).fill_(-5))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_log_sigma = nn.Parameter(torch.Tensor(out_features).fill_(-5))

        self.prior_std = prior_std

    def forward(self, x):
        weight_sigma = torch.exp(self.weight_log_sigma)
        bias_sigma = torch.exp(self.bias_log_sigma)

        # Sample weights
        weight_eps = torch.randn_like(weight_sigma)
        bias_eps = torch.randn_like(bias_sigma)
        weight = self.weight_mu + weight_sigma * weight_eps
        bias = self.bias_mu + bias_sigma * bias_eps

        self.kl = self._kl_divergence(weight, self.weight_mu, weight_sigma) + \
                  self._kl_divergence(bias, self.bias_mu, bias_sigma)

        return F.linear(x, weight, bias)

    def _kl_divergence(self, q_sample, mu, sigma):
        # KL divergence between posterior N(mu, sigma^2) and prior N(0, prior_std^2)
        prior_sigma = self.prior_std
        return torch.sum(
            torch.log(prior_sigma / sigma) +
            (sigma**2 + mu**2) / (2 * prior_sigma**2) - 0.5
        )

# ----- 2. Bayesian Neural Network -----
class BNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = BayesianLinear(28 * 28, 400)
        self.b2 = BayesianLinear(400, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.b1(x))
        x = self.b2(x)
        return x

    def kl_loss(self):
        return self.b1.kl + self.b2.kl

# ----- 3. SEU Injection into Sampled Weights -----
def flip_bit_in_tensor(tensor, bit_position=10, flip_count=1):
    flat = tensor.view(-1)
    idx = torch.randint(0, flat.numel(), (flip_count,))
    for i in idx:
        val = flat[i].item()
        int_val = np.frombuffer(np.float32(val).tobytes(), dtype=np.uint32)[0]
        flipped = int_val ^ (1 << bit_position)
        flipped_val = np.frombuffer(np.uint32(flipped).tobytes(), dtype=np.float32)[0]
        flat[i] = torch.tensor(flipped_val)
    return tensor

def inject_seu_layer(layer, bit_position=10, flip_count=1):
    with torch.no_grad():
        # Flip bits in the weight_mu tensor
        layer.weight_mu.data = flip_bit_in_tensor(layer.weight_mu.data.clone(), bit_position, flip_count)

# ----- 4. Training & Evaluation -----
def elbo_loss_fn(output, target, kl, beta=1.0):
    ce = F.cross_entropy(output, target, reduction='mean')
    return ce + beta * kl / len(target)

def train_bnn(model, loader, optimizer, epoch):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #data, target = data.to(device), target.to(device)
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        kl = model.kl_loss()
        loss = elbo_loss_fn(output, target, kl)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

    return loss.item()

def evaluate_bnn(model, loader, samples=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #data, target = data.to(device), target.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            #preds = torch.zeros(data.size(0), 10)
            data, target = data.to(device), target.to(device)
            preds = torch.zeros(data.size(0), 10, device=data.device)
            for _ in range(samples):
                out = model(data)
                preds += F.softmax(out, dim=1)
            preds /= samples
            predicted = preds.argmax(1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    return 100 * correct / total

# ----- 5. Main Routine -----
def main():
    transform = transforms.ToTensor()
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = BNN()
    print(f"Using device: {device}")
    model = BNN().to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) #might be the place to put the SGLD optimizer??
    optimizer = SGLD(model.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer_name = optimizer.__class__.__name__

    designated_epochs = 100
    loss_history = []

    start_time = time.time()
    print("Training...")
    for epoch in range(designated_epochs):
        avg_loss = train_bnn(model, train_loader, optimizer, epoch+1)
        loss_history.append(avg_loss)
    end_time = time.time()

    print(f"Training completed for {designated_epochs} epochs in {end_time - start_time:.2f} seconds")

    print("\nEvaluating clean model:")
    acc_before_seu = evaluate_bnn(model, test_loader)

    print("\nInjecting SEU into b1 weight_mu (bit 10)...")
    inject_seu_layer(model.b1, bit_position=10, flip_count=10)

    print("Evaluating after SEU:")
    acc_after_seu = evaluate_bnn(model, test_loader)

    # Save loss plot and JSON
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # print the title to the fourt decimal place
    acc_before_seu_text = f"{acc_before_seu:.2f}%"
    acc_after_seu_text = f"{acc_after_seu:.2f}%"
    plt.title(optimizer_name + "\n" +acc_before_seu_text + " -> " + acc_after_seu_text)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_filename = f"{timestamp}.png"
    json_filename = f"{timestamp}.json"
    plt.savefig(png_filename)
    plt.close()
    
    with open(json_filename, "w") as f:
        json.dump({"loss": loss_history}, f)

    #print the weight before and after SEU injection
    #print("\nWeight before SEU injection:")
    #print(model.b1.weight_mu.data)
    #print("\nWeight after SEU injection:")
    #inject_seu_layer(model.b1, bit_position=10, flip_count=10)  # Reapply to see the change
    #print(model.b1.weight_mu.data)

if __name__ == "__main__":
    main()

# Add at the top of your main()

# In your train_bnn and evaluate_bnn functions, move data and target to device:
