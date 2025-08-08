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
import csv

from torchvision.datasets import ImageFolder

dataset = ImageFolder(
    root="data/shipsnet/foldered",
    transform=transforms.ToTensor()
)

device = torch.device("cuda")

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=1)

shipsnet_mean = [0.4119, 0.4243, 0.3724]
shipsnet_std = [0.1899, 0.1569, 0.1515]

def load_data_withval(batch_size=16, val_split=0.1):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=shipsnet_mean, std=shipsnet_std)
    ])

    dataset = ImageFolder(
        root="data/shipsnet/foldered",
        transform=transform
    )

    torch.manual_seed(42)

    with open('datasplit/shipsnet_split_indices.pkl', 'rb') as f:
        split = pickle.load(f)
        full_train_dataset = Subset(dataset, split['train'])
        test_dataset = Subset(dataset, split['test'])

    # Split the full train dataset into train and val
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size   = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=4, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import torch
import torch.nn as nn
import torch.nn.functional as F

class SmartPool(nn.Module):
    """
    A “smart” max‐pool that detects outliers (values > threshold) and, if desired,
    replaces them with the 2nd‐largest value in the window.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        stride: int = 2,
        threshold: float = 10.0,
        detect_only: bool = False
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.threshold = threshold
        self.detect_only = detect_only

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W)
        N, C, H, W = x.shape
        ks, st = self.kernel_size, self.stride

        # Unfold into patches of shape (N, C, ks*ks, L) where L = #windows per image
        patches = F.unfold(x, kernel_size=ks, stride=st)  # → (N, C*ks*ks, L)
        patches = patches.view(N, C, ks*ks, -1)           # → (N, C, ks*ks, L)

        # Find top‐2 values in each patch
        top2_vals, _ = torch.topk(patches, 2, dim=2)      # → (N, C, 2, L)
        max1 = top2_vals[:, :, 0, :]                      # → (N, C, L)
        max2 = top2_vals[:, :, 1, :]                      # → (N, C, L)

        # Detect any spikes above threshold
        spikes = max1 > self.threshold                    # → (N, C, L)
        #if spikes.any():
        #    warnings.warn(f"SmartPool: detected {int(spikes.sum())} pooled values above threshold={self.threshold}")

        # If correction is off, just return the regular max‐pooled result
        if self.detect_only:
            out = max1
        else:
            # Replace each spike with the 2nd‐largest value
            out = torch.where(spikes, max2, max1)

        # Fold back to (N, C, H_out, W_out)
        H_out, W_out = (H // ks, W // ks)
        out = out.view(N, C * 1, -1)                      # → (N, C, L)
        out = out.view(N, C, H_out, W_out)
        return out
    
class ShipsCNNCustom(nn.Module):
    def __init__(self, num_classes=2, 
                 activation='relu',
                 smartpool_switch=False,
                 pool_threshold=10.0,
                 pool_detect_only=False,
                 dropout_switch=False,
                 dropout_p=0.5):
        super().__init__()

        # Activation setup (same as BayesShipsCNN)
        act_map = {
            'relu': F.relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'sin': torch.sin,
            'relu6': F.relu6,
            #'leaky_relu': F.leaky_relu,
            #'selu': F.selu,
            'actWG': self.actWG,
            'actRWG': self.actRWG,
        }
        if activation not in act_map:
            raise ValueError(f"Unsupported activation: {activation}")
        self.activation_fn = act_map[activation]

        # Layers: Same as BayesShipsCNN (2 conv layers + pooling)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        
        # SmartPool or standard MaxPool
        if smartpool_switch:
            self.pool = SmartPool(
                kernel_size=2,
                stride=2,
                threshold=pool_threshold,
                detect_only=pool_detect_only
            )
        else:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Dropout
        self.dropout_switch = dropout_switch
        if dropout_switch:
            self.dropout = nn.Dropout(p=dropout_p)

        # Fully connected layer
        # BayesShipsCNN flattens [B,64,16,16] → fc1: 64*16*16 → 2
        self.fc1 = nn.Linear(64 * 16 * 16, num_classes)

    def forward(self, x):
        x = self.activation_fn(self.conv1(x))
        x = self.pool(x)
        x = self.activation_fn(self.conv2(x))
        x = self.pool(x)

        if self.dropout_switch and self.training:
            x = self.dropout(x)

        x = x.view(x.size(0), -1)
        logits = self.fc1(x)
        return logits

    def actWG(self, x, alpha=1.0):
        return x * torch.exp(-alpha * x ** 2)

    def actRWG(self, x, alpha=1.0):
        wg = x * torch.exp(-alpha * x ** 2)
        return torch.max(torch.zeros_like(wg), wg)


import argparse

if __name__ == "__main__":
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = load_data_withval(16)

    parser = argparse.ArgumentParser(description='Train Deterministic Neural Net on Shipsnet')

    parser.add_argument('--epoch', type=int, nargs='?', action='store', default='20',
            help='Number of epoch. Default: 20.')
    parser.add_argument('--model_variant', type=str, nargs='?', action='store', default='00',
                help='Model to run. Default: \'00\'.')
    args = parser.parse_args()

    lr          = 1e-3
    num_epochs  = args.epochs #default 20

    model_variant = args.model_variant

    for aoi in ['tanh', 'sigmoid', 'sin', 'relu6', 'actWG', 'actRWG']:

        weight_decay_target = 0

        if model_variant == "00":
            model = ShipsCNNCustom(activation=aoi).to(device)
        elif model_variant == "01":
            model = ShipsCNNCustom(activation=aoi,
                                smartpool_switch=True).to(device)
        elif model_variant == "02":
            model = ShipsCNNCustom(activation=aoi,
                                dropout_switch=True).to(device)
        elif model_variant == "03":
            weight_decay_target = 1e-4
        else:
            raise ValueError(f"Unsupported model variant: {model_variant}")


        best_val_acc = 0.0
        save_dir = "results_shipsnet_deterministic_" + model_variant
        os.makedirs(save_dir, exist_ok=True)

        # ─── 6. Loss, Optimizer & Scheduler ───────────────────────────────────────────
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay_target)
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        log_filename = os.path.join(save_dir, f"training_log_{model.activation_fn.__name__}_{timestamp}.csv")

        with open(log_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['model_variant','act_fn','epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

        # ─── 7. Training Loop ─────────────────────────────────────────────────────────
        for epoch in range(1, num_epochs + 1):
            # — Train —
            model.train()
            running_loss = running_corrects = 0
            for imgs, labels in train_loader:
                imgs  = imgs.to(device, non_blocking=True)
                labels= labels.to(device, non_blocking=True)

                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                preds = outputs.argmax(dim=1)
                running_loss    += loss.item() * imgs.size(0)
                running_corrects+= (preds == labels).sum().item()

            epoch_loss = running_loss / len(train_ds)
            epoch_acc  = running_corrects / len(train_ds)

            # — Validate —
            model.eval()
            val_loss = val_corrects = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs   = imgs.to(device)
                    labels = labels.to(device)
                    outputs= model(imgs)
                    loss   = criterion(outputs, labels)
                    preds  = outputs.argmax(dim=1)

                    val_loss     += loss.item() * imgs.size(0)
                    val_corrects += (preds == labels).sum().item()

            val_loss = val_loss / len(val_ds)
            val_acc  = val_corrects / len(val_ds)
            #scheduler.step()

            print(f"Epoch {epoch:2d}/{num_epochs} "
                f"Train: loss={epoch_loss:.4f}, acc={epoch_acc:.4f} | "
                f"Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # add the model activation function to the filename
                model_name = os.path.join(save_dir,  f"best_model_{model.activation_fn.__name__}_{timestamp}.pth")
                torch.save(model.state_dict(), model_name)
                print(f"New best val accuracy: {best_val_acc:.4f} — model saved.")

            with open(log_filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([model_variant, aoi, epoch, epoch_loss, epoch_acc, val_loss, val_acc])
