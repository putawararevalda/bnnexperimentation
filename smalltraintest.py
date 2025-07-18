#!/usr/bin/env python
# bnn_eurosat_train.py
# ------------------------------------------------------------
import math, pickle, random, pathlib, warnings

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import pyro, pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, TraceMeanField_ELBO
from pyro.optim.clipped import Adam as ClippedAdam
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import Predictive

# ------------------------------------------------------------
# 0.  Hyper-params & switches
# ------------------------------------------------------------
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTIVATION  = "tanh"          # "relu" or "tanh"
BATCH_SIZE  = 64
NUM_EPOCHS  = 40
LR          = 1e-3
CLIP_NORM   = 5.0
BETA        = 0.1             # KL weight
PRIOR_STD   = 0.1             # σ of Normal prior over weights
GUIDE_INIT  = 0.05            # initial posterior σ
NUM_WORKERS = 4

# ------------------------------------------------------------
# 1.  Dataset loader ---------------------------------------------------------
# EuroSAT RGB 64×64 already
EUROSAT_MEAN = (0.344, 0.380, 0.408)
EUROSAT_STD  = (0.190, 0.137, 0.115)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(EUROSAT_MEAN, EUROSAT_STD),
])

root = "./data"
full_ds = datasets.EuroSAT(root=root, transform=transform, download=True)

# (Optional) deterministic split from .pkl  ------------------
# If you have a fixed split file, load it; else make an 80/20 split once.
split_file = pathlib.Path("datasplit/split_indices.pkl")
if split_file.is_file():
    with open(split_file, "rb") as f:
        split = pickle.load(f)
    train_ds = Subset(full_ds, split["train"])
    val_ds   = Subset(full_ds, split["test"])
else:
    rng = torch.Generator().manual_seed(42)
    train_sz = int(0.8 * len(full_ds))
    val_sz   = len(full_ds) - train_sz
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_sz, val_sz], rng)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

NUM_CLASSES = len(full_ds.classes)

# ------------------------------------------------------------
# 2.  Bayesian CNN model -------------------------------------
def make_activation():
    if ACTIVATION == "relu":
        return torch.relu
    elif ACTIVATION == "tanh":
        return torch.tanh
    else:
        raise ValueError("ACTIVATION must be 'relu' or 'tanh'")

class BayesianCNNSingleFC(PyroModule):
    def __init__(self, num_classes: int):
        super().__init__()
        prior_mu  = 0.
        prior_std = PRIOR_STD
        act = make_activation()

        # -------- conv1
        self.conv1 = PyroModule[nn.Conv2d](3, 32, kernel_size=5, padding=2)
        self.conv1.weight = PyroSample(
            dist.Normal(prior_mu, prior_std).expand([32, 3, 5, 5]).to_event(4))
        self.conv1.bias   = PyroSample(
            dist.Normal(prior_mu, prior_std).expand([32]).to_event(1))

        # -------- conv2
        self.conv2 = PyroModule[nn.Conv2d](32, 64, kernel_size=5, padding=2)
        self.conv2.weight = PyroSample(
            dist.Normal(prior_mu, prior_std).expand([64, 32, 5, 5]).to_event(4))
        self.conv2.bias   = PyroSample(
            dist.Normal(prior_mu, prior_std).expand([64]).to_event(1))

        self.pool = nn.MaxPool2d(2)
        self.gap  = nn.AdaptiveAvgPool2d((1, 1))

        # -------- fully-connected
        self.fc1 = PyroModule[nn.Linear](64, num_classes)
        self.fc1.weight = PyroSample(
            dist.Normal(prior_mu, prior_std).expand([num_classes, 64]).to_event(2))
        self.fc1.bias   = PyroSample(
            dist.Normal(prior_mu, prior_std).expand([num_classes]).to_event(1))

        self._act = act

    def forward(self, x, y=None):
        x = self.pool(self._act(self.conv1(x)))   # -> [B,32,32,32]
        x = self.pool(self._act(self.conv2(x)))   # -> [B,64,16,16]
        x = self.gap(x).flatten(1)                # -> [B,64]
        logits = self.fc1(x)                      # -> [B,C]

        if y is not None:
            with pyro.plate("data", x.size(0)):
                pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits

# ------------------------------------------------------------
# 3.  Model, guide, optimiser, ELBO ---------------------------
pyro.clear_param_store()          #  << BEFORE building model / guide
model = BayesianCNNSingleFC(NUM_CLASSES).to(DEVICE)

guide = AutoDiagonalNormal(
    model, init_loc_fn=pyro.infer.init_to_mean,
    init_scale=GUIDE_INIT
)

optim = ClippedAdam({"lr": LR, "clip_norm": CLIP_NORM})
elbo  = TraceMeanField_ELBO(num_particles=1, beta=BETA)
svi   = SVI(model, guide, optim, elbo)

# ------------------------------------------------------------
# 4.  Training / evaluation utils ----------------------------
def accuracy(loader, predictive):
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = predictive(xb)              # deterministic (loc) weights
            pred   = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total   += yb.size(0)
    return correct / total

predictive_det = Predictive(model, guide=guide, num_samples=0)

# ------------------------------------------------------------
# 5.  Training loop ------------------------------------------
for epoch in range(1, NUM_EPOCHS + 1):
    model.train(), guide.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        loss = svi.step(xb, yb)
        running_loss += loss

    epoch_loss = running_loss / len(train_loader.dataset)   # per-sample ELBO
    model.eval(), guide.eval()
    train_acc = accuracy(train_loader, predictive_det)
    val_acc   = accuracy(val_loader, predictive_det)

    # ---- basic σ diagnostics (mean/95-quantile) ------------
    with torch.no_grad():
        scales = torch.cat([
            p.detach().exp().view(-1)
            for n, p in pyro.get_param_store().items()
            if ".scale" in n
        ])
        sigma_mean = scales.mean().item()
        sigma_p95  = scales.quantile(0.95).item()

    print(f"Epoch {epoch:3d} | loss {epoch_loss:9.2f} |"
          f" train {train_acc*100:5.1f}% | val {val_acc*100:5.1f}% |"
          f" σ̄ {sigma_mean:.3f}  p95 {sigma_p95:.3f}")

# ------------------------------------------------------------
# 6.  Save the guide checkpoint (optional) -------------------
torch.save(pyro.get_param_store().get_state(), "smalltraintest.pt")
