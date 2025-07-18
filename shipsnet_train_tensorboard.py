import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import pickle
import pandas as pd
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer.autoguide.initialization import init_to_median
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# -- Model Definition --
class BayesShipsCNN(PyroModule):
    def __init__(
        self,
        num_classes=2,
        device=torch.device("cuda"),
        activation='relu',
        prior_dist='gaussian',
        mu=0.0,
        b=1.0,
        prior_params=None
    ):
        super().__init__()
        self.device = device
        # Activation lookup
        act_map = {
            'relu': F.relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'sinusoidal': torch.sin,
            'relu6': F.relu6,
            'leaky_relu': F.leaky_relu,
            'selu': F.selu,
            'wg': self.actWG,
            'rwg': self.actRWG
        }
        self.activation_fn = act_map.get(activation, None)
        if self.activation_fn is None:
            raise ValueError(f"Unsupported activation: {activation}")

        # Prior setup
        self.prior_dist = prior_dist
        default_params = {'mu': mu, 'b': b}
        params = default_params if prior_params is None else prior_params
        self.prior_mu = torch.tensor(params['mu'], device=device)
        self.prior_b  = torch.tensor(params['b'], device=device)
        print(f"Using prior: {self.prior_dist} (mu={self.prior_mu.item()}, b={self.prior_b.item()})")

        # Convolutional layers
        self.conv1 = PyroModule[nn.Conv2d](3, 32, kernel_size=3, padding=1)
        self.conv1.weight = PyroSample(self._make_prior([32,3,3,3]))
        self.conv1.bias   = PyroSample(self._make_prior([32]))
        self.conv2 = PyroModule[nn.Conv2d](32, 64, kernel_size=3, padding=1)
        self.conv2.weight = PyroSample(self._make_prior([64,32,3,3]))
        self.conv2.bias   = PyroSample(self._make_prior([64]))
        self.conv3 = PyroModule[nn.Conv2d](64, 128, kernel_size=3, padding=1)
        self.conv3.weight = PyroSample(self._make_prior([128,64,3,3]))
        self.conv3.bias   = PyroSample(self._make_prior([128]))
        self.pool = nn.MaxPool2d(2,2)
        self.gap  = nn.AdaptiveAvgPool2d((1,1))

        # Fully connected
        self.fc1 = PyroModule[nn.Linear](128,256)
        self.fc1.weight = PyroSample(self._make_prior([256,128]))
        self.fc1.bias   = PyroSample(self._make_prior([256]))
        self.fc2 = PyroModule[nn.Linear](256,num_classes)
        self.fc2.weight = PyroSample(self._make_prior([num_classes,256]))
        self.fc2.bias   = PyroSample(self._make_prior([num_classes]))

    def actWG(self, x, alpha=1.0):
        return x * torch.exp(-alpha * x**2)
    def actRWG(self, x, alpha=1.0):
        wg = x * torch.exp(-alpha * x**2)
        return torch.max(torch.zeros_like(wg), wg)

    def _make_prior(self, shape):
        if self.prior_dist == 'gaussian':
            base = dist.Normal(self.prior_mu, self.prior_b)
        elif self.prior_dist == 'laplace':
            base = dist.Laplace(self.prior_mu, self.prior_b)
        elif self.prior_dist == 'uniform':
            base = dist.Uniform(-self.prior_b, self.prior_b)
        else:
            raise ValueError(f"Unsupported prior: {self.prior_dist}")
        return base.expand(shape).to_event(len(shape))

    def forward(self, x, y=None):
        x = self.activation_fn(self.conv1(x.to(self.device)))
        x = self.pool(x)
        x = self.activation_fn(self.conv2(x))
        x = self.pool(x)
        x = self.activation_fn(self.conv3(x))
        x = self.pool(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.activation_fn(self.fc1(x))
        logits = self.fc2(x)
        if y is not None:
            with pyro.plate("data", x.size(0)):
                pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits

# -- Data Loading --
shipsnet_mean = [0.4119, 0.4243, 0.3724]
shipsnet_std  = [0.1899, 0.1569, 0.1515]

def load_data(batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=shipsnet_mean, std=shipsnet_std)
    ])
    dataset = ImageFolder(root="data/shipsnet/foldered", transform=transform)
    with open('datasplit/shipsnet_split_indices.pkl','rb') as f:
        split = pickle.load(f)
    train_ds = Subset(dataset, split['train'])
    test_ds  = Subset(dataset, split['test'])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, num_workers=4, pin_memory=True)
    return train_loader, test_loader

# -- Training with TensorBoard Logging --
def train_svi_with_stats(
    model, guide, svi, train_loader, device,
    num_epochs=10, save_dir='results', tensorboard_dir='runs'
):
    # Setup
    act_name = model.activation_fn.__name__
    prior_name = model.prior_dist
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.join(tensorboard_dir, f"shipsnet_{act_name}_{prior_name}_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)

    pyro.clear_param_store(); model.to(device)
    best_acc = 0.0
    epoch_losses, epoch_accs = [], []

    for epoch in range(1, num_epochs+1):
        model.train(); total_loss=0; batches=0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            total_loss += svi.step(imgs, labels)
            batches += 1
        avg_loss = total_loss / batches
        epoch_losses.append(avg_loss)
        writer.add_scalar('Train/ELBO_Loss', avg_loss, epoch)

        # Evaluate
        model.eval(); guide.eval()
        correct=0; total=0
        with torch.no_grad():
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                trace = pyro.poutine.trace(guide).get_trace(imgs)
                logits = pyro.poutine.replay(model, trace=trace)(imgs)
                preds = logits.argmax(dim=1)
                correct += (preds==labels).sum().item()
                total += len(labels)
        acc = correct/total
        epoch_accs.append(acc)
        writer.add_scalar('Train/Accuracy', acc, epoch)

        # Log parameter histograms
        for name, param in pyro.get_param_store().items():
            writer.add_histogram(name, param.detach().cpu().numpy(), epoch)

        # Save best
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_best_{act_name}_{prior_name}.pth"))
            pyro.get_param_store().save(os.path.join(save_dir, f"params_best_{act_name}_{prior_name}.pkl"))

    writer.close()
    print(f"TensorBoard logs saved to {log_dir}")
    print(f"Start TensorBoard with:\n  tensorboard --logdir {tensorboard_dir} --host 0.0.0.0 --port 6006")
    return epoch_losses, epoch_accs

# -- Prediction & Utils --
def predict_data(model, guide, loader, device, num_samples=10):
    model.eval(); guide.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluating"):
            imgs, labels = imgs.to(device), labels.to(device)
            logits_mc = torch.zeros(num_samples, imgs.size(0), model.fc1.out_features).to(device)
            for i in range(num_samples):
                trace = pyro.poutine.trace(guide).get_trace(imgs)
                logits_mc[i] = pyro.poutine.replay(model, trace)(imgs)
            preds = logits_mc.mean(0).argmax(-1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    return all_labels, all_preds

def save_predictions(labels, preds, path):
    df = pd.DataFrame({'True': labels, 'Pred': preds})
    df.to_csv(path, index=False)
    print(f"Saved predictions to {path}")

# -- Main --
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_data(batch_size=16)

    for prior_dist in ['gaussian','laplace','uniform']:
        for activation in ['relu','tanh','sigmoid','sinusoidal','relu6','wg','rwg']:
            for b in [10.0,1.0,0.1]:
                pyro.clear_param_store()
                print(f"Running exp: act={activation}, prior={prior_dist}, b={b}")
                model = BayesShipsCNN(2, device, activation, prior_dist, mu=0.0, b=b)
                guide = AutoDiagonalNormal(model, init_loc_fn=init_to_median(1), init_scale=0.1)
                svi = SVI(model, guide, Adam({'lr':1e-3, 'weight_decay':1e-4}), Trace_ELBO())

                losses, accs = train_svi_with_stats(model, guide, svi, train_loader, device,
                                                   num_epochs=100, save_dir='results', tensorboard_dir='runs')

                labels, preds = predict_data(model, guide, test_loader, device)
                cm = confusion_matrix(labels, preds)
                acc = np.trace(cm) / cm.sum()
                print(f"Test accuracy: {acc*100:.2f}%")
                save_predictions(labels, preds, f"results/predictions_{activation}_{prior_dist}_{int(acc*100)}.csv")
