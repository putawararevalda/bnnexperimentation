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

import pandas as pd

device = torch.device("cuda")

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

from pyro.infer.autoguide.initialization import init_to_median

import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

from torchvision.datasets import ImageFolder

from dotenv import load_dotenv
import requests
import os

import warnings

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

class BayesShipsCNNSmartpool(PyroModule):
    def __init__(
        self,
        num_classes=2,   # now 2 for Categorical
        device=torch.device("cuda"),
        activation='relu',
        prior_dist='gaussian',
        mu=0.0,
        b=1.0,
        prior_params=None,
        smartpool_switch = False,
        pool_threshold=10.0,
        pool_detect_only=False,
        dropout_switch=False,
        dropout_p=0.5
    ):
        super().__init__()
        self.device = device

        # Activation setup
        if isinstance(activation, str):
            act_map = {
                'relu': F.relu,
                'tanh': torch.tanh,
                'sigmoid': torch.sigmoid,
                'sinusoidal': torch.sin,
                'relu6': F.relu6,
                'leaky_relu': F.leaky_relu,
                'selu': F.selu,
                'wg': self.actWG,
                'rwg': self.actRWG,
            }
            self.activation_fn = act_map[activation]
        elif callable(activation):
            self.activation_fn = activation
        else:
            raise ValueError("activation must be a string or callable")

        # Prior setup
        self.prior_dist = prior_dist
        params = {'mu': mu, 'b': b} if prior_params is None else prior_params
        self.prior_mu = torch.tensor(params['mu'], device=device, dtype=torch.float32)
        self.prior_b  = torch.tensor(params['b'], device=device, dtype=torch.float32)

        print(f"[INFO] Using prior: {self.prior_dist} (mu={self.prior_mu.item()}, b={self.prior_b.item()})")

        # Layers
        self.conv1 = PyroModule[nn.Conv2d](3, 32, kernel_size=3, padding=1)
        self.conv1.weight = PyroSample(self._make_prior([32, 3, 3, 3]))
        self.conv1.bias   = PyroSample(self._make_prior([32]))

        self.conv2 = PyroModule[nn.Conv2d](32, 64, kernel_size=3, padding=1)
        self.conv2.weight = PyroSample(self._make_prior([64, 32, 3, 3]))
        self.conv2.bias   = PyroSample(self._make_prior([64]))

        if not smartpool_switch:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        elif smartpool_switch:

            self.pool = SmartPool(
                kernel_size=2,
                stride=2,
                threshold=pool_threshold,
                detect_only=pool_detect_only
            )

        self.dropout_switch = dropout_switch

        if self.dropout_switch:
            self.dropout = nn.Dropout(p=dropout_p)

        self.fc1 = PyroModule[nn.Linear](64 * 16 * 16, num_classes)
        self.fc1.weight = PyroSample(self._make_prior([num_classes, 64 * 16 * 16]))
        self.fc1.bias   = PyroSample(self._make_prior([num_classes]))

    def actWG(self, x, alpha=1.0):
        return x * torch.exp(-alpha * x ** 2)

    def actRWG(self, x, alpha=1.0):
        wg = x * torch.exp(-alpha * x ** 2)
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
        x = self.activation_fn(self.conv1(x))
        x = self.pool(x)
        x = self.activation_fn(self.conv2(x))
        x = self.pool(x)

        if self.dropout_switch:
            x = self.dropout(x)

        x = x.view(x.size(0), -1)
        logits = self.fc1(x)  # shape [batch, 2]

        if y is not None:
            with pyro.plate("data", x.size(0)):
                pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits


shipsnet_mean = [0.4119, 0.4243, 0.3724]
shipsnet_std = [0.1899, 0.1569, 0.1515]

old_mean = [0.3444, 0.3803, 0.4078]
old_std = [0.0914, 0.0651, 0.0552]


def load_data(batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=shipsnet_mean, 
                             std=shipsnet_std)
    ])

    #dataset = datasets.EuroSAT(root='./data', transform=transform, download=True)
    dataset = ImageFolder(
    root="data/shipsnet/foldered",
    transform=transform
    )
    torch.manual_seed(42)

    #train_size = int(0.8 * len(dataset))
    #test_size = len(dataset) - train_size
    #train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    with open('datasplit/shipsnet_split_indices.pkl', 'rb') as f:
        split = pickle.load(f)
        train_dataset = Subset(dataset, split['train'])
        test_dataset = Subset(dataset, split['test'])

    # Add num_workers and pin_memory for faster data loading
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    return train_loader, test_loader

# training SVI function

import os
import torch
import pyro
from tqdm import tqdm
import numpy as np

def train_svi_with_stats(
    model,
    guide,
    svi,
    train_loader,
    device,
    num_epochs=10,
    save_epochs=None,
    save_dir='results',
    model_filename_pattern='model_{activation}_{prior}_epoch_{epoch}_{timestamp}.pth',
    guide_filename_pattern='guide_{activation}_{prior}_epoch_{epoch}_{timestamp}.pth',
    param_store_filename_pattern='param_store_{activation}_{prior}_epoch_{epoch}_{timestamp}.pkl',
    accuracies_filename_pattern='accuracy_results_{activation}_{prior}_{timestamp}.csv',
    losses_filename_pattern='losses_{activation}_{prior}_{timestamp}.csv',
    model_config_filename_pattern='config_{activation}_{prior}_{timestamp}.json'
):
    """
    Train the SVI model, track losses/accuracies, and
    save artifacts only when accuracy improves, naming files
    like `model_relu_gaussian_epoch_3.pth`.
    """
    
    # Pull names off the model if available, else fall back
    #act_name  = getattr(model, 'activation', getattr(model, 'activation_name', 'act'))
    act_name = model.activation_fn.__name__ if hasattr(model.activation_fn, '__name__') else str(model.activation_fn)
    prior_name = getattr(model, 'prior_dist', 'prior')
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    os.makedirs(save_dir, exist_ok=True)
    save_epochs = set(save_epochs or range(1, num_epochs+1))

    pyro.clear_param_store()
    model.to(device)

    epoch_losses, epoch_accuracies, accuracy_epochs = [], [], []
    loc_stats = {'epochs': [], 'means': [], 'stds': []}
    scale_stats   = {'epochs': [], 'means': [], 'stds': []}
    best_acc = 0.0

    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0.0
        batches = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device).long()
            total_loss += svi.step(images, labels)
            batches += 1

        avg_loss = total_loss / batches
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch} - ELBO Loss: {avg_loss:.4f}")

        if epoch == 1 or epoch % 10 == 0 or epoch == num_epochs:
            model.eval(); guide.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in tqdm(train_loader, desc=f"Acc check epoch {epoch}"):
                    images, labels = images.to(device), labels.to(device)
                    trace = pyro.poutine.trace(guide).get_trace(images)
                    replayed = pyro.poutine.replay(model, trace=trace)
                    logits = replayed(images)
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            acc = correct/total
            epoch_accuracies.append(acc); accuracy_epochs.append(epoch)
            print(f"Epoch {epoch} - Train Acc: {acc*100:.2f}%")

            # record stats...
            w_means, w_stds, b_means, b_stds = [], [], [], []
            for name, param in pyro.get_param_store().items():
                if 'loc' or 'low' in name:
                    w_means.append(param.mean().item()); w_stds.append(param.std(unbiased=False).item())
                elif 'scale' or 'width' in name:
                    b_means.append(param.mean().item()); b_stds.append(param.std(unbiased=False).item())
            loc_stats['epochs'].append(epoch)
            loc_stats['means'].append(w_means)
            loc_stats['stds'].append(w_stds)
            scale_stats['epochs'].append(epoch)
            scale_stats['means'].append(b_means)
            scale_stats['stds'].append(b_stds)

            #for name, param in pyro.get_param_store().items():
            #    if 'loc' in name or 'scale' in name:
            #        print(f"{name}: {param.detach().cpu().numpy()}")

            # only save when accuracy improves
            if acc > best_acc:
                best_acc = acc
                fname_model = model_filename_pattern.format(activation=act_name, prior=prior_name, epoch="best", timestamp=timestamp)
                fname_guide = guide_filename_pattern.format(activation=act_name, prior=prior_name, epoch="best", timestamp=timestamp)
                fname_ps    = param_store_filename_pattern.format(activation=act_name, prior=prior_name, epoch="best", timestamp=timestamp)

                torch.save(model.state_dict(), os.path.join(save_dir, fname_model))
                torch.save(guide.state_dict(), os.path.join(save_dir, fname_guide))
                pyro.get_param_store().save(os.path.join(save_dir, fname_ps))
                print(f"  ↳ Saved: {fname_model}, {fname_guide}, {fname_ps}")

    # save losses per epoch in a csv file, with consistent file naming
    accuracies_df = pd.DataFrame({
        'epoch': accuracy_epochs,
        'accuracy': epoch_accuracies
    })
    accuracies_df.to_csv(os.path.join(save_dir,accuracies_filename_pattern.format(activation=act_name, prior=prior_name, timestamp=timestamp)), index=False)

    loss_df = pd.DataFrame({
        'epoch': list(range(1, epoch + 1)),
        'loss': epoch_losses
    })
    loss_df.to_csv(os.path.join(save_dir,losses_filename_pattern.format(activation=act_name, prior=prior_name, timestamp=timestamp)), index=False)
            
    # save model configuration in a json file
    config = {
        'activation': act_name,
        'prior': prior_name,
        'num_epochs': num_epochs,
        'best_accuracy_at_epoch': accuracy_epochs[np.argmax(epoch_accuracies)],
        'best_accuracy': best_acc,
        'batch_size': train_loader.batch_size,
        'train_size': len(train_loader.dataset),
        'prior_params': {
            'mu': model.prior_mu.item(),
            'b': model.prior_b.item()
        },
    }
    config_filename = model_config_filename_pattern.format(activation=act_name, prior=prior_name, timestamp=timestamp)

    with open(os.path.join(save_dir, config_filename), 'w') as f:
        import json
        json.dump(config, f, indent=4)
        print(f"Configuration saved to {config_filename}")

    return epoch_losses, epoch_accuracies, accuracy_epochs, loc_stats, scale_stats, os.path.join(save_dir, fname_model), os.path.join(save_dir, fname_guide), os.path.join(save_dir, fname_ps), timestamp


def plot_training_results_with_stats(losses, accuracies, accuracy_epochs, loc_stats, scale_stats, act_name, prior_name, timestamp):
    """Plot training results with weight and bias statistics"""
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Training Loss
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(losses) + 1), losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('ELBO Loss')
    plt.grid(True)
    
    # Plot 2: Training Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(accuracy_epochs, accuracies, 'o-')
    plt.title('Training Accuracy (Every 10 Epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # Plot 3: Weight Statistics Boxplot
    plt.subplot(2, 2, 3)
    loc_data = []
    loc_labels = []
    
    for i, epoch in enumerate(loc_stats['epochs']):
        # Combine means and stds for this epoch
        epoch_data = loc_stats['means'][i] + loc_stats['stds'][i]
        loc_data.append(epoch_data)
        loc_labels.append(f'Epoch {epoch}')
    
    if loc_data:
        bp1 = plt.boxplot(loc_data, labels=loc_labels, patch_artist=True)
        for patch in bp1['boxes']:
            patch.set_facecolor('lightblue')
    
    plt.title('LOC Statistics Distribution')
    plt.xlabel('Epoch')
    plt.ylabel('LOC Values')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Scale Statistics Boxplot
    plt.subplot(2, 2, 4)
    scale_data = []
    scale_labels = []
    
    for i, epoch in enumerate(scale_stats['epochs']):
        # Combine means and stds for this epoch
        epoch_data = scale_stats['means'][i] + scale_stats['stds'][i]
        scale_data.append(epoch_data)
        scale_labels.append(f'Epoch {epoch}')
    
    if scale_data:
        #bp2 = plt.boxplot(scale_data, tick_labels=scale_labels, patch_artist=True)
        bp2 = plt.boxplot(scale_data, labels=scale_labels, patch_artist=True)
        for patch in bp2['boxes']:
            patch.set_facecolor('lightcoral')
    
    plt.title('SCALE Statistics Distribution')
    plt.xlabel('Epoch')
    plt.ylabel('SCALE Values')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir,f'bayesian_cnn_training_results_{act_name}_{prior_name}_{timestamp}.png'))
    #plt.show()

import numpy as np
from sklearn.metrics import confusion_matrix


def predict_data(model, loader_of_interest, num_samples=10):
    model.eval()
    guide.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in tqdm(loader_of_interest, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            logits_mc = torch.zeros(num_samples, images.size(0), model.fc1.out_features).to(device)

            for i in range(num_samples):
                guide_trace = pyro.poutine.trace(guide).get_trace(images)
                replayed_model = pyro.poutine.replay(model, trace=guide_trace)
                logits = replayed_model(images)
                logits_mc[i] = logits

            avg_logits = logits_mc.mean(dim=0)
            predictions = torch.argmax(avg_logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    return all_labels, all_predictions

def save_predictions_to_csv(labels, predictions, filename='predictions.csv'):
    df = pd.DataFrame({'True Label': labels, 'Predicted Label': predictions})
    df.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")

def send_telegram_message(title, message):
    load_dotenv('.env')
    token = os.getenv('TELEGRAM_BOT_TOKEN')

    try:
        response = requests.post(f'https://api.telegram.org/bot{token}/sendMessage', data={
            'chat_id': os.getenv('TELEGRAM_CHAT_ID'),
            'text': f'{title}\n{message}',
            #'parse_mode': 'Markdown'
        })
    except requests.exceptions.RequestException as e:
        print(f"Error sending message: {e}")
        return None


import torch
import pyro
import pyro.distributions as dist
from pyro.nn.module import PyroModule, PyroParam
from pyro.infer.autoguide import AutoGuide
from pyro.infer.autoguide.initialization import InitMessenger, init_to_feasible
from pyro.distributions import constraints
from contextlib import ExitStack

from pyro.distributions.util       import sum_rightmost
from pyro.ops.tensor_utils         import periodic_repeat
from pyro.distributions.transforms import biject_to
from pyro.infer.autoguide.utils    import (
    deep_setattr,
    deep_getattr,
    helpful_support_errors,
)

import pyro.poutine as poutine


class AutoLaplace(AutoGuide):
    """
    An AutoGuide that uses a Laplace(loc, scale) marginal for each latent.
    """
    scale_constraint = constraints.softplus_positive

    def __init__(
        self, model, *, init_loc_fn=init_to_feasible, init_scale=0.1, create_plates=None
    ):
        self.init_loc_fn = init_loc_fn
        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError(f"Expected init_scale > 0, got {init_scale}")
        self._init_scale = init_scale

        model = InitMessenger(self.init_loc_fn)(model)
        super().__init__(model, create_plates=create_plates)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)

        self._event_dims = {}
        self.locs = PyroModule()
        self.scales = PyroModule()

        for name, site in self.prototype_trace.iter_stochastic_nodes():
            # ← use helpful_support_errors directly
            with helpful_support_errors(site):
                init_loc = (
                    biject_to(site["fn"].support)
                    .inv(site["value"].detach())
                    .detach()
                )
            event_dim = site["fn"].event_dim + init_loc.dim() - site["value"].dim()
            self._event_dims[name] = event_dim

            # handle subsampling plates
            for frame in site["cond_indep_stack"]:
                full_size = frame.full_size or frame.size
                if full_size != frame.size:
                    dim = frame.dim - event_dim
                    init_loc = periodic_repeat(init_loc, full_size, dim).contiguous()

            init_scale = torch.full_like(init_loc, self._init_scale)

            deep_setattr(
                self.locs, name, PyroParam(init_loc, constraints.real, event_dim)
            )
            deep_setattr(
                self.scales,
                name,
                PyroParam(init_scale, self.scale_constraint, event_dim),
            )

    def _get_loc_and_scale(self, name):
        site_loc = deep_getattr(self.locs, name)
        site_scale = deep_getattr(self.scales, name)
        return site_loc, site_scale

    def forward(self, *args, **kwargs):
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        plates = self._create_plates(*args, **kwargs)
        result = {}

        for name, site in self.prototype_trace.iter_stochastic_nodes():
            transform = biject_to(site["fn"].support)

            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    if frame.vectorized:
                        stack.enter_context(plates[frame.name])

                site_loc, site_scale = self._get_loc_and_scale(name)

                unconstrained = pyro.sample(
                    f"{name}_unconstrained",
                    dist.Laplace(site_loc, site_scale)
                        .to_event(self._event_dims[name]),
                    infer={"is_auxiliary": True},
                )

                value = transform(unconstrained)
                if poutine.get_mask() is False:
                    log_density = 0.0
                else:
                    log_density = transform.inv.log_abs_det_jacobian(
                        value, unconstrained
                    )
                    log_density = sum_rightmost(
                        log_density,
                        log_density.dim() - value.dim() + site["fn"].event_dim,
                    )
                delta = dist.Delta(
                    value,
                    log_density=log_density,
                    event_dim=site["fn"].event_dim,
                )
                result[name] = pyro.sample(name, delta)

        return result

    @torch.no_grad()
    def median(self, *args, **kwargs):
        medians = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            site_loc, _ = self._get_loc_and_scale(name)
            med = biject_to(site["fn"].support)(site_loc)
            medians[name] = med.clone() if med is site_loc else med
        return medians

    @torch.no_grad()
    def quantiles(self, quantiles, *args, **kwargs):
        results = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            site_loc, site_scale = self._get_loc_and_scale(name)
            qs = torch.tensor(quantiles, dtype=site_loc.dtype, device=site_loc.device)
            qs = qs.reshape((-1,) + (1,) * site_loc.dim())
            qvals = dist.Laplace(site_loc, site_scale).icdf(qs)
            results[name] = biject_to(site["fn"].support)(qvals)
        return results


import torch
import pyro
from pyro.infer.autoguide import AutoGuide
from pyro.nn.module import PyroModule
from pyro.distributions import Uniform, Delta, constraints
from pyro.distributions.transforms import biject_to
from pyro import poutine
from contextlib import ExitStack

## TODO THIS CLASS IS CHANGED FROM THE ORIGINAL AUTO-UNIFORM
class AutoUniform(AutoGuide):
    """
    A drop‑in replacement for AutoUniform that never re‑initializes
    any already‑registered pyro.param() entries, so your SEU perturbations stick.
    """
    width_constraint = constraints.softplus_positive

    def __init__(self, model, *, init_loc_fn=init_to_feasible, init_scale: float, create_plates=None):
        assert init_scale > 0
        self._init_scale = init_scale
        
        # wrap model so that a single prior draw seeds init_loc
        model = pyro.poutine.block(model)  # block any poutine interference
        super().__init__(model, create_plates=create_plates)
        self._init_lows = {}
        self._init_widths = {}
        self._event_dims = {}

    def _setup_prototype(self, *args, **kwargs):
        # Run the usual prototype logic exactly once
        print(">>> _setup_prototype running!")
        super()._setup_prototype(*args, **kwargs)
        # Walk the trace and save init_low/width for each site
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            # invert‐transform the one “prototype” sample
            loc = biject_to(site["fn"].support).inv(site["value"].detach())
            init_loc = loc.detach()
            # reconstruct any subsampling
            event_dim = site["fn"].event_dim + init_loc.dim() - site["value"].dim()
            self._event_dims[name] = event_dim
            for frame in site["cond_indep_stack"]:
                full = frame.full_size or frame.size
                if full != frame.size:
                    dim = frame.dim - event_dim
                    init_loc = init_loc.repeat_interleave(full // frame.size, dim)
            # stash
            self._init_lows[name]   = init_loc - self._init_scale
            self._init_widths[name] = torch.full_like(init_loc, 2.0 * self._init_scale)

    def forward(self, *args, **kwargs):
        # on first call, set up prototype
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        plates = self._create_plates(*args, **kwargs)
        out = {}

        for name, site in self.prototype_trace.iter_stochastic_nodes():
            # fetch-or-create in ParamStore
            low = pyro.param(
                f"uniform_low::{name}",
                self._init_lows[name],
                constraint=constraints.real
            )
            width = pyro.param(
                f"uniform_width::{name}",
                self._init_widths[name],
                constraint=self.width_constraint
            )
            transform = biject_to(site["fn"].support)

            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    if frame.vectorized:
                        stack.enter_context(plates[frame.name])

                # sample unconstrained, then transform
                u = pyro.sample(
                    f"{name}_unconstrained",
                    Uniform(low, low + width).to_event(self._event_dims[name]),
                    infer={"is_auxiliary": True}
                )
                value = transform(u)

                # correct log‑density via the Jacobian
                if poutine.get_mask() is False:
                    logp = 0.0
                else:
                    logp = transform.inv.log_abs_det_jacobian(value, u)
                    # sum out the rightmost event dimensions
                    sum_axes = tuple(range(logp.dim() - site["fn"].event_dim))
                    logp = logp.sum(sum_axes)
                out[name] = pyro.sample(
                    name,
                    Delta(value, log_density=logp, event_dim=site["fn"].event_dim)
                )

        return out

    @torch.no_grad()
    def median(self, *args, **kwargs):
        med = {}
        for name in self._init_lows:
            low   = pyro.param(f"uniform_low::{name}")
            width = pyro.param(f"uniform_width::{name}")
            med_val = biject_to(self.prototype_trace.nodes[name]["fn"].support)(
                low + 0.5 * width
            )
            med[name] = med_val
        return med

    @torch.no_grad()
    def quantiles(self, qs, *args, **kwargs):
        out = {}
        q = torch.tensor(qs)
        for name in self._init_lows:
            low   = pyro.param(f"uniform_low::{name}")
            width = pyro.param(f"uniform_width::{name}")
            vals = Uniform(low, low + width).icdf(
                q.view(-1, *([1] * low.dim()))
            )
            out[name] = biject_to(self.prototype_trace.nodes[name]["fn"].support)(vals)
        return out


from pyro.infer.autoguide import AutoNormal
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from tqdm import tqdm
import pandas as pd

import argparse

if __name__ == "__main__":
    num_classes = 2

    parser = argparse.ArgumentParser(description='Train Bayesian Neural Net on Shipsnet with Variational Inference')
    parser.add_argument('--prior', type=str, nargs='?', action='store', default='Gaussian_prior',
                    help='Model to run. Options are \'Gaussian_prior\', \'Laplace_prior\','
                         ' \'Uniform_prior\'. Default: \'Gaussian_prior\'.')
    parser.add_argument('--epoch', type=int, nargs='?', action='store', default='10',
                help='Number of epoch. Default: 10.')
    # --smartpool will set smartpool=True
    parser.add_argument(
        '--smartpool',
        dest='smartpool',
        action='store_true',
        help='Activate the smartpool layer.'
    )
    # --no-smartpool will set smartpool=False
    parser.add_argument(
        '--no-smartpool',
        dest='smartpool',
        action='store_false',
        help='Deactivate the smartpool layer.'
    )
    # default=False if neither flag is present
    parser.set_defaults(smartpool=False)

    # --wd will set weight_decay=True
    parser.add_argument(
        '--wd',
        dest='weight_decay',
        action='store_true',
        help='Activate the optimizers weight decay.'
    )
    # --no-wd will set weight_decay=False
    parser.add_argument(
        '--no-wd',
        dest='weight_decay',
        action='store_false',
        help='Deactivate the optimizers weight decay.'
    )
    # default=False if neither flag is present
    parser.set_defaults(weight_decay=False)

    parser.add_argument('--save-dir', type=str, nargs='?', action='store', default='results_GP_shipsnet_experiment',
            help='Save file directory. Default: \'results_GP_shipsnet_experiment\'.')
    
    parser.add_argument(
        '--trial-mode',
        dest='trial_mode',
        action='store_true',
        help='Activate trial mode.'
    )
    # --no-wd will set weight_decay=False
    parser.add_argument(
        '--no-trial-mode',
        dest='trial_mode',
        action='store_false',
        help='Deactivate trial mode.'
    )
    # default=False if neither flag is present
    parser.set_defaults(trial_mode=False)

    parser.add_argument(
        '--dropout-mode',
        dest='dropout_mode',
        action='store_true',
        help='Activate dropout in the model.'
    )
    # --no-wd will set weight_decay=False
    parser.add_argument(
        '--no-dropout-mode',
        dest='dropout_mode',
        action='store_false',
        help='Deactivate dropout in the model.'
    )
    # default=False if neither flag is present
    parser.set_defaults(dropout_mode=False)

    parser.add_argument('--b-set', type=str, nargs='?', action='store', default='full',
        help='Sets of b values that wanted to be test. Options are :\'full\', \'single\' Default: \'full\'.')
    
    #act_fn_list = ['gaussian', 'laplace', 'uniform']
    args = parser.parse_args()

    if args.prior == 'Gaussian_prior':
        act_fn_list = ['gaussian']
    elif args.prior == 'Laplace_prior':
        act_fn_list = ['laplace']
    elif args.prior == 'Uniform_prior':
        act_fn_list = ['uniform']

    prior_list = ['relu','tanh','sigmoid','sinusoidal','relu6','wg','rwg']
    #prior_list = ['wg','rwg']

    if args.b_set == 'full':
        b_list = [10.0, 1.0, 0.1]
        #b_list = [0.1]
    elif args.b_set == 'single':
        b_list = [1.0]

    if args.trial_mode:
        # For trial mode, reduce the number of combinations
        act_fn_list = act_fn_list[:1]
        prior_list = prior_list[:1]
        b_list = b_list[:1]

    #count how many combinations we have
    total_combinations = len(act_fn_list) * len(prior_list) * len(b_list)
    print(f"Total combinations to run: {total_combinations}")

    

    print("==========================================")
    print(f"PROJECT Configuration:")
    print(f"Trial Mode      : {args.trial_mode}")
    print(f"Prior           : {args.prior}")
    print(f"Dropout Mode    : {args.dropout_mode}")
    print(f"B Set           : {args.b_set}")
    print(f"Epochs          : {args.epoch}")
    print(f"Smartpool       : {args.smartpool}")
    print(f"Weight Decay    : {args.weight_decay}")
    print(f"Save Directory  : {args.save_dir}")
    print("==========================================")

    experiment_number = 0

    #'rwg','wg',
    for activation_iter in act_fn_list:
        for prior_iter in prior_list:
            for b_iter in b_list:

                experiment_number += 1
                experiment_time_start = time.time()
                #send telegram message to announce the start of the experiment (x/total combinations)
                send_telegram_message(
                    title=f"Experiment {experiment_number}/{total_combinations}",
                    message=f"Running with activation={activation_iter}, prior={prior_iter}, b={b_iter}"
                )

                pyro.clear_param_store()

                # print log to annoounce what experiment is running
                print("==========================================")
                print(f"Running experiment with activation={prior_iter}, prior={activation_iter}, b={b_iter}")
                print("==========================================")
                bayesian_model = BayesShipsCNNSmartpool(num_classes,
                        device,
                        activation=prior_iter,
                        prior_dist=activation_iter,
                        mu = 0.0,
                        b= b_iter,
                        smartpool_switch = args.smartpool,
                        pool_threshold=10.0,
                        pool_detect_only=False,
                        dropout_switch=args.dropout_mode,
                        dropout_p=0.5
                        #prior_params={'mu': 0.0, 'b': b_iter})
                        )
                
                # 1) construct your guide so its locs start at p(w).mean=0
                #guide = AutoDiagonalNormal(
                #    bayesian_model,
                #    init_loc_fn=init_to_median(num_samples=1),   # all μ_q ← prior mean (0)
                #    init_scale=0.1               # set initial σ_q=0.1
                #)

                if activation_iter == 'gaussian':
                    guide = AutoNormal(bayesian_model, init_scale=0.05).to(device)
                elif activation_iter == 'laplace':
                    guide = AutoLaplace(bayesian_model, init_scale=0.05).to(device)
                elif activation_iter == 'uniform':
                    guide = AutoUniform(bayesian_model, init_scale=0.05).to(device)
                else:
                    raise ValueError(f"Unsupported activation: {activation_iter}")
                
                if args.weight_decay:
                    optimizer = ClippedAdam({"lr": 1e-3, "weight_decay": 1e-4}) #  0.0001
                elif not args.weight_decay:
                    optimizer = ClippedAdam({"lr": 1e-3, "weight_decay": 0.0})

                #optimizer = Adam({"lr": 1e-3,
                #                  "weight_decay": 1e-4,
                #                  })  # Increased from 1e-4 to 1e-3, weight decay added
                svi = pyro.infer.SVI(model=bayesian_model,
                                    guide=guide,
                                    optim=optimizer,
                                    loss=pyro.infer.Trace_ELBO(num_particles=1,
                                                                )) #TODO

                pyro.clear_param_store()

                # Ensure model and guide are on the correct device
                bayesian_model.to(device)
                guide.to(device)

                train_loader, test_loader = load_data(batch_size=16)
                
                losses, accuracies, accuracy_epochs, loc_stats, scale_stats, best_model_path, best_guide_path, best_param_store_path, experiment_timestamp = train_svi_with_stats(
                bayesian_model,
                guide,
                svi,
                train_loader,
                device,
                num_epochs=args.epoch,
                save_epochs=None,
                save_dir=args.save_dir,)
                
                act_name = bayesian_model.activation_fn.__name__ if hasattr(bayesian_model.activation_fn, '__name__') else str(bayesian_model.activation_fn)
                prior_name = getattr(bayesian_model, 'prior_dist', 'prior')

                plot_training_results_with_stats(losses, accuracies, accuracy_epochs, loc_stats, scale_stats, act_name, prior_name, experiment_timestamp)

                all_labels, all_predictions = predict_data(bayesian_model, test_loader, num_samples=10)
                cm = confusion_matrix(all_labels, all_predictions)
                #print accuracy from confusion matrix
                accuracy = np.trace(cm) / np.sum(cm)
                print(f"Accuracy from confusion matrix: {accuracy * 100:.6f}%")

                experiment_time_finish = time.time()

                save_predictions_to_csv(all_labels, all_predictions, os.path.join(args.save_dir, f'predictions_{act_name}_{prior_name}_{experiment_timestamp}_{accuracy * 100:.0f}.csv'))

                send_telegram_message(
                    title=f"Experiment {experiment_number}/{total_combinations} Finished",
                    message=f"Activation: {prior_iter}, Prior: {activation_iter}, b: {b_iter}\n"
                            f"Best Model Test Accuracy: {accuracy * 100:.2f}%\n"
                            f"Time taken: {experiment_time_finish - experiment_time_start:.2f} seconds"
                )

                