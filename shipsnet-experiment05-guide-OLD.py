import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
import time
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split


import torch
import pyro
import pyro.distributions as dist
from pyro.nn.module import PyroModule, PyroParam
from pyro.infer.autoguide import AutoGuide
from pyro.infer.autoguide.initialization import InitMessenger, init_to_feasible
from pyro.distributions import constraints
from contextlib import ExitStack

import pickle
from tqdm import tqdm
import copy

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoNormal

import pandas as pd

import numpy as np
from sklearn.metrics import confusion_matrix

from bitflip import bitflip_float32

from torchvision.datasets import ImageFolder

import os

import json

import argparse

from dotenv import load_dotenv
import requests

parser = argparse.ArgumentParser(description='Train Bayesian Neural Net on Shipsnet with Variational Inference')
parser.add_argument('--prior', type=str, nargs='?', action='store', default='Gaussian_prior',
                help='Model to run. Options are \'Gaussian_prior\', \'Laplace_prior\','
                        ' \'Uniform_prior\'. Default: \'Gaussian_prior\'.')
parser.add_argument('--save-dir', type=str, nargs='?', action='store', default='shipsnet_newslate_guide_seu_result_experiment',
                help='Where to save the file. Default: \'shipsnet_newslate_guide_seu_result_experiment\'.')
parser.add_argument('--search-dir', type=str, nargs='?', action='store', default='results_GP_shipsnet_newslate_guide',
                help='Where are the files to load. Default: \'results_GP_shipsnet_newslate_guide\'.')

parser.add_argument(
    '--multivariate-guide',
    dest='multivariate_guide',
    action='store_true',
    help='Use Multivarite Guide (AutoLowRankMultivariateNormal) for SEU'
)
# --no-wd will set weight_decay=False
parser.add_argument(
    '--auto-guide',
    dest='multivariate_guide',
    action='store_false',
    help='Use AutoNormal guide for SEU'
)
# default=False if neither flag is present
parser.set_defaults(multivariate_guide=False)


parser.add_argument(
    '--limited-mode',
    dest='limited_mode',
    action='store_true',
    help='Only do the SEU to std 1 models'
)
# --no-wd will set weight_decay=False
parser.add_argument(
    '--no-limited-mode',
    dest='limited_mode',
    action='store_false',
    help='SEU done to all models, not just std 1'
)
# default=False if neither flag is present
parser.set_defaults(limited_mode=False)

#act_fn_list = ['gaussian', 'laplace', 'uniform']
args = parser.parse_args()

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

shipsnet_mean = [0.4119, 0.4243, 0.3724]
shipsnet_std = [0.1899, 0.1569, 0.1515]

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

from pyro.distributions.util       import sum_rightmost
from pyro.ops.tensor_utils         import periodic_repeat
from pyro.distributions.transforms import biject_to
from pyro.infer.autoguide.utils    import (
    deep_setattr,
    deep_getattr,
    helpful_support_errors,
)

import pyro.poutine as poutine

from pyro.infer.autoguide import AutoGuideList  #, AutoLowRankMultivariateNormal\
from pyro.infer.autoguide import AutoLowRankMultivariateNormal


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

# Added
class UniformReal(dist.Uniform):
    @property
    def support(self):
        return constraints.real


class AutoUniform(AutoGuide):
    """
    An AutoGuide that uses a Uniform(low, low+width) marginal for each latent.
    """
    # `width` must be positive
    width_constraint = constraints.softplus_positive

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
        self.lows = PyroModule()
        self.widths = PyroModule()

        for name, site in self.prototype_trace.iter_stochastic_nodes():
            # 1. get an unconstrained init_loc (inverse‐transform of site["value"])
            with helpful_support_errors(site):
                init_loc = (
                    biject_to(site["fn"].support)
                    .inv(site["value"].detach())
                    .detach()
                )
            event_dim = site["fn"].event_dim + init_loc.dim() - site["value"].dim()
            self._event_dims[name] = event_dim

            # 2. if subsampled, expand back to full size
            for frame in site["cond_indep_stack"]:
                full_size = frame.full_size or frame.size
                if full_size != frame.size:
                    dim = frame.dim - event_dim
                    init_loc = periodic_repeat(init_loc, full_size, dim).contiguous()

            # 3. build initial low & width around that init_loc
            init_low   = init_loc - self._init_scale
            init_width = torch.full_like(init_loc, 2.0 * self._init_scale)

            # 4. register as PyroParams
            deep_setattr(
                self.lows,  name,
                PyroParam(init_low,   constraints.real,             event_dim),
            )
            deep_setattr(
                self.widths, name,
                PyroParam(init_width, self.width_constraint,       event_dim),
            )

    def _get_low_and_width(self, name):
        low   = deep_getattr(self.lows,  name)
        width = deep_getattr(self.widths, name)
        return low, width

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

                low, width = self._get_low_and_width(name)
                # draw unconstrained latent from Uniform(low, low + width)
                unconstrained = pyro.sample(
                    f"{name}_unconstrained",
                    #dist.Uniform(low, low + width).to_event(self._event_dims[name]),
                    UniformReal(low, low + width).to_event(self._event_dims[name]),
                    infer={"is_auxiliary": True},
                )

                # map into constrained space
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
        """
        Posterior median is just the 0.5‐quantile of Uniform = low + 0.5*width
        """
        medians = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            low, width = self._get_low_and_width(name)
            med = biject_to(site["fn"].support)(low + 0.5 * width)
            medians[name] = med.clone() if med is low else med
        return medians

    @torch.no_grad()
    def quantiles(self, quantiles, *args, **kwargs):
        """
        Posterior quantiles via Uniform.icdf(q).
        """
        results = {}
        qs = torch.tensor(quantiles)
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            low, width = self._get_low_and_width(name)
            # shape: [len(quantiles), *low.shape]
            #qvals = dist.Uniform(low, low + width).icdf(qs.reshape((-1,) + (1,) * low.dim()))
            qvals = UniformReal(low, low + width).icdf(qs.reshape((-1,) + (1,) * low.dim()))
            results[name] = biject_to(site["fn"].support)(qvals)
        return results


import math
import torch.nn as nn
import pyro.distributions as dist
from pyro.nn import PyroParam
from pyro.distributions import constraints
from pyro.infer.autoguide import AutoContinuous
from pyro.infer.autoguide.initialization import init_to_median

class AutoLowRankMultivariateLaplace(AutoContinuous):
    """
    Low-rank-plus-diagonal multivariate Laplace guide.
    Usage::
        guide = AutoLowRankLaplace(model, rank=10)
        svi = SVI(model, guide, ...)
    """
    scale_constraint = constraints.softplus_positive

    def __init__(self, model, init_loc_fn=init_to_median, init_scale=0.1, rank=None):
        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError(f"Expected init_scale > 0 but got {init_scale}")
        if not (rank is None or isinstance(rank, int) and rank > 0):
            raise ValueError(f"Expected rank > 0 but got {rank}")
        self._init_scale = init_scale
        self.rank = rank
        super().__init__(model, init_loc_fn=init_loc_fn)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)
        # location
        self.loc = nn.Parameter(self._init_loc())
        # default rank ≈ sqrt(latent_dim)
        if self.rank is None:
            self.rank = int(round(self.latent_dim**0.5))
        # base (diagonal) scale
        self.scale_base = PyroParam(
            self.loc.new_full((self.latent_dim,), 0.5**0.5 * self._init_scale),
            constraint=self.scale_constraint
        )
        # low-rank factor
        self.cov_factor = nn.Parameter(
            self.loc.new_empty(self.latent_dim, self.rank)
                .normal_(0, 1 / math.sqrt(self.rank))
        )

    def get_posterior(self, *args, **kwargs):
        """
        Returns a multivariate Laplace with ‘effective’ scale
        incorporating low-rank + diagonal structure.
        """
        base = self.scale_base
        # apply scale to factor → [latent_dim x rank]
        factor = self.cov_factor * base.unsqueeze(-1)
        # effective per-dimension scale: base * sqrt(1 + sum_j factor^2_ij)
        eff_scale = base * (factor.pow(2).sum(-1) + 1).sqrt()
        return dist.Laplace(self.loc, eff_scale)

    def _loc_scale(self, *args, **kwargs):
        base = self.scale_base
        factor = self.cov_factor * base.unsqueeze(-1)
        eff_scale = base * (factor.pow(2).sum(-1) + 1).sqrt()
        return self.loc, eff_scale


class AutoLowRankMultivariateUniform(AutoContinuous):
    """
    Low-rank-plus-diagonal multivariate Uniform guide,
    parameterized by center (loc) ± half‐range.
    Usage::
        guide = AutoLowRankUniform(model, rank=10)
        svi = SVI(model, guide, ...)
    """
    range_constraint = constraints.positive

    def __init__(self, model, init_loc_fn=init_to_median, init_range=0.1, rank=None):
        if not isinstance(init_range, float) or not (init_range > 0):
            raise ValueError(f"Expected init_range > 0 but got {init_range}")
        if not (rank is None or isinstance(rank, int) and rank > 0):
            raise ValueError(f"Expected rank > 0 but got {rank}")
        self._init_range = init_range
        self.rank = rank
        super().__init__(model, init_loc_fn=init_loc_fn)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)
        # center
        self.loc = nn.Parameter(self._init_loc())
        if self.rank is None:
            self.rank = int(round(self.latent_dim**0.5))
        # base half‐range (diagonal)
        self.range_base = PyroParam(
            self.loc.new_full((self.latent_dim,), self._init_range),
            constraint=self.range_constraint
        )
        # low-rank factor
        self.cov_factor = nn.Parameter(
            self.loc.new_empty(self.latent_dim, self.rank)
                .normal_(0, 1 / math.sqrt(self.rank))
        )

    def get_posterior(self, *args, **kwargs):
        """
        Returns a multivariate Uniform(loc - Δ, loc + Δ)
        where Δ = base_range * sqrt(1 + sum_j (cov_factor * base_range)^2).
        """
        base = self.range_base
        factor = self.cov_factor * base.unsqueeze(-1)
        # effective half‐range
        half_range = base * (factor.pow(2).sum(-1) + 1).sqrt()
        #return dist.Uniform(self.loc - half_range, self.loc + half_range)
        return UniformReal(self.loc - half_range, self.loc + half_range)

    def _loc_scale(self, *args, **kwargs):
        base = self.range_base
        factor = self.cov_factor * base.unsqueeze(-1)
        half_range = base * (factor.pow(2).sum(-1) + 1).sqrt()
        return self.loc, half_range


class BayesShipsCNN(PyroModule):
    def __init__(
        self,
        num_classes=2,   # now 2 for Categorical
        device=torch.device("cuda"),
        activation='relu',
        prior_dist='gaussian',
        mu=0.0,
        b=1.0,
        prior_params=None
    ):
        super().__init__()
        self.device = device

        # Activation setup
        if isinstance(activation, str):
            act_map = {
                'relu': F.relu,
                'tanh': torch.tanh,
                'sigmoid': torch.sigmoid,
                'sin': torch.sin,
                'relu6': F.relu6,
                'leaky_relu': F.leaky_relu,
                'selu': F.selu,
                'actWG': self.actWG,
                'actRWG': self.actRWG,
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

        self.pool = nn.MaxPool2d(2, 2)

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
            #base = dist.Uniform(-self.prior_b, self.prior_b)
            base = UniformReal(-self.prior_b, self.prior_b)
        else:
            raise ValueError(f"Unsupported prior: {self.prior_dist}")
        return base.expand(shape).to_event(len(shape))

    def forward(self, x, y=None):
        x = self.activation_fn(self.conv1(x))
        x = self.pool(x)
        x = self.activation_fn(self.conv2(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        logits = self.fc1(x)  # shape [batch, 2]

        if y is not None:
            with pyro.plate("data", x.size(0)):
                pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits


def load_model(timestamp):
    config_path = os.path.join(search_dir, config_files[timestamp])
    guide_path = os.path.join(search_dir, guide_files[timestamp])
    model_path = os.path.join(search_dir, model_files[timestamp])
    param_path = os.path.join(search_dir, param_files[timestamp])

    print(f"Loading model with config_path: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    model = BayesShipsCNN(
        num_classes=num_classes,
        device=device,
        activation=config['activation'],
        prior_dist=config['prior'],
        mu=config['prior_params']['mu'],
        b=config['prior_params'],
        prior_params=config.get('prior_params', None)
    ).to(device)

    # Load the guide
    #guide = AutoDiagonalNormal(model)

    # Load the model state
    model.load_state_dict(torch.load(model_path))
    
    # Load the guide state
    #guide.load_state_dict(torch.load(guide_path))

    return model, param_path

class NewInjector:
    def __init__(self, trained_model, device, test_loader, num_samples, multivariate_flag=False):
        """
        Initializes SEU injector
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trained_model = trained_model.to(self.device)
        self.test_loader = test_loader
        self.trained_model.eval()
        self.num_samples = num_samples

        #self.guide = AutoDiagonalNormal(self.trained_model).to(self.device)
        if self.trained_model.prior_dist == 'gaussian':
            if multivariate_flag:
                self.guide = AutoGuideList(bayesian_model)

                # 1) conv1.weight
                self.guide.add(
                    AutoLowRankMultivariateNormal(
                        poutine.block(bayesian_model, expose=["conv1.weight"]),
                        rank=20,
                        init_scale=0.05,
                        #prefix="AutoGuideList.conv1.weight"
                    )
                )

                # 2) conv1.bias
                self.guide.add(
                    AutoLowRankMultivariateNormal(
                        poutine.block(bayesian_model, expose=["conv1.bias"]),
                        rank=5,
                        init_scale=0.05,
                        #prefix="AutoGuideList.conv1.bias"
                    )
                )

                # 3) conv2.weight
                self.guide.add(
                    AutoLowRankMultivariateNormal(
                        poutine.block(bayesian_model, expose=["conv2.weight"]),
                        rank=20,
                        init_scale=0.05,
                        #prefix="AutoGuideList.conv2.weight"
                    )
                )

                # 4) conv2.bias
                self.guide.add(
                    AutoLowRankMultivariateNormal(
                        poutine.block(bayesian_model, expose=["conv2.bias"]),
                        rank=5,
                        init_scale=0.05,
                        #prefix="AutoGuideList.conv2.bias"
                    )
                )

                # 5) fc1.weight
                self.guide.add(
                    AutoLowRankMultivariateNormal(
                        poutine.block(bayesian_model, expose=["fc1.weight"]),
                        rank=20,
                        init_scale=0.05,
                        #prefix="AutoGuideList.fc1.weight"
                    )
                )

                # 6) fc1.bias
                self.guide.add(
                    AutoLowRankMultivariateNormal(
                        poutine.block(bayesian_model, expose=["fc1.bias"]),
                        rank=5,
                        init_scale=0.05,
                        #prefix="AutoGuideList.fc1.bias"
                    )
                )
            else:
                self.guide = AutoNormal(self.trained_model, init_scale=0.05).to(self.device)
        elif self.trained_model.prior_dist == 'laplace':
            self.guide = AutoLaplace(self.trained_model, init_scale=0.05).to(self.device)
        elif self.trained_model.prior_dist == 'uniform':
            self.guide = AutoUniform(self.trained_model, init_scale=0.05).to(self.device)
        else:
            raise ValueError(f"Unsupported prior: {self.trained_model.prior_dist}")
        
        pyro.get_param_store().clear()
        pyro.get_param_store().set_state(torch.load(pyro_param_store_path, weights_only=False))

        initial_labels, initial_predictions, initial_logits, initial_probs = self.predict_data_probs(self.num_samples)
        self.initial_accuracy = self.return_accuracy(initial_labels, initial_predictions)
        self.initial_probs = np.array(initial_probs)

        print(f"Initial accuracy: {self.initial_accuracy}")

    def predict_data_probs(self, num_samples=10):
        all_labels = []
        all_predictions = []
        all_logits = []
        all_probs = []

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Evaluating"):
                images, labels = images.to(self.device), labels.to(self.device)
                logits_mc = torch.zeros(num_samples, images.size(0), self.trained_model.fc1.out_features).to(self.device)

                for i in range(num_samples):
                    guide_trace = pyro.poutine.trace(self.guide).get_trace(images)
                    replayed_model = pyro.poutine.replay(self.trained_model, trace=guide_trace)
                    logits = replayed_model(images)
                    logits_mc[i] = logits

                avg_logits = logits_mc.mean(dim=0)
                predictions = torch.argmax(avg_logits, dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_logits.extend(avg_logits.cpu().numpy())
                all_probs.extend(F.softmax(avg_logits, dim=1).cpu().numpy())

        return all_labels, all_predictions, all_logits, all_probs

    def return_accuracy(self, all_labels, all_predictions):
        cm = confusion_matrix(all_labels, all_predictions)
        return np.trace(cm) / np.sum(cm)

    def compute_softmax_difference(self, before_probs, after_logits, penalty=1.0):
        """
        before_probs: list or array, shape (N, C), all finite probabilities
        after_logits: list or array, shape (N, C), raw logits (may contain ±inf)
        penalty: float, the per‑example penalty to use if logits are nonfinite
        
        Returns the mean over N examples of either
        - max_i |before_probs[n,i] − after_probs[n,i]|,  if after_logits[n] is finite
        - penalty,                                    otherwise
        """
        before = np.asarray(before_probs, dtype=np.float32)
        after_logits = torch.from_numpy(np.asarray(after_logits, dtype=np.float32))
        N, C = after_logits.shape

        # 1) detect which rows of after_logits are all finite
        finite_mask = torch.isfinite(after_logits).all(dim=1).numpy()  # shape (N,)

        # 2) safe‑softmax only on the finite ones
        safe_after_probs = torch.zeros_like(after_logits)
        if finite_mask.any():
            good_logits = after_logits[finite_mask]
            # (you can optionally do the “stable” shift here)
            safe_after_probs[finite_mask] = F.softmax(good_logits, dim=1)
        safe_after_probs = safe_after_probs.numpy()

        # 3) compute per‑example diff, using the penalty where needed
        diffs = np.empty(N, dtype=np.float32)
        for n in range(N):
            if not finite_mask[n]:
                diffs[n] = penalty
            else:
                diffs[n] = np.max(np.abs(before[n] - safe_after_probs[n]))
        return diffs.mean()

    def compute_difference(self, original_val, modified_val):
        return abs(original_val - modified_val)
    
    def run_seu_multivariate(self, location_index, parameter_name, ll_module_index, bit_i, num_samples):
        assert parameter_name in ["loc", "scale"]
        
        pyro.get_param_store().set_state(torch.load(pyro_param_store_path, weights_only=False))

        #param_store_name_initial = f"{param_unique}.{parameter_name}.{layer}.{layer_module}"
        #TODO translate the param_store_name
        # turn things like AutoNormal.locs.conv1.weight into  AutoGuideList.0.loc
        # 0 is conv1.weight, 1 is conv1.bias, etc

        #if f"{layer}.{layer_module}" == "conv1.weight":
        #    layer_module_number = "0"
        #elif f"{layer}.{layer_module}" == "conv1.bias":
        #    layer_module_number = "1"
        #elif f"{layer}.{layer_module}" == "conv2.weight":
        #    layer_module_number = "2"
        #elif f"{layer}.{layer_module}" == "conv2.bias":
        #    layer_module_number = "3"
        #elif f"{layer}.{layer_module}" == "fc1.weight":
        #    layer_module_number = "4"
        #elif f"{layer}.{layer_module}" == "fc1.bias":
        #    layer_module_number = "5"

        #print(f"{layer}.{layer_module}")

        param_store_name = f"AutoGuideList.{ll_module_index}.{parameter_name}"

        with torch.no_grad():
            param = pyro.get_param_store().get_param(param_store_name)
            new_param = param.clone()
            new_param = new_param.view(-1) #flatten new param
            original_val = new_param[location_index].cpu().item()
            seu_val = bitflip_float32(original_val, bit_i)
            abs_diff = self.compute_difference(original_val, seu_val)
            new_param[location_index] = seu_val
            # return new_param to original shape
            new_param = new_param.view(param.shape)
            pyro.get_param_store().__setitem__(param_store_name, new_param)

            print(f"Original value: {original_val}, SEU value: {seu_val}, Abs difference: {abs_diff}")

        guide = AutoGuideList(bayesian_model)

        # 1) conv1.weight
        guide.add(
            AutoLowRankMultivariateNormal(
                poutine.block(bayesian_model, expose=["conv1.weight"]),
                rank=20,
                init_scale=0.05,
                #prefix="AutoGuideList.conv1.weight"
            )
        )

        # 2) conv1.bias
        guide.add(
            AutoLowRankMultivariateNormal(
                poutine.block(bayesian_model, expose=["conv1.bias"]),
                rank=5,
                init_scale=0.05,
                #prefix="AutoGuideList.conv1.bias"
            )
        )

        # 3) conv2.weight
        guide.add(
            AutoLowRankMultivariateNormal(
                poutine.block(bayesian_model, expose=["conv2.weight"]),
                rank=20,
                init_scale=0.05,
                #prefix="AutoGuideList.conv2.weight"
            )
        )

        # 4) conv2.bias
        guide.add(
            AutoLowRankMultivariateNormal(
                poutine.block(bayesian_model, expose=["conv2.bias"]),
                rank=5,
                init_scale=0.05,
                #prefix="AutoGuideList.conv2.bias"
            )
        )

        # 5) fc1.weight
        guide.add(
            AutoLowRankMultivariateNormal(
                poutine.block(bayesian_model, expose=["fc1.weight"]),
                rank=20,
                init_scale=0.05,
                #prefix="AutoGuideList.fc1.weight"
            )
        )

        # 6) fc1.bias
        guide.add(
            AutoLowRankMultivariateNormal(
                poutine.block(bayesian_model, expose=["fc1.bias"]),
                rank=5,
                init_scale=0.05,
                #prefix="AutoGuideList.fc1.bias"
            )
        )

        try:
            after_labels, after_predictions, after_logits, after_probs = self.predict_data_probs(num_samples)
            accuracy_after = self.return_accuracy(after_labels, after_predictions)
            softmax_diff = self.compute_softmax_difference(self.initial_probs, after_probs)
        except:
            print("Error during prediction after SEU.")
            accuracy_after = np.nan
            softmax_diff = np.nan

        print(f"Accuracy after SEU: {accuracy_after}")
        print("===================================")

        return {
            "accuracy_change": accuracy_after - self.initial_accuracy,
            "softmax_difference": softmax_diff,
            "absolute_difference": abs_diff
        }
    
    def run_seu_old(self, location_index, param_unique, parameter_name, layer, layer_module, bit_i, num_samples):
        assert parameter_name in ["locs", "scales", "lows", "widths"], "Parameter name must be 'locs' or 'scales'."
        assert bit_i in range(0, 33), "Bit index must be between 0 and 32."

        param_store_name = f"{param_unique}.{parameter_name}.{layer}.{layer_module}"
        pyro.get_param_store().set_state(torch.load(pyro_param_store_path, weights_only=False))

        with torch.no_grad():
            param = pyro.get_param_store().get_param(param_store_name)
            new_param = param.clone()
            new_param = new_param.view(-1) #flatten new param
            original_val = new_param[location_index].cpu().item()
            seu_val = bitflip_float32(original_val, bit_i)
            abs_diff = self.compute_difference(original_val, seu_val)
            new_param[location_index] = seu_val
            # return new_param to original shape
            new_param = new_param.view(param.shape)
            pyro.get_param_store().__setitem__(param_store_name, new_param)

            print(f"Original value: {original_val}, SEU value: {seu_val}, Abs difference: {abs_diff}")


        if param_unique == "AutoNormal":
            self.guide = AutoNormal(self.trained_model, init_scale=0.05).to(self.device)
        elif param_unique == "AutoLaplace":
            self.guide = AutoLaplace(self.trained_model, init_scale=0.05).to(self.device)
        elif param_unique == "AutoUniform":
            self.guide = AutoUniform(self.trained_model, init_scale=0.05).to(self.device)
        else:
            raise ValueError(f"Unsupported parameter unique: {param_unique}")

        #after_labels, after_predictions, after_logits, after_probs = self.predict_data_probs(num_samples)
        #accuracy_after = self.return_accuracy(after_labels, after_predictions)
        #softmax_diff = self.compute_softmax_difference(self.initial_probs, after_probs)

        try:
            after_labels, after_predictions, after_logits, after_probs = self.predict_data_probs(num_samples)
            accuracy_after = self.return_accuracy(after_labels, after_predictions)
            softmax_diff = self.compute_softmax_difference(self.initial_probs, after_probs)
        except:
            print("Error during prediction after SEU.")
            accuracy_after = np.nan
            softmax_diff = np.nan

        print(f"Accuracy after SEU: {accuracy_after}")
        print("===================================")

        return {
            "accuracy_change": accuracy_after - self.initial_accuracy,
            "softmax_difference": softmax_diff,
            "absolute_difference": abs_diff
        }


    #AutoNormal.locs.conv1.weight
    def run_seu(
        self,
        location_index: int,
        param_unique: str,
        parameter_name: str,
        layer: str,
        layer_module: str,
        bit_i: int,
        num_samples: int,
    ):
        assert parameter_name in ["locs", "scales", "lows", "widths"], \
            "Parameter name must be one of 'locs', 'scales', 'lows', or 'widths'."
        assert 0 <= bit_i < 32, "Bit index must be between 0 and 31."

        # Construct the Pyro ParamStore key for this tensor
        param_store_name = f"{param_unique}.{parameter_name}.{layer}.{layer_module}"

        # insert remarks if any width adjustment (in uniform) or anything is done
        remarks = ""

        # Reload the saved ParamStore so we start from your trained guide
        pyro.clear_param_store()

        pyro.get_param_store().set_state(
            torch.load(pyro_param_store_path, weights_only=False)
        )

        with torch.no_grad():
            # 1) Grab the original tensor and flatten it
            param = pyro.get_param_store().get_param(param_store_name)
            flat  = param.clone().view(-1)

            # 2) Extract, flip one bit, and wrap back as a tensor
            orig_val = flat[location_index].cpu().item()
            flipped  = bitflip_float32(orig_val, bit_i)      # Python float

            # if the parameter is 'widths' or 'scales' and the flipped value is negative, we skip
            if parameter_name in ["widths", "scales"] and flipped < 0:
                accuracy_after = np.nan
                softmax_diff = np.nan
                abs_diff = np.nan
                remarks += "SEU width or scale was negative; "
                print(
                    f"Skipping SEU at '{param_store_name}[{location_index}]': "
                    f"original value {orig_val:.3e}, flipped value {flipped:.3e}."
                )
                return {
                    "accuracy_change":    accuracy_after - self.initial_accuracy,
                    "softmax_difference": softmax_diff,
                    "absolute_difference": abs_diff,
                    "remarks": remarks
                }

            # Check for NaN or Inf and clip if necessary
            if np.isnan(flipped) or np.isinf(flipped):
                remarks += "SEU went to NaN or Inf clipping to finite range; "
                print(f"Available clip values range is '{-np.finfo(np.float32).max} {np.finfo(np.float32).max}'")
                #flipped = np.clip(flipped, -np.finfo(np.float32).max, np.finfo(np.float32).max)
                flipped = np.nan_to_num(
                    flipped,
                    nan= np.finfo(np.float32).max if orig_val >= 0 else -np.finfo(np.float32).max,
                    posinf=np.finfo(np.float32).max,
                    neginf=-np.finfo(np.float32).max,
                )
                print(
                    f"Warning: Flipped value at '{param_store_name}[{location_index}]' is NaN or Inf! "
                    f"Original value was {orig_val:.3e}, "
                    f"flipped value is clipped to {flipped:.3e}. Clipping to finite range."
                )

            seu_val  = torch.tensor(
                flipped, dtype=param.dtype, device=param.device
            )
            abs_diff = self.compute_difference(orig_val, flipped)

            # 3) Write the flipped value into the ParamStore
            flat[location_index] = seu_val
            pyro.get_param_store().__setitem__(
                param_store_name, flat.view(param.shape)
            )

            # 4) If this is an AutoUniform guide, enforce
            #    width >= nextafter(low) – low so Uniform(low, low+width) stays valid.
            if param_unique == "AutoUniform":
                
                # (a) get the current lows tensor (after any flip)
                low_name = f"{param_unique}.lows.{layer}.{layer_module}"
                low_param = pyro.get_param_store() \
                                .get_param(low_name) \
                                .view(-1)

                # determine the 'low' value we should use at this index:
                if parameter_name == "lows":
                    low_val = seu_val
                else:  # we just flipped a width, so low stays original
                    low_val = low_param[location_index]

                # check and print if low_val is nan
                if torch.isnan(low_val):
                    print(
                        f"Warning: low value at '{low_name}[{location_index}]' is NaN! "
                        f"Original value was {orig_val:.3e}, flipped value is {flipped:.3e}."
                    )

                # compute the smallest positive increment (ULP) at low_val
                delta = (
                    torch.nextafter(
                        low_val,
                        torch.tensor(float("inf"), dtype=low_val.dtype, device=low_val.device),
                    )
                    - low_val
                )

                # (b) now clamp the corresponding width
                width_name  = f"{param_unique}.widths.{layer}.{layer_module}"
                width_param = pyro.get_param_store().get_param(width_name)
                wflat       = width_param.clone().view(-1)

                # original width at that index (after any SEU if param was 'widths')
                orig_width = wflat[location_index].cpu().item()
                # check whether the orig_width is nan
                if np.isnan(orig_width):
                    print(
                        f"Warning: original width at '{width_name}[{location_index}]' is NaN! "
                        f"Original width is {orig_width:.3e}"
                    )
                # enforce the minimum
                new_width  = torch.max(wflat[location_index], delta)
                wflat[location_index] = new_width

                # check whether the new width is nan
                if torch.isnan(new_width):
                    print(
                        f"Warning: new width at '{width_name}[{location_index}]' is NaN! "
                        f"Original width was {orig_width:.3e}, delta is {delta:.3e}."
                    )

                # write back the clamped widths
                pyro.get_param_store().__setitem__(
                    width_name, wflat.view(width_param.shape)
                )

                if orig_width != new_width:
                    remarks += f"AutoUniform width adjusted; "
                    print(
                        f"Adjusted width at '{width_name}[{location_index}]': "
                        f"{orig_width:.3e} → {new_width:.3e}"
                    )
                else:
                    pass

                if parameter_name == "widths" and seu_val < 0:
                    accuracy_after = np.nan
                    softmax_diff = np.nan
                    abs_diff = np.nan
                    remarks += "SEU width was negative; "
                    # continue or break to skip this?
                    return {
                        "accuracy_change":    accuracy_after - self.initial_accuracy,
                        "softmax_difference": softmax_diff,
                        "absolute_difference": abs_diff,
                        "remarks": remarks
                    }
                    

            # 5) Report the flip
            print(
                f"Parameter '{param_store_name}[{location_index}]': "
                f"{orig_val:.6g} → {flipped:.6g}  "
                f"(abs diff {abs_diff:.3g})"
            )

        # 6) Re‑instantiate your guide so Predictive will pick up the perturbed params
        if param_unique == "AutoNormal":
            self.guide = AutoNormal(self.trained_model, init_scale=0.05).to(self.device)
        elif param_unique == "AutoLaplace":
            self.guide = AutoLaplace(self.trained_model, init_scale=0.05).to(self.device)
        elif param_unique == "AutoUniform":
            self.guide = AutoUniform(self.trained_model, init_scale=0.05).to(self.device)
        else:
            raise ValueError(f"Unsupported guide type: {param_unique}")

        # 7) Re‑run inference & evaluation
        after_labels, after_preds, after_logits, after_probs = self.predict_data_probs(num_samples)
        accuracy_after = self.return_accuracy(after_labels, after_preds)
        softmax_diff  = self.compute_softmax_difference(self.initial_probs, after_probs)

        print(f"Accuracy after SEU: {accuracy_after:.3%}")
        print("===================================")

        return {
            "accuracy_change":    accuracy_after - self.initial_accuracy,
            "softmax_difference": softmax_diff,
            "absolute_difference": abs_diff,
            "remarks": remarks
        }


    def run_seu_autodiagonal_normal_multi(self, location_indices, bit_i, parameter_name="loc",
                                          attack_ratio=1.0, num_samples=10, seed=None):
        assert parameter_name in ["loc", "scale"], "Parameter name must be 'loc' or 'scale'."
        assert bit_i in range(0, 33), "Bit index must be between 0 and 32."
        assert 0.0 <= attack_ratio <= 1.0, "Attack ratio must be between 0.0 and 1.0."

        if isinstance(location_indices, int):
            location_indices = [location_indices]

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        num_attacks = max(1, int(len(location_indices) * attack_ratio))
        attack_locations = np.random.choice(location_indices, size=num_attacks, replace=False)
        param_store_name = f"AutoDiagonalNormal.{parameter_name}"
        pyro.get_param_store().set_state(torch.load(pyro_param_store_path, weights_only=False))

        abs_differences = []

        with torch.no_grad():
            param = pyro.get_param_store().get_param(param_store_name)
            new_param = param.clone()

            #print(f"Attacking {num_attacks} out of {len(location_indices)} locations:")

            for location_index in attack_locations:
                original_val = new_param[location_index].cpu().item()
                seu_val = bitflip_float32(original_val, bit_i)
                abs_diff = self.compute_difference(original_val, seu_val)
                abs_differences.append(abs_diff)
                new_param[location_index] = seu_val
                print(f"  Location {location_index}: {original_val} -> {seu_val}, Log diff: {abs_diff}")

            pyro.get_param_store().__setitem__(param_store_name, new_param)

        self.guide = AutoDiagonalNormal(self.trained_model).to(self.device)

        try:
            after_labels, after_predictions, after_logits, after_probs = self.predict_data_probs(num_samples)
            accuracy_after = self.return_accuracy(after_labels, after_predictions)
            softmax_diff = self.compute_softmax_difference(self.initial_probs, after_probs)
            mean_abs_diff = float(np.mean(abs_differences))
        except:
            accuracy_after = np.nan
            softmax_diff = np.nan
            mean_abs_diff = np.nan

        #print(f"Accuracy after SEU: {accuracy_after}")
        #print("===================================")

        return {
            "accuracy_change": accuracy_after - self.initial_accuracy,
            "softmax_difference": softmax_diff,
            "mean_abs_difference": mean_abs_diff
        }

def load_model_config(timestamp):
    config_path = os.path.join(search_dir, config_files[timestamp])

    with open(config_path, 'r') as f:
        model_config = json.load(f)

    return model_config

if __name__ == "__main__":
    train_loader, test_loader = load_data(batch_size=16)
    device = torch.device("cuda")
    num_classes = 2
    

    search_dir = args.search_dir
    #list all .json files in the directory
    all_files = [f for f in os.listdir(search_dir)]
    json_files = [f for f in os.listdir(search_dir) if f.endswith('.json')]

    # excluding the format, get the last 16 characters of each filename
    timestamps = [f[:-5][-16:] for f in json_files]
    print("Timestamps found count:", len(timestamps))

    # remove some timestamps that are not needed
    # those are the ones that are not in the shipsnet_seu_result directory, without the .csv extension
    excluded_timestamps = [f[:-4][-16:] for f in [f for f in os.listdir(args.save_dir) if f.endswith('.csv')]]
    
    timestamps = [ts for ts in timestamps if ts not in excluded_timestamps]
    #timestamps = timestamps[:1]
    print("After excluding, timestamps count:", len(timestamps))

    # for each timestamp, look for every other files in the directory that contains the timestamp

    config_files = {}
    guide_files = {}
    model_files = {}
    param_files = {}

    for timestamp in timestamps:
        config_files[timestamp] = [f for f in all_files if timestamp in f and f.endswith('.json')][0]
        guide_files[timestamp] = [f for f in all_files if timestamp in f and f.startswith('guide')][0]
        model_files[timestamp] = [f for f in all_files if timestamp in f and f.startswith('model')][0]
        param_files[timestamp] = [f for f in all_files if timestamp in f and f.startswith('param')][0]

    # create a list that maps timestamps and the output of load_model_config['prior']
    prior_list = []
    for ts in timestamps:
        model_config = load_model_config(ts)
        prior_list.append((ts, model_config['prior']))

    # remove timestamps which prior is not stated in args.prior
    # translate what args.prior specified
    prior_map = {
        'Gaussian_prior': 'gaussian',
        'Laplace_prior': 'laplace',
        'Uniform_prior': 'uniform'
    }
    timestamps = [ts for ts, prior in prior_list if prior == prior_map[args.prior]]
    print(f"After filtering by prior '{args.prior}', timestamps count: {len(timestamps)}")

    #check the prior_params b, if each timestamps, if it is not 1.0, remove the timestamp from the list
    if args.limited_mode:
        timestamps = [ts for ts in timestamps if load_model_config(ts)['prior_params']['b'] == 1.0]
        print(f"After filtering by prior_params['b'] == 1.0, timestamps count: {len(timestamps)}")

    experiment_iteration = 0

    for ts_idx in range(len(timestamps)):
        # clear pyro's param store
        pyro.clear_param_store()
        experiment_iteration += 1
        # send a telegram message to show progress experiment_iteration per total timestamps
        send_telegram_message(            title="ShipsNet SEU Experiment Progress",
            message=f"Running experiment {experiment_iteration}/{len(timestamps)} for timestamp {timestamps[ts_idx]}. Save directory: {args.save_dir}. Prior: {args.prior}, Limited mode: {args.limited_mode}"
        )
        
        bayesian_model, pyro_param_store_path = load_model(timestamps[ts_idx])


        if args.multivariate_guide:
            newinj = NewInjector(trained_model=bayesian_model, device=device, test_loader=test_loader, num_samples=10, multivariate_flag=True)
        else:
            newinj = NewInjector(trained_model=bayesian_model, device=device, test_loader=test_loader, num_samples=10)
   
        model_config = load_model_config(timestamps[ts_idx])        

        ## MAIN LOOP CODE

        results_df = pd.DataFrame(columns=["activation_fn",
                                        "prior",
                                        "best_accuracy",
                                        "prior_mu",
                                        "prior_b",
                                        "param_type",
                                        "location_index",
                                        "location_layer",
                                        "location_module",
                                        "bit_index", 
                                        "initial_accuracy", 
                                        "accuracy_after_seu", 
                                        "accuracy_change", 
                                        "softmax_difference", 
                                        "mean_abs_difference",
                                        "remarks"
                                        ])

        # get the initial accuracy from the newinj object
        initial_accuracy = newinj.initial_accuracy

        attack_locations = ["beginning", "end"]

        if args.multivariate_guide:
            layer_list = ["conv1", "conv2", "fc1"]  # List of layers to attack
        else:
            layer_list = []
            
            for name, value in pyro.get_param_store().items():
                layer_list.append(name.split('.')[2])  # Extract the layer name from the param store key
                layer_list = list(dict.fromkeys(layer_list))
        #bit_i = 2

        for attack_location in attack_locations:
            for layer_list_iter in layer_list:
                for module_iter in ["weight", "bias"]:
                
                    if attack_location == "beginning":
                        target_index = 0
                    elif attack_location == "end":
                        target_index = -1

                    #print(f"Running SEU on {layer}.{param_name} at index {target_index} with bit flip 0")

                    #for bit_iter in [1]:
                    #for bit_iter in [0, 1, 3, 6, 10, 15, 21]:
                    for bit_iter in [0, 1, 3, 10]:
                    #for bit_iter in [6, 15, 21]:
                        if model_config['prior'] == 'uniform':
                            parameter_name_list = ["lows", "widths"]
                            param_unique_target = "AutoUniform"
                        elif model_config['prior'] == 'laplace':
                            parameter_name_list = ["locs", "scales"]
                            param_unique_target = "AutoLaplace"
                        elif model_config['prior'] == 'gaussian':
                            if args.multivariate_guide:
                                parameter_name_list = ["loc", "scale"]
                                param_unique_target = "AutoGuideList"
                            else:
                                parameter_name_list = ["locs", "scales"]
                                param_unique_target = "AutoNormal"

                        for parameter_name in parameter_name_list :
                            print(f"Running SEU for {layer_list_iter}.{module_iter} bit index {bit_iter}  for parameter {parameter_name}")
                            #result = newinj.run_seu_autodiagonal_normal(location_index=target_index, bit_i=bit_iter, parameter_name=parameter_name, num_samples=10)
                            if args.multivariate_guide:
                                #location_index, parameter_name, ll_module_index, bit_i, num_samples
                                #turn layer_list_iter, module_iter, into ll_module_index
                                if f"{layer_list_iter}.{module_iter}" == "conv1.weight":
                                    ll_module_index = "0"
                                elif f"{layer_list_iter}.{module_iter}" == "conv1.bias":
                                    ll_module_index = "1"
                                elif f"{layer_list_iter}.{module_iter}" == "conv2.weight":
                                    ll_module_index = "2"
                                elif f"{layer_list_iter}.{module_iter}" == "conv2.bias":
                                    ll_module_index = "3"
                                elif f"{layer_list_iter}.{module_iter}" == "fc1.weight":
                                    ll_module_index = "4"
                                elif f"{layer_list_iter}.{module_iter}" == "fc1.bias":
                                    ll_module_index = "5"
                                
                                print(f"{layer_list_iter}.{module_iter}")

                                #run_seu_multivariate(location_index, parameter_name, ll_module_index, bit_i, num_samples)

                                result = newinj.run_seu_multivariate(target_index, parameter_name, ll_module_index, bit_iter, num_samples=10)
                            else:
                                result = newinj.run_seu(target_index, param_unique_target, parameter_name, layer_list_iter, module_iter, bit_iter, num_samples=10)
                            #run_seu(self, location_index, param_unique, parameter_name, layer, layer_module, bit_i, num_samples)
                            # use concat to save the result to a dataframe
                            #print(initial_accuracy)
                            iter_df = pd.DataFrame({
                                "activation_fn": model_config['activation'],
                                "prior": model_config['prior'],
                                "best_accuracy": model_config['best_accuracy'],
                                "prior_mu": model_config['prior_params']['mu'],
                                "prior_b": model_config['prior_params']['b'],
                                "param_type": parameter_name,
                                "location_index": target_index,
                                "location_layer": layer_list_iter,
                                "location_module": module_iter,
                                "bit_index": bit_iter,
                                "initial_accuracy": initial_accuracy,
                                "accuracy_after_seu": initial_accuracy + result["accuracy_change"],
                                "accuracy_change": result["accuracy_change"],
                                "softmax_difference": result["softmax_difference"],
                                "mean_abs_difference": result["absolute_difference"],
                                "remarks": result["remarks"]
                            }, index=[0])

                            results_df = pd.concat([results_df, iter_df], ignore_index=True)


        results_df.to_csv(os.path.join(args.save_dir,f'{timestamps[ts_idx]}.csv'), index=False)
        print(f"Results saved for timestamp {timestamps[ts_idx]}")

        send_telegram_message(title="ShipsNet SEU Experiment",
            message=f"Finished experiment {experiment_iteration}/{len(timestamps)}. Save directory: {args.save_dir}. Timestamp: {timestamps[ts_idx]}. Prior: {model_config['prior']}, Limited mode: {args.limited_mode}")