import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

from src.models.components import SmartPool, UniformReal


class BayesShipsCNN(PyroModule):
    """
    Bayesian CNN for ShipsNet binary classification (ship / no-ship).
    Architecture: conv1(3→32,k=3) → pool → conv2(32→64,k=3) → pool → fc1(64*16*16→num_classes)
    All weights and biases are Pyro random variables drawn from the chosen prior.

    Args:
        num_classes: Number of output classes (default 2).
        device: Torch device.
        activation: Activation function name string or callable.
                    Supported strings: relu, tanh, sigmoid, sinusoidal, relu6, wg, rwg.
        prior_dist: Prior distribution family — 'gaussian', 'laplace', or 'uniform'.
        mu: Prior center (mean/loc) parameter.
        b: Prior spread (std/scale/half-width) parameter.
        prior_params: Optional dict {'mu': ..., 'b': ...} overriding mu and b.
        smartpool_switch: Replace MaxPool with SmartPool (SEU-resilient pooling).
        pool_threshold: SmartPool spike threshold (default 10.0).
        pool_detect_only: If True, SmartPool detects spikes but does not replace them.
        dropout_switch: Add Dropout(p=dropout_p) after conv2 during training.
        dropout_p: Dropout probability (default 0.5).
    """

    def __init__(
        self,
        num_classes: int = 2,
        device=torch.device("cuda"),
        activation: str = 'relu',
        prior_dist: str = 'gaussian',
        mu: float = 0.0,
        b: float = 1.0,
        prior_params=None,
        smartpool_switch: bool = False,
        pool_threshold: float = 10.0,
        pool_detect_only: bool = False,
        dropout_switch: bool = False,
        dropout_p: float = 0.5,
    ):
        super().__init__()
        self.device = device
        self.prior_dist = prior_dist
        self.dropout_switch = dropout_switch
        # Likelihood scale for minibatch ELBO: set to N/batch_size so the data term
        # is weighted to the full dataset. Default 1.0 preserves legacy behavior.
        self.obs_scale = 1.0

        # Activation
        if isinstance(activation, str):
            act_map = {
                'relu': F.relu,
                'tanh': torch.tanh,
                'sigmoid': torch.sigmoid,
                'sinusoidal': torch.sin,
                'relu6': F.relu6,
                'wg': self._actWG,
                'rwg': self._actRWG,
                # Legacy aliases: EuroSAT configs store the activation function's
                # __name__ (e.g. from model.activation_fn.__name__), not the key.
                'sin': torch.sin,
                '_actWG': self._actWG,
                '_actRWG': self._actRWG,
            }
            if activation not in act_map:
                raise ValueError(f"Unsupported activation '{activation}'. Choose from: {list(act_map)}")
            self.activation_fn = act_map[activation]
        elif callable(activation):
            self.activation_fn = activation
        else:
            raise ValueError("activation must be a string or callable")

        # Prior parameters
        params = {'mu': mu, 'b': b} if prior_params is None else prior_params
        self.prior_mu = torch.tensor(params['mu'], device=device, dtype=torch.float32)
        self.prior_b = torch.tensor(params['b'], device=device, dtype=torch.float32)

        print(f"[BayesShipsCNN] prior={prior_dist}, mu={self.prior_mu.item()}, b={self.prior_b.item()}, "
              f"activation={activation}, smartpool={smartpool_switch}, dropout={dropout_switch}")

        # Layers
        self.conv1 = PyroModule[nn.Conv2d](3, 32, kernel_size=3, padding=1)
        self.conv1.weight = PyroSample(self._make_prior([32, 3, 3, 3]))
        self.conv1.bias = PyroSample(self._make_prior([32]))

        self.conv2 = PyroModule[nn.Conv2d](32, 64, kernel_size=3, padding=1)
        self.conv2.weight = PyroSample(self._make_prior([64, 32, 3, 3]))
        self.conv2.bias = PyroSample(self._make_prior([64]))

        self.pool = (
            SmartPool(kernel_size=2, stride=2, threshold=pool_threshold, detect_only=pool_detect_only)
            if smartpool_switch
            else nn.MaxPool2d(kernel_size=2, stride=2)
        )

        if dropout_switch:
            self.dropout = nn.Dropout(p=dropout_p)

        self.fc1 = PyroModule[nn.Linear](64 * 16 * 16, num_classes)
        self.fc1.weight = PyroSample(self._make_prior([num_classes, 64 * 16 * 16]))
        self.fc1.bias = PyroSample(self._make_prior([num_classes]))

    def _actWG(self, x, alpha=1.0):
        return x * torch.exp(-alpha * x ** 2)

    def _actRWG(self, x, alpha=1.0):
        wg = x * torch.exp(-alpha * x ** 2)
        return torch.max(torch.zeros_like(wg), wg)

    def _make_prior(self, shape):
        if self.prior_dist == 'gaussian':
            base = dist.Normal(self.prior_mu, self.prior_b)
        elif self.prior_dist == 'laplace':
            base = dist.Laplace(self.prior_mu, self.prior_b)
        elif self.prior_dist == 'uniform':
            base = UniformReal(-self.prior_b, self.prior_b)
        else:
            raise ValueError(f"Unsupported prior '{self.prior_dist}'. Choose: gaussian, laplace, uniform")
        return base.expand(shape).to_event(len(shape))

    def forward(self, x, y=None):
        x = self.activation_fn(self.conv1(x))
        x = self.pool(x)
        x = self.activation_fn(self.conv2(x))
        x = self.pool(x)

        if self.dropout_switch and self.training:
            x = self.dropout(x)

        x = x.view(x.size(0), -1)
        logits = self.fc1(x)

        if y is not None:
            with pyro.poutine.scale(scale=self.obs_scale):
                with pyro.plate("data", x.size(0)):
                    pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits
