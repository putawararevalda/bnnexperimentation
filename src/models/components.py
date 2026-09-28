import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro.distributions as dist
from pyro.distributions import constraints


class WeightedGaussian(nn.Module):
    """WG activation: x * exp(-x^2)"""
    def forward(self, x):
        return x * torch.exp(-x ** 2)


class WeightedGaussianActivation(nn.Module):
    """Learnable WG activation with mu, sigma, weight parameters."""
    def __init__(self, mu=0.0, sigma=1.0, weight=1.0, learnable=False):
        super().__init__()
        if learnable:
            self.mu = nn.Parameter(torch.tensor(mu))
            self.sigma = nn.Parameter(torch.tensor(sigma))
            self.weight = nn.Parameter(torch.tensor(weight))
        else:
            self.register_buffer('mu', torch.tensor(mu))
            self.register_buffer('sigma', torch.tensor(sigma))
            self.register_buffer('weight', torch.tensor(weight))

    def forward(self, x):
        return self.weight * torch.exp(-((x - self.mu) ** 2) / (2 * self.sigma ** 2))


class SmartPool(nn.Module):
    """
    Max-pool that replaces spike values (> threshold) with the 2nd-largest
    value in each window, preventing SEU-induced outliers from propagating.
    Inspired by Santos et al. (2019).
    """
    def __init__(self, kernel_size: int = 2, stride: int = 2,
                 threshold: float = 10.0, detect_only: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.threshold = threshold
        self.detect_only = detect_only

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        ks = self.kernel_size

        patches = F.unfold(x, kernel_size=ks, stride=self.stride)  # (N, C*ks*ks, L)
        patches = patches.view(N, C, ks * ks, -1)                   # (N, C, ks*ks, L)

        top2_vals, _ = torch.topk(patches, 2, dim=2)
        max1 = top2_vals[:, :, 0, :]
        max2 = top2_vals[:, :, 1, :]

        spikes = max1 > self.threshold
        out = max1 if self.detect_only else torch.where(spikes, max2, max1)

        H_out, W_out = H // ks, W // ks
        out = out.view(N, C, H_out, W_out)
        return out


class UniformReal(dist.Uniform):
    """Uniform distribution with unconstrained support for use as a Pyro prior."""
    @property
    def support(self):
        return constraints.real
