import torch
import torch.nn as nn

class WeightedGaussianActivation(nn.Module):
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
        return self.weight * torch.exp(-((x - self.mu)**2) / (2 * self.sigma**2))
    
class WeightedGaussian(nn.Module):
    def forward(self, x):
        return x * torch.exp(-x**2)
    
if __name__ == "__main__":
    # if imported as a module, this block will not run
    pass

import torch.nn as nn

