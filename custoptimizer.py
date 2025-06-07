import torch
from torch.optim.optimizer import Optimizer
import math

class SGLD(Optimizer):
    def __init__(self, params, lr=1e-2, weight_decay=0.0, addnoise=True):
        defaults = dict(lr=lr, weight_decay=weight_decay, addnoise=addnoise)
        super(SGLD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            addnoise = group['addnoise']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                if weight_decay != 0:
                    grad = grad + weight_decay * p

                noise = torch.randn_like(p) * math.sqrt(lr) if addnoise else 0.0
                p.add_( -0.5 * lr * grad + noise )