import torch
from torch import nn
#from torch.distributions import constraints, transforms, Laplace
from pyro.nn import PyroParam
from pyro.infer.autoguide import AutoContinuous
from pyro.infer.autoguide.initialization import init_to_median

import torchvision.transforms as transforms
import pyro


class AutoDiagonalLaplace(AutoContinuous):
    """
    This implementation uses a Laplace distribution with diagonal scale to construct
    a guide over the entire latent space, approximating a Laplace posterior.

    Usage::

        guide = AutoDiagonalLaplace(model)
        svi = SVI(model, guide, ...)

    By default, the mean vector is initialized to zero and the scale (b parameter) 
    is initialized to a small positive value.

    :param callable model: A generative model.
    :param callable init_loc_fn: A per-site initialization function.
    :param float init_scale: Initial scale for the standard deviation of each
        latent variable (b parameter of Laplace).
    """

    scale_constraint = torch.distributions.constraints.positive  # scale b > 0 for Laplace

    def __init__(self, model, init_loc_fn=init_to_median, init_scale=0.1):
        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        self._init_scale = init_scale
        super().__init__(model, init_loc_fn=init_loc_fn)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)
        self.loc = nn.Parameter(self._init_loc())
        self.scale = PyroParam(
            self.loc.new_full((self.latent_dim,), self._init_scale),
            self.scale_constraint,
        )

    def get_base_dist(self):
        # Base distribution is standard Laplace centered at 0 with scale=1
        return pyro.distributions.Laplace(
            torch.zeros_like(self.loc), 
            torch.ones_like(self.loc)
        ).to_event(1)

    def get_transform(self, *args, **kwargs):
        # Affine transform: x -> loc + scale * x
        # Laplace is closed under affine transforms.
        return transforms.AffineTransform(self.loc, self.scale)

    def get_posterior(self, *args, **kwargs):
        """
        Returns a diagonal Laplace posterior distribution.
        """
        return pyro.distributions.Laplace(self.loc, self.scale).to_event(1)

    def _loc_scale(self, *args, **kwargs):
        return self.loc, self.scale

if __name__ == "__main__":
    # if imported as a module, this block will not run
    pass