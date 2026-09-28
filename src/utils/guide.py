"""
Custom Pyro variational guides for Bayesian inference.

AutoLaplace  — diagonal Laplace posterior (one Laplace per latent variable)
AutoUniform  — diagonal Uniform posterior (one Uniform per latent variable)

Both are drop-in replacements for AutoNormal and follow the same interface.
"""
from contextlib import ExitStack

import torch
import pyro
import pyro.distributions as dist
from pyro.nn.module import PyroModule, PyroParam
from pyro.infer.autoguide import AutoGuide
from pyro.infer.autoguide.initialization import InitMessenger, init_to_feasible
from pyro.distributions import constraints
from pyro.distributions.util import sum_rightmost
from pyro.ops.tensor_utils import periodic_repeat
from pyro.distributions.transforms import biject_to
from pyro.infer.autoguide.utils import deep_setattr, deep_getattr, helpful_support_errors
import pyro.poutine as poutine


class AutoLaplace(AutoGuide):
    """
    Diagonal Laplace variational guide.
    Each latent variable is approximated by an independent Laplace(loc, scale).
    """
    scale_constraint = constraints.softplus_positive

    def __init__(self, model, *, init_loc_fn=init_to_feasible, init_scale=0.1, create_plates=None):
        if not isinstance(init_scale, float) or init_scale <= 0:
            raise ValueError(f"Expected init_scale > 0, got {init_scale}")
        self._init_scale = init_scale
        self.init_loc_fn = init_loc_fn
        model = InitMessenger(self.init_loc_fn)(model)
        super().__init__(model, create_plates=create_plates)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)
        self._event_dims = {}
        self.locs = PyroModule()
        self.scales = PyroModule()

        for name, site in self.prototype_trace.iter_stochastic_nodes():
            with helpful_support_errors(site):
                init_loc = biject_to(site["fn"].support).inv(site["value"].detach()).detach()
            event_dim = site["fn"].event_dim + init_loc.dim() - site["value"].dim()
            self._event_dims[name] = event_dim

            for frame in site["cond_indep_stack"]:
                full_size = frame.full_size or frame.size
                if full_size != frame.size:
                    init_loc = periodic_repeat(init_loc, full_size, frame.dim - event_dim).contiguous()

            deep_setattr(self.locs, name, PyroParam(init_loc, constraints.real, event_dim))
            deep_setattr(self.scales, name,
                         PyroParam(torch.full_like(init_loc, self._init_scale),
                                   self.scale_constraint, event_dim))

    def _get_loc_and_scale(self, name):
        return deep_getattr(self.locs, name), deep_getattr(self.scales, name)

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
                    dist.Laplace(site_loc, site_scale).to_event(self._event_dims[name]),
                    infer={"is_auxiliary": True},
                )
                value = transform(unconstrained)
                if poutine.get_mask() is False:
                    log_density = 0.0
                else:
                    log_density = sum_rightmost(
                        transform.inv.log_abs_det_jacobian(value, unconstrained),
                        transform.inv.log_abs_det_jacobian(value, unconstrained).dim()
                        - value.dim() + site["fn"].event_dim,
                    )
                result[name] = pyro.sample(name, dist.Delta(value, log_density=log_density,
                                                             event_dim=site["fn"].event_dim))
        return result

    @torch.no_grad()
    def median(self, *args, **kwargs):
        return {name: biject_to(site["fn"].support)(self._get_loc_and_scale(name)[0]).clone()
                for name, site in self.prototype_trace.iter_stochastic_nodes()}

    @torch.no_grad()
    def quantiles(self, quantiles, *args, **kwargs):
        results = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            loc, scale = self._get_loc_and_scale(name)
            qs = torch.tensor(quantiles, dtype=loc.dtype, device=loc.device)
            qs = qs.reshape((-1,) + (1,) * loc.dim())
            results[name] = biject_to(site["fn"].support)(dist.Laplace(loc, scale).icdf(qs))
        return results


class AutoUniform(AutoGuide):
    """
    Diagonal Uniform variational guide.
    Each latent variable is approximated by Uniform(low, low + width).
    """
    width_constraint = constraints.softplus_positive

    def __init__(self, model, *, init_loc_fn=init_to_feasible, init_scale=0.1, create_plates=None):
        if not isinstance(init_scale, float) or init_scale <= 0:
            raise ValueError(f"Expected init_scale > 0, got {init_scale}")
        self._init_scale = init_scale
        self.init_loc_fn = init_loc_fn
        model = InitMessenger(self.init_loc_fn)(model)
        super().__init__(model, create_plates=create_plates)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)
        self._event_dims = {}
        self.lows = PyroModule()
        self.widths = PyroModule()

        for name, site in self.prototype_trace.iter_stochastic_nodes():
            with helpful_support_errors(site):
                init_loc = biject_to(site["fn"].support).inv(site["value"].detach()).detach()
            event_dim = site["fn"].event_dim + init_loc.dim() - site["value"].dim()
            self._event_dims[name] = event_dim

            for frame in site["cond_indep_stack"]:
                full_size = frame.full_size or frame.size
                if full_size != frame.size:
                    init_loc = periodic_repeat(init_loc, full_size, frame.dim - event_dim).contiguous()

            deep_setattr(self.lows, name, PyroParam(init_loc, constraints.real, event_dim))
            deep_setattr(self.widths, name,
                         PyroParam(torch.full_like(init_loc, self._init_scale),
                                   self.width_constraint, event_dim))

    def _get_low_and_width(self, name):
        return deep_getattr(self.lows, name), deep_getattr(self.widths, name)

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
                unconstrained = pyro.sample(
                    f"{name}_unconstrained",
                    dist.Uniform(low, low + width).to_event(self._event_dims[name]),
                    infer={"is_auxiliary": True},
                )
                value = transform(unconstrained)
                if poutine.get_mask() is False:
                    log_density = 0.0
                else:
                    log_density = sum_rightmost(
                        transform.inv.log_abs_det_jacobian(value, unconstrained),
                        transform.inv.log_abs_det_jacobian(value, unconstrained).dim()
                        - value.dim() + site["fn"].event_dim,
                    )
                result[name] = pyro.sample(name, dist.Delta(value, log_density=log_density,
                                                             event_dim=site["fn"].event_dim))
        return result

    @torch.no_grad()
    def median(self, *args, **kwargs):
        return {name: biject_to(site["fn"].support)(
                    self._get_low_and_width(name)[0] + self._get_low_and_width(name)[1] / 2).clone()
                for name, site in self.prototype_trace.iter_stochastic_nodes()}
