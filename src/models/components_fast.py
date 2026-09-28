"""
Performance-optimised drop-in replacements for components in
`src.models.components`.

The originals stay untouched and remain the reference implementation; anything
here must produce numerically identical output. Opt in explicitly (see
`SmartPoolFast` below) rather than by import-shadowing, so a run's pooling
implementation is always an explicit choice.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmartPoolFast(nn.Module):
    """
    Numerically identical to `src.models.components.SmartPool`, but avoids the
    `F.unfold` + `topk` path that dominates SEU sweep runtime.

    The original materialises a (N, C*ks*ks, L) tensor -- ks*ks times the input
    -- then runs `topk(k=2)` over it. Both the largest and 2nd-largest can
    instead come from two fused `max_pool2d` calls:

      1. `max1` (and its argmax index) via `max_pool2d(..., return_indices=True)`
      2. mask out the argmax position, then `max_pool2d` again -> `max2`

    This keeps memory at O(input) instead of O(ks^2 * input) and uses cuDNN's
    fused pooling kernels rather than a general-purpose sort.

    Correctness notes:
      * Only valid when windows do not overlap (stride >= kernel_size).
        Overlapping windows share input positions, so masking one window's
        argmax would corrupt its neighbours -- `_overlapping` guards this and
        falls back to the original algorithm.
      * Ties are handled correctly: if a window's top two values are equal,
        `max_pool2d` masks only one of them and the second `max_pool2d`
        recovers the other, matching `topk`'s [v, v].
      * Masking uses `-inf` so the masked position can never win the second
        pass. A window always has ks*ks >= 2 elements with exactly one masked,
        so `max2` is never spuriously `-inf`.
    """

    def __init__(self, kernel_size: int = 2, stride: int = 2,
                 threshold: float = 10.0, detect_only: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.threshold = threshold
        self.detect_only = detect_only
        self._overlapping = stride < kernel_size

    def _forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        """Original unfold+topk path, used for overlapping windows."""
        N, C, H, W = x.shape
        ks = self.kernel_size

        patches = F.unfold(x, kernel_size=ks, stride=self.stride)
        patches = patches.view(N, C, ks * ks, -1)

        top2_vals, _ = torch.topk(patches, 2, dim=2)
        max1 = top2_vals[:, :, 0, :]
        max2 = top2_vals[:, :, 1, :]

        spikes = max1 > self.threshold
        out = max1 if self.detect_only else torch.where(spikes, max2, max1)

        H_out, W_out = H // ks, W // ks
        return out.view(N, C, H_out, W_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._overlapping:
            return self._forward_reference(x)

        N, C, H, W = x.shape
        ks = self.kernel_size

        max1, idx = F.max_pool2d(
            x, kernel_size=ks, stride=self.stride, return_indices=True
        )

        # detect_only never consults max2, so skip the second pass entirely.
        if self.detect_only:
            return max1

        # Mask each window's argmax, then re-pool to get the 2nd largest.
        # idx holds flat indices into the H*W spatial plane, per (N, C).
        flat = x.reshape(N, C, H * W).clone()
        flat.scatter_(2, idx.reshape(N, C, -1), float("-inf"))
        max2 = F.max_pool2d(flat.view(N, C, H, W), kernel_size=ks, stride=self.stride)

        spikes = max1 > self.threshold
        return torch.where(spikes, max2, max1)
