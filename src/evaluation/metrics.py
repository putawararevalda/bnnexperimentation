import numpy as np
import torch


def absolute_accuracy_difference(acc_before: float, acc_after: float) -> float:
    """AAD: absolute change in accuracy before/after SEU. Lower = more robust."""
    return abs(acc_after - acc_before)


def softmax_difference(logits_before: torch.Tensor, logits_after: torch.Tensor) -> float:
    """
    Mean L-inf distance between softmax output distributions before and after SEU.
    Lower = more robust. Inputs shape: (N, num_classes).
    """
    probs_before = torch.softmax(logits_before, dim=-1)
    probs_after = torch.softmax(logits_after, dim=-1)
    l_inf = (probs_before - probs_after).abs().max(dim=-1).values
    return l_inf.mean().item()


def aggregate_robustness_index(aad: float, smd: float) -> float:
    """
    ARIn: RMS of AAD and Softmax Difference.
    Proposed in the ICAART 2026 paper as a single scalar robustness measure.
    Lower = more robust.
    """
    return float(np.sqrt((aad ** 2 + smd ** 2) / 2))
