import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import SmartPool


class ShipsCNNCustom(nn.Module):
    """
    Deterministic CNN baseline for ShipsNet binary classification.
    Mirrors the architecture of BayesShipsCNN for fair comparison.
    Architecture: conv1(3→32,k=3) → pool → conv2(32→64,k=3) → pool → fc1(64*16*16→num_classes)

    Model variants (matching the paper's 4 designs):
        Variant 00: base model (MaxPool, no dropout, no weight decay)
        Variant 01: SmartPool instead of MaxPool
        Variant 02: Dropout(p=0.5) after conv2 during training
        Variant 03: Base model + weight decay via optimizer (configured externally)

    Args:
        num_classes: Number of output classes (default 2).
        activation: Activation name string. Supported: relu, tanh, sigmoid, sin, relu6, actWG, actRWG.
        smartpool_switch: Use SmartPool instead of MaxPool.
        pool_threshold: SmartPool spike threshold (default 10.0).
        pool_detect_only: SmartPool detect-only mode (no replacement).
        dropout_switch: Apply Dropout(p=dropout_p) after conv2 during training.
        dropout_p: Dropout probability (default 0.5).
    """

    def __init__(
        self,
        num_classes: int = 2,
        activation: str = 'relu',
        smartpool_switch: bool = False,
        pool_threshold: float = 10.0,
        pool_detect_only: bool = False,
        dropout_switch: bool = False,
        dropout_p: float = 0.5,
    ):
        super().__init__()
        self.dropout_switch = dropout_switch

        act_map = {
            'relu': F.relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'sin': torch.sin,
            'relu6': F.relu6,
            'actWG': self._actWG,
            'actRWG': self._actRWG,
        }
        if activation not in act_map:
            raise ValueError(f"Unsupported activation '{activation}'. Choose from: {list(act_map)}")
        self.activation_fn = act_map[activation]

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = (
            SmartPool(kernel_size=2, stride=2, threshold=pool_threshold, detect_only=pool_detect_only)
            if smartpool_switch
            else nn.MaxPool2d(kernel_size=2, stride=2)
        )

        if dropout_switch:
            self.dropout = nn.Dropout(p=dropout_p)

        self.fc1 = nn.Linear(64 * 16 * 16, num_classes)

    def _actWG(self, x, alpha=1.0):
        return x * torch.exp(-alpha * x ** 2)

    def _actRWG(self, x, alpha=1.0):
        wg = x * torch.exp(-alpha * x ** 2)
        return torch.max(torch.zeros_like(wg), wg)

    def forward(self, x):
        x = self.activation_fn(self.conv1(x))
        x = self.pool(x)
        x = self.activation_fn(self.conv2(x))
        x = self.pool(x)

        if self.dropout_switch and self.training:
            x = self.dropout(x)

        x = x.view(x.size(0), -1)
        return self.fc1(x)
