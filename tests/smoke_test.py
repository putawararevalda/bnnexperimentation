"""Phase 1 smoke test — verifies all src/ imports and basic forward passes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

def test_imports():
    from src.models.components import SmartPool, WeightedGaussian, WeightedGaussianActivation, UniformReal
    from src.models.bayesian_cnn import BayesShipsCNN
    from src.models.deterministic_cnn import ShipsCNNCustom
    from src.models import BayesShipsCNN, ShipsCNNCustom
    from src.data.shipsnet import load_data, load_data_withval
    from src.data.eurosat import load_data as eurosat_load_data
    from src.training.svi import train_svi_with_stats, predict_data, plot_training_results_with_stats
    from src.evaluation.seu import bitflip_float32, bitflip_float32_with_original, float32_to_binary, binary_to_float32
    from src.evaluation.metrics import absolute_accuracy_difference, softmax_difference, aggregate_robustness_index
    from src.utils.notify import send_telegram_message
    from src.utils.guide import AutoLaplace, AutoUniform
    print("  [PASS] All imports OK")

def test_components():
    from src.models.components import SmartPool, WeightedGaussian

    wg = WeightedGaussian()
    x = torch.randn(4, 32, 16, 16)
    out = wg(x)
    assert out.shape == x.shape, f"WeightedGaussian shape mismatch: {out.shape}"

    pool = SmartPool(kernel_size=2, stride=2, threshold=10.0)
    out = pool(x)
    assert out.shape == (4, 32, 8, 8), f"SmartPool shape mismatch: {out.shape}"
    print("  [PASS] Components (WeightedGaussian, SmartPool) forward pass OK")

def test_deterministic_cnn():
    from src.models.deterministic_cnn import ShipsCNNCustom

    for act in ['relu', 'sigmoid', 'actWG', 'actRWG']:
        model = ShipsCNNCustom(activation=act)
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        assert out.shape == (2, 2), f"DNN output shape wrong for {act}: {out.shape}"
    print("  [PASS] ShipsCNNCustom forward pass OK (relu, sigmoid, actWG, actRWG)")

def test_bayesian_cnn():
    from src.models.bayesian_cnn import BayesShipsCNN

    device = torch.device("cpu")
    for prior in ['gaussian', 'laplace', 'uniform']:
        model = BayesShipsCNN(prior_dist=prior, device=device, b=1.0)
        x = torch.randn(2, 3, 64, 64)
        import pyro
        pyro.clear_param_store()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 2), f"BNN output shape wrong for {prior}: {out.shape}"
    print("  [PASS] BayesShipsCNN forward pass OK (gaussian, laplace, uniform)")

def test_seu():
    from src.evaluation.seu import (
        bitflip_float32, bitflip_float32_with_original,
        float32_to_binary, binary_to_float32
    )

    original = 1.5
    bits = float32_to_binary(original)
    assert len(bits) == 32
    recovered = binary_to_float32(bits)
    assert abs(recovered - original) < 1e-6, f"float32 roundtrip failed: {recovered}"

    flipped = bitflip_float32(original, bit_i=5)
    assert isinstance(flipped, float)
    assert flipped != original

    flipped2, orig_bit = bitflip_float32_with_original(original, bit_i=5)
    assert orig_bit in ('0', '1')
    print("  [PASS] SEU bitflip functions OK")

def test_metrics():
    from src.evaluation.metrics import absolute_accuracy_difference, softmax_difference, aggregate_robustness_index

    aad = absolute_accuracy_difference(0.95, 0.90)
    assert abs(aad - 0.05) < 1e-6, f"AAD wrong: {aad}"

    logits_before = torch.tensor([[2.0, 1.0], [0.5, 1.5]])
    logits_after  = torch.tensor([[1.5, 1.5], [0.5, 1.5]])
    smd = softmax_difference(logits_before, logits_after)
    assert 0.0 <= smd <= 1.0, f"Softmax diff out of range: {smd}"

    arin = aggregate_robustness_index(0.05, 0.10)
    assert arin > 0, f"ARIn should be positive: {arin}"
    print("  [PASS] Metrics (AAD, SoftmaxDiff, ARIn) OK")

if __name__ == "__main__":
    print("Running Phase 1 smoke tests...\n")
    failures = []
    for test_fn in [test_imports, test_components, test_deterministic_cnn,
                    test_bayesian_cnn, test_seu, test_metrics]:
        try:
            test_fn()
        except Exception as e:
            print(f"  [FAIL] {test_fn.__name__}: {e}")
            failures.append(test_fn.__name__)

    print()
    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)
    else:
        print("All tests passed.")
