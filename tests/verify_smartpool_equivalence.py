"""
Verifies SmartPoolFast is bit-identical to SmartPool, then benchmarks it.

Covers the cases where a max_pool-based rewrite could plausibly diverge from
unfold+topk: exact ties, spikes above threshold, values straddling the
threshold, infinities, negatives, detect_only, and overlapping windows
(which must fall back to the reference path).
"""
import os
import sys
import time

os.chdir("D:/bnnexperimentation")
sys.path.insert(0, "D:/bnnexperimentation")

import torch

from src.models.components import SmartPool
from src.models.components_fast import SmartPoolFast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}\n")

THR = 10.0
fails = 0


def compare(name, x, threshold=THR, detect_only=False, ks=2, stride=2):
    global fails
    ref = SmartPool(kernel_size=ks, stride=stride, threshold=threshold,
                    detect_only=detect_only).to(device)
    fast = SmartPoolFast(kernel_size=ks, stride=stride, threshold=threshold,
                         detect_only=detect_only).to(device)
    with torch.no_grad():
        a = ref(x)
        b = fast(x)
    same_shape = a.shape == b.shape
    # equal_nan so NaN-vs-NaN counts as a match rather than a spurious failure
    identical = same_shape and torch.equal(a, b)
    close = same_shape and torch.allclose(a, b, rtol=0, atol=0, equal_nan=True)
    ok = identical or close
    if not ok:
        fails += 1
        diff = (a - b).abs()
        finite = torch.isfinite(diff)
        maxdiff = diff[finite].max().item() if finite.any() else float("nan")
        print(f"  FAIL {name}: shapes {a.shape} vs {b.shape}, max|diff|={maxdiff}")
    else:
        print(f"  ok   {name}")
    return ok


print("=== Correctness ===")

torch.manual_seed(0)

# 1. Plain random activations, nothing above threshold.
compare("random, no spikes", torch.randn(8, 16, 32, 32, device=device))

# 2. Random with a few injected spikes above threshold.
x = torch.randn(8, 16, 32, 32, device=device)
x.view(-1)[torch.randint(0, x.numel(), (200,), device=device)] = 1e6
compare("random + spikes", x)

# 3. Exact ties everywhere -- topk gives [v, v]; max_pool masks only one.
compare("all-equal (ties)", torch.full((4, 8, 16, 16), 3.0, device=device))

# 4. Ties at the max, above threshold (tie AND spike interact).
x = torch.randn(4, 8, 16, 16, device=device)
x[:, :, 0::2, 0::2] = 50.0
x[:, :, 0::2, 1::2] = 50.0
compare("tied maxima above threshold", x)

# 5. Values exactly at threshold (boundary: `>` not `>=`).
compare("exactly at threshold", torch.full((4, 8, 16, 16), THR, device=device))

# 6. Straddling threshold.
x = torch.linspace(-20, 20, 4 * 8 * 16 * 16, device=device).reshape(4, 8, 16, 16)
compare("linspace straddling threshold", x)

# 7. Infinities (SEU bitflips routinely produce these).
x = torch.randn(4, 8, 16, 16, device=device)
x.view(-1)[:50] = float("inf")
x.view(-1)[50:100] = float("-inf")
compare("with +/-inf", x)

# 8. All negative (max2 selection with negatives).
compare("all negative", -torch.rand(4, 8, 16, 16, device=device) * 100)

# 9. Very large finite values near fp32 max.
x = torch.full((4, 8, 16, 16), 3.4e38, device=device)
compare("near fp32 max", x)

# 10. detect_only=True (must skip max2 entirely).
x = torch.randn(4, 8, 16, 16, device=device)
x.view(-1)[:100] = 1e6
compare("detect_only", x, detect_only=True)

# 11. Non-default threshold.
compare("threshold=0.5", torch.randn(4, 8, 16, 16, device=device), threshold=0.5)

# 12. Overlapping windows: the ORIGINAL SmartPool is already broken here
#     (it computes H_out = H//ks, which disagrees with unfold's output count
#     whenever stride != kernel_size). The project only ever uses ks=2,
#     stride=2. SmartPoolFast delegates to the same code path, so it must
#     fail the same way rather than silently returning something different.
_x = torch.randn(4, 8, 16, 16, device=device)
_ref_err = _fast_err = None
try:
    SmartPool(kernel_size=3, stride=2, threshold=THR).to(device)(_x)
except Exception as e:
    _ref_err = type(e).__name__
try:
    SmartPoolFast(kernel_size=3, stride=2, threshold=THR).to(device)(_x)
except Exception as e:
    _fast_err = type(e).__name__
if _ref_err == _fast_err and _ref_err is not None:
    print(f"  ok   overlapping ks=3 stride=2 (both raise {_ref_err}, "
          f"pre-existing limitation of the original)")
else:
    fails += 1
    print(f"  FAIL overlapping: orig={_ref_err} fast={_fast_err}")

# 13. Larger non-overlapping kernel.
compare("ks=4 stride=4", torch.randn(4, 8, 32, 32, device=device), ks=4, stride=4)

# 14. Actual EuroSAT-shaped tensors from the real model path.
#     conv1 out: (54, 32, 64, 64); conv2 out: (54, 64, 32, 32)
compare("eurosat conv1 shape", torch.randn(54, 32, 64, 64, device=device))
compare("eurosat conv2 shape", torch.randn(54, 64, 32, 32, device=device))

# 15. Randomised fuzz across many seeds/scales.
fuzz_ok = True
for seed in range(50):
    torch.manual_seed(seed)
    scale = 10 ** (seed % 8 - 3)
    xf = torch.randn(4, 8, 16, 16, device=device) * scale
    if seed % 3 == 0:
        xf.view(-1)[torch.randint(0, xf.numel(), (20,), device=device)] = 1e7
    ref = SmartPool(threshold=THR).to(device)
    fast = SmartPoolFast(threshold=THR).to(device)
    with torch.no_grad():
        if not torch.equal(ref(xf), fast(xf)):
            fuzz_ok = False
            print(f"  FAIL fuzz seed={seed} scale={scale}")
            fails += 1
            break
if fuzz_ok:
    print("  ok   fuzz 50 seeds x varying scales")

print(f"\n=== Correctness result: {'ALL IDENTICAL' if fails == 0 else f'{fails} FAILURES'} ===\n")

if fails:
    sys.exit(1)

print("=== Benchmark (EuroSAT shapes, GPU) ===")


def bench(mod, x, iters=200):
    with torch.no_grad():
        for _ in range(20):  # warmup
            mod(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            mod(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000


for shape in [(54, 32, 64, 64), (54, 64, 32, 32)]:
    x = torch.randn(*shape, device=device)
    ref = SmartPool(threshold=THR).to(device)
    fast = SmartPoolFast(threshold=THR).to(device)
    t_ref = bench(ref, x)
    t_fast = bench(fast, x)
    print(f"  {str(shape):20s}  orig {t_ref:7.3f} ms   fast {t_fast:7.3f} ms   "
          f"speedup {t_ref / t_fast:5.2f}x")

# Also time a full model forward, which is what actually gates the sweep.
print("\n=== Full model forward (MC-10 equivalent) ===")
from src.models.bayesian_cnn import BayesShipsCNN
from src.models.components_fast import SmartPoolFast as _SPF

for label, use_fast in [("orig SmartPool", False), ("SmartPoolFast", True)]:
    torch.manual_seed(0)
    m = BayesShipsCNN(num_classes=10, device=device, activation="relu6",
                      prior_dist="gaussian", mu=0.0, b=1.0,
                      smartpool_switch=True, dropout_switch=False).to(device)
    if use_fast:
        m.pool = _SPF(kernel_size=2, stride=2, threshold=10.0)
    img = torch.randn(54, 3, 64, 64, device=device)
    with torch.no_grad():
        for _ in range(5):
            m(img)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            m(img)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 50 * 1000
    print(f"  {label:16s}  {dt:7.3f} ms / forward")
