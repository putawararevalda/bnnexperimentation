"""
Verifies eval_seu_eurosat_seeded.py is actually deterministic:

  A. Two independent SeededInjector runs -> identical baseline + flip metrics.
  B. Order-independence: evaluating flips in a DIFFERENT order still yields the
     same per-flip values (this is what makes resume safe).
  C. Contrast: the unseeded NewInjector does NOT reproduce, confirming the
     seeding is what's doing the work and the test isn't vacuous.
"""
import os
import sys

os.chdir("D:/bnnexperimentation")
sys.path.insert(0, "D:/bnnexperimentation")

import torch

from scripts.eval_seu_eurosat import NewInjector, load_model
from scripts.eval_seu_eurosat_seeded import SeededInjector
from src.data.eurosat import load_data

SEARCH = "results/eurosat/bayesian/results_eurosat_v02_01"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_files = os.listdir(SEARCH)
ts = [f[:-5][-16:] for f in all_files if f.endswith(".json")][0]
cf = {ts: next(f for f in all_files if ts in f and f.endswith(".json"))}
mf = {ts: next(f for f in all_files if ts in f and f.startswith("model"))}
pf = {ts: next(f for f in all_files if ts in f and f.startswith("param"))}
print(f"config: {ts}\n")

_, test_loader = load_data(batch_size=54)

FLIPS = [
    (0, "conv1", "weight", 0, "locs"),
    (0, "conv2", "bias", 6, "locs"),
    (-1, "fc1", "weight", 1, "locs"),
]
KEYS = ["accuracy_change", "softmax_difference", "absolute_difference"]


def run(injector_cls, flips, **kw):
    model, param_path, _ = load_model(
        ts, SEARCH, cf, mf, pf, device,
        smartpool_switch=True, dropout_switch=False, fast_smartpool=True,
    )
    inj = injector_cls(
        trained_model=model, device=device, test_loader=test_loader,
        num_samples=10, pyro_param_store_path=param_path, **kw,
    )
    out = {}
    for (loc, layer, module, bit, pname) in flips:
        r = inj.run_seu(loc, "AutoNormal", pname, layer, module, bit, 10)
        out[(loc, layer, module, bit, pname)] = {k: r.get(k) for k in KEYS}
    return inj.initial_accuracy, out


def same(d1, d2):
    if set(d1) != set(d2):
        return False
    for k in d1:
        for key in KEYS:
            a, b = d1[k][key], d2[k][key]
            if not (a == b or (a != a and b != b)):
                return False
    return True


print("=== A. Two independent SEEDED runs (same order) ===")
i1, r1 = run(SeededInjector, FLIPS, base_seed=42, timestamp=ts)
i2, r2 = run(SeededInjector, FLIPS, base_seed=42, timestamp=ts)
print(f"  initial_accuracy: {i1!r} vs {i2!r}")
a_ok = (i1 == i2) and same(r1, r2)
print(f"  -> {'IDENTICAL' if a_ok else 'DIFFERS'}")
for k in FLIPS:
    print(f"     {k[1]}.{k[2]} bit={k[3]}: {r1[k]['accuracy_change']!r} | {r2[k]['accuracy_change']!r}")

print("\n=== B. Seeded, flips evaluated in REVERSED order (resume-safety) ===")
i3, r3 = run(SeededInjector, list(reversed(FLIPS)), base_seed=42, timestamp=ts)
b_ok = (i3 == i1) and same(r1, r3)
print(f"  initial_accuracy: {i1!r} vs {i3!r}")
print(f"  -> {'ORDER-INDEPENDENT' if b_ok else 'ORDER-DEPENDENT (resume unsafe)'}")
for k in FLIPS:
    print(f"     {k[1]}.{k[2]} bit={k[3]}: {r1[k]['accuracy_change']!r} | {r3[k]['accuracy_change']!r}")

print("\n=== C. Contrast: two UNSEEDED runs (should differ) ===")
u1, ur1 = run(NewInjector, FLIPS)
u2, ur2 = run(NewInjector, FLIPS)
c_differs = not ((u1 == u2) and same(ur1, ur2))
print(f"  initial_accuracy: {u1!r} vs {u2!r}")
print(f"  -> {'DIFFERS (expected)' if c_differs else 'identical (test may be vacuous!)'}")

print("\n" + "=" * 60)
print(f"A seeded reproducible : {'PASS' if a_ok else 'FAIL'}")
print(f"B order-independent   : {'PASS' if b_ok else 'FAIL'}")
print(f"C unseeded differs    : {'PASS' if c_differs else 'WARN'}")
print("=" * 60)
