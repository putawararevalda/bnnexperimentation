import json
from pathlib import Path

root = Path("results/shipsnet/bayesian")
dirs = {}
for cfg in sorted(root.rglob("config_*.json")):
    d = cfg.parent.name
    if d not in dirs:
        dirs[d] = {"acts": set(), "priors": set(), "bs": set(), "count": 0, "dates": []}
    with open(cfg) as f:
        c = json.load(f)
    dirs[d]["acts"].add(c.get("activation", "?"))
    dirs[d]["priors"].add(c.get("prior", "?"))
    b = c.get("prior_params", {}).get("b")
    if b:
        dirs[d]["bs"].add(round(float(b), 2))
    dirs[d]["count"] += 1
    # extract date from filename
    stem = cfg.stem  # e.g. config_relu_gaussian_20260608_195536
    parts = stem.split("_")
    for i, p in enumerate(parts):
        if len(p) == 8 and p.isdigit():
            dirs[d]["dates"].append(p)
            break

for d, info in sorted(dirs.items()):
    dates = sorted(set(info["dates"]))
    date_range = f"{dates[0]} -> {dates[-1]}" if dates else "?"
    print(f"{d} ({info['count']} runs, {date_range})")
    print(f"  acts:   {sorted(info['acts'])}")
    print(f"  priors: {sorted(info['priors'])}")
    print(f"  b vals: {sorted(info['bs'])}")
    print()
