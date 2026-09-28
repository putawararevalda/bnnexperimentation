"""
Apply run_set tags to all backfill runs in MLflow based on source_dir.
"""
import mlflow

RUN_SET_MAP = {
    "results_shipsnet_v02_00": "paper_final",
    "results_shipsnet_v02_01": "paper_final",
    "results_shipsnet_v02_02": "paper_final",
    "results_shipsnet_v02_03": "paper_final",
    "results_shipsnet_00": "pre_paper",
    "results_shipsnet_01": "pre_paper",
    "results_shipsnet_02": "pre_paper",
    "results_shipsnet_03": "pre_paper",
    "results_GP_shipsnet_newslate_guide": "guide_iteration",
    "results_GP_shipsnet_newslate": "guide_iteration",
    "results_GP_shipsnet": "early",
    "results_GP_shipsnet_elbo3": "ablation",
    "results_shipsnet_scale_01": "ablation",
    "results_shipsnet_00_mvrt": "ablation",
    "results_shipsnet_00_mvrt_bset": "ablation",
    "uniform_test": "ablation",
    "bayesian": "test",
}

client = mlflow.MlflowClient()
runs = client.search_runs("1", max_results=2000)
print(f"Total runs: {len(runs)}")

counts = {}
skipped = 0
for run in runs:
    source_dir = run.data.tags.get("source_dir", "")
    run_set = RUN_SET_MAP.get(source_dir)
    if run_set is None:
        skipped += 1
        continue
    client.set_tag(run.info.run_id, "run_set", run_set)
    counts[run_set] = counts.get(run_set, 0) + 1

print("\nTagged:")
for label, count in sorted(counts.items()):
    print(f"  {label}: {count} runs")
if skipped:
    print(f"  (skipped {skipped} runs with no source_dir match)")
print("\nDone. Filter in UI with: tags.run_set = \"paper_final\"")
