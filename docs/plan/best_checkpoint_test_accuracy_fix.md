# Fix: evaluate test accuracy from the BEST checkpoint, not epoch 100

**Status: written, NOT APPLIED.** Do not apply while a training grid is running
— see "When to apply" below.

Written 2026-08-08. Resolves the pending item in `CLAUDE.md`
("Best-Checkpoint 'Initial Accuracy' Re-evaluation").

## The bug

`src/training/svi.py:178-195` writes `*_epoch_best_*` artifacts whenever **train**
accuracy improves, then keeps training from the live weights. Nothing reloads
them before the function returns (`svi.py:225`).

Both training entry points then evaluate the *live* objects:

```python
# scripts/train_shipsnet.py:186-207   (scripts/train_eurosat.py:183-198 identical)
(losses, accuracies, accuracy_epochs,
 loc_stats, scale_stats,
 best_model_path, best_guide_path, best_ps_path,   # <- unpacked
 ts) = train_svi_with_stats(...)
...
labels, preds = predict_data(model, guide, test_loader, device, num_samples=10)
#                            ^^^^^  ^^^^^  epoch-100 state; best_*_path never used
```

So `test_acc` and the `predictions_*_NN.csv` filename record the **last-epoch**
model. The three `best_*_path` variables are dead after line 188.

### Why it matters

- Paper **Table 1**'s BNN column is last-epoch test accuracy. It is close to the
  best-checkpoint value only for runs that were stable to epoch 100.
- The SEU scripts load the **opposite** model — `eval_seu_*.py` loads
  `*_epoch_best_*`. So SEU `initial_accuracy` (.8449, ShipsNet fold-1 base) and
  Table 1 (.8288) describe two different models. This is why Table 1 cannot be
  reproduced from the SEU CSVs.
- It is the dominant cause of ShipsNet fold-2's uniform-prior "collapse":
  relu6/uniform/b=1.0 has train-max **.796 at epoch 20** but **.54 at epoch 100**.
  The saved checkpoint is fine; only the reported number is broken.

## The fix

Load all three best artifacts before `predict_data`. Applies identically to
`scripts/train_shipsnet.py` (~line 207) and `scripts/train_eurosat.py` (~line 198).

```python
        # Evaluate the BEST checkpoint, not the epoch-100 weights. Training
        # continues past the best epoch, so `model`/`guide` and the global Pyro
        # param store are all at the last epoch by the time we get here. The SEU
        # scripts load these same *_epoch_best_* artifacts, so evaluating them
        # here keeps accuracy tables and SEU baselines on one model.
        if best_model_path and best_guide_path and best_ps_path:
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            guide.load_state_dict(torch.load(best_guide_path, map_location=device))
            pyro.clear_param_store()
            pyro.get_param_store().load(best_ps_path, map_location=device)
            eval_source = "best"
        else:
            # No epoch ever improved on the initial best_acc=0.0 sentinel.
            logger.warning("no best checkpoint saved; evaluating last-epoch weights")
            eval_source = "last"

        labels, preds = predict_data(model, guide, test_loader, device, num_samples=10)
        cm = confusion_matrix(labels, preds)
        test_acc = np.trace(cm) / np.sum(cm)
        print(f"Test accuracy ({eval_source} checkpoint): {test_acc * 100:.4f}%")
```

### Details that matter

1. **The param-store load is the important one.** `predict_data` samples the
   guide through Pyro's *global* param store. Restoring only the two module
   `state_dict`s would leave the epoch-100 variational parameters in play and
   change almost nothing. All three loads are required.

2. **Guard against `None`.** `svi.py:115` initialises
   `best_model_path = best_guide_path = best_ps_path = None`. In practice
   `best_acc` starts at 0.0 so the first evaluated epoch always saves, but the
   guard costs nothing and avoids a confusing `TypeError`.

3. **`pyro.clear_param_store()` before the load.** `ParamStoreDict.load` merges
   into the existing store rather than replacing it; clearing first guarantees
   no epoch-100 parameter survives.

4. **Both files already import what is needed** (`pyro`, `torch`, `np`,
   `confusion_matrix`). `train_shipsnet.py` has no `logger` — either add
   `logging` per the project style rule, or use `print` to match the
   surrounding code.

5. **Accuracy is sampled, not continuous.** `svi.py:142` evaluates only at
   `epoch == 1 or epoch % 10 == 0 or epoch == num_epochs`. So "best epoch" is
   always one of those, and when the final epoch happens to be the best, the
   fix is a no-op for that run.

## What this does NOT fix

- **Checkpoint selection is on TRAIN accuracy.** There is no validation split —
  `train_shipsnet.py:183` calls `load_data` (train/test only). So "best" means
  best-fitting-the-training-set, a weak and mildly optimistic criterion. Worth a
  sentence in the revision regardless of whether this patch lands.
- **The uniform-prior ELBO divergence.** Every uniform run in *both* folds has
  `inf` loss for 51-99 of 100 epochs, caused by `UniformReal`
  (`src/models/components.py:64`) overriding `support` to `real` without
  overriding `log_prob`, while `AutoUniform` is free to wander outside `[-b, b]`.
  This patch mitigates the *symptom* (last-epoch random-walk endpoint) but not
  the cause. Separate decision.

## Blast radius

Applying this changes reported test accuracy for **every future run only**.
Existing `predictions_*` files and config JSONs are untouched, so the grid would
be split across two evaluation methods unless all configs are re-evaluated.

Expected direction and size, from ShipsNet fold-2 base (train-max vs last-epoch
paired over 63 configs): mean shift **+0.02**, but up to **+0.34** for the
diverged uniform runs. Stable runs barely move.

## Recommended sequencing

1. **Wait** for the running fold-2 grid to finish (see below).
2. Apply the patch to both training scripts.
3. Write a separate re-evaluation script for the *existing* configs — the best
   checkpoints are already on disk for every config in both datasets, so this is
   a pure inference pass, no retraining. Write to a new directory; do not
   overwrite `predictions_*`.
4. Regenerate Table 1 from the re-evaluated numbers so accuracy tables and SEU
   baselines finally refer to the same model.

## When to apply

**Not while a training grid is running.** `scripts/train_shipsnet_folds2to5.bat`
launches a *fresh* `uv run python scripts/train_shipsnet.py` per variant/prior
(12 invocations per fold), so an edit lands on the next invocation and splits a
single fold across two evaluation methods. The already-running process is
unaffected — Python has imported the module — but the next one in the loop is not.

Safe point: after a fold's 12 invocations complete, or after killing the batch.
