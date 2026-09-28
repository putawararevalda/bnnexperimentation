import os
import time
import json

import numpy as np
import pandas as pd
import torch
import pyro
import matplotlib
matplotlib.use('Agg')  # headless backend: we only savefig; avoids Tk teardown errors
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


def _ml(fn, *args, **kwargs):
    """Call an mlflow function, never letting a logging failure (e.g. SQLite
    'database is locked' under concurrent runs) crash training."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"[mlflow] logging call failed (non-fatal): {e}")
        return None


def _build_config(
    act_name: str,
    prior_name: str,
    num_epochs: int,
    best_acc: float,
    best_epoch: int | None,
    batch_size: int,
    train_size: int,
    prior_mu: float | None,
    prior_b: float | None,
    extra_config: dict | None = None,
) -> dict:
    """Assemble the run config JSON, merging caller-supplied metadata."""
    config = {
        'activation': act_name,
        'prior': prior_name,
        'num_epochs': num_epochs,
        'best_accuracy': best_acc,
        'best_accuracy_at_epoch': best_epoch,
        'batch_size': batch_size,
        'train_size': train_size,
        'prior_params': {'mu': prior_mu, 'b': prior_b},
    }
    if extra_config:
        config.update(extra_config)
    return config


def train_svi_with_stats(
    model,
    guide,
    svi,
    train_loader,
    device,
    num_epochs: int = 10,
    save_dir: str = 'results',
    model_filename_pattern: str = 'model_{activation}_{prior}_epoch_{epoch}_{timestamp}.pth',
    guide_filename_pattern: str = 'guide_{activation}_{prior}_epoch_{epoch}_{timestamp}.pth',
    param_store_filename_pattern: str = 'param_store_{activation}_{prior}_epoch_{epoch}_{timestamp}.pkl',
    accuracies_filename_pattern: str = 'accuracy_results_{activation}_{prior}_{timestamp}.csv',
    losses_filename_pattern: str = 'losses_{activation}_{prior}_{timestamp}.csv',
    model_config_filename_pattern: str = 'config_{activation}_{prior}_{timestamp}.json',
    extra_config: dict | None = None,
):
    """
    Train a Pyro SVI model, tracking ELBO loss and train accuracy.
    Saves model/guide/param-store artifacts only when train accuracy improves.
    Accuracy is checked at epoch 1, every 10 epochs, and the final epoch.

    Returns:
        (losses, accuracies, accuracy_epochs, loc_stats, scale_stats,
         best_model_path, best_guide_path, best_param_store_path, timestamp)
    """
    act_name = (model.activation_fn.__name__
                if hasattr(model.activation_fn, '__name__')
                else str(model.activation_fn))
    prior_name = getattr(model, 'prior_dist', 'prior')
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    os.makedirs(save_dir, exist_ok=True)

    if _MLFLOW_AVAILABLE:
        _ml(mlflow.set_experiment, "bnn-seu-shipsnet")
        _ml(mlflow.start_run, run_name=f"{act_name}_{prior_name}_{timestamp}")
        _ml(mlflow.log_params, {
            "activation": act_name,
            "prior": prior_name,
            "prior_mu": model.prior_mu.item() if hasattr(model, "prior_mu") else None,
            "prior_b": model.prior_b.item() if hasattr(model, "prior_b") else None,
            "num_epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "train_size": len(train_loader.dataset),
            "smartpool": getattr(model, "smartpool_switch", False),
            "dropout": getattr(model, "dropout_switch", False),
        })

    pyro.clear_param_store()
    model.to(device)

    epoch_losses, epoch_accuracies, accuracy_epochs = [], [], []
    loc_stats = {'epochs': [], 'means': [], 'stds': []}
    scale_stats = {'epochs': [], 'means': [], 'stds': []}
    best_acc = 0.0
    best_model_path = best_guide_path = best_ps_path = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss, batches = 0.0, 0
        return_inf_value = None

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device).long()

            loss_val = svi.evaluate_loss(images, labels)
            loss_tensor = torch.tensor(loss_val)
            if torch.isinf(loss_tensor) or torch.isnan(loss_tensor):
                if return_inf_value is None:
                    return_inf_value = float('inf') if loss_val > 0 else float('-inf')
                continue

            total_loss += svi.step(images, labels)
            batches += 1

        avg_loss = (total_loss / batches) if batches > 0 else (
            return_inf_value if return_inf_value is not None else float('nan'))
        epoch_losses.append(avg_loss)
        if _MLFLOW_AVAILABLE:
            _ml(mlflow.log_metric, "loss_elbo", avg_loss, step=epoch)
        print(f"Epoch {epoch} — avg ELBO loss: {avg_loss:.4f}")

        if epoch == 1 or epoch % 10 == 0 or epoch == num_epochs:
            model.eval()
            guide.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in tqdm(train_loader, desc=f"  Acc check epoch {epoch}"):
                    images, labels = images.to(device), labels.to(device)
                    trace = pyro.poutine.trace(guide).get_trace(images)
                    replayed = pyro.poutine.replay(model, trace=trace)
                    preds = torch.argmax(replayed(images), dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            acc = correct / total
            epoch_accuracies.append(acc)
            accuracy_epochs.append(epoch)
            print(f"  Train accuracy: {acc * 100:.2f}%")
            if _MLFLOW_AVAILABLE:
                _ml(mlflow.log_metric, "train_acc", acc, step=epoch)

            # Record variational parameter statistics
            w_means, w_stds, b_means, b_stds = [], [], [], []
            for name, param in pyro.get_param_store().items():
                if 'loc' in name or 'low' in name:
                    w_means.append(param.mean().item())
                    w_stds.append(param.std(unbiased=False).item())
                elif 'scale' in name or 'width' in name:
                    b_means.append(param.mean().item())
                    b_stds.append(param.std(unbiased=False).item())
            loc_stats['epochs'].append(epoch)
            loc_stats['means'].append(w_means)
            loc_stats['stds'].append(w_stds)
            scale_stats['epochs'].append(epoch)
            scale_stats['means'].append(b_means)
            scale_stats['stds'].append(b_stds)

            if acc > best_acc:
                best_acc = acc
                fname_model = model_filename_pattern.format(
                    activation=act_name, prior=prior_name, epoch="best", timestamp=timestamp)
                fname_guide = guide_filename_pattern.format(
                    activation=act_name, prior=prior_name, epoch="best", timestamp=timestamp)
                fname_ps = param_store_filename_pattern.format(
                    activation=act_name, prior=prior_name, epoch="best", timestamp=timestamp)

                best_model_path = os.path.join(save_dir, fname_model)
                best_guide_path = os.path.join(save_dir, fname_guide)
                best_ps_path = os.path.join(save_dir, fname_ps)

                torch.save(model.state_dict(), best_model_path)
                torch.save(guide.state_dict(), best_guide_path)
                pyro.get_param_store().save(best_ps_path)
                print(f"  >> New best ({acc * 100:.2f}%) - saved artifacts")
                if _MLFLOW_AVAILABLE:
                    _ml(mlflow.log_metric, "best_train_acc", best_acc, step=epoch)

    # Save accuracy and loss CSVs
    pd.DataFrame({'epoch': accuracy_epochs, 'accuracy': epoch_accuracies}).to_csv(
        os.path.join(save_dir, accuracies_filename_pattern.format(
            activation=act_name, prior=prior_name, timestamp=timestamp)), index=False)
    pd.DataFrame({'epoch': list(range(1, num_epochs + 1)), 'loss': epoch_losses}).to_csv(
        os.path.join(save_dir, losses_filename_pattern.format(
            activation=act_name, prior=prior_name, timestamp=timestamp)), index=False)

    # Save config JSON
    config = _build_config(
        act_name=act_name,
        prior_name=prior_name,
        num_epochs=num_epochs,
        best_acc=best_acc,
        best_epoch=accuracy_epochs[int(np.argmax(epoch_accuracies))] if epoch_accuracies else None,
        batch_size=train_loader.batch_size,
        train_size=len(train_loader.dataset),
        prior_mu=model.prior_mu.item() if hasattr(model, 'prior_mu') else None,
        prior_b=model.prior_b.item() if hasattr(model, 'prior_b') else None,
        extra_config=extra_config,
    )
    config_path = os.path.join(save_dir, model_config_filename_pattern.format(
        activation=act_name, prior=prior_name, timestamp=timestamp))
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Config saved to {config_path}")

    return (epoch_losses, epoch_accuracies, accuracy_epochs,
            loc_stats, scale_stats,
            best_model_path, best_guide_path, best_ps_path, timestamp)


def predict_data(model, guide, loader, device, num_samples: int = 10):
    """
    Monte Carlo inference: average logits over `num_samples` weight samples.
    Returns (all_labels, all_predictions) as Python lists.
    """
    model.eval()
    guide.eval()
    all_labels, all_predictions = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            logits_mc = torch.zeros(num_samples, images.size(0), model.fc1.out_features, device=device)

            for i in range(num_samples):
                trace = pyro.poutine.trace(guide).get_trace(images)
                replayed = pyro.poutine.replay(model, trace=trace)
                logits_mc[i] = replayed(images)

            avg_logits = logits_mc.mean(dim=0)
            preds = torch.argmax(avg_logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())

    return all_labels, all_predictions


def plot_training_results_with_stats(losses, accuracies, accuracy_epochs,
                                     loc_stats, scale_stats,
                                     act_name: str, prior_name: str,
                                     timestamp: str, save_dir: str = 'results'):
    """Save a 4-panel training summary plot to save_dir."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].plot(range(1, len(losses) + 1), losses)
    axes[0, 0].set(title='Training Loss', xlabel='Epoch', ylabel='ELBO Loss')
    axes[0, 0].grid(True)

    axes[0, 1].plot(accuracy_epochs, accuracies, 'o-')
    axes[0, 1].set(title='Training Accuracy', xlabel='Epoch', ylabel='Accuracy')
    axes[0, 1].grid(True)

    for ax, stats, color, label in [
        (axes[1, 0], loc_stats, 'lightblue', 'LOC'),
        (axes[1, 1], scale_stats, 'lightcoral', 'SCALE'),
    ]:
        data = [m + s for m, s in zip(stats['means'], stats['stds'])]
        labels = [f'Ep {e}' for e in stats['epochs']]
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(color)
        ax.set(title=f'{label} Statistics', xlabel='Epoch', ylabel=f'{label} Values')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(save_dir, f'training_results_{act_name}_{prior_name}_{timestamp}.png')
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Plot saved to {out_path}")
