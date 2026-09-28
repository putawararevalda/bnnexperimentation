"""
Train deterministic CNN baseline on ShipsNet dataset.

Mirrors the 4 model variants from the paper:
    00 — base (MaxPool, no dropout, no weight decay)
    01 — SmartPool
    02 — Dropout(p=0.5) after conv2
    03 — base + weight decay (λ=1e-4) via Adam

Usage examples:
    uv run python scripts/train_deterministic.py --model-variant 00 --epoch 20
    uv run python scripts/train_deterministic.py --model-variant 02 --epoch 50
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from src.data.shipsnet import load_data_withval
from src.models.deterministic_cnn import ShipsCNNCustom

ACTIVATIONS = ['relu', 'tanh', 'sigmoid', 'sin', 'relu6', 'actWG', 'actRWG']


def parse_args():
    parser = argparse.ArgumentParser(description='Train deterministic CNN on ShipsNet')
    parser.add_argument('--epoch', type=int, default=20,
                        help='Number of training epochs. Default: 20')
    parser.add_argument('--model-variant', type=str, default='00',
                        choices=['00', '01', '02', '03'],
                        help='Model variant. Default: 00 (base)')
    parser.add_argument('--fold', type=int, default=None, choices=[1, 2, 3, 4, 5],
                        help='Cross-validation fold (1-5). Omit for legacy split.')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Save directory. Defaults to results/shipsnet/deterministic/[foldN/]<variant>')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.save_dir:
        save_dir = args.save_dir
    elif args.fold is not None:
        save_dir = f"results/shipsnet/deterministic/fold{args.fold}/{args.model_variant}"
    else:
        save_dir = f"results/shipsnet/deterministic/{args.model_variant}"
    os.makedirs(save_dir, exist_ok=True)

    train_loader, val_loader, test_loader = load_data_withval(batch_size=16, fold=args.fold)
    train_ds = train_loader.dataset
    val_ds = val_loader.dataset

    for activation in ACTIVATIONS:
        weight_decay = 0.0
        smartpool = args.model_variant == '01'
        dropout = args.model_variant == '02'
        if args.model_variant == '03':
            weight_decay = 1e-4

        model = ShipsCNNCustom(
            activation=activation,
            smartpool_switch=smartpool,
            dropout_switch=dropout,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        act_name = model.activation_fn.__name__

        log_path = os.path.join(save_dir, f"log_{act_name}_{timestamp}.csv")
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(['variant', 'activation', 'epoch',
                                    'train_loss', 'train_acc', 'val_loss', 'val_acc'])

        best_val_acc = 0.0
        for epoch in range(1, args.epoch + 1):
            model.train()
            running_loss = running_correct = 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * imgs.size(0)
                running_correct += (out.argmax(1) == labels).sum().item()

            train_loss = running_loss / len(train_ds)
            train_acc = running_correct / len(train_ds)

            model.eval()
            val_loss = val_correct = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    out = model(imgs)
                    val_loss += criterion(out, labels).item() * imgs.size(0)
                    val_correct += (out.argmax(1) == labels).sum().item()

            val_loss /= len(val_ds)
            val_acc = val_correct / len(val_ds)

            print(f"[{act_name}] Epoch {epoch:2d}/{args.epoch} "
                  f"train={train_acc:.4f} val={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(),
                           os.path.join(save_dir, f"best_{act_name}_{timestamp}.pth"))
                print(f"  ↳ New best val acc: {best_val_acc:.4f}")

            with open(log_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    args.model_variant, activation, epoch,
                    train_loss, train_acc, val_loss, val_acc
                ])

        print(f"[{act_name}] Best val acc: {best_val_acc:.4f}\n")


if __name__ == "__main__":
    main()
