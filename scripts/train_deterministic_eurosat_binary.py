"""
Quick deterministic baseline to verify architecture learns Forest vs. SeaLake.
If this gets <80%, the issue is data/architecture. If it gets >80%, the issue is SVI.

Usage:
    uv run python scripts/train_deterministic_eurosat_binary.py --epoch 30
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from src.data.eurosat_binary import load_data_withval
from src.models.deterministic_cnn import ShipsCNNCustom


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = load_data_withval(batch_size=32)

    model = ShipsCNNCustom(activation="relu", smartpool_switch=False, dropout_switch=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epoch + 1):
        model.train()
        correct = total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                correct += (model(imgs).argmax(1) == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        print(f"Epoch {epoch:3d}/{args.epoch}  train={train_acc:.4f}  val={val_acc:.4f}")

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total += labels.size(0)
    print(f"\nTest accuracy: {correct/total*100:.2f}%")


if __name__ == "__main__":
    main()
