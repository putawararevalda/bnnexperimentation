import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import random
import numpy as np

# ----- 1. Bayesian Layer using MC Dropout -----
class BayesianLinear(nn.Linear):
    def forward(self, x):
        return F.dropout(super().forward(x), p=0.2, training=True)  # MC Dropout

# ----- 2. BNN Model -----
class BNN_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = BayesianLinear(28 * 28, 256)
        self.fc2 = BayesianLinear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ----- 3. SEU: Bit Flip in Weights -----
def flip_bit_in_tensor(tensor, bit_position=10, flip_count=1):
    flat_tensor = tensor.view(-1)
    indices = torch.randint(0, flat_tensor.numel(), (flip_count,))
    
    for i in indices:
        val = flat_tensor[i].item()
        int_repr = np.frombuffer(np.float32(val).tobytes(), dtype=np.uint32)[0]
        flipped = int_repr ^ (1 << bit_position)
        flipped_val = np.frombuffer(np.uint32(flipped).tobytes(), dtype=np.float32)[0]
        flat_tensor[i] = torch.tensor(flipped_val)
    
    return tensor

def inject_seu(model, layer_name="fc1", bit_position=10, flip_count=1):
    layer = getattr(model, layer_name)
    with torch.no_grad():
        layer.weight.data = flip_bit_in_tensor(layer.weight.data.clone(), bit_position, flip_count)

# ----- 4. Load MNIST -----
def get_data_loader():
    transform = transforms.ToTensor()
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=transform),
        batch_size=1000, shuffle=False
    )
    return test_loader

# ----- 5. Evaluate with Uncertainty -----
def evaluate(model, data_loader, num_samples=10):
    model.eval()
    correct = 0
    total = 0
    all_preds = []

    with torch.no_grad():
        for data, target in data_loader:
            preds = torch.zeros(data.size(0), 10)
            for _ in range(num_samples):  # MC Dropout
                output = model(data)
                preds += F.softmax(output, dim=1)
            preds /= num_samples
            predicted = preds.argmax(dim=1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

    acc = correct / total
    print(f"Accuracy: {acc*100:.2f}%")
    return acc

# ----- 6. Run Experiment -----
def main():
    model = BNN_MNIST()
    test_loader = get_data_loader()

    # Dummy train (or load pretrained weights here)
    print("Evaluating clean model:")
    evaluate(model, test_loader)

    # Inject SEU
    print("\nInjecting SEU (bit flip at position 10 in fc1 weights)...")
    inject_seu(model, layer_name="fc1", bit_position=10, flip_count=5)

    print("Evaluating after SEU injection:")
    evaluate(model, test_loader)

if __name__ == "__main__":
    main()
