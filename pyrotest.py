import torch
import torch.nn.functional as F
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample, PyroParam
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import time

pyro.set_rng_seed(0)

# ----- 1. Bayesian Linear Layer -----
class BayesianLinear(PyroModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, in_features]).to_event(2))
        self.bias = PyroSample(dist.Normal(0., 1.).expand([out_features]).to_event(1))

    def forward(self, x, weight, bias):
        return F.linear(x, weight, bias)

# ----- 2. Bayesian Neural Network -----
class BNN(PyroModule):
    def __init__(self):
        super().__init__()
        self.fc1 = BayesianLinear(28*28, 256)
        self.fc2 = BayesianLinear(256, 10)

    def forward(self, x, y=None):
        x = x.view(-1, 28*28)
        w1 = pyro.sample("w1", self.fc1.weight)
        b1 = pyro.sample("b1", self.fc1.bias)
        w2 = pyro.sample("w2", self.fc2.weight)
        b2 = pyro.sample("b2", self.fc2.bias)

        x = F.relu(self.fc1(x, w1, b1))
        logits = self.fc2(x, w2, b2)

        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits

# ----- 3. Guide (Variational Posterior) -----
class Guide(PyroModule):
    def __init__(self):
        super().__init__()
        self.fc1w_loc = PyroParam(torch.randn(256, 28*28) * 0.1)
        self.fc1w_scale = PyroParam(torch.ones(256, 28*28) * 0.1, constraint=dist.constraints.positive)
        self.fc1b_loc = PyroParam(torch.randn(256) * 0.1)
        self.fc1b_scale = PyroParam(torch.ones(256) * 0.1, constraint=dist.constraints.positive)

        self.fc2w_loc = PyroParam(torch.randn(10, 256) * 0.1)
        self.fc2w_scale = PyroParam(torch.ones(10, 256) * 0.1, constraint=dist.constraints.positive)
        self.fc2b_loc = PyroParam(torch.randn(10) * 0.1)
        self.fc2b_scale = PyroParam(torch.ones(10) * 0.1, constraint=dist.constraints.positive)

    def forward(self, x, y=None):
        pyro.sample("w1", dist.Normal(self.fc1w_loc, self.fc1w_scale).to_event(2))
        pyro.sample("b1", dist.Normal(self.fc1b_loc, self.fc1b_scale).to_event(1))
        pyro.sample("w2", dist.Normal(self.fc2w_loc, self.fc2w_scale).to_event(2))
        pyro.sample("b2", dist.Normal(self.fc2b_loc, self.fc2b_scale).to_event(1))

# ----- 4. SEU Injection on Mean (loc) -----
def flip_bit_in_tensor(tensor, bit_position=10, flip_count=1):
    flat = tensor.view(-1)
    idx = torch.randint(0, flat.numel(), (flip_count,))
    for i in idx:
        val = flat[i].item()
        int_val = np.frombuffer(np.float32(val).tobytes(), dtype=np.uint32)[0]
        flipped = int_val ^ (1 << bit_position)
        flipped_val = np.frombuffer(np.uint32(flipped).tobytes(), dtype=np.float32)[0]
        flat[i] = torch.tensor(flipped_val)
    return tensor

def inject_seu(guide, layer="fc1w_loc", bit_position=10, flip_count=1):
    param = getattr(guide, layer)
    with torch.no_grad():
        new_tensor = flip_bit_in_tensor(param.data.clone(), bit_position, flip_count)
        setattr(guide, layer, PyroParam(new_tensor))

# ----- 5. Training -----
def train(model, guide, loader, svi, epochs=3):
    for epoch in range(epochs):
        epoch_loss = 0
        for x, y in loader:
            loss = svi.step(x, y)
            epoch_loss += loss
        print(f"Epoch {epoch+1} - ELBO Loss: {epoch_loss / len(loader.dataset):.4f}")

# ----- 6. Evaluation -----
def evaluate(model, guide, loader, samples=10):
    correct = 0
    total = 0
    for x, y in loader:
        probs = torch.zeros(x.size(0), 10)
        for _ in range(samples):
            guide_trace = pyro.poutine.trace(guide).get_trace(x)
            model_trace = pyro.poutine.trace(pyro.poutine.replay(model, trace=guide_trace)).get_trace(x)
            logits = model_trace.nodes["obs"]["fn"].logits
            probs += F.softmax(logits, dim=1)
        probs /= samples
        preds = probs.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# ----- 7. Run Experiment -----
def main():
    transform = transforms.ToTensor()
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000)

    model = BNN()
    guide = Guide()
    svi = SVI(model, guide, Adam({"lr": 1e-3}), loss=Trace_ELBO())

    # record the time taken for training
    designated_epochs = 3
    start_time = time.time()
    print("Training...")
    train(model, guide, train_loader, svi, epochs=designated_epochs)
    end_time = time.time()
    print(f"Training completed for {designated_epochs} epochs in {end_time - start_time:.2f} seconds")

    print("\nEvaluating clean model:")
    evaluate(model, guide, test_loader)

    print("\nInjecting SEU into fc1 weight mean (bit 10)...")
    inject_seu(guide, "fc1w_loc", bit_position=10, flip_count=10)

    print("Evaluating after SEU injection:")
    evaluate(model, guide, test_loader)

if __name__ == "__main__":
    main()
