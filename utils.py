from pyro.nn import PyroModule, PyroSample
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist

class BayesianCNN1FC(PyroModule):
    def __init__(self):
        super().__init__()

        # Bayesian Conv2d layer 1
        self.conv1 = PyroModule[nn.Conv2d](3, 32, kernel_size=5, stride=1, padding=2)
        self.conv1.weight = PyroSample(dist.Normal(0., 1.).expand([32, 3, 5, 5]).to_event(4))
        self.conv1.bias = PyroSample(dist.Normal(0., 1.).expand([32]).to_event(1))

        # Bayesian Conv2d layer 2
        self.conv2 = PyroModule[nn.Conv2d](32, 64, kernel_size=5, stride=1, padding=2)
        self.conv2.weight = PyroSample(dist.Normal(0., 1.).expand([64, 32, 5, 5]).to_event(4))
        self.conv2.bias = PyroSample(dist.Normal(0., 1.).expand([64]).to_event(1))

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer (output layer)
        self.fc1 = PyroModule[nn.Linear](64 * 16 * 16, 10)
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([10, 64 * 16 * 16]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([10]).to_event(1))

    def forward(self, x, y=None):
        x = self.pool(F.relu(self.conv1(x)))   # -> (B, 32, 32, 32)
        x = self.pool(F.relu(self.conv2(x)))   # -> (B, 64, 16, 16)
        x = x.view(x.size(0), -1)
        logits = self.fc1(x)  # No intermediate layer

        if y is not None:
            with pyro.plate("data", x.size(0)):
                pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits
    
if __name__ == "__main__":
    # if imported as a module, this block will not run
    model = BayesianCNN1FC()
    print(model)