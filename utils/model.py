from pyro.nn import PyroModule, PyroSample
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
import torch


class BayesianCNNSingleFC(PyroModule):
    def __init__(self, num_classes, device):
        super().__init__()

        prior_mu = 0.
        prior_sigma = torch.tensor(10., device=device)

        self.conv1 = PyroModule[nn.Conv2d](3, 32, kernel_size=5, stride=1, padding=2)
        self.conv1.weight = PyroSample(dist.Normal(prior_mu, prior_sigma).expand([32, 3, 5, 5]).to_event(4))
        self.conv1.bias = PyroSample(dist.Normal(prior_mu, prior_sigma).expand([32]).to_event(1))

        self.conv2 = PyroModule[nn.Conv2d](32, 64, kernel_size=5, stride=1, padding=2) #initially padding=1 kernel_size=3, without stride
        self.conv2.weight = PyroSample(dist.Normal(prior_mu, prior_sigma).expand([64, 32, 5, 5]).to_event(4))
        self.conv2.bias = PyroSample(dist.Normal(prior_mu, prior_sigma).expand([64]).to_event(1))

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = PyroModule[nn.Linear](64 * 16 * 16, num_classes)
        self.fc1.weight = PyroSample(dist.Normal(prior_mu, prior_sigma).expand([num_classes, 64 * 16 * 16]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(prior_mu, prior_sigma).expand([num_classes]).to_event(1))

    def forward(self, x, y=None):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        logits = self.fc1(x)
        
        if y is not None:
            with pyro.plate("data", x.shape[0]):
                pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        
        return logits
    

class BayesianCNNSingleFCCustom(PyroModule):
    def __init__(self, num_classes, mu, sigma, device):
        super().__init__()

        prior_mu = mu
        prior_sigma = torch.tensor(sigma, device=device)

        self.conv1 = PyroModule[nn.Conv2d](3, 32, kernel_size=5, stride=1, padding=2)
        self.conv1.weight = PyroSample(dist.Normal(prior_mu, prior_sigma).expand([32, 3, 5, 5]).to_event(4))
        self.conv1.bias = PyroSample(dist.Normal(prior_mu, prior_sigma).expand([32]).to_event(1))

        self.conv2 = PyroModule[nn.Conv2d](32, 64, kernel_size=5, stride=1, padding=2) #initially padding=1 kernel_size=3, without stride
        self.conv2.weight = PyroSample(dist.Normal(prior_mu, prior_sigma).expand([64, 32, 5, 5]).to_event(4))
        self.conv2.bias = PyroSample(dist.Normal(prior_mu, prior_sigma).expand([64]).to_event(1))

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = PyroModule[nn.Linear](64 * 16 * 16, num_classes)
        self.fc1.weight = PyroSample(dist.Normal(prior_mu, prior_sigma).expand([num_classes, 64 * 16 * 16]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(prior_mu, prior_sigma).expand([num_classes]).to_event(1))

    def forward(self, x, y=None):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        logits = self.fc1(x)
        
        if y is not None:
            with pyro.plate("data", x.shape[0]):
                pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        
        return logits
    
class LaplaceBayesianCNNSingleFCwDropout(PyroModule):
    def __init__(self, num_classes, device):
        super().__init__()

        prior_mu = 0.
        prior_b = torch.tensor(10., device=device)  # Scale parameter for Laplace

        self.conv1 = PyroModule[nn.Conv2d](3, 32, kernel_size=5, stride=1, padding=2)
        self.conv1.weight = PyroSample(dist.Laplace(prior_mu, prior_b).expand([32, 3, 5, 5]).to_event(4))
        self.conv1.bias = PyroSample(dist.Laplace(prior_mu, prior_b).expand([32]).to_event(1))

        self.conv2 = PyroModule[nn.Conv2d](32, 64, kernel_size=5, stride=1, padding=2)
        self.conv2.weight = PyroSample(dist.Laplace(prior_mu, prior_b).expand([64, 32, 5, 5]).to_event(4))
        self.conv2.bias = PyroSample(dist.Laplace(prior_mu, prior_b).expand([64]).to_event(1))

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = PyroModule[nn.Linear](64 * 16 * 16, num_classes)
        self.fc1.weight = PyroSample(dist.Laplace(prior_mu, prior_b).expand([num_classes, 64 * 16 * 16]).to_event(2))
        self.fc1.bias = PyroSample(dist.Laplace(prior_mu, prior_b).expand([num_classes]).to_event(1))

    def forward(self, x, y=None):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        logits = self.fc1(x)
        
        if y is not None:
            with pyro.plate("data", x.shape[0]):
                pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        
        return logits

class LaplaceBayesianCNNSingleFC(PyroModule):
    def __init__(self, num_classes, device):
        super().__init__()

        prior_mu = 0.
        prior_b = torch.tensor(10., device=device)  # Scale parameter for Laplace

        self.conv1 = PyroModule[nn.Conv2d](3, 32, kernel_size=5, stride=1, padding=2)
        self.conv1.weight = PyroSample(dist.Laplace(prior_mu, prior_b).expand([32, 3, 5, 5]).to_event(4))
        self.conv1.bias = PyroSample(dist.Laplace(prior_mu, prior_b).expand([32]).to_event(1))

        self.conv2 = PyroModule[nn.Conv2d](32, 64, kernel_size=5, stride=1, padding=2)
        self.conv2.weight = PyroSample(dist.Laplace(prior_mu, prior_b).expand([64, 32, 5, 5]).to_event(4))
        self.conv2.bias = PyroSample(dist.Laplace(prior_mu, prior_b).expand([64]).to_event(1))

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = PyroModule[nn.Linear](64 * 16 * 16, num_classes)
        self.fc1.weight = PyroSample(dist.Laplace(prior_mu, prior_b).expand([num_classes, 64 * 16 * 16]).to_event(2))
        self.fc1.bias = PyroSample(dist.Laplace(prior_mu, prior_b).expand([num_classes]).to_event(1))

    def forward(self, x, y=None):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        logits = self.fc1(x)
        
        if y is not None:
            with pyro.plate("data", x.shape[0]):
                pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        
        return logits
    
class LaplaceBayesianCNNSingleFCCustom(PyroModule):
    def __init__(self, num_classes, device, mu=0., b=10.):
        super().__init__()

        prior_mu = mu
        prior_b = torch.tensor(b, device=device)  # Scale parameter for Laplace

        self.conv1 = PyroModule[nn.Conv2d](3, 32, kernel_size=5, stride=1, padding=2)
        self.conv1.weight = PyroSample(dist.Laplace(prior_mu, prior_b).expand([32, 3, 5, 5]).to_event(4))
        self.conv1.bias = PyroSample(dist.Laplace(prior_mu, prior_b).expand([32]).to_event(1))

        self.conv2 = PyroModule[nn.Conv2d](32, 64, kernel_size=5, stride=1, padding=2)
        self.conv2.weight = PyroSample(dist.Laplace(prior_mu, prior_b).expand([64, 32, 5, 5]).to_event(4))
        self.conv2.bias = PyroSample(dist.Laplace(prior_mu, prior_b).expand([64]).to_event(1))

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = PyroModule[nn.Linear](64 * 16 * 16, num_classes)
        self.fc1.weight = PyroSample(dist.Laplace(prior_mu, prior_b).expand([num_classes, 64 * 16 * 16]).to_event(2))
        self.fc1.bias = PyroSample(dist.Laplace(prior_mu, prior_b).expand([num_classes]).to_event(1))

    def forward(self, x, y=None):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        logits = self.fc1(x)
        
        if y is not None:
            with pyro.plate("data", x.shape[0]):
                pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        
        return logits
    

if __name__ == "__main__":
    # if imported as a module, this block will not run
    model = BayesianCNNSingleFC()
    print(model)