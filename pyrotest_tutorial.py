import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import sklearn.metrics as metrics

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Subset
import pickle

from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from tqdm.auto import trange
pyro.clear_param_store()

from pyro.infer import Predictive

from tqdm import tqdm

class BayesianCNN(PyroModule):
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

        # Fully connected layer 1
        self.fc1 = PyroModule[nn.Linear](64 * 16 * 16, 128)
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([128, 64 * 16 * 16]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([128]).to_event(1))

        # Fully connected layer 2 (classifier)
        self.fc2 = PyroModule[nn.Linear](128, 10)
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([10, 128]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([10]).to_event(1))

    def forward(self, x, y=None):
        x = self.pool(F.relu(self.conv1(x)))   # -> (B, 32, 32, 32)
        x = self.pool(F.relu(self.conv2(x)))   # -> (B, 64, 16, 16)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)

        if y is not None:
            # During training: observe the labels
            with pyro.plate("data", x.size(0)):
                pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        # Always return logits (training or inference)
        return logits

def load_data(batch_size=54):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3444, 0.3809, 0.4082], std=[0.1809, 0.1331, 0.1137])
    ])

    dataset = datasets.EuroSAT(root='./data', transform=transform, download=True)


    #train_size = int(0.8 * len(dataset))
    #test_size = len(dataset) - train_size
    #train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    with open('datasplit/split_indices.pkl', 'rb') as f:
        split = pickle.load(f)
        train_dataset = Subset(dataset, split['train'])
        test_dataset = Subset(dataset, split['test'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader

model = BayesianCNN()

# Set Pyro random seed
pyro.set_rng_seed(42)

train_loader, test_loader = load_data()

################################################## train

mean_field_guide = AutoDiagonalNormal(model)
optimizer = pyro.optim.Adam({"lr": 0.01})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")  # Force CPU for this example
model.to(device)
mean_field_guide.to(device)
print(f"Using device: {device}")

svi = SVI(model, mean_field_guide, optimizer, loss=Trace_ELBO())
pyro.clear_param_store()

num_epochs = 1
progress_bar = trange(num_epochs)

full_batch = next(iter(train_loader))
x_train, y_train = full_batch

#for epoch in progress_bar:
#    loss = svi.step(x_train, y_train)
#    progress_bar.set_postfix(loss=f"{loss / x_train.shape[0]:.3f}")

for epoch in progress_bar:
    epoch_loss = 0
    for x_train, y_train in train_loader:
        loss = svi.step(x_train, y_train)
        epoch_loss += loss
    progress_bar.set_postfix(loss=f"{epoch_loss / len(train_loader.dataset):.3f}")

torch.save(model.state_dict(), 'pyro_result/bayesian_cnn.pth')

################################################## train

################################################## prediction

#x_test, y_test = next(iter(test_loader))
predictive = Predictive(model, guide=mean_field_guide, num_samples=5, return_sites=["_RETURN"])

all_preds = []
all_labels = []

for x_batch, y_batch in tqdm(test_loader):
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    batch_preds = predictive(x_batch)["_RETURN"]  # shape: (num_samples, batch_size, ...)
    batch_preds = torch.softmax(batch_preds, dim=-1)
    mean_preds = batch_preds.float().mean(dim=0).squeeze()  # shape: (batch_size, num_classes)
    mean_preds = mean_preds.argmax(dim=1)  # shape: (batch_size,)
    all_preds.append(mean_preds)
    all_labels.append(y_batch)

# Concatenate along the batch dimension
all_preds = torch.cat(all_preds, dim=0)   # shape: (total_test_samples, ...)
all_labels = torch.cat(all_labels, dim=0) # shape: (total_test_samples, ...)

################################################## prediction

accuracy = metrics.accuracy_score(all_labels.numpy(), all_preds.numpy())
print(f"Test Accuracy: {accuracy:.4f}")